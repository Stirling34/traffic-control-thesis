# -*- coding: utf-8 -*-
"""
Unattended experiment chain: build a job manifest, run it across N worker
processes, and leave a resumable record.

WHY IT IS BUILT THIS WAY
------------------------
The ledger records a completed 10-seed run being silently destroyed because two
processes held the same results list in memory and rewrote the whole file. So
there is exactly ONE writer per file here: worker k appends only to
`data/sweep/<exp>.shard<k>.json` and never reads or writes another worker's
shard. Merging happens at analysis time, in a separate read-only pass.

Workers do not talk to each other. Worker k takes job i whenever
`i % n_workers == k`, so no queue, no lock, no IPC to go wrong overnight. A
worker that dies takes only its own slice down; the rest keep going and the
slice is picked up on the next launch, because every shard is scanned for
already-completed job keys at startup.

Each run is wrapped: a SUMO crash, a TraCI timeout or an XML parse failure is
recorded as a failed row and the worker moves to the next job, rather than
ending a 48 h chain at hour three.

Usage:
    python control/sweep.py --manifest             # write data/sweep/jobs.json
    python control/sweep.py --worker 0 --workers 6 # run slice 0
    python control/sweep.py --status               # progress across all shards
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

SWEEP_DIR = os.path.join(_ROOT, "data", "sweep")
JOBS = os.path.join(SWEEP_DIR, "jobs.json")
MIN_FREE_GB = 2.0          # abort a worker rather than fill the disk

LONG_NET = os.path.join("sumo", "net", "plain.net3.xml")
LONG_CFG = os.path.join("sumo", "simulation3.sumocfg")
LONG_STEPS = 3500          # long road: vehicles clear well inside this
SHORT_STEPS = 2500
# straight jam-wave road, sumo/build_straight_net.py
STRAIGHT_NET = os.path.join("sumo", "net", "plain.net4.xml")
STRAIGHT_STEPS = 2700
STRAIGHT_STEPS_LONG = 3100


def job_key(j):
    """Identity of a job, for resume. Everything that changes the result.

    `max_iter` and `max_wall` are both in here deliberately. They are solver
    settings, not physics, but they demonstrably change the answer: on the long
    road the per-objective solver-failure rate correlates with the measured
    shockwave reduction at r = -0.87, so a row produced under a different solver
    budget is a different measurement and must not satisfy a resume check.
    """
    k = "|".join(str(j.get(k)) for k in
                 ("exp", "net", "cost", "seed", "Np", "instrument",
                  "max_wall", "max_iter", "b", "window", "drop_frac",
                  "model", "ctrl_interval",
                  # appended 2026-09-01. Both default to None, and `_norm_key`
                  # strips trailing Nones, so every key already on disk keeps
                  # the value it had -- see the note in `_norm_key` about the
                  # 270 finished jobs that were re-run when `model` was
                  # appended without that protection.
                  "dv_space", "alpha_cd",
                  # appended 2026-09-02, again last, for the same reason
                  "demand",
                  # appended 2026-09-11 (straight jam-wave road), last again
                  "dist", "gate",
                  # appended 2026-09-19 (rule-based comparator), last again
                  "rule",
                  # appended 2026-09-20 (deployable-actuator filter), last again
                  "quant", "hold",
                  # appended 2026-09-20 (driver response), last again
                  "lag", "comp",
                  # appended 2026-09-29 (driver-aware MPC), last again
                  "aware",
                  # appended 2026-10-05 (flat-objective test), last again
                  "vmin",
                  # appended 2026-10-05 (per-move rate limit), last again
                  "dvmax",
                  # appended 2026-10-06 (failure handling / IPOPT options)
                  "solver"))
    return _norm_key(k)


def _norm_key(k):
    """Drop trailing default (None) fields so the key stays stable as new job
    parameters are added.

    Without this, adding a parameter silently invalidates every key already on
    disk and the whole sweep re-runs: appending `model` and `ctrl_interval`
    turned every stored key into a non-match, and the workers began repeating
    270 finished jobs before it was spotted. A job that does not set a new
    parameter must keep the key it had before that parameter existed.
    """
    while k.endswith("|None"):
        k = k[:-5]
    return k


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------
def build_manifest(long_max_wall):
    """The experiment chain, in priority order.

    Ordered so that stopping at ANY point leaves a balanced, usable dataset:
    every block is complete in itself, and within a block seeds are interleaved
    across cost functions rather than run cost-by-cost, so an interruption
    costs seeds off every objective equally instead of losing whole objectives.
    """
    from control.cost_functions import COSTS
    jobs = []
    all_costs = list(COSTS)
    # the objectives worth the expensive follow-ups: the honest winner, the
    # worst metering offender, the safety-motivated pair, the incumbent
    top = ["tts", "accel_tts_queue", "accel_delay", "grad_dvsl", "risk_tts",
           "density_track"]

    def add(exp, net, cost, seed, **kw):
        j = dict(exp=exp, net=net, cost=cost, seed=seed,
                 instrument=kw.pop("instrument", False), Np=kw.pop("Np", None),
                 max_wall=kw.pop("max_wall", None),
                 max_iter=kw.pop("max_iter", None), b=kw.pop("b", None),
                 window=kw.pop("window", None))
        j.update(kw)
        jobs.append(j)

    # -- BLOCK 1: the long-road league table -------------------------------
    # The new headline. Does the ranking survive once the queue the controller
    # creates lands on measured road instead of vanishing into the origin?
    for seed in range(1, 16):
        add("long_table", "long", None, seed, max_wall=long_max_wall)
        for c in all_costs:
            add("long_table", "long", c, seed, max_wall=long_max_wall)

    # -- BLOCK 2: short road to 30 seeds -----------------------------------
    # Removes the "unequal sample sizes" limitation and makes the short-vs-long
    # comparison paired on the same seeds. Cheap: ~69 s a run, and no-control
    # plus six objectives are already at 30.
    done30 = {"nocontrol", "tts", "tts_dvsl", "delay_dvsl", "accel_delay",
              "accel_tts_queue"}
    for seed in range(11, 31):
        for c in all_costs:
            if c not in done30:
                add("short_table", "short", c, seed)

    # -- BLOCK 3: the prediction-horizon sweep -----------------------------
    # 30 s control interval x Np intervals. Np=7 (210 s) is the incumbent.
    # The ledger's earlier horizon test was run on the broken actuator, so it
    # has to be redone; doing it on both roads shows whether the answer is a
    # property of the controller or of the measurement boundary.
    for seed in range(1, 11):
        for Np in (2, 3, 4, 5, 7, 10, 14, 20):
            for c in ("tts", "accel_tts_queue"):
                add("horizon_long", "long", c, seed, Np=Np,
                    max_wall=long_max_wall)
    for seed in range(1, 11):
        for Np in (2, 3, 4, 5, 7, 10, 14, 20):
            for c in ("tts", "accel_tts_queue"):
                add("horizon_short", "short", c, seed, Np=Np)

    # -- BLOCK 4: long-road league table, seeds 16-30 ----------------------
    # Lifts block 1 to the 30 seeds the congestion metric needs (a p=0.002
    # result at n=10 already vanished once at n=30).
    for seed in range(16, 31):
        add("long_table", "long", None, seed, max_wall=long_max_wall)
        for c in all_costs:
            add("long_table", "long", c, seed, max_wall=long_max_wall)

    # -- BLOCK 5: safety instrumentation on the long road ------------------
    # Half the thesis title is safety, and every objective significantly
    # WORSENS the conflict surrogates on the short road. Whether that survives
    # once the displaced queue is on measured road is the open question.
    for seed in range(1, 31):
        add("long_ssm", "long", None, seed, instrument=True,
            max_wall=long_max_wall)
        for c in top:
            add("long_ssm", "long", c, seed, instrument=True,
                max_wall=long_max_wall)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_ref_manifest():
    """Round 2b: the constant-sign reference on the extended road.

    Every cost-function claim in this project has to be stated against a plain
    fixed speed limit -- no controller, no model -- or it is not evidence that
    closing the loop bought anything (ledger, 2026-08-19). That reference exists
    on the short road and does NOT exist on the long one, so right now no
    long-road result can be reported at all.

    Run here rather than through experiments/fixed_vsl.py so it is scored by the
    same code on the same seeds, and so it produces the whole-road/`_win` split
    the extended road needs. Both whole-run and windowed variants: a whole-run
    sign pays delay through the free-flow and drain phases where it prevents
    nothing, so the windowed one is the honest comparator for policy SHAPE while
    the whole-run one is the honest comparator for a deployable dumb policy.

    Solver-independent (there is no solver), so this is worth running whatever
    the solver-config test concludes.
    """
    # Windowing turns out to be a no-op on the extended road and is kept only as
    # a spot check. Measured on seed 1: whole-run and [120, 1850] give
    # bit-identical traffic (accel 0.2788, road 170.5 h) because vehicles do not
    # reach the controlled section until t ~ 153 s (3.6 km of approach at
    # 84.6 km/h) and the road has drained well before 1850 s, so the window
    # already contains every step at which the sign could matter. The short-road
    # distinction that made windowing necessary does not arise here. Three
    # b-values carry it at 30 seeds to confirm that rather than assume it.
    jobs = []
    for seed in range(1, 31):
        for b in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
            wins = [None, [120, 1850]] if b in (0.95, 0.85, 0.75) else [None]
            for win in wins:
                jobs.append(dict(exp="long_ref", net="long", cost=None,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=None, max_iter=None, b=b, window=win))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


FRONTIER_FAMILIES = ["accel", "accel_dens", "speed_grad", "conflict_risk",
                     "speed_var", "bneck_density"]
FRONTIER_LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def build_round3_manifest(max_wall, max_iter):
    """Round 3: the two gaps the round-1 results left open.

    (a) THE NEW OBJECTIVES. Five were added after round 1:
        `bneck`/`bneck_tts`/`bneck_tts_heavy` express "hold the merge at
        capacity", which is the control law the VSL literature is actually built
        on and which NOTHING in the original 26 could represent, since every
        term sums over all segments. `tts_grad` combines the only objective that
        improved the road with the only safety term that did not degrade it.
        `tts_road` is a diagnostic: it is `tts` with the origin queue deleted
        from the objective, and it tests whether `tts` avoids metering precisely
        BECAUSE it counts queued vehicles. That one is run on BOTH roads on
        purpose -- the prediction is that it meters hard on the short road and
        collapses back to `tts` on the long one, and only the pair demonstrates
        the mechanism.

    (b) THE WEIGHT FRONTIER. The largest stated limitation of the comparison is
        that weights were screened, not optimised: each objective was run at one
        arbitrary setting, so the table compares points, not families at their
        best. Sweeping each family's weight against a fixed `tts:1` traces a
        proper trade-off curve per family, which is what actually supports a
        claim of the form "family X dominates family Y".
    """
    jobs = []
    new_costs = ["bneck", "bneck_tts", "bneck_tts_heavy", "tts_grad", "tts_road"]

    def add(exp, net, cost, seed, mw, mi):
        jobs.append(dict(exp=exp, net=net, cost=cost, seed=seed,
                         instrument=False, Np=None, max_wall=mw, max_iter=mi,
                         b=None, window=None))

    for seed in range(1, 31):
        for c in new_costs:
            add("new_long", "long", c, seed, max_wall, max_iter)
    # Short road too: the metering signature only exists there, so the
    # `tts_road` mechanism test needs both halves. The others come along
    # cheaply (~92 s a run) and complete the short-road table.
    for seed in range(1, 31):
        for c in new_costs:
            add("new_short", "short", c, seed, None, None)

    for seed in range(1, 16):
        for fam in FRONTIER_FAMILIES:
            for lam in FRONTIER_LAMBDAS:
                spec = "%s:%g|tts:1|dvsl:0.1" % (fam, lam)
                add("frontier", "long", spec, seed, max_wall, max_iter)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round4_manifest(max_wall=15.0):
    """Round 4: three gaps that remain after the long-road comparison.

    (a) IS THE DUMB POLICY SAFER? Every objective significantly worsens every
        conflict surrogate (TET/TIT/DRAC, 30 seeds, both roads). The obvious
        next question has never been asked: does a plain constant sign do the
        same? If it does, the degradation belongs to slowing traffic at a merge
        in general and is not an indictment of MPC; if it does not, the
        controller is specifically worse and that is a finding the thesis has to
        carry. Instrumented, three sign levels spanning the range.

    (b) DOES THE RANKING DEPEND ON THE PREDICTION MODEL? The project's own
        ledger says cost-function performance is a property of the (model, cost)
        PAIR, measured at Spearman -0.17. But that was established on the BROKEN
        actuator, before `vsl_scaled`, and it is currently the single biggest
        caveat attached to every result. Re-testing it on the working
        configuration is the difference between "the ranking is model-specific"
        and "that was an artefact of a dead actuator". Same 10 objectives, same
        seeds, same road -- only the prediction model changes.

    (c) HOW OFTEN MUST THE SIGN UPDATE? 30 s throughout, never justified. This
        is the deployment-facing question a real gantry operator asks, and it is
        cheap. M is moved with the interval so the controller's own notion of a
        move matches how often the sign is actually re-posted.
    """
    jobs = []

    def add(exp, **kw):
        j = dict(exp=exp, net="long", cost=None, seed=None, instrument=False,
                 Np=None, max_wall=max_wall, max_iter=None, b=None, window=None,
                 drop_frac=0.0, model=None, ctrl_interval=None)
        j.update(kw)
        jobs.append(j)

    # (a) safety of the constant-sign reference -- instrumented, no controller
    for seed in range(1, 31):
        add("ref_ssm", seed=seed, instrument=True, b=None, max_wall=None)
        for b in (0.95, 0.85, 0.75):
            add("ref_ssm", seed=seed, instrument=True, b=b, max_wall=None)

    # (b) the same comparison on the other prediction model
    model_costs = ["tts", "tts_dvsl", "delay_dvsl", "accel_tts", "accel_delay",
                   "accel_tts_queue", "grad_dvsl", "grad_tts", "risk_tts",
                   "speedvar_tts", "density_track", "bneck_tts"]
    for seed in range(1, 16):
        add("model_s14", cost=None, seed=seed, model="s14")
        for c in model_costs:
            add("model_s14", cost=c, seed=seed, model="s14")

    # (c) control interval. 5 s model step, so these are M = 3, 6, 12, 24.
    for seed in range(1, 11):
        for iv in (15, 60, 120):
            for c in ("tts", "accel_tts_queue"):
                add("interval", cost=c, seed=seed, ctrl_interval=iv)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round5_manifest(max_wall=15.0):
    """Round 5: act on `docs/research/2026-09-01-why-the-frontier-is-one-dimensional.md`.

    That review concluded the empty bottom-left corner of `report_tradeoff*.png`
    is a MODEL-STRUCTURE result, not a cost-function or tuning result: the
    prediction model has no mechanism by which a speed limit can raise
    throughput, so every objective is confined to a single restriction axis.
    These nine blocks test that, and fix three verified defects found on the way.

    Ordered by value per hour, because the chain may be stopped before it
    empties, and emitted seed-major within each block so an early stop costs
    seeds per point rather than whole points.

    (1) frontier_lo -- FILL THE HOLE. The accel-vs-tts frontier jumps from
        lambda=0 (-0.3% accel, -8.6 s/veh, mean sign 91.3) to lambda=0.25
        (-8.2%, +42.2 s/veh, mean sign 78.5). ONE weight step moves the mean
        posted limit 12.8 km/h, and the zero-delay crossing -- the best
        available win-win point -- lies inside that unsampled interval. Every
        "no objective improves both" claim rests on a curve with a hole in the
        middle of it. Cheapest real chance of a bottom-left point.

    (2) bneck_fix -- THE CANONICAL VSL LAW WAS NEVER ACTUALLY TESTED.
        `term_bottleneck_density` normalised the BOTTLENECK's density by the
        MAINLINE rho_crit (35.14) instead of the bottleneck's own (49.40), so
        the hinge charged from 71% of the density that maximises discharge in
        the model -- ALINEA's "nonnecessary underload of the freeway". Runs the
        legacy (buggy) term, the fixed term, and the fixed term at 1.15x, with
        and without a mobility term. FINDINGS.md sec. 7d's claim that the
        canonical law is significantly worse than no control depends on this.

    (3) terminal -- THE HIGHEST-INFORMATION SINGLE TEST. Minimising TTS over Np
        moves weights discharge by (Np-1-k), so at Np=7 the weights are
        7,6,5,4,3,2,1 (verified numerically, NOT 6..0 as first written): the last
        move carries 3.6% of the weight and the final 60 s of the 210 s horizon
        carries 10.7%, while anything deferred BEYOND move Np-1 carries exactly
        zero. An objective that defers delay past the
        horizon collects it free. Charging the terminal state c times is exactly
        equivalent to extending the horizon by c frozen steps. If the
        acceleration family's smoothing survives c=5 it is real; if it
        collapses, the metering story is closed.

    (4) ref_mild -- the constant-sign reference has the same sampling defect:
        b=0.95 is both the mildest setting ever tested AND the only one that
        helps. Nothing between 0.95 and 1.00. Cheap (no controller).

    (5) ttd_rew -- THE LITERATURE'S OWN REMEDY FOR THIS EXACT PATHOLOGY. Han et
        al. 2017 add a small flow reward proportional to total travel distance
        "to address the so-called holding back problem". Note the identity:
        `tts:1|ttd_reward:1` IS `delay:1`, so the remedy already exists in the
        objective set at exactly one weight, as `delay_dvsl` -- one of only two
        objectives that reduce total time spent. Sweeping the weight is the test.

    (6) capdrop -- THE ROOT-CAUSE EXPERIMENT, POSSIBLE FOR THE FIRST TIME.
        `alpha_cd` was a SILENT NO-OP on Bounded-METANET: `_capacity_drop_scale`
        was never called from `_bounded_step`, and rollouts at 0 / 0.118 / 0.25
        were bit-identical. It is now wired in (parity at alpha_cd=0 held at
        5.8e-13). Hegyi (2004) sec. 5.1.1 makes reproducing the drop a NECESSARY
        condition for model-based control to find the right signals; Han et al.
        2017 show a controller whose model UNDERSTATED the drop failed to
        resolve a jam at all. Open-loop this still shows no win-win, because the
        drop is triggered by upstream density and a VSL RAISES that -- so the
        closed-loop question (can a controller that chooses WHERE and WHEN do
        better than a uniform sign?) is exactly what needs measuring.

    (7) spatial -- SAFETY AS A CONSTRAINT, NOT A COST. Measured: with no spatial
        bound the controller posts a 63 km/h drop between adjacent 200 m signs.
        Every source in the literature bounds this HARD (Hegyi 10 km/h over
        500 m; Carlson Delta b <= 0.2; I-24 and Oregon 10 mph) and none
        optimises a gradient cost -- which is why the `speed_grad` family was
        flat across a 16-fold weight sweep. 4 km/h per 200 m is Hegyi's rate.

    (8) bflow -- maximise bottleneck discharge, time-weighted so it is
        TTS-consistent rather than the degenerate flat form.

    (9) horizon2 -- the existing horizon block is n=10 and noisy, and `tts`'s
        mobility gain looked LARGEST at short horizons (Np=3: -13.3 s/veh
        against -2.2 at Np=7). 210 s is at Hegyi's stated minimum. Re-measure
        the two mobility objectives at 30 seeds.
    """
    jobs = []

    def add(exp, **kw):
        j = dict(exp=exp, net="long", cost=None, seed=None, instrument=False,
                 Np=None, max_wall=max_wall, max_iter=None, b=None, window=None,
                 drop_frac=0.0, model=None, ctrl_interval=None,
                 dv_space=None, alpha_cd=None)
        j.update(kw)
        jobs.append(j)

    # (1) the unsampled interval between lambda 0 and 0.25
    for seed in range(1, 31):
        for lam in ("0.02", "0.05", "0.1", "0.15", "0.2"):
            add("frontier_lo", cost="accel:%s|tts:1|dvsl:0.1" % lam, seed=seed)

    # (2) the bottleneck setpoint: buggy / fixed / fixed-and-raised
    for seed in range(1, 31):
        for c in ("bneck_legacy:1|dvsl:0.1",
                  "bneck_density:1|dvsl:0.1",
                  "bneck_hi:1|dvsl:0.1",
                  "bneck_legacy:1|tts:1|dvsl:0.1",
                  "bneck_density:1|tts:1|dvsl:0.1",
                  "bneck_hi:1|tts:1|dvsl:0.1"):
            add("bneck_fix", cost=c, seed=seed)

    # (3) terminal cost. c=0 is `accel_tts` / `tts_dvsl`, already measured.
    for seed in range(1, 31):
        for c in ("accel:1|tts:1|terminal:2|dvsl:0.1",
                  "accel:1|tts:1|terminal:5|dvsl:0.1",
                  "accel:1|tts:1|terminal:10|dvsl:0.1",
                  "tts:1|terminal:5|dvsl:0.1",
                  "accel_dens:2|tts:1|terminal:5|dvsl:0.1"):
            add("terminal", cost=c, seed=seed)

    # (4) the mild end of the constant-sign reference (no controller, cheap)
    for seed in range(1, 31):
        for b in (0.99, 0.98, 0.97, 0.96):
            add("ref_mild", seed=seed, b=b, max_wall=None)

    # (5) the anti-holding-back reward. 0 = accel_tts, 1 = accel_delay.
    for seed in range(1, 31):
        for c in ("accel:1|tts:1|ttd_reward:0.25|dvsl:0.1",
                  "accel:1|tts:1|ttd_reward:0.5|dvsl:0.1",
                  "accel:1|tts:1|ttd_reward:2|dvsl:0.1",
                  "accel_dens:2|tts:1|ttd_reward:1|dvsl:0.1"):
            add("ttd_rew", cost=c, seed=seed)

    # (6) the capacity drop, live at last. nocontrol does not use the model, so
    # the existing long_table baseline is the correct pairing partner.
    for seed in range(1, 21):
        for acd in (0.12, 0.25):
            for c in ("tts", "tts_dvsl", "accel_tts_queue",
                      "bneck_density:1|tts:1|dvsl:0.1"):
                add("capdrop", cost=c, seed=seed, alpha_cd=acd)

    # (7) the spatial gradient bound
    for seed in range(1, 21):
        for dvs in (4.0, 8.0):
            for c in ("tts", "accel_tts_queue", "accel_dens_tts", "grad_dvsl"):
                add("spatial", cost=c, seed=seed, dv_space=dvs)

    # (8) bottleneck discharge maximisation
    for seed in range(1, 31):
        for c in ("bneck_flow:1|dvsl:0.1",
                  "bneck_flow:1|tts:1|dvsl:0.1",
                  "bneck_flow:2|tts:1|dvsl:0.1"):
            add("bflow", cost=c, seed=seed)

    # (9) horizon, properly powered this time
    for seed in range(1, 31):
        for np_ in (3, 5, 10):
            add("horizon2", cost="tts", seed=seed, Np=np_)
    for seed in range(1, 21):
        for np_ in (3, 10):
            add("horizon2", cost="accel_tts_queue", seed=seed, Np=np_)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round6_manifest(max_wall=15.0):
    """Round 6: is the terminal-cost result a design principle or one lucky weight?

    Round 5 turned up the project's first free smoothing. `tts:1|terminal:5|dvsl:0.1`
    against `tts_dvsl`, directly paired on 30 common seeds:

        shockwave  -1.67%  p=0.0037   (22/30 seeds improved)
        congestion -2.35%  p=0.0405
        road time  -2.70 s/veh  p=0.31  (nominally BETTER)
        mean posted sign  -0.01 km/h  p=0.87  (IDENTICAL)

    That last line is why it matters. Every other objective in the project buys
    smoothing by moving along the restriction axis and pays for it in delay. This
    one does not move along that axis at all -- same mean sign to two decimals --
    and still smooths. It corroborates the 2026-08-19 finding that d_accel tracks
    WHERE and WHEN the restriction is placed rather than how low it is set.

    But it rests on a SINGLE weight. Weight 5 was chosen arbitrarily, and the
    project's own history is full of single settings that did not survive
    (`density_track`, `speedvar_tts`). The one thing that would make it a
    reportable design result rather than a lucky draw is MONOTONICITY: if the
    effect grows with the terminal weight and then saturates, it is a mechanism;
    if it is a spike at 5, it is noise dressed up.

    (a) terminal weight on `tts`, at 1 / 2 / 10 / 20 (5 already measured).
    (b) does it transfer to the OTHER mild objective that improves the road?
        `delay_dvsl` is the only other one, so `delay:1|terminal:5|dvsl:0.1`.
    (c) does it stack with the best-behaved safety term? `tts_grad` never got a
        terminal cost.
    """
    jobs = []

    def add(exp, **kw):
        j = dict(exp=exp, net="long", cost=None, seed=None, instrument=False,
                 Np=None, max_wall=max_wall, max_iter=None, b=None, window=None,
                 drop_frac=0.0, model=None, ctrl_interval=None,
                 dv_space=None, alpha_cd=None)
        j.update(kw)
        jobs.append(j)

    for seed in range(1, 31):
        for c in ("tts:1|terminal:1|dvsl:0.1",
                  "tts:1|terminal:2|dvsl:0.1",
                  "tts:1|terminal:10|dvsl:0.1",
                  "tts:1|terminal:20|dvsl:0.1",
                  "delay:1|terminal:5|dvsl:0.1",
                  "tts:1|speed_grad:1|terminal:5|dvsl:0.1"):
            add("terminal2", cost=c, seed=seed)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round7_manifest(max_wall=15.0):
    """Round 7: is the mild constant-sign effect real, or an artefact of the
    TraCI call itself?

    The four mild signs give -1.73 / -1.71 / -1.65 / -1.86 % for cuts of 1, 2, 3
    and 4 percent. That is FLAT: a four-fold change in restriction produces no
    change in outcome. A causal "lower limit smooths traffic" story predicts a
    dose-response and there is none, so the effect is not proportional to the
    speed reduction at all -- it behaves like a THRESHOLD that is already
    saturated at a 1% cut.

    One mechanism would explain a flat response: it is not the speed reduction
    doing the work, it is `traci.edge.setMaxSpeed` being CALLED. At b = 1.00 the
    runner never calls it (`tgt` equals the already-applied legal limit to within
    1e-9 and the write is skipped), so b=1.00 is provably bit-identical to no
    control and is useless as a placebo. b = 0.999 is a 0.09 km/h cut -- far
    below SUMO's Krauss dawdle amplitude and a twentieth of one speedFactor
    standard deviation, i.e. physically nothing -- but it DOES trigger the call.

    So: if b=0.999 also lands near -1.7%, the mild-sign result is an artefact of
    the actuation path and not of the speed limit, and every mild constant-sign
    number in this project is void. If it lands near zero, the effect is real and
    has a threshold somewhere between a 0.1% and a 1% cut, which is itself a
    finding worth reporting.
    """
    jobs = []
    for seed in range(1, 31):
        for b in (0.999, 0.995):
            jobs.append(dict(exp="placebo_mild", net="long", cost=None,
                             seed=seed, instrument=False, Np=None,
                             max_wall=None, max_iter=None, b=b, window=None,
                             drop_frac=0.0, model=None, ctrl_interval=None,
                             dv_space=None, alpha_cd=None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round8_manifest(max_wall=15.0):
    """Round 8: DOES THE ANSWER DEPEND ON HOW BUSY THE ROAD IS?

    Every result in this project was measured at ONE loading -- 4000 veh/h
    against a measured free-flow bottleneck capacity of 2434, i.e. d/C = 1.64.
    At that demand the queue is unavoidable, so the mechanism the VSL literature
    actually relies on (prevent breakdown, and with it the ~12% capacity drop)
    is not on the table at all. Round 5 measured the consequence directly:
    queued discharge does not respond to the posted limit, and switching the
    capacity-drop term on in the prediction model makes things WORSE, because
    the incentive is false at this demand.

    Two questions follow, and only a demand sweep answers either.

    (a) IS THE NULL A PROPERTY OF THE CONTROLLER OR OF THE SCENARIO? Near d/C = 1
    breakdown is marginal, so whether it happens at all is genuinely decidable by
    control -- and the capacity drop becomes recoverable, because the queue
    becomes avoidable. If the objectives do something there and nothing at 1.64,
    the negative headline is a statement about over-saturation, not about VSL.

    (b) IS ANY OF THIS DEPLOYABLE? Real demand is not constant; a fixed sign
    programme meets 85% and 140% of capacity within the same hour. A policy that
    helps at 120% and HURTS at 85% is unusable, and nothing in this project would
    currently detect that. This is the robustness axis, and it is arguably more
    important than the ranking.

    Five demands spanning d/C 0.86 to 1.64. Policies: the two mobility winners,
    the best smoothing objective, and BOTH references (no control, and the
    constant sign) at every demand -- the reference has to move with the demand
    or the comparison is meaningless.

    NOTE: vehicle count scales with demand (1000 s window), so per-vehicle
    metrics must be normalised by each run's own `throughput`, never by 1112.
    d=4000 is the control arm: its route file reproduces route.long.xml exactly,
    so those rows must match the existing `long_table` results.
    """
    jobs = []
    demands = [2100, 2440, 2920, 3400, 4000]
    costs = [None,                            # no control
             "tts",                           # best mobility
             "tts:1|terminal:5|dvsl:0.1",     # best mild, round-5 repair
             "accel_tts_queue"]               # best smoothing with a guard
    for seed in range(1, 16):
        for d in demands:
            for c in costs:
                jobs.append(dict(exp="demand", net="long", cost=c, seed=seed,
                                 instrument=False, Np=None, max_wall=max_wall,
                                 max_iter=None, b=None, window=None,
                                 drop_frac=0.0, model=None, ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=d))
            # the dumb-sign reference, at the same demand
            jobs.append(dict(exp="demand", net="long", cost=None, seed=seed,
                             instrument=False, Np=None, max_wall=None,
                             max_iter=None, b=0.95, window=None,
                             drop_frac=0.0, model=None, ctrl_interval=None,
                             dv_space=None, alpha_cd=None, demand=d))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round9_manifest(max_wall=15.0):
    """Round 9: rank every cost function on WAVE metrics, not on accel_rms.

    The research question is which cost function best reduces SHOCKWAVES. Every
    ranking so far has used `accel_rms`, which decomposes 93% into its SPATIAL
    term on the uncontrolled baseline -- the speed step a vehicle drives into.
    That is congestion, not wave propagation, and it is gameable by simply
    slowing traffic down (the acceleration family's headline falls almost
    exactly in proportion to how much it slows traffic).

    `validation/wave_metrics.py` now measures the congestion FRONT directly:
    how fast it propagates upstream, how far it reaches, how fast it recovers,
    and when it peaks. Measured on the uncontrolled long road, 4 seeds: the
    front travels upstream at -9.6 km/h, reaches 2.7 km, and a recovery wave
    sweeps forward at +10.2 km/h. That is the diagonal on a space-time plot and
    it is a shockwave in the ordinary kinematic-wave sense.

    Verified before spending runs: growth rate recovers EXACTLY on planted
    synthetic waves (-15/-9/-20 all recovered to the decimal), reach is accurate
    to one cell and is identical across v_c = 20/30/40, and recovery is accurate
    provided the wave does not hit the domain boundary. v_c=30 is primary;
    v_c=40 is too permissive (growth drifts to -5.8) and is a diagnostic only.

    None of the ~4800 stored rows carries these metrics, so the whole league
    table has to be re-run. Same objectives, same seeds, same road as
    `long_table`, so the two are directly comparable and the old accel_rms
    ranking can be set beside the new wave ranking objective for objective.

    15 seeds rather than 30: this is 27 objectives, and the front metrics are
    geometric rather than noisy aggregates -- reach was identical to 0.01 km
    across four seeds. Re-run the shortlist at 30 once the ranking is known.
    """
    from control.cost_functions import COSTS
    jobs = []
    costs = [None] + list(COSTS)
    for seed in range(1, 16):
        for c in costs:
            jobs.append(dict(exp="waves", net="long", cost=c, seed=seed,
                             instrument=False, Np=None, max_wall=max_wall,
                             max_iter=None, b=None, window=None,
                             drop_frac=0.0, model=None, ctrl_interval=None,
                             dv_space=None, alpha_cd=None, demand=None))
        # the two references have to be on the same figure
        for b in (0.95, 0.99):
            jobs.append(dict(exp="waves", net="long", cost=None, seed=seed,
                             instrument=False, Np=None, max_wall=None,
                             max_iter=None, b=b, window=None,
                             drop_frac=0.0, model=None, ctrl_interval=None,
                             dv_space=None, alpha_cd=None, demand=None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_round10_manifest(max_wall=15.0):
    """Round 10: the shockwave chain. Everything needed to answer the research
    question -- which cost function best reduces SHOCKWAVES -- and nothing else.

    Scope discipline, because this project has repeatedly drifted: no demand
    variation, no plant parameters (sigma, compliance), no activation logic, no
    capacity-drop mechanism, no new road. Those are scenario characterisation
    and belong in the limitations chapter. Every block below changes only the
    COST FUNCTION and is scored on wave metrics.

    Blocks, in priority order so an early stop still leaves a usable dataset:

    (1) waves      -- all 26 existing objectives + both references, on the wave
                      metrics for the first time. Seeds 1-15 are round 9 and
                      resume for free; this extends every one to 30, because the
                      front GROWTH rate varied -7.6 to -11.8 km/h across four
                      no-control seeds (about +-20%) even though REACH was
                      stable to 0.01 km. Growth needs the seeds; reach does not.

    (2) wavecost   -- THE NOVEL CONTRIBUTION. Six objectives built 2026-09-05
                      that target wave propagation rather than acceleration
                      magnitude. Nothing in the project has done this:
                        `accel_temporal`  the oscillation only, dropping the
                                          spatial term that is 93% of the
                                          headline and is standing-queue
                                          deceleration rather than a wave;
                        `upstream_queue`  congestion weighted by how far
                                          upstream it has spread -- aimed at
                                          front REACH, the most
                                          threshold-robust wave measure;
                        `dens_rate`       squared d(rho)/dt -- aimed at front
                                          GROWTH RATE, the speed of the wave.

    (3) wavefront  -- weight frontier on the two most promising new families.
                      Every previous frontier was swept on `accel`, which we now
                      know is mostly a slowing-traffic term.

    (4) wavessm    -- safety on the wave shortlist. The stated aim is mobility
                      AND safety, and no safety measurement exists for any
                      objective on the wave metrics. Instrumented, so slow;
                      last, so it degrades gracefully.
    """
    from control.cost_functions import COSTS
    jobs = []

    def add(exp, cost, seed, **kw):
        j = dict(exp=exp, net="long", cost=cost, seed=seed, instrument=False,
                 Np=None, max_wall=max_wall, max_iter=None, b=None, window=None,
                 drop_frac=0.0, model=None, ctrl_interval=None,
                 dv_space=None, alpha_cd=None, demand=None)
        j.update(kw)
        jobs.append(j)

    WAVE = ["accel_temporal:1|tts:1|dvsl:0.1",
            "accel_temporal:1|dvsl:0.1",
            "upstream_queue:1|tts:1|dvsl:0.1",
            "upstream_queue:2|tts:1|dvsl:0.1",
            "dens_rate:1|tts:1|dvsl:0.1",
            "dens_rate:1|accel_temporal:1|tts:1|dvsl:0.1"]

    # (1) the full league table on wave metrics, 30 seeds
    for seed in range(1, 31):
        for c in [None] + list(COSTS):
            add("waves", c, seed)
        for b in (0.95, 0.99):
            add("waves", None, seed, b=b, max_wall=None)

    # (2) the wave-targeting objectives
    for seed in range(1, 31):
        for c in WAVE:
            add("wavecost", c, seed)

    # (3) weight frontier on the two new families that have a natural scale
    for seed in range(1, 21):
        for lam in ("0.25", "0.5", "1", "2", "4"):
            add("wavefront", "accel_temporal:%s|tts:1|dvsl:0.1" % lam, seed)
            add("wavefront", "upstream_queue:%s|tts:1|dvsl:0.1" % lam, seed)

    # (4) safety on the shortlist, instrumented
    SSM = [None, "tts", "accel_tts_queue", "grad_dvsl", "density_track",
           "accel_temporal:1|tts:1|dvsl:0.1", "upstream_queue:1|tts:1|dvsl:0.1"]
    for seed in range(1, 21):
        for c in SSM:
            add("wavessm", c, seed, instrument=True)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


# THE DISTURBANCE IS ONE SLOW VEHICLE IN ONE LANE (the user's form), AND THE
# LOADING IS WHAT MAKES IT BITE. Measured, 3 seeds, 240 s hold:
#   at ~73% of capacity a single slow vehicle does nothing at 50% or 25% of the
#   limit, and gives a 91 s / 0.9 veh-h nothing at 15%;
#   at ~82% the SAME 15% vehicle gives a 640 s jam reaching 1.47 km (4600 veh/h)
#   and 1035 s / 3.40 km (4800), because arriving flow now matches the jam's
#   outflow -- the metastable condition the VSL jam-wave literature works in.
# 50% never produces congestion at any loading reachable here, in one lane or
# both: a platoon following at 42 km/h still passes ~2400 veh/h per lane against
# ~2350 arriving. The threshold is ~22% of the limit; 15% is the setting used.
# Getting there needed departSpeed="max" (see sumo/build_straight_net.py): with
# the merge's "speedLimit" the road could not be loaded past ~3530 veh/h at all.
#
# (Superseded, kept because the reasoning still applies to any closure-like
#  disturbance:)
# THE DISTURBANCE IS A SLOW ZONE, NOT A CLOSURE, AND THAT MATTERS.
# The first version stopped both lanes dead for 180 s. It plants a good jam, but
# a closure passes NO traffic, so the 0.8 km of modelled road downstream of it
# drains; the activation gate then switched every controller off for the whole
# jam and the sweep measured nothing (ledger 2026-09-11). Dropping the zone to
# 10 km/h instead still passes ~2400 veh/h against 2800-3500 arriving, so a queue
# builds and a wave forms, while the road stays occupied end to end. Measured,
# 3 seeds: jam 413-467 s, reach 0.8-1.07 km, 11-15 veh-h at 240 s hold -- the
# same size as the closure gave -- and the NMPC now solves in most of those
# states (14/21 against 0/12 under the closure).
# The plant clears every planted jam on its own (SUMO jam outflow ~4700 veh/h),
# so the test is whether a policy makes it clear FASTER / SMALLER, not whether it
# can remove a jam that would otherwise persist. 3500 is ~capacity: at 3600 the
# realised flow is already capped (~3525 veh/h).
STRAIGHT_DEMANDS = [4200, 4600, 4800]
STRAIGHT_DIST = dict(t=600, edge="d1", pos=100.0, v_frac=0.15, hold_s=240,
                     mode="vehicle", lanes=[0])
# Average the gate's density over one control interval. The instantaneous rule
# switches the controller off whenever ANY modelled cell is momentarily empty,
# which at 200 m happens in flowing traffic: 108-283 s of a 480 s window, gaps up
# to 25 s. Averaged: 0 s off in 5 of 6 seeds. See run_closed_loop.
STRAIGHT_GATE_AVG_S = 30


def build_straight_manifest(max_wall=15.0):
    """Round 11: the best cost functions on a road that CAN recover.

    Every earlier result is a 2 -> 1 merge at 164% of capacity, where the queue is
    physically unavoidable while demand lasts. Here the road is two lanes end to
    end (sumo/build_straight_net.py), demand is below capacity, and the wave is
    planted: at t = 600 s both lanes drop to STRAIGHT_DIST["v_kmh"] km/h for
    STRAIGHT_DIST["hold_s"] seconds just downstream of the signed section, then
    release. The zone still passes traffic (see STRAIGHT_DIST), so the road never
    empties and the controller stays active through the jam. The question is the one
    Hegyi et al. (2005, IEEE T-ITS) and SPECIALIST pose -- can signs upstream of a
    jam shorten it, i.e. does anything shorten the jam's life, reach or the
    vehicle-hours spent in it, relative to leaving it alone?

    Policies are the least-harmful objectives on the merge wave league table
    (`tts_road`, `tts_dvsl`, `dens_rate`, `tts`+terminal, `tts`, `delay_dvsl`),
    the old acceleration-metric winner with its guard (`accel_tts_queue`), both
    constant-sign references, and no control. Same controller settings as the
    `waves` block (Np 7, 30 s moves, max_wall 15, no spatial bound), so the only
    thing that differs from the merge table is the road. Demand is a quarter per
    flow, 2400 s window, per-vehicle metrics normalised by own throughput.
    """
    jobs = []
    costs = [None, "tts_road", "tts_dvsl", "dens_rate:1|tts:1|dvsl:0.1",
             "tts:1|terminal:5|dvsl:0.1", "tts", "delay_dvsl", "accel_tts_queue"]
    for seed in range(1, 11):
        for d in STRAIGHT_DEMANDS:
            for c in costs:
                jobs.append(dict(exp="straight", net="straight", cost=c,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=max_wall if c else None,
                                 max_iter=None, b=None, window=None,
                                 drop_frac=0.0, model=None, ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=d,
                                 dist=dict(STRAIGHT_DIST),
                                 gate=STRAIGHT_GATE_AVG_S))
            for b in (0.95, 0.99):
                jobs.append(dict(exp="straight", net="straight", cost=None,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=None, max_iter=None, b=b, window=None,
                                 drop_frac=0.0, model=None, ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=d,
                                 dist=dict(STRAIGHT_DIST),
                                 gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


STRAIGHT2_POLICIES = [None, "tts", "tts_dvsl", "dens_rate:1|tts:1|dvsl:0.1",
                      "tts:1|terminal:5|dvsl:0.1", "delay_dvsl",
                      "accel_tts_queue"]


def build_straight2_manifest(max_wall=15.0):
    """Round 12: the fuller picture on the straight road -- how the answer moves
    with LOADING and with DISTURBANCE SIZE.

    Round 11 measured three loadings and one disturbance, and found `tts` cutting
    jam vehicle-hours 41-59% (p <= 0.004) while beating the constant sign, with
    the effect growing as the road filled. Three things follow, and each is a
    block here.

    (a) `ladder` -- four more loadings (3800, 4000, 4400, 5000 veh/h) so the
        effect can be plotted against realised flow rather than asserted from
        three points. 5000 is the insertion ceiling (~4780 realised); below 4200
        the wave gets small, which is itself the interesting end: the claim is
        that control pays only once the road is loaded enough for a jam to
        persist, and that is testable here rather than assumed.

    (b) `strength` -- the disturbance at 10% and 20% of the limit beside the 15%
        already run, at 4600 veh/h. If the ranking holds across a disturbance
        three times the size, it is a property of the objectives; if it flips,
        round 11 measured one incident, not a policy.

    (c) `hold` -- the same 15% vehicle held for 120 s and 480 s against the 240 s
        already run. Duration and severity are different axes: a brief severe
        blockage and a long mild one make different jams.

    Policies are round 11's, minus `tts_road` (a diagnostic, and at 4800 it made
    the jam +116% while adding 40 veh-h of congestion elsewhere), plus both
    constant signs as references in the ladder. `accel_tts_queue` is kept
    precisely because it cheats -- it shrinks the planted jam while creating
    congestion elsewhere, and the `post` column is what shows that.
    """
    jobs = []

    def add(demand, cost, seed, dist, b=None):
        jobs.append(dict(exp="straight2", net="straight", cost=cost, seed=seed,
                         instrument=False, Np=None,
                         max_wall=max_wall if cost else None, max_iter=None,
                         b=b, window=None, drop_frac=0.0, model=None,
                         ctrl_interval=None, dv_space=None, alpha_cd=None,
                         demand=demand, dist=dist, gate=STRAIGHT_GATE_AVG_S))

    base = dict(STRAIGHT_DIST)
    for seed in range(1, 11):
        # (a) the loading ladder, at the round-11 disturbance
        for d in (3800, 4000, 4400, 5000):
            for c in STRAIGHT2_POLICIES:
                add(d, c, seed, dict(base))
            for b in (0.95, 0.99):
                add(d, None, seed, dict(base), b=b)
        # (b) disturbance size, at the round-11 loading
        for frac in (0.10, 0.20):
            dist = dict(base, v_frac=frac)
            for c in STRAIGHT2_POLICIES:
                add(4600, c, seed, dist)
            add(4600, None, seed, dist, b=0.95)
        # (c) disturbance duration
        for hold in (120, 480):
            dist = dict(base, hold_s=hold)
            for c in ("tts", "tts_dvsl", None):
                add(4600, c, seed, dist)
            add(4600, None, seed, dist, b=0.95)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight3_manifest(max_wall=15.0):
    """Round 13: is `tts`'s win a TUNING artefact?

    Round 12 leaves `tts` cutting jam vehicle-hours 41-58% across seven loadings
    while every other objective decays to nothing. Two ways that could be less
    than it looks, and both are cheap to test at the two loadings where the
    effect is largest and the entrance queue is still zero (4600, 4800):

    (a) HORIZON. The merge work found the horizon decides which axis a controller
        wins on -- short horizons bought mobility, long ones bought smoothing --
        and Np=7 has been fixed throughout this project. If `tts`'s advantage is
        really a horizon effect, other objectives at a better horizon should
        close the gap, and `tts` itself should move a lot with Np.

    (b) WEIGHTS. Every objective was run at ONE arbitrary weighting, the standing
        limitation of the whole comparison. The restraint term (`dvsl`) is swept
        25-fold, a terminal term is added, and a small acceleration term is
        blended in -- if any of those beats plain `tts`, the league table is
        comparing points rather than families at their best.

    Baselines are not re-run: these share the round-11/12 disturbance and demands,
    so the no-control and constant-sign rows in those blocks are the pairing
    partners (`exp` keeps the blocks apart in analysis).
    """
    jobs = []

    def add(demand, cost, seed, Np=None):
        jobs.append(dict(exp="straight3", net="straight", cost=cost, seed=seed,
                         instrument=False, Np=Np, max_wall=max_wall,
                         max_iter=None, b=None, window=None, drop_frac=0.0,
                         model=None, ctrl_interval=None, dv_space=None,
                         alpha_cd=None, demand=demand, dist=dict(STRAIGHT_DIST),
                         gate=STRAIGHT_GATE_AVG_S))

    WEIGHTS = ["tts:1|dvsl:0.02",            # restraint 5x lighter
               "tts:1|dvsl:0.5",             # restraint 5x heavier
               "tts:1|terminal:10|dvsl:0.1",  # the round-5 repair, heavier
               "accel:0.25|tts:1|dvsl:0.1"]  # a little smoothing blended in
    for seed in range(1, 11):
        for d in (4600, 4800):
            for np_ in (3, 5, 10):           # 7 is the default, already run
                add(d, "tts", seed, Np=np_)
            for c in WEIGHTS:
                add(d, c, seed)
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight4_manifest(max_wall=15.0):
    """Round 14: is the league table really about the RESTRAINT TERM?

    Round 13 found the `dvsl` weight maps monotonically onto the result for the
    `tts` family (0 -> -58%, 0.02 -> -24%, 0.1 -> -11%, 0.5 -> -14% at 4600), and
    plain `tts` -- the one objective in the table carrying NO dvsl term -- is
    also the only one that works. That is either a property of the tts term or a
    property of being free to move the signs quickly, and the two have never been
    separated: every other objective in the table carries `dvsl:0.1`.

    So run the others WITHOUT it. If they rise towards `tts`, the ranking was
    largely a restraint artefact and the write-up has to say so; if they stay
    flat, `tts` really is the better objective.
    """
    jobs = []
    pairs = [("dens_rate:1|tts:1|dvsl:0.1", "dens_rate:1|tts:1"),
             ("delay_dvsl", "delay:1"),
             ("accel_tts_queue", "accel:1|tts:1|queue:0.5"),
             ("tts:1|terminal:5|dvsl:0.1", "tts:1|terminal:5")]
    for seed in range(1, 11):
        for d in (4600, 4800):
            for _, bare in pairs:
                jobs.append(dict(exp="straight4", net="straight", cost=bare,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=max_wall, max_iter=None, b=None,
                                 window=None, drop_frac=0.0, model=None,
                                 ctrl_interval=None, dv_space=None,
                                 alpha_cd=None, demand=d,
                                 dist=dict(STRAIGHT_DIST),
                                 gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight5_manifest(max_wall=15.0):
    """Round 15: does the answer hold under a model that predicts THIS road?

    Every straight-road result so far runs `PARAMS_BM`, fitted on the merge.
    `PARAMS_BM_STRAIGHT` is the refit: held-out objective 1.92 -> 1.46 and 210 s
    prediction RRMSE 0.25-0.32 -> 0.18-0.24, with capacity corrected from 3714 to
    5900 veh/h. Its jam discharge is still ~4x too slow, so this is not a fixed
    model -- it is a better one, and the question is whether the controller's
    advantage depends on which of the two it plans with.

    The four objectives that actually resolve the jam (round 14: the restraint
    term removed), at the two clean loadings. No-control and constant-sign rows
    are model-independent and already exist in rounds 11-12, so they are not
    re-run; the analysis pairs against them.
    """
    jobs = []
    costs = ["tts", "tts:1|terminal:5", "dens_rate:1|tts:1",
             "accel:1|tts:1|queue:0.5"]
    for seed in range(1, 11):
        for d in (4600, 4800):
            for c in costs:
                jobs.append(dict(exp="straight5", net="straight", cost=c,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=max_wall, max_iter=None, b=None,
                                 window=None, drop_frac=0.0,
                                 model="bm_straight", ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=d,
                                 dist=dict(STRAIGHT_DIST),
                                 gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


# The objectives that actually resolve the jam once the restraint term is gone
# (round 14). The `dvsl`-carrying versions are handicapped, so the round-12
# severity and duration blocks -- which used them -- have to be redone with
# these before their conclusions mean anything.
STRAIGHT_FREE = ["tts", "tts:1|terminal:5", "dens_rate:1|tts:1"]


def build_straight6_manifest(max_wall=15.0):
    """Round 16: the disturbance axes, done properly, plus the missing horizons.

    What rounds 11-15 left open, and why each matters:

    (a) SEVERITY, with objectives that work. Round 12 swept 10/15/20% but with
        the dvsl-carrying objectives, which round 14 showed do almost nothing --
        so it measured handicapped controllers.
    (b) DURATION, same problem: 120/240/480 s was run for `tts` and `tts_dvsl`
        only.
    (c) ONSET RATE -- a new axis, never measured. Every disturbance so far brakes
        at the vehicle's limit, i.e. abruptly. `ramp_s` eases the same vehicle to
        the same speed over 30 or 90 s, which is the same severity and duration
        with a different shape. A controller with a 210 s horizon might be
        expected to handle a gradual onset better; nothing tests that yet.
    (d) HORIZON for the OTHER two objectives that work. Np was swept on `tts`
        only, and `tts:1|terminal:5` is the cheapest policy found (-57% for
        +9.5 s/veh), so its horizon sensitivity matters as much.

    All at 4600 veh/h, the loading with the largest effect and a zero entrance
    queue. Baselines: (a) and (b) reuse round 12's no-control rows in those
    cells; (c) is a new cell, so it carries its own no-control and constant-sign
    references.
    """
    jobs = []

    def add(cost, seed, dist, b=None, Np=None):
        jobs.append(dict(exp="straight6", net="straight", cost=cost, seed=seed,
                         instrument=False, Np=Np,
                         max_wall=max_wall if cost else None, max_iter=None,
                         b=b, window=None, drop_frac=0.0, model=None,
                         ctrl_interval=None, dv_space=None, alpha_cd=None,
                         demand=4600, dist=dist, gate=STRAIGHT_GATE_AVG_S))

    base = dict(STRAIGHT_DIST)
    for seed in range(1, 11):
        # (a) severity
        for frac in (0.10, 0.20):
            for c in STRAIGHT_FREE:
                add(c, seed, dict(base, v_frac=frac))
        # (b) duration
        for hold in (120, 480):
            for c in STRAIGHT_FREE:
                add(c, seed, dict(base, hold_s=hold))
        # (c) onset rate -- new cells, so they need their own references
        for ramp in (30, 90):
            d = dict(base, ramp_s=ramp)
            for c in STRAIGHT_FREE:
                add(c, seed, d)
            add(None, seed, d)
            add(None, seed, d, b=0.95)
        # (d) horizon for the other two
        for c in ("tts:1|terminal:5", "dens_rate:1|tts:1"):
            for np_ in (5, 10):
                add(c, seed, dict(base), Np=np_)

    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight7_manifest(max_wall=15.0):
    """Round 17: HOW QUICKLY the disturbance arrives, measured properly.

    Round 16 tried this on the standard road and the result was an artefact: a
    vehicle easing to 15% of the limit over 90 s covers ~1.2 km getting there,
    but the standard road has only 0.9 km past the trigger, so it drove off the
    end and NO jam formed at all (measured: 0 s below 30 km/h, against 413 s for
    the abrupt case; the 30 s ramp was truncated too, 334 s). That is a property
    of the road's length, not of onset rate.

    So this runs on `straight_long` (24 downstream segments, 4.5 km past the
    trigger -- room for the full ramp plus the 240 s hold), with the ABRUPT case
    re-run on the same road as the control. The modelled stretch is unchanged, so
    the controller problem is identical; only the scored road is longer, which is
    why these rows are a separate `net` and never pool with the 6.6 km ones.
    """
    jobs = []
    for seed in range(1, 11):
        for ramp in (0, 30, 90):
            d = dict(STRAIGHT_DIST, ramp_s=ramp)
            for c in STRAIGHT_FREE:
                jobs.append(dict(exp="straight7", net="straight_long", cost=c,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=max_wall, max_iter=None, b=None,
                                 window=None, drop_frac=0.0, model=None,
                                 ctrl_interval=None, dv_space=None,
                                 alpha_cd=None, demand=4600, dist=d,
                                 gate=STRAIGHT_GATE_AVG_S))
            for b in (None, 0.95):
                jobs.append(dict(exp="straight7", net="straight_long", cost=None,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=None, max_iter=None, b=b, window=None,
                                 drop_frac=0.0, model=None, ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=4600,
                                 dist=d, gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight8_manifest(max_wall=15.0):
    """Round 18: SAFETY on the recoverable-jam road.

    Half the project title is safety and this scenario has no safety numbers.
    The merge road's answer was that every objective makes every conflict
    surrogate significantly WORSE, with a mechanism attached: r(TIT, front
    reach) = +0.66, i.e. a longer front puts more vehicles at a queue tail. Here
    the front is SHORTER under control, so that mechanism predicts the OPPOSITE
    sign. That is a sharp, falsifiable prediction and the reason this is the
    most valuable run left.

    Instrumented (SSM device) at both clean loadings, on the objectives that
    resolve the jam plus the one that cheats and both references. Np is left at
    the project default so these pair with the round 11-16 rows.

    SSM files are ~90 MB and go to %TEMP% (never the OneDrive tree -- ledger
    2026-09-01); `run_closed_loop` deletes them in a `finally`.
    """
    jobs = []
    costs = [None] + STRAIGHT_FREE + ["accel:1|tts:1|queue:0.5"]
    for seed in range(1, 11):
        for d in (4600, 4800):
            for c in costs:
                jobs.append(dict(exp="straight8", net="straight", cost=c,
                                 seed=seed, instrument=True, Np=None,
                                 max_wall=max_wall if c else None,
                                 max_iter=None, b=None, window=None,
                                 drop_frac=0.0, model=None, ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=d,
                                 dist=dict(STRAIGHT_DIST),
                                 gate=STRAIGHT_GATE_AVG_S))
            jobs.append(dict(exp="straight8", net="straight", cost=None,
                             seed=seed, instrument=True, Np=None, max_wall=None,
                             max_iter=None, b=0.95, window=None, drop_frac=0.0,
                             model=None, ctrl_interval=None, dv_space=None,
                             alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
                             gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight9_manifest(max_wall=15.0):
    """Round 19: THE RULE-BASED COMPARATOR, tuned.

    Everything measured so far is MPC against no control or against a constant
    sign, and a constant sign never moves, so it resolves nothing at any onset.
    The benchmark a reviewer will expect is the field's own heuristic: detect
    the jam, post a low limit on the signs behind its upstream front, release
    when it clears (SPECIALIST, Hegyi & Hoogendoorn 2008). This is that rule --
    not a reimplementation of the published algorithm (no solvability test, no
    wave-speed geometry, the zone does not travel).

    The comparison is only worth anything if the rule is at its BEST, so its
    three free parameters are swept rather than guessed: how low to post, how
    many signs, at both clean loadings. A rule run uses no model and no solver,
    so these are ~50 s each.

    Paired against the round 11/16 rows on the same seeds, same demands, same
    disturbance -- nothing in this block changes the scenario.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for v_post in (40.0, 60.0, 80.0):
                for n_signs in (2, 4, 9):
                    jobs.append(dict(
                        exp="straight9", net="straight", cost=None, seed=seed,
                        instrument=False, Np=None, max_wall=None,
                        max_iter=None, b=None, window=None, drop_frac=0.0,
                        model=None, ctrl_interval=None, dv_space=None,
                        alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
                        gate=STRAIGHT_GATE_AVG_S,
                        rule=dict(v_post=v_post, n_signs=n_signs, hold_s=120,
                                  v_det=35.0, avg_s=30)))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight10_manifest(max_wall=15.0):
    """Round 20: REPEATED disturbances.

    Every straight-road result so far plants ONE slowdown on an otherwise clean
    road. Real motorways produce waves in sequence, and the interesting failure
    mode is compounding: a controller that resolves wave 1 by holding traffic
    upstream may hand wave 2 a worse road than it would have had. Three
    patterns at the same severity, so only the spacing changes:
      double  420 s apart -- the first jam has cleared
      triple  300 s apart x3 -- barely time to recover between them
      close   180 s apart -- the second lands while the first is still alive

    One loading (4600, the headline cell) to keep this block to 150 runs; the
    policies are the three that resolve the single jam plus both references.
    """
    jobs = []
    pats = {
        "double": [600, 1020],
        "triple": [600, 900, 1200],
        "close": [600, 780],
    }
    for seed in range(1, 11):
        for name in ("double", "triple", "close"):
            ds = [dict(STRAIGHT_DIST, t=t) for t in pats[name]]
            base = dict(exp="straight10", net="straight", seed=seed,
                        instrument=False, Np=None, max_iter=None, window=None,
                        drop_frac=0.0, model=None, ctrl_interval=None,
                        dv_space=None, alpha_cd=None, demand=4600, dist=ds,
                        gate=STRAIGHT_GATE_AVG_S, pattern=name)
            jobs.append(dict(base, cost=None, b=None, max_wall=None))
            for c in STRAIGHT_FREE:
                jobs.append(dict(base, cost=c, b=None, max_wall=max_wall))
            jobs.append(dict(base, cost=None, b=0.95, max_wall=None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight11_manifest(max_wall=15.0):
    """Round 21: WHERE the disturbance happens.

    The headline result plants the slowdown at d1, 0.3 km downstream of the
    last sign: the jam forms off the signed road and grows back onto it, which
    is the case VSL is deployed for. Two things are unmeasured and both are
    obvious questions:
      * does control still work when the jam starts INSIDE the signed stretch
        (c4, c8), where there is little or no road left upstream to meter with
        -- or does the sign then simply sit inside the queue?
      * does it decay with distance (d3, d5: 0.7 and 1.1 km further on), where
        the wave has longer to establish before it reaches any sign?

    Four locations x five policies x ten seeds at 4600. d1 itself is not
    repeated -- those rows already exist and are the reference column.
    """
    jobs = []
    for seed in range(1, 11):
        for edge in ("c4", "c8", "d3", "d5"):
            ds = dict(STRAIGHT_DIST, edge=edge)
            base = dict(exp="straight11", net="straight", seed=seed,
                        instrument=False, Np=None, max_iter=None, window=None,
                        drop_frac=0.0, model=None, ctrl_interval=None,
                        dv_space=None, alpha_cd=None, demand=4600, dist=ds,
                        gate=STRAIGHT_GATE_AVG_S)
            jobs.append(dict(base, cost=None, b=None, max_wall=None))
            for c in STRAIGHT_FREE:
                jobs.append(dict(base, cost=c, b=None, max_wall=max_wall))
            jobs.append(dict(base, cost=None, b=0.95, max_wall=None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight12_manifest(max_wall=15.0):
    """Round 22: report item 6 -- 30 SEEDS on the headline cells, not 10.

    Insurance, not discovery: the effect is large and every "without dvsl"
    entry in Table restraint is p=0.002, the ceiling a 10-seed paired Wilcoxon
    can report (every pair the same sign). That is also exactly the situation
    that burned this project once already -- a p=0.002 shockwave result at
    n=10 that reversed sign at n=30 (ledger, merge road, 2026-08). This does
    not re-run those 10 seeds; it adds seeds 11-30 to the same cells so the
    existing rows and these pool into one 30-seed comparison.

    The headline cells: STRAIGHT_FREE (the 3 objectives without a restraint
    term -- the ones the restraint-term finding says actually matter) plus no
    control and b=0.95, at the two loadings Table restraint reports (4600 and
    4800). 5 policies x 2 demands x 20 new seeds = 200 runs, matching the
    report's estimate.
    """
    jobs = []
    costs = [None] + STRAIGHT_FREE
    for seed in range(11, 31):
        for d in (4600, 4800):
            for c in costs:
                jobs.append(dict(exp="straight12", net="straight", cost=c,
                                 seed=seed, instrument=False, Np=None,
                                 max_wall=max_wall if c else None,
                                 max_iter=None, b=None, window=None,
                                 drop_frac=0.0, model=None, ctrl_interval=None,
                                 dv_space=None, alpha_cd=None, demand=d,
                                 dist=dict(STRAIGHT_DIST),
                                 gate=STRAIGHT_GATE_AVG_S))
            jobs.append(dict(exp="straight12", net="straight", cost=None,
                             seed=seed, instrument=False, Np=None, max_wall=None,
                             max_iter=None, b=0.95, window=None, drop_frac=0.0,
                             model=None, ctrl_interval=None, dv_space=None,
                             alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
                             gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight13_manifest(max_wall=15.0):
    """Round 23: the rule comparator, scored HONESTLY, plus a release sweep.

    Round 19 picked the rule's parameters on the same 10 seeds it was then
    scored on, which is selection on the test set and flatters it. The MPC now
    has 30 seeds at the headline cells (round 22), so the clean split is
    available for free: TUNE on seeds 1-10 (already run), SCORE on 11-30, where
    both sides are out of sample for the rule's parameter choice.

    The hold sweep is the second half. `hold_s` was fixed at 120 s on a guess,
    and the whole finding -- that a brief, deep, local restriction matches the
    MPC at no delay -- is a claim about release timing, so it has to be shown
    that it is not an artefact of one value. 60 s and 240 s on the two best
    zone shapes.
    """
    jobs = []

    def add(seed, d, v_post, n_signs, hold):
        jobs.append(dict(
            exp="straight13", net="straight", cost=None, seed=seed,
            instrument=False, Np=None, max_wall=None, max_iter=None, b=None,
            window=None, drop_frac=0.0, model=None, ctrl_interval=None,
            dv_space=None, alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
            gate=STRAIGHT_GATE_AVG_S,
            rule=dict(v_post=v_post, n_signs=n_signs, hold_s=hold,
                      v_det=35.0, avg_s=30)))

    # (a) the full round-19 grid on the held-out seeds
    for seed in range(11, 31):
        for d in (4600, 4800):
            for v_post in (40.0, 60.0, 80.0):
                for n_signs in (2, 4, 9):
                    add(seed, d, v_post, n_signs, 120)
    # (b) release timing, on the two zone shapes that won at 4600/4800
    for seed in range(1, 11):
        for d in (4600, 4800):
            for v_post, n_signs in ((40.0, 9), (60.0, 9)):
                for hold in (60, 240):
                    add(seed, d, v_post, n_signs, hold)
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight14_manifest(max_wall=15.0):
    """Round 24: does a LONGER HORIZON make the MPC surgical?

    Measured mechanism behind round 19 (4600 veh/h): the rule holds a mean
    posted limit of 82.1 km/h against the legal 84.6 -- it restricts almost
    nothing on average and 40 km/h exactly where and when the jam is. `tts`
    holds 74.2 km/h, a broad restriction over the whole road for the whole run.
    Both remove the jam; only the MPC charges +14 s/veh for it. So the gap is
    not optimisation quality, it is TARGETING.

    Np=10 is the test. Round 14 found it gives the same jam reduction for a
    third of the delay, which is the same axis: a longer horizon should let the
    controller wait and act late rather than restrain early. If Np=10 closes the
    delay gap to the rule, the story is "the horizon was too short"; if it does
    not, the story is that a model-based controller optimising a road-wide
    objective cannot help but restrict road-wide, and that is the finding.

    All three restraint-free objectives, both loadings, 10 seeds. These pair
    with the Np=7 rows already on disk.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for c in STRAIGHT_FREE:
                jobs.append(dict(
                    exp="straight14", net="straight", cost=c, seed=seed,
                    instrument=False, Np=10, max_wall=max_wall, max_iter=None,
                    b=None, window=None, drop_frac=0.0, model=None,
                    ctrl_interval=None, dv_space=None, alpha_cd=None,
                    demand=d, dist=dict(STRAIGHT_DIST),
                    gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight15_manifest(max_wall=15.0):
    """Round 25: IS THE RESULT ABOUT AN ACTUATOR THAT CAN EXIST?

    The straight-road headline is measured with a sign that moves at 73 % of
    30 s steps by a median of 2.4 km/h. No gantry can show 2.4 km/h. Filtering
    the posted value to 10 km/h steps held 60 s cuts `tts` from -75 % to -30 %
    on one seed, while the rule-based comparator is untouched (it already posts
    one 44.6 km/h jump held a minute) -- so on deployable actuation the rule
    wins by a much wider margin than on idealised actuation.

    But that filter is an UNFAIR test and must not be the last word: the
    controller plans a continuous limit and reality then rounds it, so it is
    optimising for an actuator it does not have. The honest question is whether
    an actuator-AWARE controller recovers the result. Four blocks:

    (a) `quant`  -- 10 km/h steps, no hold. Magnitude granularity alone.
    (b) `hold`   -- continuous values, 60 s minimum hold. Timing granularity
                    alone. (a) vs (b) says WHICH granularity the result needs,
                    which no measurement currently distinguishes.
    (c) `both`   -- the realistic sign.
    (d) `block`  -- MOVE BLOCKING: the controller replans on a 60/120 s grid, so
                    the sign only changes when it actually can and the NLP knows
                    it. `M` is adjusted with the interval (see run_closed_loop),
                    so the horizon covers the same wall-clock time. This is the
                    fair actuator-aware test, and the straight road has NO
                    control-interval variants at all today.

    Plus the rule under (c), which should be unaffected and is the control on
    the whole experiment, and SAFETY for the rule, which has zero instrumented
    runs despite currently being the best-performing policy on this road.
    """
    jobs = []

    def add(cost, seed, d, **kw):
        j = dict(exp="straight15", net="straight", cost=cost, seed=seed,
                 instrument=False, Np=None, max_wall=max_wall if cost else None,
                 max_iter=None, b=None, window=None, drop_frac=0.0, model=None,
                 ctrl_interval=None, dv_space=None, alpha_cd=None, demand=d,
                 dist=dict(STRAIGHT_DIST), gate=STRAIGHT_GATE_AVG_S)
        j.update(kw)
        jobs.append(j)

    for seed in range(1, 11):
        for d in (4600, 4800):
            for c in STRAIGHT_FREE:
                add(c, seed, d, quant=10.0)                      # (a)
                add(c, seed, d, hold=60.0)                       # (b)
                add(c, seed, d, quant=10.0, hold=60.0)           # (c)
                add(c, seed, d, ctrl_interval=60)                # (d)
                add(c, seed, d, ctrl_interval=120)
            # the comparator through the same filter: it should not move
            for v_post, n_signs in ((40.0, 9), (60.0, 9)):
                add(None, seed, d, quant=10.0, hold=60.0,
                    rule=dict(v_post=v_post, n_signs=n_signs, hold_s=120,
                              v_det=35.0, avg_s=30))
            # SAFETY for the rule -- it has none, and it is currently winning
            add(None, seed, d, instrument=True,
                rule=dict(v_post=40.0 if d == 4600 else 60.0, n_signs=9,
                          hold_s=120, v_det=35.0, avg_s=30))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight16_manifest(max_wall=15.0):
    """Round 26: the `restrict` term at weights that can actually trade off.

    The pilot put `restrict` at 0.5 and `restrict_l1` at 0.5/2. All three drove
    the mean posted limit to the legal 84.6 km/h -- the term simply switched the
    controller off, and the jam reduction fell to -8 %. That is the term working
    as written and the weight being far too high: it prices ALL restriction
    equally, including the restriction that is paying for itself, so the
    cheapest move is to stop controlling.

    So this is a one-dimensional weight sweep, two orders of magnitude below the
    pilot, to find where the trade-off actually bites -- and whether any weight
    reaches the rule's operating point (mean posted ~82 km/h, jam -64 %, delay
    ~0) which is the whole point of adding the term.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for form in ("restrict", "restrict_l1"):
                for w in (0.01, 0.03, 0.1, 0.3):
                    jobs.append(dict(
                        exp="straight16", net="straight",
                        cost="tts:1|%s:%g" % (form, w), seed=seed,
                        instrument=False, Np=None, max_wall=max_wall,
                        max_iter=None, b=None, window=None, drop_frac=0.0,
                        model=None, ctrl_interval=None, dv_space=None,
                        alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
                        gate=STRAIGHT_GATE_AVG_S))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_mergerule_manifest(max_wall=15.0):
    """Round 27: the rule-based comparator ON THE MERGE ROAD.

    The thesis compares two roads against the same baselines, and the reactive
    rule currently exists on only one of them. Without this the 2x2 has a hole
    exactly where the interesting asymmetry should be: the rule wins on the
    straight road, and the merge road's jam is a standing queue at a fixed
    bottleneck rather than a travelling wave, which is NOT what the rule was
    designed for. A rule that fails there while the MPC's modest merge gains
    survive would be the cleanest statement of when the model-based machinery
    earns its cost.

    No planted disturbance: the merge jam forms on its own from the demand, so
    the rule simply sees it and responds. Needs no solver, ~35 s a run.
    """
    jobs = []
    for seed in range(1, 11):
        for v_post in (40.0, 60.0, 80.0):
            for n_signs in (2, 4, 9):
                jobs.append(dict(
                    exp="mergerule", net="long", cost=None, seed=seed,
                    instrument=False, Np=None, max_wall=None, max_iter=None,
                    b=None, window=None, drop_frac=0.0, model=None,
                    ctrl_interval=None, dv_space=None, alpha_cd=None,
                    demand=None, dist=None, gate=None,
                    rule=dict(v_post=v_post, n_signs=n_signs, hold_s=120,
                              v_det=35.0, avg_s=30)))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight17_manifest(max_wall=15.0):
    """Round 28: DO DRIVERS HAVE TO OBEY INSTANTLY FOR THIS TO WORK?

    docs/REPORT-MUST-DISCUSS.md #1: SUMO adopts a posted limit on the next
    simulation step, exactly, with no reaction time. The straight-road headline
    comes from a sign moving at 73 % of 30 s steps by a median 2.4 km/h, which
    under instant adoption is a continuously variable flow valve. Pilot, one
    seed: a 20 s adoption lag takes `tts` from 5.68 to 8.30 jam veh-h -- it
    costs nearly half the benefit.

    Two axes, deliberately kept separate:
      lag          how long the fleet takes to reach a newly posted limit
                   (10/20/40 s). Tests the instant-adoption idealisation.
      compliance   what fraction of the posted limit drivers actually target,
                   on top of SUMO's existing 0.89x speed factor (0.8/0.9).
                   Tests the perfect-obedience idealisation.

    The rule-based comparator is included at every setting, and that is the
    point rather than a courtesy: the rule makes ~9 % as many sign moves, so if
    the MPC degrades faster under lag than the rule does, the MPC's advantage
    on idealised actuation was substantially an artefact of it, and the
    comparison flips further toward the rule on a real road. No-control needs
    no variants -- it posts nothing -- so the existing baseline rows pair with
    all of these.

    This does NOT model detector realism (sparse, noisy loops): declared out of
    scope 2026-09-20, and stated as a limitation rather than claimed.
    """
    jobs = []
    conds = ([dict(lag=x) for x in (10.0, 20.0, 40.0)]
             + [dict(comp=x) for x in (0.8, 0.9)])
    for seed in range(1, 11):
        for d in (4600, 4800):
            for cond in conds:
                for c in STRAIGHT_FREE:
                    jobs.append(dict(
                        exp="straight17", net="straight", cost=c, seed=seed,
                        instrument=False, Np=None, max_wall=max_wall,
                        max_iter=None, b=None, window=None, drop_frac=0.0,
                        model=None, ctrl_interval=None, dv_space=None,
                        alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
                        gate=STRAIGHT_GATE_AVG_S,
                        lag=cond.get("lag"), comp=cond.get("comp")))
                jobs.append(dict(
                    exp="straight17", net="straight", cost=None, seed=seed,
                    instrument=False, Np=None, max_wall=None, max_iter=None,
                    b=None, window=None, drop_frac=0.0, model=None,
                    ctrl_interval=None, dv_space=None, alpha_cd=None,
                    demand=d, dist=dict(STRAIGHT_DIST),
                    gate=STRAIGHT_GATE_AVG_S,
                    rule=dict(v_post=40.0 if d == 4600 else 60.0, n_signs=9,
                              hold_s=120, v_det=35.0, avg_s=30),
                    lag=cond.get("lag"), comp=cond.get("comp")))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight18_manifest(max_wall=15.0):
    """Round 29: THE BEST DEPLOYABLE MPC, and the deployable headline at 30 seeds.

    Round 25 measured each actuator constraint separately and never combined
    move blocking with quantisation -- yet that combination IS the deployable
    controller: plan on the grid the sign can actually move on, and post only
    values it can actually show. Every cell here is a controller that could be
    installed. The reference is the rule, which is invariant to all of this
    (-64.2 % ideal and -64.2 % filtered, identical to the decimal).

    Also extends the deployable comparison to 30 seeds, because it is about to
    become the headline and the project has been burnt once by a p=0.002 result
    at n=10 (ledger, merge road).
    """
    jobs = []

    def add(cost, seed, d, **kw):
        j = dict(exp="straight18", net="straight", cost=cost, seed=seed,
                 instrument=False, Np=None, max_wall=max_wall if cost else None,
                 max_iter=None, b=None, window=None, drop_frac=0.0, model=None,
                 ctrl_interval=None, dv_space=None, alpha_cd=None, demand=d,
                 dist=dict(STRAIGHT_DIST), gate=STRAIGHT_GATE_AVG_S)
        j.update(kw)
        jobs.append(j)

    for seed in range(1, 11):
        for d in (4600, 4800):
            for c in STRAIGHT_FREE:
                add(c, seed, d, ctrl_interval=60, quant=10.0)
                add(c, seed, d, ctrl_interval=120, quant=10.0)
                add(c, seed, d, ctrl_interval=60, quant=10.0, hold=60.0)
    # 30 seeds on the deployable comparison: best-so-far deployable MPC
    # (block 60 + quant) and the rule under the same filter, against the
    # baselines that already have 30 seeds.
    for seed in range(11, 31):
        for d in (4600, 4800):
            add("tts", seed, d, ctrl_interval=60, quant=10.0)
            add(None, seed, d, quant=10.0, hold=60.0,
                rule=dict(v_post=40.0 if d == 4600 else 60.0, n_signs=9,
                          hold_s=120, v_det=35.0, avg_s=30))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight19_manifest(max_wall=15.0):
    """Round 30: DOES THE RULE'S WIN GENERALISE, or is it one scenario?

    The rule has only ever been run on the standard disturbance at d1. The MPC
    has been run across locations (round 21), repeats (round 20), severities
    and durations (round 12). Until the rule is run on those same conditions,
    "the rule beats the MPC" is a claim about a single incident, and the
    honest comparison the thesis needs does not exist.

    This deliberately includes the conditions where the MPC FAILED -- notably
    c4, the disturbance inside the signed stretch, where every objective is
    neutral or worse because the jam grows back over the signs. A rule that
    also fails there confirms the deployment boundary is a property of the
    ROAD, not of the controller; a rule that survives it means the MPC's
    failure was its own.

    Also sweeps the rule's detection threshold, which was fixed at 35 km/h on a
    guess and never varied -- the one free parameter never examined.
    """
    jobs = []

    def add(seed, d, dist, rule):
        jobs.append(dict(exp="straight19", net="straight", cost=None, seed=seed,
                         instrument=False, Np=None, max_wall=None,
                         max_iter=None, b=None, window=None, drop_frac=0.0,
                         model=None, ctrl_interval=None, dv_space=None,
                         alpha_cd=None, demand=d, dist=dist,
                         gate=STRAIGHT_GATE_AVG_S, rule=rule))

    R = dict(v_post=40.0, n_signs=9, hold_s=120, v_det=35.0, avg_s=30)
    for seed in range(1, 11):
        # (a) location -- pairs with round 21
        for edge in ("c4", "c8", "d3", "d5"):
            add(seed, 4600, dict(STRAIGHT_DIST, edge=edge), dict(R))
        # (b) repeated waves -- pairs with round 20
        for pat in ([600, 1020], [600, 900, 1200], [600, 780]):
            add(seed, 4600, [dict(STRAIGHT_DIST, t=t) for t in pat], dict(R))
        # (c) severity and duration -- pairs with round 12
        for vf in (0.10, 0.20):
            add(seed, 4600, dict(STRAIGHT_DIST, v_frac=vf), dict(R))
        for hs in (120, 480):
            add(seed, 4600, dict(STRAIGHT_DIST, hold_s=hs), dict(R))
        # (d) the detection threshold, never swept
        for d in (4600, 4800):
            for v_det in (25.0, 30.0, 45.0):
                add(seed, d, dict(STRAIGHT_DIST), dict(R, v_det=v_det))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight20_manifest(max_wall=15.0):
    """Round 31: `restrict` at weights that actually trade off -- and under a
    REAL sign, which is the interesting half.

    Round 26 swept 0.01-0.3 and every one switched the controller off (mean
    posted 84.1-84.6 of a legal 84.6). A one-seed pilot located the transition
    two orders lower: 1e-5 gives jam 8.76 at mean posted 76.9, 1e-4 gives 18.85
    at 81.8. So the live range is 1e-6 to 1e-4 and round 26's grid was simply
    off the end of it.

    The real question is not whether `restrict` can match `tts` on an idealised
    sign -- it cannot, it is trading jam for restraint by construction. It is
    whether a controller PUSHED TOWARD SPARSE, DEEP CONTROL degrades less under
    quantisation. `tts` collapses from -77.7 % to -4.6 % under a real sign
    precisely because its strategy is fine-grained; the rule is invariant
    because its strategy is coarse. If `restrict` moves the MPC along that axis,
    the deployable number should improve even as the idealised one gets worse --
    which would make this term the useful contribution rather than a failed
    experiment.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for w in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
                for filt in (False, True):
                    jobs.append(dict(
                        exp="straight20", net="straight",
                        cost="tts:1|restrict:%g" % w, seed=seed,
                        instrument=False, Np=None, max_wall=max_wall,
                        max_iter=None, b=None, window=None, drop_frac=0.0,
                        model=None, ctrl_interval=None, dv_space=None,
                        alpha_cd=None, demand=d, dist=dict(STRAIGHT_DIST),
                        gate=STRAIGHT_GATE_AVG_S,
                        quant=10.0 if filt else None,
                        hold=60.0 if filt else None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


# ---------------------------------------------------------------------------
# A DETERMINISTIC SOLVER BUDGET (2026-09-22).
#
# `max_wall` is a WALL-CLOCK cap, so whether a marginal solve converges depends
# on how busy the machine is. Measured: of 210 cell/seed pairs run twice across
# the sweeps, 196 agree bit-identically and 14 disagree -- every one of the 14
# an MPC row, 11 with a different failure count -- and 2.5 % of all solves exit
# `Maximum_WallTime_Exceeded`. An abandoned solve holds the previous limits, so
# the trajectory diverges from there on. Verified on an idle machine that the
# same cell run twice IS identical, and that a stored row produced under a
# 6-worker load is NOT reproduced (12.7 % on jam veh-h), which is the load
# dependence stated directly.
#
# Removing the cap means passing a huge value, not None: `ipopt_max_wall`
# DEFAULTS to 8.0 s in NmpcVslController, which is tighter than the 15 s the
# sweeps pass. With the wall clock out of the way `max_iter` is the only
# budget, and iteration count does not care how busy the machine is -- which
# also makes it safe to oversubscribe the cores.
#
# The 2026-08-30 ledger entry already established that the MEASUREMENT is
# invariant to the budget size (accel_rms moved < 0.5 % while the failure rate
# moved 4 points), so this costs nothing scientifically. It does change the
# control, so rows produced under it carry different job keys and must be
# compared against baselines re-run the same way -- which is what round 37 is
# for. No-control, constant-sign and rule rows have no solver and are
# unaffected.
DET_MAX_WALL = 1e6
DET_MAX_ITER = 300


def _straight_job(exp, cost, seed, demand, **kw):
    """A straight-road job on the deterministic budget."""
    j = dict(exp=exp, net="straight", cost=cost, seed=seed,
             instrument=False, Np=None,
             max_wall=(DET_MAX_WALL if cost else None),
             max_iter=(DET_MAX_ITER if cost else None),
             b=None, window=None, drop_frac=0.0, model=None,
             ctrl_interval=None, dv_space=None, alpha_cd=None,
             demand=demand, dist=dict(STRAIGHT_DIST), gate=STRAIGHT_GATE_AVG_S)
    j.update(kw)
    if j.get("rule") is not None and j["cost"] is None:
        j["max_wall"] = j["max_iter"] = None
    return j


def _rule(v_det=35.0, v_post=40.0, n_signs=9, hold_s=120):
    return dict(v_post=v_post, n_signs=n_signs, hold_s=hold_s,
                v_det=v_det, avg_s=30)


def build_straight21_manifest(max_wall=15.0):
    """Round 33: GIVE THE DEPLOYABLE MPC A FAIR HORIZON.

    Round 29 concluded that the MPC collapses under a real gantry (-55.8 % ->
    -12.0 %). That conclusion is CONFOUNDED, and the confound is in this file.
    `ctrl_interval` correctly changes the controller's move length (M), but Np
    stayed at 7, so the prediction horizon silently went from Np*M*T =
    7*6*5 = 210 s to 7*12*5 = **420 s** at a 60 s interval and **840 s** at
    120 s. Solver failures duly went 13 % -> 27 % -> 48 %. So the deployable
    controller was handed a two-to-four times longer, much harder NLP at the
    same moment its sign got coarser, and half its solves failed -- and a
    failed solve holds the previous limits, which is exactly what "the
    controller stopped working" looks like.

    Here the horizon is held in the range the project has always used by
    sweeping Np at the deployable interval: at 60 s moves, Np 3/4/5/7/10 is
    180/240/300/420/600 s, so Np=3 is horizon-matched to every earlier result.
    If a horizon-matched deployable MPC recovers, round 29's headline is an
    artefact of the sweep design; if it does not, the negative result stands
    with its most obvious objection answered.

    `tts` also gets Np 2/3/5 at a 120 s interval (240/360/480 s) for the same
    reason.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for c in STRAIGHT_FREE:
                for np_ in (3, 4, 5, 7, 10):
                    jobs.append(_straight_job(
                        "straight21", c, seed, d, Np=np_,
                        ctrl_interval=60, quant=10.0))
            for np_ in (2, 3, 5):
                jobs.append(_straight_job(
                    "straight21", "tts", seed, d, Np=np_,
                    ctrl_interval=120, quant=10.0))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight22_manifest(max_wall=15.0):
    """Round 34: SAFETY ON THE DEPLOYABLE SET -- the title obligation.

    The project is "decongestion AND safety" and there is no safety number for
    the reactive rule on this road at all, nor for any deployable configuration
    on either road. Round 18 instrumented the IDEALISED-sign MPC only.

    Pre-registered prediction, from the merge road's mechanism (r(TIT, front
    reach) = +0.66, 2026-09-06): conflicts track how far the congestion front
    reaches, so the policy that shrinks the jam most should show the biggest
    TET/TIT reduction. That is the RULE, and it would be the first time in the
    project that a policy improved a safety surrogate at all -- every objective
    on the merge made every surrogate worse. If instead the rule's hard
    40 km/h step raises conflicts at the queue tail, that is a real and
    reportable cost of the comparator, and it is the one thing that could still
    favour the MPC's gentler profile.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for kw in (dict(cost=None),
                       dict(cost=None, rule=_rule(35.0)),
                       dict(cost=None, rule=_rule(30.0)),
                       dict(cost="tts", Np=10),
                       dict(cost="tts", Np=7, ctrl_interval=60, quant=10.0),
                       dict(cost="tts", Np=3, ctrl_interval=60, quant=10.0)):
                kw = dict(kw)
                jobs.append(_straight_job("straight22", kw.pop("cost"), seed, d,
                                          instrument=True, **kw))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight23_manifest(max_wall=15.0):
    """Round 35: UNDER-COMPLIANCE -- drivers ignoring a low limit.

    Round 28 swept `comp` at 0.8 and 0.9, which is OVER-compliance: drivers
    obeying harder than asked. It made every policy look better on the jam and
    cost +125 to +223 s/veh, i.e. it is just more restriction bought
    expensively. The real-world failure mode is the opposite and has never been
    simulated.

    This is the sharpest remaining threat to the rule specifically, which is
    why it is worth the runs: the rule's whole strategy is one hard step to
    40 km/h on a 84.6 km/h road, and a 40 km/h advisory is far likelier to be
    ignored than the MPC's ~75 km/h. If the rule's advantage survives drivers
    doing 1.5x the posted value it is robust; if it evaporates, the honest
    finding is that the comparator wins only where it is obeyed.

    `_comply` caps the result at the legal limit (phase_d, 2026-09-22), so a
    released sign cannot post above the law; the cap is inert at comp <= 1, so
    round 28's rows are untouched.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for comp in (1.1, 1.25, 1.5):
                jobs.append(_straight_job("straight23", "tts", seed, d,
                                          comp=comp))
                jobs.append(_straight_job("straight23", "tts", seed, d,
                                          comp=comp, ctrl_interval=60,
                                          quant=10.0))
                for v_det in (35.0, 30.0):
                    jobs.append(_straight_job("straight23", None, seed, d,
                                              comp=comp, rule=_rule(v_det)))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight24_manifest(max_wall=15.0):
    """Round 36: THE RULE'S DETECTION THRESHOLD, SCORED OUT OF SAMPLE.

    v_det was fixed at 35 km/h on a guess and swept for the first time in round
    30, where 30 km/h read -81.0 % / -84.1 % against 35's -64.9 % / -61.9 %.
    That sweep was scored on the same ten seeds it was chosen from, which is
    precisely the mistake round 23 was built to correct for the rule's other
    parameters. The rule is now the project's comparator, so its tuning has to
    be at least as honest as the MPC's.

    Seeds 11-30, which already have no-control and constant-sign baselines, so
    no new baseline runs are needed. v_det 28/30/32 fills in around the round-30
    optimum; 35 is already on disk at these seeds.
    """
    jobs = []
    for seed in range(11, 31):
        for d in (4600, 4800):
            for v_det in (28.0, 30.0, 32.0):
                jobs.append(_straight_job("straight24", None, seed, d,
                                          rule=_rule(v_det)))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_straight25_manifest(max_wall=15.0):
    """Round 37: RE-BASELINE THE IDEAL-SIGN MPC ON THE DETERMINISTIC BUDGET.

    Every stored MPC row was produced under a wall-clock budget on a loaded
    machine and is therefore not reproducible (see DET_MAX_WALL). The
    deployable rounds above all run on the deterministic budget, so the
    ideal-sign arm they are compared against has to be re-run the same way --
    otherwise "ideal vs deployable" would also be "old budget vs new budget".

    30 seeds, the three unrestrained objectives at the project default Np=7,
    plus `tts` at Np=10 (round 24's cheaper and better setting). The
    no-control, constant-sign and rule baselines are NOT re-run: they have no
    solver, and validation/repro_check.py confirms this machine reproduces them
    to every decimal.
    """
    jobs = []
    for seed in range(1, 31):
        for d in (4600, 4800):
            for c in STRAIGHT_FREE:
                jobs.append(_straight_job("straight25", c, seed, d))
            jobs.append(_straight_job("straight25", "tts", seed, d, Np=10))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_driver_manifest(max_wall=15.0):
    """Round 38 (2026-09-29): DOES TELLING THE MPC HOW ITS DRIVERS RESPOND
    RECOVER WHAT LAG AND NON-COMPLIANCE COST IT? Plus the rounding-cost check.

    Rounds 28/35 made the plant's drivers lag (first-order, time constant L)
    or do 1.5x the posted value, and the MPC collapsed (ideal sign -52.5 % ->
    -9.8 % at 1.5x) while the rule held. But the MPC's prediction model was
    never told: it planned for instant, exact adoption. `aware=True` gives it
    the plant's own driver model (NmpcVslController driver_lag_s/driver_comp,
    validated in validation/test_driver_aware.py), so each condition is run
    unaware AND aware, on the ideal sign and on the deployable one.

    The ROUNDING-COST CHECK is the `block` cell against `deploy`: both replan
    on the 60 s grid the sign can move on (so the NLP knows the timing), and
    differ ONLY in whether what is posted is rounded to 10 km/h. It answers
    whether a controller that plans in 10 km/h steps (a mixed-integer NMPC)
    is worth building: if rounding costs little on top of move blocking, it
    is not.

    Everything MPC-side is run HERE, including the unaware arms and baselines
    that already exist from the desktop, because MPC rows are NOT reproducible
    across machines even on the deterministic budget: measured 2026-09-29, the
    laptop gives jam 6.17 veh-h / 10 failed solves on the cell the desktop
    stored as 6.95 / 14, while reproducing ITSELF exactly (same value twice).
    Pairing across machines would mix a machine effect into every contrast.
    No-control and rule rows ARE machine-independent (repro_check.py) and are
    read from data/sweep_straight; only the rule under lag at v_det=30 is new.

    Own directory (--dir data/sweep_driver), not the cumulative manifest: the
    two machines share data/sweep_straight through OneDrive and its shards are
    per-worker files, so running here would rewrite shards the desktop wrote.
    """
    jobs = []
    ideal = dict()
    deploy = dict(ctrl_interval=60, quant=10.0)
    for seed in range(1, 11):
        for d in (4600, 4800):
            # baselines on this machine, and the rounding-cost pair
            jobs.append(_straight_job("straight26", "tts", seed, d, **ideal))
            jobs.append(_straight_job("straight26", "tts", seed, d,
                                      ctrl_interval=60))
            jobs.append(_straight_job("straight26", "tts", seed, d, **deploy))
            for cond in (dict(lag=20.0), dict(lag=40.0), dict(comp=1.5)):
                for cfg in (ideal, deploy):
                    for aware in (None, True):
                        jobs.append(_straight_job("straight26", "tts", seed, d,
                                                  aware=aware, **cond, **cfg))
            # the tuned rule under the same lag (it exists under comp already)
            for lag in (20.0, 40.0):
                jobs.append(_straight_job("straight26", None, seed, d,
                                          lag=lag, rule=_rule(30.0)))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_vmin_manifest(max_wall=15.0):
    """Round 39 (2026-10-05): IS THE STRAIGHT-ROAD MPC RESULT SET BY THE SOLVER?

    Ledger 2026-09-29: under PARAMS_BM the straight-road objective is flat
    (jam) or rising (elsewhere) in the posted value, and raising the NLP's
    never-reached lower bound 20 -> 40 km/h cut `tts`'s jam reduction from
    -55/-50/-75 % to -33/-27/-47 % on 3 seeds. That is mechanism evidence
    only; this round puts it at 10 seeds before it goes in the report.

      A1  `tts` (ideal sign, Np=7), v_min default (20) vs 40, both demands.
      A2  the two objectives that "jumped to -55 % without dvsl" (round 14),
          same contrast at 4600 -- if they move too, the straight-road
          objective ranking is substantially solver behaviour.
      B   the deployable `tts` (60 s moves, 10 km/h steps) on the
          deterministic budget, on the LAPTOP -- the machine round 29's
          wall-clock -12.0 % rows came from -- so budget is separated from
          machine (ledger 2026-09-29: round 33 changed both at once).

    Laptop only, own directory, because MPC rows do not pair across machines.
    """
    jobs = []
    for seed in range(1, 11):
        for d in (4600, 4800):
            for vmin in (None, 40.0):
                jobs.append(_straight_job("straight27", "tts", seed, d,
                                          vmin=vmin))
            jobs.append(_straight_job("straight27", "tts", seed, d,
                                      ctrl_interval=60, quant=10.0))
        for c in ("tts:1|terminal:5", "dens_rate:1|tts:1"):
            for vmin in (None, 40.0):
                jobs.append(_straight_job("straight27", c, seed, 4600,
                                          vmin=vmin))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


RETEST_OBJECTIVES = [
    # the unrestrained set that worked on the ideal sign (round 14) ...
    "tts", "tts:1|terminal:5", "dens_rate:1|tts:1", "delay:1",
    "accel:1|tts:1|queue:0.5",
    # ... and two restrained originals from the round-11/12 league table, so
    # the "does a smoothness term help on a coarse sign?" question is covered
    "tts_dvsl", "accel_tts_queue"]


def build_objretest_manifest(max_wall=15.0):
    """Round 40 (2026-10-05): RE-RANK THE OBJECTIVES UNDER REALISTIC CONDITIONS.

    The objective ranking was only ever done on the idealised sign with
    obedient drivers. The realistic-sign rounds (29/33) re-ran just three
    objectives and the under-compliance round (35) only `tts`. This re-runs
    seven objectives under four conditions, all on the laptop so objectives
    can be compared with each other (MPC rows do not pair across machines):

      ideal sign, drivers obey      (the within-machine reference)
      realistic sign, drivers obey  (60 s moves, 10 km/h steps)
      ideal sign, drivers 1.5x
      realistic sign, drivers 1.5x

    4600 veh/h, seeds 1-10. No-control and rule rows are machine-independent
    and already exist under every one of these conditions.

    READ WITH LEDGER 2026-09-29: under PARAMS_BM the objective is flat in the
    sign, so differences between objectives may be solver behaviour rather
    than what each objective asks for. Report the ranking as closed-loop
    behaviour of each controller, not as evidence about the objective.

    REORDERED 2026-10-05 23:45 (same job dicts, so finished rows still match):
    PRIORITY by condition -- realistic sign, ideal at 1.5x, realistic at 1.5x,
    then the ideal-obey reference -- and SEED-MAJOR inside each, so a night
    that runs short leaves every objective with the same number of seeds. The
    ideal-obey cells of `tts`, `tts:1|terminal:5` and `dens_rate:1|tts:1` are
    DROPPED: round 39 ran exactly those cells (same machine, budget, demand,
    seeds), and the analysis pools by cell, not by round.
    """
    jobs = []
    conds = (dict(ctrl_interval=60, quant=10.0), dict(comp=1.5),
             dict(ctrl_interval=60, quant=10.0, comp=1.5), dict())
    in_r39 = ("tts", "tts:1|terminal:5", "dens_rate:1|tts:1")
    for cond in conds:
        for seed in range(1, 11):
            for c in RETEST_OBJECTIVES:
                if not cond and c in in_r39:
                    continue
                jobs.append(_straight_job("straight28", c, seed, 4600, **cond))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_bmjam_manifest(max_wall=15.0):
    """Round 41 (2026-10-05): A JAM OBJECTIVE ON A MODEL THAT CAN CLEAR A JAM.

    What the evening established (offline, ledger 2026-10-05):
      * PARAMS_BM_JAM (calibration/calibrate_jam.py) clears SUMO-seeded jams on
        SUMO's timescale on its fit runs, with general prediction unchanged.
      * On it, restricting ALWAYS raises predicted total time -- as SUMO does:
        every jam-reducing policy on this road adds delay. So `tts` correctly
        does not restrict and stays near-flat in the jam (7/17 flat states).
      * It DOES predict that a time-limited 40 km/h restriction shrinks an
        established jam 33-44 %. So `term_jam` (predicted jam veh-h, the
        primary metric) gives the controller a real reason to restrict.
      * The 7.2 km/h-per-move rate limit (physically inert in SUMO) stops the
        MPC reaching that restriction inside its horizon; lifted (dvmax 60)
        the jam objective restricts in proportion to its weight.

    SMOKE TEST (seed 1, 4600, jam:5, rate limit lifted, v_min 20): jam 5.48
    veh-h vs no control 13.65 (-60 %) for +22.5 s/veh; restricts hard in the jam
    (mean ~48, min 20 km/h at 840-900 s) and releases to ~80 after it. 30/78
    solves failed. The sign reached the 20 km/h floor, which no motorway posts
    and which is below the rule's 40 -- so the main cells use the rule's own
    floor, v_min 40, for a like-for-like comparison.

    Jobs in PRIORITY ORDER (each worker runs its slice in order, so if the night
    runs out the earlier cells are complete):
      1  bm_jam, tts:1|jam:5,  rate limit lifted, floor 40, ideal   -- main
      2  same, deployable sign (60 s, 10 km/h)                      -- vs rule
      3  same as 1, drivers 1.5x the posted value                   -- robustness
      4  bm_jam, tts:1|jam:20, floor 40, ideal                      -- weight
      5  bm_jam, tts:1|jam:5,  floor 20, ideal                      -- floor
      6  PARAMS_BM, tts:1|jam:5, floor 40, ideal                    -- model?
      7  bm_jam, tts, default rate limit and floor, ideal           -- tts
    4600 and 4800 veh/h, seeds 1-10, laptop, deterministic budget.

    SWITCHED TO FIT B (2026-10-05 ~00:15, `bm_jam_dis` = PARAMS_BM_JAM_DIS):
    tau 24 s, discharge 4687 veh/h (SUMO ~4700), held-out clearance error 0.34
    vs fit A's 0.54. Offline from logged SUMO jam states it predicts a 40 km/h
    restriction cuts the jam 63-82 % (fit A 37-44 %, with spurious self-made
    queues) and a 120 s hold at 40 costs ~0 to +1.4 % travel time -- the same
    trade the rule exploits in SUMO. One fit-A cell kept at low priority.
    """
    jobs = []
    J = "tts:1|jam:5"
    base = dict(model="bm_jam_dis", dvmax=60.0, vmin=40.0)
    cells = [
        dict(base, cost=J),
        dict(base, cost=J, ctrl_interval=60, quant=10.0),
        dict(base, cost=J, comp=1.5),
        dict(base, cost="tts:1|jam:20"),
        dict(cost=J, model="bm_jam_dis", dvmax=60.0),
        dict(cost=J, model=None, dvmax=60.0, vmin=40.0),
        dict(cost="tts", model="bm_jam_dis"),
        dict(cost=J, model="bm_jam", dvmax=60.0, vmin=40.0),
    ]
    # Group C, appended 2026-10-06 ~06:00 after the first two groups finished:
    # the OLD model with the jam objective matched fit B on the ideal sign
    # (-80/-79%) at higher delay (+41/+86 s/veh median), so the model's
    # contribution has to be measured where it should matter -- a realistic
    # sign and drivers at 1.5x.
    old = dict(cost=J, model=None, dvmax=60.0, vmin=40.0)
    cells += [dict(old, ctrl_interval=60, quant=10.0), dict(old, comp=1.5)]
    # priority groups, SEED-MAJOR inside each (see the round-40 note)
    for group in (cells[:3] + cells[6:7], cells[3:6] + cells[7:8],
                  cells[8:]):
        for seed in range(1, 11):
            for cell in group:
                for d in (4600, 4800):
                    kw = dict(cell)
                    jobs.append(_straight_job("straight29", kw.pop("cost"),
                                              seed, d, **kw))
    # Group D, appended 2026-10-06: fit B on the REALISTIC sign with drivers at
    # 1.5x -- the one missing cell of the obey/1.5x x ideal/realistic table.
    # DEMAND-MAJOR (4600 seeds 1-10 first) so that six workers run 4600 seeds
    # 1-6 as the first parallel wave, giving a >= 6-seed estimate fastest.
    for d in (4600, 4800):
        for seed in range(1, 11):
            jobs.append(_straight_job("straight29", J, seed, d,
                                      **dict(base, ctrl_interval=60,
                                             quant=10.0, comp=1.5)))
    # Group E, appended 2026-10-06 after the supervisor meeting: SAFETY for the
    # new MPC (SSM-instrumented), idealised and realistic sign. The no-control,
    # rule and old-MPC instrumented rows exist from round 34 (seeds 1-10, both
    # demands) and do not depend on the machine for the solver-free ones.
    for seed in range(1, 11):
        for cfg in (dict(), dict(ctrl_interval=60, quant=10.0)):
            for d in (4600, 4800):
                jobs.append(_straight_job("straight29", J, seed, d,
                                          instrument=True, **dict(base, **cfg)))
    # Group F, appended 2026-10-06: the SAME new MPC with improved failure
    # handling. Diagnosed: 23 of 27 logged failures are
    # Maximum_Iterations_Exceeded, 4 Error_In_Step_Computation. Offline on the
    # failed states, IPOPT mu_strategy=adaptive + max_iter 1000 rescued 14/19
    # (ideal) and 4/8 (realistic) while keeping every successful solve, and
    # adaptive alone was faster than the default. Plus: retry once from a
    # fresh rollout on a failed warm solve, and fall back to the last good
    # plan's scheduled move instead of freezing the sign.
    solver = dict(ipopt_max_iter=1000, ipopt_extra={"mu_strategy": "adaptive"},
                  retry_cold=True, on_fail="shift")
    for seed in range(1, 11):
        for cfg in (dict(), dict(ctrl_interval=60, quant=10.0)):
            for d in (4600, 4800):
                kw = dict(base, **cfg)
                kw["max_iter"] = 1000           # the job budget field, too
                jobs.append(_straight_job("straight29", J, seed, d,
                                          solver=solver, **kw))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


# ---------------------------------------------------------------------------
# Round 42 (2026-10-09): THE FINAL COST-FUNCTION SEARCH, both roads, realistic
# sign, final controller configuration.
# ---------------------------------------------------------------------------
# MATCHED WEIGHTS. The terms in cost_functions.py are normalised by fixed
# constants (tts_scale 100, accel_scale 2000, ...) that were set on the short
# merge road, so a weight of 1.0 does NOT mean "as important as tts". Measured
# offline (scratch term_scale.py, 2026-10-09): from every logged solve state of
# one closed-loop run per road, the model was rolled 420 s (Np=7 x 60 s moves,
# the realistic-sign NLP) under "hold the current sign" and "all signs at 40",
# and each term's response to that change was compared with tts's. The weight
# at which a term responds to the sign as strongly as tts does (w*, jam states
# only, i.e. where control matters; queue on the merge uses all states because
# the origin queue does not move inside a standing jam) is:
#
#   term            straight (PARAMS_BM_JAM_DIS)   merge (PARAMS_BM)
#   jam                    2.6                          1.3
#   accel                  0.11                         0.32
#   accel_temporal         6.0                          9.0
#   accel_dens             0.023                        0.0021
#   dens_rate              3.6e4                        1.2e3
#   upstream_queue         3.2                          0.089
#   speed_var              1.2                          0.63
#   speed_grad             58                           28
#   conflict_risk          55                           77
#   density_sq             2.3                          0.67
#   density_track          4.2                          2.8
#   terminal               36                           214
#   queue                  0.13                         5.3
#   dvsl                   1.7                          0.55
#   bneck_density           -                           38
#   bneck_flow              -                           6.9e-4
#
# So at the weights used all project, `dens_rate:1` was ~1e3-1e4x too weak to
# matter, `speed_grad`/`conflict_risk` 30-80x too weak, and `accel_dens` /
# `bneck_flow` 100-1000x too STRONG. Here every term enters at its own road's
# matched weight (x1), the six families the research gap is about are swept at
# x0.25 and x4, and the historical literature weightings are kept as their own
# entries so the old results can be placed on the same plot.
FINAL_W_STRAIGHT = dict(jam=2.5, accel=0.1, accel_temporal=6.0,
                        accel_dens=0.025, dens_rate=3e4, upstream_queue=3.0,
                        speed_var=1.2, speed_grad=60.0, conflict_risk=60.0,
                        density_sq=2.0, density_track=4.0, terminal=40.0,
                        queue=0.1, dvsl=1.5)
FINAL_W_MERGE = dict(jam=1.3, accel=0.3, accel_temporal=9.0,
                     accel_dens=0.002, dens_rate=1200.0, upstream_queue=0.09,
                     speed_var=0.6, speed_grad=30.0, conflict_risk=80.0,
                     density_sq=0.7, density_track=3.0, terminal=200.0,
                     queue=5.0, dvsl=0.55, bneck_density=40.0,
                     bneck_flow=0.0007)
FINAL_SWEPT = ("jam", "accel", "accel_temporal", "dens_rate", "speed_var",
               "speed_grad")


def final_costs(W, merge):
    """The round-42 cost list for one road, given its matched weights W."""
    g = lambda x: "%g" % x                                      # noqa: E731
    core = ["tts", "delay:1"] + [
        "tts:1|%s:%s" % (t, g(W[t])) for t in
        ("terminal", "jam", "accel", "accel_temporal", "accel_dens",
         "dens_rate", "upstream_queue", "speed_var", "speed_grad",
         "conflict_risk", "density_sq", "density_track")]
    swept = ["tts:1|%s:%s" % (t, g(W[t] * m)) for t in FINAL_SWEPT
             for m in (0.25, 4.0)]
    literature = [
        "tts:1|dvsl:0.1",                    # Hegyi 2005, the field standard
        "speed_var:1|tts:1|dvsl:0.1",        # Hadiuzzaman / Yang pair (+ restraint)
        "density_track:1|dvsl:0.1",          # the Assessment-2 incumbent
        "accel:1|tts:1|queue:0.5|dvsl:0.1",  # accel_tts_queue as run on the merge
        "dens_rate:1|tts:1",                 # as run on the straight road in Sept
    ]
    pure = ["jam:1", "accel:1"]              # no mobility term: degeneracy demos
    combos = [
        "tts:1|accel:%s|queue:%s" % (g(W["accel"]), g(W["queue"])),
        "tts:1|jam:%s|speed_grad:%s" % (g(W["jam"]), g(W["speed_grad"])),
        "tts:1|jam:%s|accel:%s" % (g(W["jam"]), g(W["accel"])),
        "tts:1|jam:%s|dvsl:%s" % (g(W["jam"]), g(W["dvsl"])),
    ]
    extra = (["tts:1|bneck_density:%s" % g(W["bneck_density"]),
              "tts:1|bneck_flow:%s" % g(W["bneck_flow"])] if merge else [])
    out = core + swept + literature + pure + combos + extra
    assert len(out) == len(set(out)), "duplicate cost string"
    return out


def build_final_manifest(max_wall=15.0):
    """Round 42: every cost function, realistic sign, final configuration,
    on BOTH roads, scored on the validated wave and safety metrics.

    Why: the research question is which objective reduces the shockwave, and
    the only broad comparison on the merge (116 cost strings, rounds 1-10) used
    the idealised sign, the 7.2 km/h rate limit and the old solver budget; on
    the straight road the final configuration (jam-calibrated model, rate limit
    lifted, 40 km/h floor) has only ever run `tts` and `tts:1|jam:5`.

    Final configuration, both roads: realistic sign (replan every 60 s, 10 km/h
    steps -- the round-41 definition, NOT round 32's post-hoc filter), v_min 40,
    dv_max 60, Np=7, deterministic budget (max_iter 300, no wall clock), default
    solver. Straight road on PARAMS_BM_JAM_DIS at 4600 and 4800 veh/h; merge on
    PARAMS_BM (the jam-calibrated fits are straight-road fits) at its 4000 veh/h.
    Self-contained baselines: no control and the rule (v_det 30, v_post 40,
    9 signs, 120 s), solver-free, so machine-independent.

    SEED-MAJOR across both roads, so a stop at any point leaves every objective
    on both roads with the same number of seeds. Laptop, 6 workers.
    """
    jobs = []
    cm = final_costs(FINAL_W_MERGE, merge=True)
    cs_ = final_costs(FINAL_W_STRAIGHT, merge=False)
    rule = _rule(v_det=30.0, v_post=40.0, n_signs=9, hold_s=120)
    merge_base = dict(exp="final_merge", net="long", instrument=False, Np=None,
                      b=None, window=None, drop_frac=0.0, model=None,
                      dv_space=None, alpha_cd=None, demand=None, dist=None,
                      gate=None)
    real = dict(ctrl_interval=60, quant=10.0, vmin=40.0, dvmax=60.0)
    for seed in range(1, 11):
        for c in [None, "RULE"] + cm:
            j = dict(merge_base, seed=seed)
            if c is None or c == "RULE":
                j.update(cost=None, max_wall=None, max_iter=None,
                         ctrl_interval=None, rule=(rule if c else None))
            else:
                j.update(cost=c, max_wall=DET_MAX_WALL, max_iter=DET_MAX_ITER,
                         **real)
            jobs.append(j)
        for c in [None, "RULE"] + cs_:
            for d in (4600, 4800):
                if c is None:
                    jobs.append(_straight_job("final_straight", None, seed, d))
                elif c == "RULE":
                    jobs.append(_straight_job("final_straight", None, seed, d,
                                              rule=rule))
                else:
                    jobs.append(_straight_job("final_straight", c, seed, d,
                                              model="bm_jam_dis", **real))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_final_ssm_manifest(max_wall=15.0):
    """Round 42b (2026-10-12): SAFETY on the three headline straight-road
    objectives of round 42, realistic sign, same final configuration,
    SSM-instrumented. Kept in its own directory so instrumented rows can never
    enter the round-42 pool (they are not bit-identical to plain rows, ledger
    2026-10-06). No-control / rule instrumented rows exist from round 34
    (solver-free, machine-independent); `tts:1|jam:5` from round 41 group E.
    Seed-major so a stop leaves every arm with the same number of seeds."""
    W = FINAL_W_STRAIGHT
    # 2026-10-20: + the literature standard `tts:1|dvsl:0.1` (Hegyi 2005), at
    # the user's request, so the safety table covers it too. Appended last:
    # the three original costs keep their job keys, so their finished rows are
    # skipped on resume and only the 20 new jobs run.
    costs = ["tts:1|speed_var:%g" % W["speed_var"],
             "tts:1|conflict_risk:%g" % W["conflict_risk"],
             "tts:1|jam:%g" % (4.0 * W["jam"]),
             "tts:1|dvsl:0.1"]
    real = dict(ctrl_interval=60, quant=10.0, vmin=40.0, dvmax=60.0)
    jobs = []
    for seed in range(1, 11):
        for c in costs:
            for d in (4600, 4800):
                jobs.append(_straight_job("final_ssm", c, seed, d,
                                          model="bm_jam_dis", instrument=True,
                                          **real))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_mergedeploy_manifest(max_wall=15.0):
    """Round 32: the MERGE road under a realistic sign.

    Every merge-road number in this project was also measured with the
    idealised actuator. The surviving merge claim -- `tts` buys ~10 s/veh and
    1.7-3.0 % genuine smoothing -- has never been tested against a sign that
    can only show 10 km/h steps held 60 s. If it evaporates the way the
    straight-road result does, then the honest project-wide conclusion is that
    MPC-VSL benefits in this study are an artefact of idealised actuation
    everywhere, not just on one road. That is a much stronger and much more
    useful statement than either road alone, and it is 100 runs.
    """
    jobs = []
    for seed in range(1, 11):
        for c in ("tts", "tts_dvsl", "accel_tts_queue", "delay_dvsl"):
            for filt in (False, True):
                jobs.append(dict(
                    exp="mergedeploy", net="long", cost=c, seed=seed,
                    instrument=False, Np=None, max_wall=max_wall,
                    max_iter=None, b=None, window=None, drop_frac=0.0,
                    model=None, ctrl_interval=None, dv_space=None,
                    alpha_cd=None, demand=None, dist=None, gate=None,
                    quant=10.0 if filt else None,
                    hold=60.0 if filt else None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_placebo_manifest(max_wall=15.0):
    """Round 2c: does solver failure CAUSE the apparent smoothing?

    The confound, restated: across 26 objectives on the long road, the
    per-objective solver-failure rate predicts measured shockwave reduction at
    r = -0.87 (R2 = 0.75) and road time at +0.86. The acceleration family fails
    47% of solves and posts -13 to -17%; the tts/grad/risk family fails 32% and
    posts about -2%. Since a failed solve holds the previous limits, failures
    are not a neutral loss of control -- they freeze the sign low, which is
    indistinguishable from what an aggressive objective does deliberately.

    Raising the solver budget does not settle it: max_wall 15 -> 30 with
    max_iter 300 -> 3000 moved failures only 44% -> 41% (and doubled run time),
    so the failures are genuine convergence difficulty, not a budget shortfall,
    and they cannot simply be removed.

    So test the causal claim directly instead. Take two objectives that
    currently fail LEAST and smooth least (`tts`, `grad_dvsl`) and inject
    placebo failures by discarding a fixed fraction of successful solves. Their
    natural failure rates are ~36% and ~32%, so drop fractions of 0.15/0.30/0.45
    carry them across the acceleration family's ~47%.

      * If d_accel becomes markedly more negative and road time rises as the
        drop fraction increases, then freezing the sign IS the mechanism, and
        the acceleration family's lead is substantially an artefact of its own
        solver trouble. The long-road ranking would then have to be reported
        with that caveat, or re-run under matched failure rates.
      * If they barely move, the correlation is a symptom -- aggressive control
        drives the model into stiff states -- and the ranking stands.

    Cheap and decisive: 2 objectives x 3 fractions x 10 seeds.
    """
    jobs = []
    for seed in range(1, 11):
        for c in ("tts", "grad_dvsl"):
            for df in (0.15, 0.30, 0.45):
                jobs.append(dict(exp="placebo", net="long", cost=c, seed=seed,
                                 instrument=False, Np=None, max_wall=max_wall,
                                 max_iter=None, b=None, window=None,
                                 drop_frac=df))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


def build_solver_manifest():
    """Round 2a: find a solver budget that makes the comparison fair.

    THE PROBLEM THIS EXISTS TO SOLVE. On the long road, per-objective solver
    failure rate correlates with measured shockwave reduction at r = -0.87
    (R2 = 0.75) and with road time at r = +0.86, across the 26 objectives. The
    acceleration family fails 47% of solves and posts the biggest smoothing;
    the tts/grad/risk family fails 32% and posts almost none. A failed solve
    holds the previous limits, so failures are not neutral -- they freeze the
    sign low and mimic exactly the behaviour being measured. Until the failure
    rate is both LOW and roughly EQUAL across objectives, the league table
    cannot be separated from its own solver trouble.

    What is already known, and why neither previous test settled it:
      * max_wall 8 s, max_iter 300  -> 44% fail, 15/20 Maximum_Iterations
      * max_wall 8 s, max_iter 800/2000/5000 -> 44% fail, unchanged; the
        iteration failures simply became Maximum_WallTime (wall was binding)
      * max_wall 15 s, max_iter 300 (the round-1 sweep) -> 30% overall, and
        19.5% of all solves still exit Maximum_Iterations (iters now binding)
    Each test raised ONE cap and hit the other. This raises them together.

    Three costs spanning the observed failure range, so the answer is not
    tuned to one objective.
    """
    jobs = []
    costs = ["accel_tts_queue",   # 47% fail, -14.4%
             "tts",               # 36% fail,  -3.2%
             "grad_dvsl"]         # 32% fail,  -3.5%
    configs = [(15, 300),         # round-1 setting, for a paired reference
               (30, 3000),
               (60, 20000)]
    for seed in range(1, 6):
        for mw, mi in configs:
            jobs.append(dict(exp="solvercfg", net="long", cost=None, seed=seed,
                             instrument=False, Np=None, max_wall=mw,
                             max_iter=mi, b=None, window=None))
            for c in costs:
                jobs.append(dict(exp="solvercfg", net="long", cost=c, seed=seed,
                                 instrument=False, Np=None, max_wall=mw,
                                 max_iter=mi, b=None, window=None))
    for i, j in enumerate(jobs):
        j["i"] = i
    return jobs


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def run_worker(k, n_workers):
    from model.metanet_model import (PARAMS_BM, PARAMS_BM_STRAIGHT, PARAMS_S14,
                                     MetanetModel, MetanetParams, load_geometry,
                                     model_geometry, straight_model_geometry)
    from control.phase_d import run_closed_loop
    sys.path.insert(0, os.path.join(_ROOT, "sumo"))
    import build_straight_net as bsn

    jobs = json.load(open(JOBS))
    shard = os.path.join(SWEEP_DIR, "shard%d.json" % k)
    rows = json.load(open(shard)) if os.path.exists(shard) else []
    # Resume against EVERY shard, not just this worker's: a relaunch with a
    # different worker count would otherwise redo work another slice finished.
    # Only SUCCESSFUL, uncorrupted rows count as done. Two ways a relaunch
    # would otherwise silently skip work that never actually completed:
    #   * a row with ok=False is a recorded failure, not a result -- treating
    #     its key as done means a transient SUMO crash permanently removes that
    #     job from the sweep;
    #   * a row whose wall time is absurd was produced across a machine suspend,
    #     so its solver statistics are fiction and it has to be redone.
    done = set()
    for f in os.listdir(SWEEP_DIR):
        if f.startswith("shard") and f.endswith(".json"):
            try:
                for r in json.load(open(os.path.join(SWEEP_DIR, f))):
                    if r.get("ok") and r.get("wall_s", 0) <= 3000:
                        done.add(_norm_key(r["key"]))
            except Exception:
                pass

    # Build each geometry/model once per worker, not once per run: constructing
    # the CasADi NLP is the expensive part and it does not depend on the seed.
    cache = {}

    def get(net, which="bounded", alpha_cd=None, demand=None):
        # alpha_cd is part of the cache key because it changes the MODEL, not
        # the controller: building one model and mutating it would silently
        # apply the last job's capacity drop to every job after it.
        if (net, which, alpha_cd, demand) not in cache and net in ("straight",
                                                                   "straight_long"):
            # Straight jam-wave road: no bottleneck, so the model is the signed
            # stretch plus the unsigned downstream stretch where the disturbance
            # happens, as one road (MetanetModel n_ctrl mode). Same prediction
            # model and actuator as every long-road result.
            if alpha_cd is not None:
                raise ValueError("alpha_cd is not wired for the straight road")
            # "straight_long" adds downstream road for a TRAVELLING disturbance
            # (an onset ramp) to run into. The MODELLED stretch is the same 9+6
            # segments either way, so the NLP and the calibration are unchanged
            # and only the scored road differs.
            tag = "_long" if net == "straight_long" else ""
            geom = load_geometry(bsn.net_path(tag), allow_straight=True)
            mgeom, off = straight_model_geometry(geom, bsn.N_UP,
                                                 n_seg=bsn.N_CTRL + bsn.N_DOWN)
            # "bm_straight" = Bounded-METANET refitted to THIS road
            # (calibration/calibrate_straight.py). See the caveats on
            # PARAMS_BM_STRAIGHT: better prediction and capacity, jam discharge
            # still ~4x too slow.
            # "bm_jam" = Bounded-METANET calibrated on JAM CLEARANCE
            # (calibration/calibrate_jam.py, 2026-10-05). Looked up lazily so
            # nothing else needs the constant to exist.
            if which in ("bm_jam", "bm_jam_dis"):
                import model.metanet_model as _mm
                pars = (_mm.PARAMS_BM_JAM if which == "bm_jam"
                        else _mm.PARAMS_BM_JAM_DIS)
            else:
                pars = (PARAMS_S14 if which == "s14"
                        else PARAMS_BM_STRAIGHT if which == "bm_straight"
                        else PARAMS_BM)
            model = MetanetModel(mgeom, MetanetParams(**pars, vsl_scaled=True),
                                 n_ctrl=bsn.N_CTRL)
            cache[(net, which, alpha_cd, demand)] = (geom, model, off,
                                                     bsn.cfg_path(demand, tag))
        if (net, which, alpha_cd, demand) not in cache:
            if net == "long":
                geom = load_geometry(os.path.join(_ROOT, LONG_NET))
                # `demand` selects a config whose routes carry a different
                # vehsPerHour. The NETWORK is identical, so geometry, the nine
                # controlled segments and the prediction model are unchanged --
                # only the arrival rate differs. demand=None keeps the original
                # 4000 veh/h config so every pre-existing job is untouched.
                if demand is None:
                    cfg = os.path.join(_ROOT, LONG_CFG)
                else:
                    cfg = os.path.join(_ROOT, "sumo",
                                       "simulation3.d%d.sumocfg" % int(demand))
            else:
                geom, cfg = load_geometry(), None
            mgeom, off = model_geometry(geom, n_ctrl=9)
            # Both prediction models run with vsl_scaled=True. The actuator fix
            # is not a property of either model -- it is what makes the sign
            # bind at all -- so comparing them without it would compare two
            # broken controllers.
            pars = PARAMS_S14 if which == "s14" else PARAMS_BM
            mp = MetanetParams(**pars, vsl_scaled=True)
            if alpha_cd is not None:
                # The capacity drop. Until 2026-09-01 this was a SILENT NO-OP on
                # Bounded-METANET: `_capacity_drop_scale` was never called from
                # `_bounded_step`, so setting the parameter changed nothing and
                # rollouts at 0 / 0.118 / 0.25 were bit-identical. It is now
                # wired in, with parity at alpha_cd=0 held at 5.8e-13.
                # rho_cd_ref is the density at which the drop is fully realised;
                # 2x the bottleneck's own critical density puts the midpoint
                # inside the congested range actually visited.
                mp.alpha_cd = float(alpha_cd)
                mp.rho_cd_ref = 2.0 * (mp.rho_crit_bneck or mp.rho_crit)
            model = MetanetModel(mgeom, mp)
            cache[(net, which, alpha_cd, demand)] = (geom, model, off, cfg)
        return cache[(net, which, alpha_cd, demand)]

    mine = [j for j in jobs if j["i"] % n_workers == k]
    print("[w%d] %d jobs in slice, %d already done overall"
          % (k, len(mine), len(done)), flush=True)

    for j in mine:
        key = job_key(j)
        if key in done:
            continue
        if free_gb(_ROOT) < MIN_FREE_GB:
            print("[w%d] STOP: free disk %.1f GB < %.1f GB"
                  % (k, free_gb(_ROOT), MIN_FREE_GB), flush=True)
            return
        geom, model, off, cfg = get(j["net"], j.get("model") or "bounded",
                                    j.get("alpha_cd"), j.get("demand"))
        steps = {"long": LONG_STEPS, "straight": STRAIGHT_STEPS,
                 # 10.2 km takes ~434 s to traverse, so the road needs longer to
                 # clear after demand stops at 2400 s
                 "straight_long": STRAIGHT_STEPS_LONG}.get(j["net"], SHORT_STEPS)
        t0 = time.time()
        try:
            r = run_closed_loop(
                j["cost"], j["seed"], instrument=j["instrument"], steps=steps,
                model=model, geom=geom, offset=off, config=cfg, Np=j["Np"],
                max_wall=j["max_wall"], max_iter=j.get("max_iter"),
                fixed_b=j.get("b"), window=j.get("window"),
                drop_frac=j.get("drop_frac", 0.0) or 0.0,
                ctrl_interval=j.get("ctrl_interval"),
                dv_space=j.get("dv_space"),
                demand_end_s=(bsn.DEMAND_END_S if j["net"] == "straight"
                              else None),
                disturbance=j.get("dist"), gate_avg_s=j.get("gate"),
                rule=j.get("rule"),
                sign_quant_kmh=j.get("quant"),
                sign_min_hold_s=j.get("hold"),
                actuation_lag_s=j.get("lag"), compliance=j.get("comp"),
                driver_aware=bool(j.get("aware")),
                v_min_kmh=j.get("vmin"), dv_max_kmh=j.get("dvmax"),
                solver_opts=j.get("solver"),
                tag="w%d_%s_%s_s%d" % (k, j["exp"], j["cost"] or "nc", j["seed"]))
            r["ok"] = True
        except Exception as e:
            r = {"cost": j["cost"] or "nocontrol", "seed": j["seed"],
                 "ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                 "trace": traceback.format_exc()[-500:]}
            print("[w%d] FAIL %s" % (k, r["error"]), flush=True)
            # A run that dies mid-simulation leaves traci's MODULE-LEVEL
            # connection registered, so the next job in this slice fails at
            # startup with "Connection 'default' is already active" -- and so
            # does the one after it. One dead SUMO would otherwise take out
            # every remaining job in the worker. Measured: that is exactly what
            # happened when SUMO processes were killed mid-run, 3 of the 7
            # errors being the cascade rather than the original fault.
            try:
                import traci
                traci.close(False)
            except Exception:
                pass
        r.update(key=key, exp=j["exp"], net=j["net"], Np_req=j["Np"],
                 instrumented=j["instrument"], max_wall=j["max_wall"],
                 max_iter_req=j.get("max_iter"),
                 model_name=j.get("model") or "bounded",
                 interval_req=j.get("ctrl_interval"),
                 dv_space_req=j.get("dv_space"), alpha_cd_req=j.get("alpha_cd"),
                 demand_req=j.get("demand"), dist_req=j.get("dist"),
                 gate_req=j.get("gate"),
                 # round 19-21 provenance. `pattern` is only a NAME for the
                 # disturbance list already in `dist_req`; the analysis groups
                 # on it so it never has to re-parse the list.
                 rule_req=j.get("rule"), pattern=j.get("pattern"),
                 quant_req=j.get("quant"), hold_req=j.get("hold"),
                 lag_req=j.get("lag"), comp_req=j.get("comp"),
                 aware_req=j.get("aware"), vmin_req=j.get("vmin"),
                 dvmax_req=j.get("dvmax"), solver_req=j.get("solver"),
                 wall_s=round(time.time() - t0, 1))
        rows.append(r)
        # Rewrite this worker's own shard only. One writer, one file.
        tmp = shard + ".tmp"
        json.dump(rows, open(tmp, "w"), indent=1)
        # The repo is OneDrive-synced, and OneDrive briefly locks a file for
        # its own read while syncing the previous write -- measured 2026-09-19:
        # os.replace hit WinError 5 (Access is denied) on shard0 after 348
        # clean writes, with nothing else on the machine holding the file, and
        # it was uncaught, so it killed the whole worker (and every job left in
        # its slice) rather than just that one save. Retry with backoff before
        # giving up; a real, permanent lock (another sweep process on the same
        # shard) still fails loudly after these run out.
        for attempt in range(6):
            try:
                os.replace(tmp, shard)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        print("[w%d] %-13s %-18s seed %-3d %s %6.0fs  (%d/%d)"
              % (k, j["exp"], j["cost"] or "nocontrol", j["seed"],
                 "ok " if r["ok"] else "ERR", r["wall_s"],
                 len(rows), len(mine)), flush=True)
    print("[w%d] SLICE COMPLETE" % k, flush=True)


def status():
    jobs = json.load(open(JOBS)) if os.path.exists(JOBS) else []
    rows = []
    for f in sorted(os.listdir(SWEEP_DIR)):
        if f.startswith("shard") and f.endswith(".json"):
            try:
                rows += json.load(open(os.path.join(SWEEP_DIR, f)))
            except Exception:
                pass
    ok = [r for r in rows if r.get("ok")]
    print("manifest %d jobs | done %d | errors %d | free disk %.1f GB"
          % (len(jobs), len(ok), len(rows) - len(ok), free_gb(_ROOT)))
    by = {}
    for r in rows:
        e = by.setdefault(r.get("exp", "?"), [0, 0, 0.0])
        e[0] += 1
        e[1] += 0 if r.get("ok") else 1
        e[2] += r.get("wall_s", 0.0)
    tot = {}
    for j in jobs:
        tot[j["exp"]] = tot.get(j["exp"], 0) + 1
    print("%-14s %8s %8s %8s %10s" % ("block", "done", "of", "errors", "mean s"))
    for e in tot:
        d, err, w = by.get(e, [0, 0, 0.0])
        print("%-14s %8d %8d %8d %10.0f"
              % (e, d, tot[e], err, w / max(d, 1)))
    if ok:
        el = sum(r.get("wall_s", 0) for r in rows)
        rem = len(jobs) - len(ok)
        print("\n~%.1f core-hours spent, %d jobs left (~%.1f core-hours)"
              % (el / 3600, rem, rem * (el / max(len(rows), 1)) / 3600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--round", default="main",
                    choices=("main", "solver", "ref", "solver+ref", "round3",
                             "placebo", "round3+placebo", "round4",
                             "round3+round4", "round5", "round5b", "round6",
                             "round7", "round8", "round9", "round10", "round10b",
                             "straight", "straight2", "straight3", "straight4",
                             "straight5", "straight6", "straight7", "straight8",
                             "straight9", "straight10", "straight11", "straight12",
                             "straight13", "straight14", "straight15",
                             "straight16", "straight17", "straight18",
                             "straight19", "straight20", "straight21",
                             "straight22", "straight23", "straight24",
                             "straight25", "mergerule", "driver", "vmin",
                             "objretest", "bmjam", "final", "final_ssm"))
    ap.add_argument("--r3-max-wall", type=float, default=15.0)
    ap.add_argument("--r3-max-iter", type=int, default=300)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--long-max-wall", type=float, default=8.0)
    ap.add_argument("--dir", default=None,
                    help="sweep directory (manifest + shards), relative to the "
                         "project root. Default data/sweep. A separate directory "
                         "keeps an unrelated experiment out of the 5000-row "
                         "resume scan and out of every analysis glob.")
    a = ap.parse_args()
    global SWEEP_DIR, JOBS
    if a.dir:
        SWEEP_DIR = os.path.join(_ROOT, a.dir)
        JOBS = os.path.join(SWEEP_DIR, "jobs.json")
    os.makedirs(SWEEP_DIR, exist_ok=True)
    if a.manifest:
        if a.round == "solver":
            jobs = build_solver_manifest()
        elif a.round == "ref":
            jobs = build_ref_manifest()
        elif a.round == "round3":
            jobs = build_round3_manifest(a.r3_max_wall, a.r3_max_iter)
        elif a.round == "placebo":
            jobs = build_placebo_manifest(a.r3_max_wall)
        elif a.round == "round4":
            jobs = build_round4_manifest(a.r3_max_wall)
        elif a.round == "round5":
            jobs = build_round5_manifest(a.r3_max_wall)
        elif a.round == "round6":
            jobs = build_round6_manifest(a.r3_max_wall)
        elif a.round == "round7":
            jobs = build_round7_manifest(a.r3_max_wall)
        elif a.round == "round8":
            jobs = build_round8_manifest(a.r3_max_wall)
        elif a.round == "round9":
            jobs = build_round9_manifest(a.r3_max_wall)
        elif a.round == "round10":
            jobs = build_round10_manifest(a.r3_max_wall)
        elif a.round == "straight":
            if not a.dir:
                ap.error("--round straight must use --dir (keep it out of data/sweep)")
            jobs = build_straight_manifest(a.r3_max_wall)
        elif a.round in ("straight3", "straight4", "straight5", "straight6",
                         "straight7", "straight8", "straight9",
                         "straight10", "straight11", "straight12",
                         "straight13", "straight14", "straight15",
                         "straight16", "straight17", "straight18",
                         "straight19", "straight20", "straight21",
                         "straight22", "straight23", "straight24",
                         "straight25"):
            if not a.dir:
                ap.error("--round %s must use --dir" % a.round)
            jobs = build_straight_manifest(a.r3_max_wall) + \
                build_straight2_manifest(a.r3_max_wall) + \
                build_straight3_manifest(a.r3_max_wall)
            # CUMULATIVE: a later round always carries every earlier round's
            # jobs. They are skipped on resume because their keys are already
            # on disk, and carrying them means a round that was left part-done
            # is finished rather than silently dropped out of the manifest.
            n_round = int(a.round.replace("straight", "") or 1)
            if n_round >= 4:
                jobs += build_straight4_manifest(a.r3_max_wall)
            if n_round >= 5:
                jobs += build_straight5_manifest(a.r3_max_wall)
            if n_round >= 6:
                jobs += build_straight6_manifest(a.r3_max_wall)
            if n_round >= 7:
                jobs += build_straight7_manifest(a.r3_max_wall)
            if n_round >= 8:
                jobs += build_straight8_manifest(a.r3_max_wall)
            if n_round >= 9:
                jobs += build_straight9_manifest(a.r3_max_wall)
            if n_round >= 10:
                jobs += build_straight10_manifest(a.r3_max_wall)
            if n_round >= 11:
                jobs += build_straight11_manifest(a.r3_max_wall)
            if n_round >= 12:
                jobs += build_straight12_manifest(a.r3_max_wall)
            if n_round >= 13:
                jobs += build_straight13_manifest(a.r3_max_wall)
            if n_round >= 14:
                jobs += build_straight14_manifest(a.r3_max_wall)
            if n_round >= 15:
                jobs += build_straight15_manifest(a.r3_max_wall)
            if n_round >= 16:
                jobs += build_straight16_manifest(a.r3_max_wall)
            if n_round >= 17:
                jobs += build_straight17_manifest(a.r3_max_wall)
                # the merge-road rule baseline completes the thesis 2x2
                # (two roads x {no control, constant sign, reactive rule}).
                # net="long", so every analysis that filters on net ignores it.
                jobs += build_mergerule_manifest(a.r3_max_wall)
            if n_round >= 18:
                jobs += build_straight18_manifest(a.r3_max_wall)
            if n_round >= 19:
                jobs += build_straight19_manifest(a.r3_max_wall)
            if n_round >= 20:
                jobs += build_straight20_manifest(a.r3_max_wall)
                jobs += build_mergedeploy_manifest(a.r3_max_wall)
            # Rounds 33-37 (2026-09-22). Ordered so that a stop leaves the most
            # decision-relevant block finished: round 33 can overturn round 29's
            # headline, so it runs first; the re-baseline it will be compared
            # against runs last because it is insurance, not discovery.
            if n_round >= 21:
                jobs += build_straight21_manifest(a.r3_max_wall)
            if n_round >= 22:
                jobs += build_straight22_manifest(a.r3_max_wall)
            if n_round >= 23:
                jobs += build_straight23_manifest(a.r3_max_wall)
            if n_round >= 24:
                jobs += build_straight24_manifest(a.r3_max_wall)
            if n_round >= 25:
                jobs += build_straight25_manifest(a.r3_max_wall)
            for i, j in enumerate(jobs):
                j["i"] = i
        elif a.round == "mergerule":
            if not a.dir:
                ap.error("--round mergerule must use --dir")
            jobs = build_mergerule_manifest(a.r3_max_wall)
        elif a.round == "driver":
            if a.dir != os.path.join("data", "sweep_driver") and \
                    a.dir.replace("\\", "/") != "data/sweep_driver":
                ap.error("--round driver must use --dir data/sweep_driver")
            jobs = build_driver_manifest(a.r3_max_wall)
        elif a.round == "vmin":
            if (a.dir or "").replace("\\", "/") != "data/sweep_vmin":
                ap.error("--round vmin must use --dir data/sweep_vmin")
            jobs = build_vmin_manifest(a.r3_max_wall)
        elif a.round == "objretest":
            if (a.dir or "").replace("\\", "/") != "data/sweep_objretest":
                ap.error("--round objretest must use --dir data/sweep_objretest")
            jobs = build_objretest_manifest(a.r3_max_wall)
        elif a.round == "bmjam":
            if (a.dir or "").replace("\\", "/") != "data/sweep_bmjam":
                ap.error("--round bmjam must use --dir data/sweep_bmjam")
            jobs = build_bmjam_manifest(a.r3_max_wall)
        elif a.round == "final":
            if (a.dir or "").replace("\\", "/") != "data/sweep_final":
                ap.error("--round final must use --dir data/sweep_final")
            jobs = build_final_manifest(a.r3_max_wall)
        elif a.round == "final_ssm":
            if (a.dir or "").replace("\\", "/") != "data/sweep_final_ssm":
                ap.error("--round final_ssm must use --dir data/sweep_final_ssm")
            jobs = build_final_ssm_manifest(a.r3_max_wall)
        elif a.round == "straight2":
            if not a.dir:
                ap.error("--round straight2 must use --dir")
            # round 11 first, so a stop leaves the headline table complete
            jobs = build_straight_manifest(a.r3_max_wall) + \
                build_straight2_manifest(a.r3_max_wall)
            for i, j in enumerate(jobs):
                j["i"] = i
        elif a.round == "round10b":
            # Same jobs as round10, RE-ORDERED mid-run. Completed keys resume,
            # so this only changes what runs NEXT.
            #
            # `wavecost` -- the six objectives that target wave propagation --
            # is the novel contribution and was queued BEHIND ~600 remaining
            # runs of the existing league table. The laptop has now slept twice
            # during this chain (8 suspend-corrupted rows in `waves` alone), so
            # the tail of the queue is genuinely at risk, and losing the tail
            # currently means losing the one block the thesis needs most.
            # Re-ordered: novel objectives first, then finish the league table,
            # then the weight frontier, then safety.
            jobs = build_round10_manifest(a.r3_max_wall)
            prio = {"wavecost": 0, "waves": 1, "wavefront": 2, "wavessm": 3}
            jobs.sort(key=lambda j: prio.get(j["exp"], 9))
            for i, j in enumerate(jobs):
                j["i"] = i
        elif a.round == "round5b":
            # Same jobs as round5, RE-ORDERED mid-run. Completed keys are
            # skipped on resume, so this only changes what runs NEXT.
            #
            # Why: (1) the laptop slept ~7 h overnight (6 rows at ~25,050 s,
            # one per worker, simultaneous -- the signature the ledger records),
            # so roughly half the wall-clock budget bought no compute; and
            # (2) the day's own results changed the value ordering underneath
            # the original manifest. `capdrop` and `bflow` were written when a
            # capacity-drop prize still looked plausible. It is now measured to
            # be absent -- queued discharge is invariant to the posted limit
            # (1.8% span, n.s., 10 seeds) -- so giving the model a capacity-drop
            # term cannot manufacture one, and a discharge-maximising objective
            # has nothing to maximise. Both are now confirmatory nulls.
            # `spatial` meanwhile became the most valuable block left: with no
            # bound the controller posts a 63 km/h drop across a 200 m boundary,
            # and every source in the literature constrains that hard, so what
            # it costs to impose is directly reportable.
            jobs = build_round5_manifest(a.r3_max_wall)
            prio = {"ttd_rew": 0, "spatial": 1, "horizon2": 2,
                    "capdrop": 3, "bflow": 4}
            jobs.sort(key=lambda j: prio.get(j["exp"], -1))
            for i, j in enumerate(jobs):
                j["i"] = i
        elif a.round == "round3+round4":
            jobs = build_placebo_manifest(a.r3_max_wall) + \
                build_round3_manifest(a.r3_max_wall, a.r3_max_iter) + \
                build_round4_manifest(a.r3_max_wall)
            # ORDER BY VALUE PER HOUR, NOT BY WHEN IT WAS WRITTEN, because the
            # run may well be stopped before the queue empties.
            # `frontier` is the largest block (450 runs at ~330 s) and would
            # otherwise starve the three cheaper blocks behind it -- the
            # prediction-model robustness check in particular reframes the
            # single biggest caveat on every result in the project, for a third
            # of the frontier's cost. It also degrades the most gracefully: its
            # jobs are emitted seed-major, so cutting it short costs seeds per
            # point rather than whole points, and the trade-off curve is still
            # traced. So it goes last.
            prio = {"placebo": 0, "new_long": 1, "new_short": 2, "interval": 3,
                    "model_s14": 4, "ref_ssm": 5, "frontier": 6}
            jobs.sort(key=lambda j: prio.get(j["exp"], 9))
            for i, j in enumerate(jobs):
                j["i"] = i
        elif a.round == "round3+placebo":
            # Placebo first: it decides whether the round-1 long-road ranking
            # can be reported at all, so its answer is wanted before the
            # frontier sweep spends 20+ core-hours extending that ranking.
            jobs = build_placebo_manifest(a.r3_max_wall) + \
                build_round3_manifest(a.r3_max_wall, a.r3_max_iter)
            for i, j in enumerate(jobs):
                j["i"] = i
        elif a.round == "solver+ref":
            # Solver test first: it gates whether the league table has to be
            # re-run, so its answer is wanted soonest. The reference block is
            # solver-independent and fills the rest of the machine behind it.
            jobs = build_solver_manifest() + build_ref_manifest()
            for i, j in enumerate(jobs):
                j["i"] = i
        else:
            jobs = build_manifest(a.long_max_wall)
        json.dump(jobs, open(JOBS, "w"), indent=1)
        n = {}
        for j in jobs:
            n[j["exp"]] = n.get(j["exp"], 0) + 1
        print("wrote %s: %d jobs" % (JOBS, len(jobs)))
        for e, c in n.items():
            print("   %-14s %5d" % (e, c))
    elif a.status:
        status()
    elif a.worker is not None:
        run_worker(a.worker, a.workers)
    else:
        ap.error("one of --manifest / --worker / --status")


if __name__ == "__main__":
    main()
