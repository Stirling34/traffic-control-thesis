# -*- coding: utf-8 -*-
"""
Phase D: run every candidate cost function in closed loop and score it.

Each run is SUMO + the NMPC talking over TraCI, exactly as `main_metanet_v1.py`
does, but with the objective swapped for one of the entries in
`control/cost_functions.py`. Scoring is always on SUMO output, never on the
model, because the model is what chose the control and cannot also be the judge.

Two passes:
  --screen    fast. TTS, shockwave acceleration, speed variation, emissions.
              About 40 s a run, so all candidates by several seeds is minutes.
  (default)   adds the SSM safety device for TET/TIT. ~28x slower in SUMO, so
              reserve it for the shortlist.

Common random numbers: every cost function sees the same seed set, so the
comparison is paired and the seed-to-seed variation cancels. That matters here,
because TET/TIT have a ~23% coefficient of variation against 3.5% for delay.

Usage:
    python control/phase_d.py --screen --seeds 3
    python control/phase_d.py --costs accel_tts,tts_dvsl --seeds 10
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import (MS_TO_KMH, PARAMS_BM, PARAMS_S14,      # noqa: E402
                                 MetanetModel, MetanetParams, load_geometry)
from control.nmpc_metanet import NmpcVslController                    # noqa: E402
from control.cost_functions import COSTS                              # noqa: E402
from validation.control_authority import shockwave_metrics            # noqa: E402
from validation.wave_metrics import jam_metrics, wave_metrics         # noqa: E402
from validation.sumo_metrics import (CONFIG, OUT, SUMO_DIR, _ssm_args,
                                     parse_drac, parse_emissions,
                                     parse_safety)              # noqa: E402

SIM_STEPS = 2500
CONTROL_INTERVAL = 30
# "vehicle" disturbance: how long to keep looking for a vehicle to slow in each
# target lane. A lane can be momentarily empty at the trigger instant even in
# flowing traffic, and firing once silently produced one-lane disturbances in 6
# of 63 pilot runs (ledger 2026-09-11).
DIST_WAIT_S = 60

# Per-run scratch (emissions + SSM XML) goes to LOCAL temp, NOT to sumo/out.
#
# The repository lives inside a OneDrive-synced folder. An instrumented run
# writes a ~90 MB SSM file there and deletes it again, but OneDrive has already
# taken it, so a 120-run safety block pushes something like 10 GB through the
# sync engine and its local caches for files that were never meant to leave the
# machine. Free space fell from 26 GB to 4.2 GB across one such block, and the
# workers self-abort below 2 GB. Writing to %TEMP% keeps the churn off OneDrive
# entirely. Falls back to sumo/out if no temp directory is available.
_SCRATCH = os.environ.get("TEMP") or os.environ.get("TMP")
SCRATCH = (os.path.join(_SCRATCH, "egh400_sumo_scratch") if _SCRATCH
           else os.path.join(_ROOT, "sumo", "out"))
DEMAND_VPH = 4000.0
DEMAND_END_S = 1000.0
RESULTS = os.path.join(_ROOT, "data", "phase_d.json")
# Results from the ported bake-off model go to their own file. They are NOT
# comparable with data/phase_d.json run-for-run: same seeds and same scoring,
# but a different prediction model, which is the whole point of the comparison.
RESULTS_S14 = os.path.join(_ROOT, "data", "phase_d_s14.json")
# The final configuration: the published Bounded-METANET prediction model plus
# the scaled VSL cap, so the sign actually constrains anything. Separate file:
# results are NOT comparable run-for-run with the other two, because the
# prediction model differs and because everything here uses the corrected 8 s
# shockwave metric.
RESULTS_BM = os.path.join(_ROOT, "data", "phase_d_bounded.json")
RESULTS_BM_SSM = os.path.join(_ROOT, "data", "phase_d_bounded_ssm.json")
# Instrumented (SSM) runs go to their own file: they are the same experiment
# with TET/TIT added, but SUMO is far slower with the device on, so they are run
# for a shortlist only and must not be mixed with the screening rows.
RESULTS_S14_SSM = os.path.join(_ROOT, "data", "phase_d_s14_ssm.json")


class _Dist:
    """One planted disturbance, self-contained so a run can carry several.

    Lifted out of `run_closed_loop` on 2026-09-19 without changing what a
    single disturbance does (the branches below are the previous inline code,
    verbatim). The point of the extraction is that `disturbance` may now be a
    LIST of specs: the repeated-wave experiment needs two or three slowdowns in
    one run, and the previous code kept its trigger state in run-level locals,
    so a second one could not exist.

    Instances share the `taken` set of vehicle ids, so two disturbances that
    overlap in space cannot command the same vehicle twice -- which would look
    like a disturbance that fired but produce only one slow car.
    """

    def __init__(self, spec, geom, edges, Lkm, lanes, taken):
        edges = list(edges)
        self.t = int(spec["t"])
        self.edge = str(spec["edge"])
        self.pos = float(spec["pos"])
        # `v_frac` (a fraction of the edge limit) may be given INSTEAD of v_kmh
        self.v_ms = float(spec.get("v_kmh", 0.0)) / MS_TO_KMH
        self.hold = int(spec["hold_s"])
        if self.edge not in edges:
            raise ValueError("disturbance edge %r is not on this road" % self.edge)
        self.j = edges.index(self.edge)
        self.x_km = float(np.sum(Lkm[:self.j]) + self.pos / 1000.0)
        self.legal_ms = float(geom.limits_ms[self.j])
        # "vehicle": slow the vehicle nearest `pos` in each target lane, with
        #   everyone else free to overtake -- a moving bottleneck of the kind a
        #   real road produces. Lane changing is switched off for it so it
        #   cannot dodge out of the way.
        # "edge": the whole edge's limit drops for hold_s -- a full
        #   cross-section blockade. Kept for the pilot comparison only; it
        #   empties the road downstream and that confounded a 300-run sweep
        #   (ledger 2026-09-12).
        self.mode = str(spec.get("mode", "vehicle"))
        if self.mode not in ("vehicle", "edge"):
            raise ValueError("disturbance mode must be 'vehicle' or 'edge'")
        # HOW QUICKLY it slows, as distinct from how much. 0 = brake at the
        # vehicle's own limit (an abrupt event); ramp_s > 0 eases the speed
        # down linearly over that many seconds via traci slowDown.
        self.ramp = int(spec.get("ramp_s", 0) or 0)
        ln = spec.get("lanes", [0])
        self.lanes = set(range(int(lanes[self.j])) if ln == "all"
                         else [int(x) for x in np.atleast_1d(ln)])
        if spec.get("v_frac") is not None:
            self.v_ms = float(spec["v_frac"]) * self.legal_ms
        # A lane can be momentarily empty at the trigger instant, which is how
        # 6 of 63 pilot runs silently slowed only one lane. So keep looking for
        # up to DIST_WAIT_S and record when each vehicle was actually caught.
        self.fired = {}
        self.ids = []
        self.taken = taken

    def apply(self, traci, t_now):
        if self.mode == "edge":
            if t_now == self.t:
                self.ids = list(traci.edge.getLastStepVehicleIDs(self.edge))
                traci.edge.setMaxSpeed(self.edge, max(self.v_ms, 0.1))
            elif t_now == self.t + self.hold:
                traci.edge.setMaxSpeed(self.edge, self.legal_ms)
            return
        if (self.t <= t_now <= self.t + DIST_WAIT_S
                and len(self.fired) < len(self.lanes)):
            best = {}
            for vid in traci.edge.getLastStepVehicleIDs(self.edge):
                ln = traci.vehicle.getLaneIndex(vid)
                if ln not in self.lanes or ln in self.fired or vid in self.taken:
                    continue
                gap = abs(traci.vehicle.getLanePosition(vid) - self.pos)
                if ln not in best or gap < best[ln][0]:
                    best[ln] = (gap, vid)
            for ln, (_, vid) in best.items():
                traci.vehicle.setLaneChangeMode(vid, 0)   # it holds its lane
                if self.ramp > 0:
                    traci.vehicle.slowDown(vid, self.v_ms, float(self.ramp))
                else:
                    traci.vehicle.setSpeed(vid, self.v_ms)
                self.fired[ln] = (vid, t_now)
                self.ids.append(vid)
                self.taken.add(vid)
        for ln, (vid, t_fire) in list(self.fired.items()):
            # end of the ramp: pin the speed so the hold means the same thing
            # whether the approach was abrupt or gradual
            if self.ramp > 0 and t_now == t_fire + self.ramp:
                if vid in set(traci.vehicle.getIDList()):
                    traci.vehicle.setSpeed(vid, self.v_ms)
            if t_now == t_fire + self.ramp + self.hold:
                # it may have driven off the end of the road by now
                if vid in set(traci.vehicle.getIDList()):
                    traci.vehicle.setSpeed(vid, -1)
                    # 1621 is SUMO's default, which is what every other vehicle
                    # runs (the route file's laneChangeMode="512" is ignored)
                    traci.vehicle.setLaneChangeMode(vid, 1621)

    def summary(self):
        return dict(dist_t=self.t, dist_edge=self.edge,
                    dist_x_km=round(self.x_km, 4),
                    dist_v_kmh=round(self.v_ms * MS_TO_KMH, 2),
                    dist_hold_s=self.hold, dist_mode=self.mode,
                    dist_ramp_s=self.ramp,
                    dist_lanes=sorted(self.lanes),
                    # when each target lane was actually caught (vehicle mode)
                    dist_fire_t=sorted(t for _, t in self.fired.values()),
                    # validity check carried on every row: in "vehicle" mode one
                    # vehicle per lane should have been disturbed; in "edge"
                    # mode this is just how many were on the edge when it closed
                    dist_n_veh=len(self.ids))


def run_closed_loop(cost, seed, instrument=False, steps=SIM_STEPS, model=None,
                    geom=None, offset=0, config=None, Np=None, tag=None,
                    max_iter=None, max_wall=None, fixed_b=None, window=None,
                    drop_frac=0.0, ctrl_interval=None, dv_space=None,
                    return_fields=False, demand_end_s=None, disturbance=None,
                    fcd_file=None, gate_avg_s=None, rule=None,
                    sign_quant_kmh=None, sign_min_hold_s=None,
                    actuation_lag_s=None, compliance=None,
                    driver_aware=False, v_min_kmh=None, dv_max_kmh=None,
                    solver_opts=None):
    """One SUMO + NMPC run. `cost=None` means no control (the baseline).

    Added 2026-09-11 for the straight-road jam-wave scenario, all defaulting to
    the previous behaviour:
      demand_end_s  when the demand the controller forecasts stops (was fixed at
                    DEMAND_END_S = 1000 s)
      disturbance   dict(t, edge, pos, v_kmh, hold_s): at time t the vehicle
                    nearest `pos` metres along `edge` in EACH lane is commanded
                    down to v_kmh (braking at its own decel limit) with lane
                    changing off, held for hold_s, then released. Every lane,
                    so nothing can overtake it -- it is a transient full
                    cross-section disturbance, and what survives it is a jam.
                    Its location and time are fixed; which vehicle it is depends
                    on the policy upstream, as it would on a real road.
                    A LIST of such dicts plants several in one run (added
                    2026-09-19 for the repeated-wave experiment); they are
                    independent and cannot command the same vehicle twice.
      fcd_file      write SUMO floating-car data (x, speed per vehicle-second),
                    for validating the cell-based wave metrics against
                    trajectories. Put it outside OneDrive (see SCRATCH).
      sign_quant_kmh / sign_min_hold_s
                    DEPLOYABILITY FILTER. Quantise what is posted to a multiple
                    of `sign_quant_kmh` (real gantries show 10 km/h steps) and
                    forbid a sign from changing again for `sign_min_hold_s`
                    after it moves. The controller still plans continuously.
                    Both default to None = the unfiltered behaviour every
                    pre-2026-09-20 result was measured under.
      actuation_lag_s / compliance
                    HOW DRIVERS RESPOND. SUMO adopts a newly posted limit
                    INSTANTLY and exactly -- no perception, no reaction time,
                    no disbelief (see docs/REPORT-MUST-DISCUSS.md #1). That is
                    the single caveat that bears hardest on the headline,
                    because the controller moves the sign at 73% of steps by a
                    median of 2.4 km/h, which under instant adoption behaves
                    like a continuously variable flow valve.
                      actuation_lag_s  the speed actually enforced on an edge
                        closes 1/L of its gap to the posted value every
                        second instead of jumping to it, i.e. a FIRST-ORDER lag
                        with time constant L: 63 % adopted after L seconds,
                        95 % after 3L. (Described as "linear over L seconds"
                        until 2026-09-29; the code was always first-order.)
                      compliance  drivers target this fraction of the posted
                        limit (on top of SUMO's own 0.89x speed factor); 1.0 is
                        the previous behaviour.
                    Both default to None = exactly the pre-2026-09-20 model.
      driver_aware  give the MPC's prediction model the SAME lag / compliance
                    the plant applies (NmpcVslController driver_lag_s /
                    driver_comp; validation/test_driver_aware.py), so it plans
                    for the drivers it actually has. False = the controller
                    assumes instant, exact adoption, as in rounds 28/35.
      rule          dict(v_post, n_signs, hold_s, v_det, avg_s): run the
                    rule-based reactive comparator INSTEAD of the MPC (no model
                    and no solver are used). Mutually exclusive with `cost` and
                    `fixed_b`. See the block in the loop.

    `geom` is the SCORED geometry: every edge SUMO simulates. `model` spans a
    sub-stretch of it starting at edge `offset` (see `model_geometry`). On the
    original network offset is 0 and the two coincide, so behaviour there is
    unchanged. On the extended-approach network offset is 18: the controller
    still sets the same nine segments, but 3.6 km of approach road in front of
    them is simulated and measured, so a queue the controller pushes upstream
    lands on instrumented road instead of vanishing into the origin.
    """
    import traci
    if rule is not None and (cost is not None or fixed_b is not None):
        raise ValueError("`rule` replaces the controller: pass it with "
                         "cost=None and fixed_b=None")
    geom = geom or load_geometry()
    model = model or MetanetModel(geom, MetanetParams())
    kw = {} if Np is None else {"Np": Np, "Nc": min(5, Np)}
    if v_min_kmh is not None:
        # The NLP's lower bound on the posted value (NmpcVslController default
        # 20 km/h). Exposed 2026-10-05 for the flat-objective test: the sign
        # never approaches 20 or 40 on the straight road, so if moving this
        # bound changes the result, the result is set by the solver, not by
        # the forecast (ledger 2026-09-29). None = the default every stored
        # run used.
        kw["v_min_kmh"] = float(v_min_kmh)
    if dv_max_kmh is not None:
        # The per-move rate limit (NmpcVslController default 7.2 km/h per
        # move, inherited from the LWR MPC). Exposed 2026-10-05: it is
        # physically inert in SUMO (ledger 2026-09-01, a sign-flicker
        # constraint, not a safety one) but it stops the MPC making the move
        # the reactive rule makes -- 84.6 -> 40 in one step -- so with it on, a
        # jam objective cannot reach the restriction its model says would help
        # before the jam has gone. None = the default every stored run used.
        kw["dv_max_kmh"] = float(dv_max_kmh)
    if solver_opts:
        # failure handling and IPOPT options (2026-10-06): any of retry_cold,
        # on_fail, ipopt_extra, ipopt_max_iter -- see NmpcVslController.
        # None = exactly the solver every stored run used.
        kw.update(solver_opts)
    if max_iter is not None:
        kw["ipopt_max_iter"] = max_iter
    if max_wall is not None:
        kw["ipopt_max_wall"] = max_wall
    if dv_space is not None:
        # hard-ish bound on the DROP between adjacent signs (soft with a heavy
        # penalty; see NmpcVslController). None reproduces every pre-2026-09-01
        # result exactly, because the constraint is simply not added.
        kw["dv_space_kmh"] = dv_space
    interval = int(ctrl_interval or CONTROL_INTERVAL)
    if ctrl_interval is not None:
        # The controller's internal notion of a move MUST match how often the
        # sign is actually re-posted, or the two disagree about what the first
        # move commits to. M is the number of model steps per control move, so
        # at T_sec = 5 s a 60 s interval is M = 12. Changing the SUMO interval
        # without changing M would leave the NLP planning 30 s moves while the
        # loop applied them every 60 s -- a silent mismatch that would look like
        # a control-interval result but would really be a modelling error.
        step_s = (model or MetanetModel(geom, MetanetParams())).p.T_sec
        if interval % int(step_s):
            raise ValueError("control interval %d s must be a multiple of the "
                             "model step %g s" % (interval, step_s))
        kw["M"] = interval // int(step_s)
    if driver_aware:
        if cost is None:
            raise ValueError("driver_aware needs an MPC (cost is None)")
        if not (actuation_lag_s or compliance):
            raise ValueError("driver_aware needs actuation_lag_s or compliance")
        kw["driver_lag_s"] = actuation_lag_s or None
        kw["driver_comp"] = compliance or None
    ctrl = None if cost is None else NmpcVslController(model, cost=cost, **kw)

    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    n_ctrl = model.n_ctrl
    ctrl_sl = slice(offset, offset + n_ctrl)        # controlled edges, full list
    seg_idx = offset + np.asarray(model.seg_edge_index)  # model seg -> full edge
    applied = limits[ctrl_sl].copy()
    # Vehicles entering the controlled section have already driven `offset`
    # segments, so the demand the model's origin sees is the source demand
    # delayed by that free-flow travel time. On the short network offset is 0
    # and this is exactly the profile used for every previous result.
    lag_s = float(np.sum(Lkm[:offset] / np.maximum(limits[:offset], 1e-9)) * 3600.0)
    # The tag names this run's scratch files, so it must be a legal filename.
    # Inline weight specs look like `accel:0.25|tts:1|dvsl:0.1`, and `:` and `|`
    # are both illegal on Windows -- every frontier run died on
    # `OSError: [Errno 22] Invalid argument` before this sanitising was added.
    tag = tag or "%s_s%d" % (cost or "nocontrol", seed)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)

    # Each run needs its OWN emissions file. All runs previously wrote and read
    # the same sumo/add/edge_emissions.xml, so a parse could land on a file
    # another run was still writing -> "not well-formed (invalid token)".
    os.makedirs(SCRATCH, exist_ok=True)
    em_path = os.path.join(SCRATCH, "em_%s.xml" % tag)
    add_path = os.path.join(SCRATCH, "em_%s.add.xml" % tag)
    with open(add_path, "w") as fh:
        fh.write(
            "<additional>"
            '<edgeData id="em" type="emissions" period="60" '
            'file="%s" excludeEmpty="true"/>' % os.path.basename(em_path)
            + "</additional>")
    args = ["sumo", "-c", config or CONFIG, "--end", str(steps), "--no-warnings",
            "--seed", str(seed), "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--device.emissions.probability", "1.0",
            "--additional-files", add_path]
    if instrument:
        args += _ssm_args(tag, SCRATCH)
    if fcd_file:
        args += ["--fcd-output", fcd_file, "--fcd-output.attributes", "x,speed"]
    os.makedirs(SCRATCH, exist_ok=True)
    traci.start(args)

    spd, dens, on_road, pend, vsl_hist, counts = [], [], [], [], [], []
    arrived = n_solve = n_fail = n_dropped = 0
    # Own RNG, seeded on the run, so a placebo-failure run is reproducible and
    # never perturbs SUMO's own stream.
    rng = np.random.default_rng((seed * 1000003) ^ int(drop_frac * 1e6))
    active = False
    t_solve = 0.0
    dists = []
    if disturbance is not None:
        specs = ([disturbance] if isinstance(disturbance, dict)
                 else list(disturbance))
        taken = set()
        dists = [_Dist(s, geom, edges, Lkm, lanes, taken) for s in specs]
        dists.sort(key=lambda d: d.t)
    # rule-based comparator state (see the block in the loop)
    rule_until = -1
    rule_zone = None
    rule_on_s = 0
    # per-sign time until which the currently posted value must be held
    sign_held_until = np.zeros(n_ctrl)
    # What the sign SAYS (vsl_target) vs what the traffic is actually doing
    # (vsl_actual). They differ only when actuation_lag_s is set; with no lag
    # every write below is identical to the pre-2026-09-20 immediate write.
    vsl_target = applied.copy()
    vsl_actual = applied.copy()

    def _comply(vals):
        """What the traffic actually does, given `compliance`.

        `compliance` < 1 is OVER-compliance (drivers slow down further than
        asked) and is what rounds 28 ran. `compliance` > 1 is the real-world
        failure mode -- drivers ignoring a low limit -- and it needs the cap
        below: without it a RELEASED sign (posting the legal limit, which is
        what an inactive controller and a released rule both do) would be
        multiplied above the legal limit and the signed stretch would run
        FASTER than the law allows, which is not non-compliance, it is a
        different road. Non-compliance means reverting toward normal driving,
        not beyond it.

        The cap is inert for compliance <= 1 -- `vals` never exceeds the legal
        limit -- so every stored round-28 row is unaffected.
        """
        if not compliance:
            return vals
        return np.minimum(vals * compliance, limits[ctrl_sl])

    def _post(vals):
        """Command the signs. Honours compliance and the actuation lag."""
        vsl_target[:] = vals
        if actuation_lag_s:
            return          # the per-step ramp below does the writing
        out = _comply(vals)
        for e, vv in zip(edges[ctrl_sl], out):
            traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
        vsl_actual[:] = vals
    for step in range(steps):
        traci.simulationStep()
        t_now = step + 1
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        vraw = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        # An empty edge reports mean speed -1, so fall back to what is posted
        # there: the applied limit on the controlled edges, the legal limit
        # everywhere else (including the uncontrolled upstream approach).
        posted = limits.copy()
        posted[ctrl_sl] = applied
        v_edges = np.where(c > 0, vraw, posted)
        spd.append(np.where(c > 0, vraw, np.nan))
        dens.append(c / (Lkm * lanes))
        counts.append(c)
        on_road.append(c.sum())
        pend.append(len(traci.simulation.getPendingVehicles()))
        arrived += traci.simulation.getArrivedNumber()
        vsl_hist.append(applied.copy())

        # THE DISTURBANCE(S), before any `continue` so they fire under every policy
        for _d in dists:
            _d.apply(traci, t_now)

        if actuation_lag_s:
            # Drivers do not adopt a new limit the instant it is posted. Move
            # what is actually enforced toward the posted value at a fixed
            # rate, so a change takes `actuation_lag_s` seconds to take full
            # effect. Written every simulation second, not every control step.
            step = (vsl_target - vsl_actual) / float(actuation_lag_s)
            near = np.abs(vsl_target - vsl_actual) <= np.abs(step)
            vsl_actual[:] = np.where(near, vsl_target, vsl_actual + step)
            out = _comply(vsl_actual)
            for e, vv in zip(edges[ctrl_sl], out):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)

        if rule is not None:
            # THE RULE-BASED COMPARATOR (added 2026-09-19).
            #
            # Every result so far is MPC against no control or against a
            # constant sign, and a constant sign is a weak benchmark: it does
            # nothing at any onset because it never moves. The question a
            # reviewer will ask is whether the OPTIMISATION earns its cost
            # against a rule an engineer could write in an afternoon. This is
            # that rule, in the spirit of SPECIALIST (Hegyi & Hoogendoorn 2008,
            # field-tested on the A12): find the upstream front of the jam,
            # post a low limit on the signs immediately behind it so the flow
            # entering the jam falls below the flow leaving it, hold, release.
            #
            # It is NOT a reimplementation of SPECIALIST -- no solvability
            # test, no theoretical wave-speed geometry, and the zone does not
            # travel with the wave. It is the generic reactive rule, and the
            # comparison is only fair if it is TUNED, so v_post, n_signs and
            # hold_s are swept rather than guessed (round 19).
            #
            # No model, no solver, no forecast: it sees the same measured
            # speeds the controller sees and nothing else.
            if t_now % interval == 0:
                k0 = max(0, len(spd) - int(rule.get("avg_s", 30)))
                _w = np.array(spd[k0:], float)
                _ok = np.isfinite(_w)
                _n = _ok.sum(axis=0)
                # an all-empty cell has no measured speed; what is posted there
                # is the best available estimate, as for the controller's input
                vbar = np.where(_n > 0,
                                np.where(_ok, _w, 0.0).sum(axis=0)
                                / np.maximum(_n, 1), posted)
                jam = np.nonzero(vbar < float(rule.get("v_det", 35.0)))[0]
                if len(jam):
                    # the most UPSTREAM jammed cell is the tail of the wave;
                    # the zone goes behind it, clipped to the signed road
                    tail = int(jam.min())
                    hi = min(offset + n_ctrl, tail)
                    lo = max(offset, hi - int(rule.get("n_signs", 4)))
                    rule_zone = (lo, hi) if hi > lo else None
                    rule_until = t_now + int(rule.get("hold_s", 120))
                want = limits[ctrl_sl].copy()
                if rule_zone is not None and t_now < rule_until:
                    want[rule_zone[0] - offset:rule_zone[1] - offset] = \
                        float(rule["v_post"])
                    rule_on_s += interval
                if np.any(np.abs(want - applied) > 1e-9):
                    applied = want
                    _post(applied)
            continue
        if fixed_b is not None:
            # THE REFERENCE EVERY RESULT MUST CLEAR: one reduced limit posted
            # on the controlled segments, no controller and no model at all.
            # Run through this same function rather than a separate script so
            # it is scored by identical code on identical seeds -- including
            # the whole-road/`_win` split, which a standalone reference would
            # not produce and without which it cannot be compared with the
            # objectives on the extended road.
            #
            # `window` restricts the sign to the interval the controller is
            # active. A whole-run sign pays delay through the free-flow and
            # drain phases where it prevents nothing, so comparing a controller
            # against one flatters the controller (ledger, 2026-08-19).
            want = fixed_b if (window is None
                               or window[0] <= t_now < window[1]) else 1.0
            tgt = want * limits[ctrl_sl]
            if np.any(np.abs(tgt - applied) > 1e-9):
                applied = tgt
                _post(applied)
            continue
        if ctrl is None or t_now % interval != 0:
            continue
        rho_seg = dens[-1][seg_idx]
        v_seg = v_edges[seg_idx]
        # Gate the controller on traffic being present, BOTH ways. The ledger
        # records that solving on a partly empty road grinds: the density
        # target is unreachable and there is nothing to control. The original
        # gate only ever switched ON, so once the road drained the controller
        # kept solving into an empty network - which is where 35 of 81 failures
        # in a 2500 s run were coming from, all after t ~ 1700 s.
        if gate_avg_s:
            # GATE ON THE AVERAGE, NOT THE INSTANT. A 200 m two-lane cell holds
            # only ~5 vehicles at these demands, so it is briefly empty even in
            # flowing traffic: measured on the straight road, the instantaneous
            # rule switches the controller off for 108-283 s of a 480 s
            # disturbance window, in gaps of up to 25 s. Averaging the density
            # over one control interval leaves it ON for the whole window in 5
            # of 6 seeds (56 s in the sixth). The controller still receives the
            # INSTANTANEOUS state; only the on/off decision is averaged, so
            # nothing else about the comparison moves.
            k0 = max(0, len(dens) - int(gate_avg_s))
            rho_gate = np.array(dens[k0:], float)[:, seg_idx].mean(axis=0)
        else:
            rho_gate = rho_seg
        active = rho_gate.min() > 1.0
        if not active:
            # RELEASE THE SIGNS. Gating only stopped the controller SOLVING; the
            # last commanded limits stayed posted in SUMO for the rest of the
            # run, so a drained road sat under a 20-40 km/h restriction that no
            # longer had any justification. Harmless for delay on an empty road,
            # but it is wrong, it is plainly visible in every VSL figure, and it
            # taxes the tail of vehicles still clearing.
            if np.any(applied < limits[ctrl_sl] - 1e-9):
                applied = limits[ctrl_sl].copy()
                _post(applied)
            continue
        t_future = t_now + np.arange(ctrl.Np * ctrl.M) * model.p.T_sec
        if offset:
            # THE MODEL'S ORIGIN IS A MID-ROAD CUT, NOT A REAL QUEUE POINT.
            # On the short network the model started where vehicles entered, so
            # the SUMO origin queue `pend` was genuinely its origin queue. Here
            # the model starts 3.6 km downstream of the entrance, so `pend` is
            # a queue somewhere else entirely; feeding it in tells the model it
            # is holding back vehicles it has no relationship with. Measured:
            # doing so drove solver failures to 31-41% against ~10% on the short
            # road, because the state handed to IPOPT was not self-consistent.
            #
            # The correct boundary condition for a cut is zero queue plus the
            # flow actually arriving there, measured over the last control
            # interval on the edge immediately upstream and held across the
            # horizon. That is also self-correcting: if the controller backs
            # traffic up past the cut, the arriving flow it sees falls.
            k0 = max(0, len(counts) - interval)
            cu = np.array(counts[k0:], float)[:, offset - 1]
            vu = np.array(spd[k0:], float)[:, offset - 1]
            q_in = float(np.nanmean(np.where(cu > 0, cu * vu / Lkm[offset - 1], 0.0)))
            w_in = 0.0
            d_end = DEMAND_END_S if demand_end_s is None else float(demand_end_s)
            d_hat = np.where(t_future < d_end + lag_s, q_in, 0.0)
        else:
            w_in = float(pend[-1])
            d_hat = np.where(t_future < DEMAND_END_S, DEMAND_VPH, 0.0)
        t0 = time.time()
        if driver_aware and actuation_lag_s:
            # the traffic is still enforcing the lagged value, not `applied`
            u, info = ctrl.solve(rho_seg, v_seg, w_in, d_hat, applied,
                                 v_enf0=vsl_actual.copy())
        else:
            u, info = ctrl.solve(rho_seg, v_seg, w_in, d_hat, applied)
        t_solve += time.time() - t0
        n_solve += 1
        # PLACEBO FAILURES, for the confound test only (drop_frac = 0 normally).
        #
        # On the long road the per-objective solver-failure rate correlates with
        # measured shockwave reduction at r = -0.87. Two readings fit that
        # equally well: failures CAUSE the smoothing (a failed solve holds the
        # previous limits, so the sign freezes low, which is exactly the
        # behaviour being measured), or aggressive objectives drive the state
        # somewhere hard and failures are a SYMPTOM. Raising the solver budget
        # does not separate them -- measured, max_wall 15->30 with max_iter
        # 300->3000 moved the failure rate 44% -> 41% while doubling run time.
        #
        # This does separate them. Discarding a controlled fraction of
        # SUCCESSFUL solves injects the failure mechanism without the stiff
        # states that normally accompany it. If a benign objective starts
        # looking like the acceleration family as the fraction rises, failures
        # cause the effect; if it does not move, the correlation is a symptom
        # and the ranking stands.
        if info["success"] and drop_frac > 0.0 and rng.random() < drop_frac:
            info = dict(info, success=False)
            n_dropped += 1
        if info["success"]:
            want = np.asarray(u, float)
            if sign_quant_kmh or sign_min_hold_s:
                # WHAT A REAL GANTRY CAN ACTUALLY DISPLAY (added 2026-09-20).
                #
                # Measured on the headline configuration: without the `dvsl`
                # penalty the controller moves the sign at 73 % of control
                # steps, and the MEDIAN move is 2.4 km/h. No deployed VSL
                # system can show 2.4 km/h -- limits are posted in 10 km/h
                # steps and held for minutes, because drivers have to read and
                # react to them. So the headline result, as measured, assumes a
                # continuously variable sign updated every 30 s, and until it
                # is re-measured through this filter it is not a claim about
                # anything deployable.
                #
                # The controller still PLANS in continuous speeds; only what
                # reaches the road is quantised and rate-held, which is exactly
                # the split a real installation has.
                if sign_quant_kmh:
                    want = np.round(want / sign_quant_kmh) * sign_quant_kmh
                    want = np.clip(want, sign_quant_kmh, limits[ctrl_sl])
                if sign_min_hold_s:
                    hold = np.asarray(sign_held_until, float)
                    keep = t_now < hold
                    want = np.where(keep, applied, want)
                    changed = np.abs(want - applied) > 1e-9
                    sign_held_until = np.where(
                        changed, t_now + float(sign_min_hold_s), hold)
            applied = want
            _post(applied)
        else:
            n_fail += 1
    traci.close()

    on_road = np.array(on_road, float)
    pend = np.array(pend, float)
    spd = np.array(spd)
    dens = np.array(dens)
    cnt = np.array(counts, float)
    out = {
        # A rule run MUST NOT be labelled "nocontrol": it has no cost function
        # because it has no optimiser, but it is a policy, and one mislabelled
        # row would poison the baseline every percentage is measured against.
        "cost": (cost if cost else
                 ("fixed_b%.2f" % fixed_b if fixed_b is not None else
                  ("rule_v%d_n%d_h%d" % (rule["v_post"], rule.get("n_signs", 4),
                                         rule.get("hold_s", 120))
                   if rule is not None else "nocontrol"))),
        "seed": seed, "fixed_b": fixed_b, "window": list(window) if window else None,
        "TTS_h": float((on_road + pend).sum() / 3600.0),
        "TTS_road_h": float(on_road.sum() / 3600.0),
        "throughput": float(arrived),
        # solver health reported with every number: the field does not do this
        "n_solves": n_solve, "n_failed": n_fail,
        "solve_time_s": round(t_solve, 1),
        "mean_vsl": float(np.mean(vsl_hist)),
        "min_vsl": float(np.min(vsl_hist)),
        # WHICH IPOPT exit, not just how many failed. A run that fails on
        # Maximum_Iterations_Exceeded is stiff and may just need more budget; one
        # that fails Infeasible_Problem_Detected or Restoration_Failed is being
        # handed a state the model cannot reconcile. The two call for opposite
        # fixes, and counting only `n_failed` cannot tell them apart.
        "solver_status": (dict(Counter(s for _, s, _, _ in ctrl.stats))
                          if ctrl is not None else {}),
        # provenance: which road, which model span, which horizon
        "drop_frac": drop_frac, "n_dropped": n_dropped,
        "ctrl_interval_s": interval,
        "offset": int(offset), "n_edges": len(edges),
        "demand_lag_s": round(lag_s, 1),
        "Np": int(ctrl.Np) if ctrl is not None else None,
    }
    out.update(shockwave_metrics(spd, dens, 1.0, Lkm, lanes))
    # SHOCKWAVE DETECTION, added 2026-09-05. `accel_rms` is 93% its spatial
    # term on the uncontrolled baseline, i.e. mostly vehicles decelerating
    # into a STANDING queue, which is not a shockwave. These metrics separate
    # moving jams (detached, propagating upstream) from the standing queue,
    # so the cost-function ranking can be stated on the thing the research
    # question is actually about. Bottleneck cells are identified from the
    # lane counts, not hard-coded, so this survives a geometry change.
    _ln = np.asarray(lanes).ravel()
    _one = np.nonzero(_ln < 1.5)[0]
    _merge = int(_one[0]) if len(_one) else len(_ln) - 1
    if return_fields:
        # the raw space-time fields, for plotting only. NOT stored by the
        # sweep -- these are ~28 x 3500 floats a run and would bloat every
        # shard. Opt-in, and only `control/plot_wave_anatomy.py` asks.
        out["_field_v"] = np.asarray(spd).tolist()
        out["_field_merge"] = _merge
        out["_field_cnt"] = np.asarray(cnt).tolist()
        out["_field_vsl"] = np.asarray(vsl_hist).tolist()
    out.update(wave_metrics(spd, Lkm, 1.0, merge_idx=_merge))
    if dists:
        # The FIRST disturbance keeps the column names every stored row and
        # every analysis script already uses; the rest are carried alongside,
        # so a one-disturbance run is scored exactly as before.
        out.update(dists[0].summary())
        out["dist_n"] = len(dists)
        if len(dists) > 1:
            out["dist_all"] = [d.summary() for d in dists]
        # the seeding window must cover the whole disturbance: with a ramp the
        # jam can start a couple of minutes after the trigger, and further
        # downstream (see jam_metrics). With several disturbances it must also
        # reach the last one, or the waves it plants are scored as background
        # congestion instead of as part of the event.
        win = max(d.t + d.ramp + d.hold for d in dists) - dists[0].t + 120.0
        out.update(jam_metrics(spd, cnt, Lkm, dists[0].x_km, dists[0].t, 1.0,
                               seed_win_s=win))
    if rule is not None:
        out.update(rule_req=dict(rule), rule_on_s=rule_on_s)
    if sign_quant_kmh or sign_min_hold_s:
        out.update(sign_quant_kmh=sign_quant_kmh,
                   sign_min_hold_s=sign_min_hold_s)
    if actuation_lag_s or compliance:
        out.update(actuation_lag_s=actuation_lag_s, compliance=compliance,
                   driver_aware=bool(driver_aware))
    if v_min_kmh is not None:
        out.update(v_min_kmh=float(v_min_kmh))
    if dv_max_kmh is not None:
        out.update(dv_max_kmh=float(dv_max_kmh))
    if solver_opts and ctrl is not None:
        out.update(solver_opts={k: v for k, v in solver_opts.items()},
                   n_retry=ctrl.n_retry, n_retry_ok=ctrl.n_retry_ok,
                   n_shift=ctrl.n_shift)
    if offset:
        # THE SAME METRICS OVER THE ORIGINAL WINDOW ONLY (the nine controlled
        # segments plus the bottleneck), suffixed `_win`.
        #
        # Both are needed and they answer different questions. The whole-road
        # figures are the honest ones — they are the entire point of extending
        # the road, because a queue pushed upstream now lands on measured
        # tarmac and is counted. But they are NOT comparable with any number
        # measured on the short network: 3.6 km of mostly free-flowing approach
        # dilutes every vehicle-weighted average, so a percentage computed over
        # 5.9 km and one computed over 2.3 km differ even when the traffic is
        # identical. The `_win` columns cover exactly the road the short-network
        # results cover, so short-vs-long is a like-for-like comparison, and the
        # gap between `_win` and whole-road is itself the measurement of how
        # much an objective displaces upstream.
        w = slice(offset, len(edges))
        out["TTS_road_win_h"] = float(cnt[:, w].sum() / 3600.0)
        out["TTS_upstream_h"] = float(cnt[:, :offset].sum() / 3600.0)
        for k, v in shockwave_metrics(spd[:, w], dens[:, w], 1.0,
                                      Lkm[w], lanes[w]).items():
            out[k + "_win"] = v
        # Does the queue still reach the network entrance? If it does, the
        # extension is too short and the metering artefact is back. Reported so
        # every run carries its own validity check rather than relying on a
        # one-off sanity run.
        rho_up = dens[:, :offset]
        out["entry_dens_max"] = float(np.nanmax(dens[:, 0]))
        out["upstream_dens_max"] = float(np.nanmax(rho_up))
        congested = rho_up > model.p.rho_crit
        reach = np.where(congested.any(axis=0))[0]
        out["queue_reach_edges_upstream"] = int(offset - reach.min()) if len(reach) else 0
    # Every per-run scratch file this run created, deleted in a `finally` no
    # matter how the parsing goes. Two defects this replaces, both of which
    # only bite on a long unattended sweep:
    #   * the emissions XML was never deleted at all (the loop that looked like
    #     it did was `for tmp in (...): pass`), leaving one ~170 kB pair per run
    #     -- 907 files had accumulated;
    #   * the ~90 MB SSM file was removed only AFTER parse_safety/parse_drac
    #     succeeded, so any parse failure leaked the whole file. A truncated SSM
    #     write (which is what a full disk produces) therefore made the next
    #     run more likely to fail, not less.
    # A full disk is the failure mode that silently ends a 48 h run, so cleanup
    # must not depend on the happy path.
    scratch = [add_path, em_path]
    if instrument:
        scratch.append(os.path.join(SCRATCH, "ssm_%s.xml" % tag))
    try:
        if os.path.exists(em_path):
            out.update(parse_emissions(em_path,
                                       {e: L for e, L in zip(edges, Lkm)}))
        if instrument:
            _ssm = os.path.join(SCRATCH, "ssm_%s.xml" % tag)
            out.update(parse_safety(_ssm))
            # DRAC as well: it needs no speed filter, so unlike TET/TIT it is
            # not biased by a controller changing how long traffic spends below
            # 5 m/s. See validation/sumo_metrics.DRAC_STAR for why that matters.
            out.update(parse_drac(_ssm))
    finally:
        for tmp in scratch:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="skip the SSM device")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--costs", type=str, default="")
    ap.add_argument("--steps", type=int, default=SIM_STEPS)
    ap.add_argument("--model", default="shipped",
                    choices=("shipped", "s14", "bounded"),
                    help="'bounded' = Bounded-METANET + scaled VSL cap, the "
                         "final configuration -> data/phase_d_bounded.json")
    a = ap.parse_args()

    names = [c for c in a.costs.split(",") if c] or list(COSTS)
    seeds = list(range(1, a.seeds + 1))
    geom = load_geometry()
    if a.model == "bounded":
        # the final configuration: published prediction model + a VSL cap that
        # actually binds
        mp = MetanetParams(**PARAMS_BM, vsl_scaled=True)
    elif a.model == "s14":
        mp = MetanetParams(**PARAMS_S14)
    else:
        mp = MetanetParams()
    model = MetanetModel(geom, mp)
    if a.model == "bounded":
        results = RESULTS_BM if a.screen else RESULTS_BM_SSM
    elif a.model == "s14":
        results = RESULTS_S14 if a.screen else RESULTS_S14_SSM
    else:
        results = RESULTS

    rows = []
    if os.path.exists(results):
        rows = json.load(open(results))
    done = {(r["cost"], r["seed"], r.get("instrumented", False)) for r in rows}

    todo = [(None, s) for s in seeds] + [(c, s) for c in names for s in seeds]
    print("%d runs, instrument=%s" % (len(todo), not a.screen))
    for cost, seed in todo:
        key = (cost or "nocontrol", seed, not a.screen)
        if key in done:
            print("  skip %-18s seed %d (done)" % (key[0], seed))
            continue
        t0 = time.time()
        try:
            r = run_closed_loop(cost, seed, instrument=not a.screen,
                                steps=a.steps, model=model, geom=geom)
        except Exception as e:
            print("  FAIL %-18s seed %d: %s" % (cost, seed, str(e)[:70]))
            continue
        r["instrumented"] = not a.screen
        rows.append(r)
        json.dump(rows, open(results, "w"), indent=1)
        print("  %-18s seed %d  TTS %7.1f  accel %6.3f  cvs %5.3f  "
              "fails %d/%d  [%.0fs]"
              % (r["cost"], seed, r["TTS_h"], r["accel_rms"], r["cvs"],
                 r["n_failed"], r["n_solves"], time.time() - t0))
    print("\nwrote %s (%d rows)" % (results, len(rows)))


if __name__ == "__main__":
    main()
