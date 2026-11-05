# -*- coding: utf-8 -*-
"""
Rounds 23-32: the DEPLOYABILITY programme (straight13..straight20, mergerule,
mergedeploy). Read-only.

These are the rounds that ask whether the straight-road headline survives an
actuator and a driver that could exist: a gantry that shows 10 km/h steps and
holds them, a controller that replans on that grid, a reaction lag, imperfect
compliance -- and whether a reactive rule (no model, no solver) does the same
job. `analyse_straight_r18.py` covers rounds 18-22; this file covers everything
after it.

Conventions, identical to the rest of the project:
  * paired on seed WITHIN demand and within disturbance, two-sided Wilcoxon
    signed-rank, reported against no control AND the constant b=0.95 sign;
  * change is  sum(policy - baseline) / sum(baseline)  over the paired seeds
    (a ratio of totals, not a mean of ratios -- the latter blows up when a
    baseline seed has almost no jam);
  * primary metric is jam vehicle-hours at v_c=30 (`jam_vc30_veh_h`),
    pre-registered 2026-09-11; guards are total delay per vehicle including
    the origin queue, the entrance-queue component on its own, and total
    post-disturbance congestion.

A CELL is identified by (cost, Np, model, demand, disturbance, control
interval, sign quantisation, sign minimum hold, actuation lag, compliance,
rule parameters). Every one of those is an experiment, not an incidental:
rounds 25/29 run the SAME objective at 30/60/120 s intervals with and without
a 10 km/h / 60 s sign filter, and round 30 sweeps the rule's detection
threshold without changing its label. Pooling on anything less silently blends
them.

Prediction model for every MPC row here: Bounded-METANET PARAMS_BM, vsl_scaled,
Np 7 unless stated (round 24 is the Np=10 block), 30 s moves unless stated.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from control.sweep import STRAIGHT_DIST, STRAIGHT_FREE                   # noqa: E402

DIRS = [os.path.join(_ROOT, "data", "sweep_straight"),
        os.path.join(_ROOT, "data", "sweep")]
PRIMARY = "jam_vc30_veh_h"
SUSPEND_S = 3000.0                       # see analyse_straight_r18.load()


# --------------------------------------------------------------------- loading
def load(net):
    rows = []
    for d in DIRS:
        for f in glob.glob(os.path.join(d, "shard*.json")):
            rows += json.load(open(f))
    return [r for r in rows if r.get("ok") and r.get("net") == net
            and (r.get("wall_s") or 0.0) <= SUSPEND_S]


def dkey(d):
    return json.dumps(d, sort_keys=True)


D0 = dkey(STRAIGHT_DIST)
DNONE = dkey(None)


def _no_solver(cost):
    """No-control, constant-sign and rule rows run no NLP at all."""
    c = str(cost)
    return (cost is None or c == "nocontrol" or c.startswith("rule_")
            or c.startswith("fixed_b"))


def _budget(cost, max_wall, max_iter):
    """Solver budget as part of a cell's identity.

    A row produced under a different budget is a different measurement -- the
    job key in sweep.py has said so since 2026-08-30, and rounds 33-37 run with
    the wall clock disabled (`max_wall` 1e6) while everything before them ran
    under a 15 s wall cap. Without this in the key the re-baselined `tts` cell
    of round 37 would silently collide with the round-13 one and only one of
    them would survive.

    Rows with no solver carry None, so a single baseline serves both budgets.
    """
    if _no_solver(cost):
        return None
    if max_wall is None:
        return "wall15"                      # the pre-2026-09-22 default
    return "det" if max_wall > 1e5 else "wall%g" % max_wall


def C(cost, demand=None, dist=DNONE, Np=7, model="bounded", interval=30,
      quant=None, hold=None, lag=None, comp=None, rule=None, budget="wall15"):
    """The canonical cell key. Keep the argument order stable.

    `budget` defaults to the wall-clock setting every pre-round-33 result used,
    so existing call sites are unchanged; rounds 33-37 pass budget="det".
    """
    return (cost, Np, model, demand, dist, interval, quant, hold, lag, comp,
            dkey(rule), None if _no_solver(cost) else budget)


def cell_of(r):
    return C(r["cost"], r.get("demand_req"), dkey(r.get("dist_req")),
             r.get("Np") or 7, r.get("model_name") or "bounded",
             r.get("ctrl_interval_s") or 30, r.get("sign_quant_kmh"),
             r.get("sign_min_hold_s"), r.get("lag_req"), r.get("comp_req"),
             r.get("rule_req"),
             _budget(r["cost"], r.get("max_wall"), r.get("max_iter_req")))


def pool(rows, instrumented=False):
    """{(cell, seed): row}, checking that duplicate cells agree exactly.

    Duplicates are expected and welcome: shared cells (no control, `tts`,
    b=0.95) were re-run in several blocks, and agreement is the check that
    pooling across blocks is legitimate. It has caught a real blend before
    (analyse_straight_r18, Np).

    Where a duplicate DISAGREES it is kept and reported, not asserted away --
    see the reproducibility section in main(). Selection is deterministic
    (lowest `exp` then lowest `i`) so this file's numbers do not depend on
    shard iteration order.
    """
    out, agree, conflict = {}, 0, []
    # Prefer a row that actually carries the wave metrics: the same long-road
    # cell exists in blocks run before and after the wave detector was built,
    # and the pre-detector row would otherwise win on name order and empty the
    # whole merge comparison.
    def rank(x):
        return (0 if x.get("wave_vc30_front_reach") is not None else 1,
                str(x.get("exp")), x.get("i", 0))

    for r in sorted(rows, key=rank):
        if bool(r.get("instrumented")) != instrumented:
            continue
        k = cell_of(r) + (r["seed"],)
        if k in out:
            a, b = out[k], r
            if all(abs((a.get(m) or 0.0) - (b.get(m) or 0.0)) < 1e-9
                   for m in (PRIMARY, "TTS_h")):
                agree += 1
            else:
                conflict.append((k, a, b))
            continue
        out[k] = r
    return out, agree, conflict


# --------------------------------------------------------------------- metrics
def spv(r):
    """Total delay per vehicle, origin queue INCLUDED (the guard metric)."""
    return r["TTS_h"] * 3600.0 / max(r["throughput"], 1.0)


def road_spv(r):
    return r["TTS_road_h"] * 3600.0 / max(r["throughput"], 1.0)


def queue_spv(r):
    """The entrance-queue component on its own -- the metering channel."""
    return (r["TTS_h"] - r["TTS_road_h"]) * 3600.0 / max(r["throughput"], 1.0)


def failpct(r):
    return 100.0 * (r.get("n_failed") or 0) / max(r.get("n_solves") or 1, 1)


def get(r, m):
    """Missing metric -> nan, never KeyError.

    Long-road rows predating the wave detector (round 10, 2026-09-05) carry no
    `wave_*` fields, and straight-road rows carry no merge metrics. A row that
    never measured a quantity must drop out of that comparison rather than
    crash it or, worse, be silently treated as zero.
    """
    if callable(m):
        return m(r)
    v = r.get(m)
    return float("nan") if v is None else v


# --------------------------------------------------------------------- pairing
def paired(P, pol, base, metric, seeds=range(1, 31), minn=6):
    xs, ys = [], []
    for s in seeds:
        a, b = P.get(pol + (s,)), P.get(base + (s,))
        if a is None or b is None:
            continue
        xs.append(get(a, metric))
        ys.append(get(b, metric))
    xs, ys = np.array(xs, float), np.array(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < minn:
        return None
    d = xs - ys
    try:
        p = wilcoxon(xs, ys).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    pct = 100.0 * d.sum() / ys.sum() if ys.sum() else float("nan")
    return dict(n=len(xs), pct=pct, dmean=float(d.mean()),
                base=float(ys.mean()), val=float(xs.mean()), p=p,
                wins=int((d < 0).sum()))


def mean(P, cell, metric, seeds=range(1, 31)):
    v = [get(P[cell + (s,)], metric) for s in seeds if cell + (s,) in P]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def fmt(res):
    if res is None:
        return "    n/a       "
    return "%+7.1f%% p=%.3f n=%-2d" % (res["pct"], res["p"], res["n"])


def dfmt(res, unit=" s"):
    if res is None:
        return "   n/a  "
    return "%+7.1f%s" % (res["dmean"], unit)


def head(s):
    print("\n" + "=" * 92)
    print(s)
    print("=" * 92)


# ---------------------------------------------------------- straight-road rows
def row(P, pol, demand, dist=D0, label=None, base_nc="nocontrol",
        seeds=range(1, 31)):
    """One line: jam vs no control, jam vs the sign, delay, queue, reach."""
    nc = C(base_nc, demand, dist)
    sg = C("fixed_b0.95", demand, dist)
    a = paired(P, pol, nc, PRIMARY, seeds)
    b = paired(P, pol, sg, PRIMARY, seeds)
    d = paired(P, pol, nc, spv, seeds)
    q = paired(P, pol, nc, queue_spv, seeds)
    po = paired(P, pol, nc, "jam_vc30_post_veh_h", seeds)
    print("   %-34s jam %s | vs sign %s | delay %s queue %s | post-cong %s | sign %.1f fail %.0f%%"
          % (label or pol[0], fmt(a), fmt(b), dfmt(d), dfmt(q), fmt(po),
             mean(P, pol, "mean_vsl", seeds), mean(P, pol, failpct, seeds)))
    return a


# =============================================================================
def main():
    S = load("straight")
    PS, agree, conflict = pool(S)
    print("straight: %d ok rows -> %d pooled cell/seed pairs (%d exact duplicates)"
          % (len(S), len(PS), agree))

    head("REPRODUCIBILITY -- the MPC closed loop is NOT bit-reproducible, and\n"
         "the comparators are. Measured on cells that were run twice.")
    print(" %d duplicate cell/seed pairs agree to 1e-9; %d DISAGREE." % (agree, len(conflict)))
    for k, a, b in conflict[:12]:
        print("   %-28s d=%-5s seed %-2d  %s/%s  jam %7.3f vs %7.3f (%+.1f%%)"
              "  fails %s vs %s"
              % (str(k[0])[:28], k[3], k[-1], a.get("exp"), b.get("exp"),
                 a[PRIMARY], b[PRIMARY],
                 100.0 * (a[PRIMARY] - b[PRIMARY]) / b[PRIMARY] if b[PRIMARY] else float("nan"),
                 a.get("n_failed"), b.get("n_failed")))
    ismpc = [c for c in conflict
             if c[1].get("cost") and not str(c[1]["cost"]).startswith("rule_")
             and c[1].get("fixed_b") is None]
    print(" of %d disagreements, %d are MPC rows and %d are no-control / fixed-sign /"
          " rule rows." % (len(conflict), len(ismpc), len(conflict) - len(ismpc)))
    st = {}
    for r in S:
        if isinstance(r.get("solver_status"), dict):
            for kk, vv in r["solver_status"].items():
                st[kk] = st.get(kk, 0) + vv
    tot = sum(st.values()) or 1
    print(" solver exit status over every straight-road solve (n=%d): %s"
          % (tot, ", ".join("%s %.1f%%" % (kk, 100.0 * vv / tot)
                            for kk, vv in sorted(st.items(), key=lambda x: -x[1])[:4])))
    print(" MECHANISM: `max_wall` is a WALL-CLOCK budget, so a solve that is close\n"
          " to it succeeds on an idle machine and is abandoned on a busy one. The\n"
          " abandoned solve holds the previous limits, which changes the control and\n"
          " therefore the whole trajectory. The plant is deterministic; the\n"
          " CONTROLLER is not. This is a real caveat for every MPC number and it\n"
          " does not apply to the rule or the constant sign, which are exact.")

    FREE = list(STRAIGHT_FREE)
    DEM = (4600, 4800)

    # ---------------------------------------------------------------- round 23
    head("ROUND 23 (straight13) -- THE RULE COMPARATOR, SCORED HONESTLY\n"
         "Parameters chosen on seeds 1-10, then scored on the HELD-OUT seeds 11-20.\n"
         "Round 19 picked v_post/n_signs on the same seeds it then reported.")
    for d in DEM:
        rules = sorted({r["cost"] for r in S
                        if str(r.get("cost", "")).startswith("rule_")
                        and r.get("demand_req") == d
                        and dkey(r.get("dist_req")) == D0
                        and r.get("exp") == "straight13"})
        nc = C("nocontrol", d, D0)
        print(" demand %d" % d)
        sel = []
        for pol in rules:
            cell = C(pol, d, D0, rule=dict(
                v_post=float(pol.split("_")[1][1:]), n_signs=int(pol.split("_")[2][1:]),
                hold_s=int(pol.split("_")[3][1:]), v_det=35.0, avg_s=30))
            tune = paired(PS, cell, nc, PRIMARY, range(1, 11))
            test = paired(PS, cell, nc, PRIMARY, range(11, 21))
            dly = paired(PS, cell, nc, spv, range(11, 21))
            if tune is None:
                continue
            sel.append((tune["pct"], pol, cell, test, dly))
            print("   %-20s tune(1-10) %s   held-out(11-20) %s  delay %s"
                  % (pol, fmt(tune), fmt(test), dfmt(dly)))
        if not sel:
            continue
        sel.sort()
        pct, pol, cell, test, dly = sel[0]
        print("   -> BEST ON TUNING SEEDS: %s (%.1f%%);  HELD-OUT %s, delay %s"
              % (pol, pct, fmt(test), dfmt(dly)))
        for mpc in ("tts", "tts:1|terminal:5"):
            m = C(mpc, d, D0)
            print("      MPC %-18s held-out(11-20) %s  delay %s"
                  % (mpc, fmt(paired(PS, m, nc, PRIMARY, range(11, 21))),
                     dfmt(paired(PS, m, nc, spv, range(11, 21)))))
            print("      MPC vs best rule, paired:  jam %s  delay %s"
                  % (fmt(paired(PS, m, cell, PRIMARY, range(11, 21))),
                     dfmt(paired(PS, m, cell, spv, range(11, 21)))))

    # ---------------------------------------------------------------- round 24
    head("ROUND 24 (straight14) -- DOES Np=10 MAKE THE MPC SURGICAL?\n"
         "Np=10 against the same objective at the project default Np=7.")
    for d in DEM:
        print(" demand %d" % d)
        for cost in ("tts", "tts:1|terminal:5", "dens_rate:1|tts:1"):
            c10, c7 = C(cost, d, D0, Np=10), C(cost, d, D0, Np=7)
            row(PS, c10, d, label="%s  Np=10" % cost)
            row(PS, c7, d, label="%s  Np=7" % cost)
            print("      Np=10 vs Np=7 paired:  jam %s  delay %s  sign %.1f -> %.1f"
                  % (fmt(paired(PS, c10, c7, PRIMARY)),
                     dfmt(paired(PS, c10, c7, spv)),
                     mean(PS, c7, "mean_vsl"), mean(PS, c10, "mean_vsl")))

    # ---------------------------------------------------------------- round 25
    head("ROUND 25 (straight15) -- IS THE RESULT ABOUT AN ACTUATOR THAT CAN EXIST?\n"
         "Each constraint separately: slower moves (60/120 s), 10 km/h steps,\n"
         "a 60 s minimum hold. The rule is run through the same filter.")
    variants = [("ideal            ", dict()),
                ("interval 60 s    ", dict(interval=60)),
                ("interval 120 s   ", dict(interval=120)),
                ("quantised 10 km/h", dict(quant=10.0)),
                ("min hold 60 s    ", dict(hold=60.0)),
                ("quant + hold     ", dict(quant=10.0, hold=60.0))]
    for d in DEM:
        print(" demand %d" % d)
        for cost in ("tts", "tts:1|terminal:5", "dens_rate:1|tts:1"):
            print("  -- %s" % cost)
            for nm, kw in variants:
                row(PS, C(cost, d, D0, **kw), d, label="   " + nm)
        rl = "rule_v40_n9_h120" if d == 4600 else "rule_v60_n9_h120"
        rp = dict(v_post=40.0 if d == 4600 else 60.0, n_signs=9, hold_s=120,
                  v_det=35.0, avg_s=30)
        print("  -- %s (reactive rule)" % rl)
        row(PS, C(rl, d, D0, rule=rp), d, label="   ideal            ")
        row(PS, C(rl, d, D0, quant=10.0, hold=60.0, rule=rp), d,
            label="   quant + hold     ")

    # ---------------------------------------------------------------- round 26
    head("ROUND 26 (straight16) -- `restrict` / `restrict_l1` AT 0.01-0.3")
    for d in DEM:
        print(" demand %d" % d)
        for cost in sorted({r["cost"] for r in S if r.get("exp") == "straight16"
                            and r.get("demand_req") == d}):
            row(PS, C(cost, d, D0), d)

    # ---------------------------------------------------------------- round 28
    head("ROUND 28 (straight17) -- DO DRIVERS HAVE TO OBEY INSTANTLY?\n"
         "Actuation lag 10/20/40 s; compliance 0.8/0.9 of the posted value.\n"
         "The rule is exposed to exactly the same plant.")
    for d in DEM:
        print(" demand %d" % d)
        rl = "rule_v40_n9_h120" if d == 4600 else "rule_v60_n9_h120"
        rp = dict(v_post=40.0 if d == 4600 else 60.0, n_signs=9, hold_s=120,
                  v_det=35.0, avg_s=30)
        for cost in ("tts", "tts:1|terminal:5", "dens_rate:1|tts:1", rl):
            rule = rp if cost.startswith("rule_") else None
            print("  -- %s" % cost)
            base = C(cost, d, D0, rule=rule)
            row(PS, base, d, label="   ideal            ")
            for lag in (10.0, 20.0, 40.0):
                row(PS, C(cost, d, D0, lag=lag, rule=rule), d,
                    label="   lag %4.0f s        " % lag)
            for comp in (0.9, 0.8):
                row(PS, C(cost, d, D0, comp=comp, rule=rule), d,
                    label="   compliance %.1f   " % comp)

    # ---------------------------------------------------------------- round 29
    head("ROUND 29 (straight18) -- THE BEST DEPLOYABLE MPC\n"
         "Move blocking (replan on the actuator's grid) COMBINED with 10 km/h\n"
         "steps -- the combination round 25 never ran. Headline at 30 seeds.")
    depl = [("block 60  + quant      ", dict(interval=60, quant=10.0)),
            ("block 120 + quant      ", dict(interval=120, quant=10.0)),
            ("block 60  + quant+hold ", dict(interval=60, quant=10.0, hold=60.0))]
    for d in DEM:
        print(" demand %d" % d)
        for cost in ("tts", "tts:1|terminal:5", "dens_rate:1|tts:1"):
            print("  -- %s" % cost)
            row(PS, C(cost, d, D0), d, label="   ideal sign        ")
            for nm, kw in depl:
                row(PS, C(cost, d, D0, **kw), d, label="   " + nm)
        rl = "rule_v40_n9_h120" if d == 4600 else "rule_v60_n9_h120"
        rp = dict(v_post=40.0 if d == 4600 else 60.0, n_signs=9, hold_s=120,
                  v_det=35.0, avg_s=30)
        print("  -- %s (reactive rule, same filter)" % rl)
        row(PS, C(rl, d, D0, quant=10.0, hold=60.0, rule=rp), d,
            label="   quant + hold     ")
        print("   [30-seed head-to-head, deployable MPC vs deployable rule]")
        mm = C("tts", d, D0, interval=60, quant=10.0)
        rr = C(rl, d, D0, quant=10.0, hold=60.0, rule=rp)
        print("      MPC(block60+quant) vs rule(quant+hold): jam %s  delay %s"
              % (fmt(paired(PS, mm, rr, PRIMARY)), dfmt(paired(PS, mm, rr, spv))))
        for c, nm in ((mm, "MPC block60+quant"), (rr, "rule quant+hold")):
            a10 = paired(PS, c, C("nocontrol", d, D0), PRIMARY, range(1, 11))
            a30 = paired(PS, c, C("nocontrol", d, D0), PRIMARY)
            print("      %-20s seeds 1-10 %s -> all %s" % (nm, fmt(a10), fmt(a30)))

    # ---------------------------------------------------------------- round 30
    head("ROUND 30 (straight19) -- DOES THE RULE'S WIN GENERALISE?\n"
         "The rule on the conditions the MPC was already run on (location,\n"
         "repeats, severity, duration), and its detection threshold swept.")
    R40 = dict(v_post=40.0, n_signs=9, hold_s=120, v_det=35.0, avg_s=30)
    conds = []
    for edge in ("c4", "c8", "d1", "d3", "d5"):
        conds.append(("location %-3s" % edge,
                      dkey(dict(STRAIGHT_DIST, edge=edge))))
    for pat in ([600, 1020], [600, 900, 1200], [600, 780]):
        conds.append(("repeat %-13s" % str(pat),
                      dkey([dict(STRAIGHT_DIST, t=t) for t in pat])))
    for vf in (0.10, 0.20):
        conds.append(("severity %.2f" % vf, dkey(dict(STRAIGHT_DIST, v_frac=vf))))
    for hs in (120, 480):
        conds.append(("duration %3d s" % hs, dkey(dict(STRAIGHT_DIST, hold_s=hs))))
    for nm, dk in conds:
        nc = C("nocontrol", 4600, dk)
        base = mean(PS, nc, PRIMARY)
        if not np.isfinite(base):
            continue
        if base < 1.0:
            # Project rule (plot_straight_effect, 2026-09-15): below 1 veh-h
            # there is no jam to reduce and a percentage on that baseline is
            # noise on a zero -- d5 and severity 0.20 are exactly that case.
            print(" %s  NO JAM FORMS (no-control %.2f veh-h) -- percentages "
                  "meaningless, cell dropped" % (nm, base))
            continue
        print(" %s  (no-control jam %.1f veh-h)" % (nm, base))
        row(PS, C("rule_v40_n9_h120", 4600, dk, rule=R40), 4600, dk,
            label="   rule v40 n9 h120 ")
        for cost in ("tts", "tts:1|terminal:5"):
            row(PS, C(cost, 4600, dk), 4600, dk, label="   MPC %-14s" % cost)
    print("\n DETECTION THRESHOLD v_det (standard disturbance).")
    print(" NOTE: round 30 swept v_det on the v_post=40 rule at BOTH demands, so")
    print(" the 4800 rows here are v40, not the v60 rule used as the 4800 default.")
    for d in DEM:
        for v_det in (25.0, 30.0, 35.0, 45.0):
            row(PS, C("rule_v40_n9_h120", d, D0, rule=dict(R40, v_det=v_det)), d,
                label="   d=%d v40 v_det %4.0f" % (d, v_det))

    # ---------------------------------------------------------------- round 31
    head("ROUND 31 (straight20) -- `restrict` AT WEIGHTS THAT TRADE OFF,\n"
         "with and without the real sign. The question is whether pushing the\n"
         "MPC toward sparse, deep control makes it degrade LESS under filtering.")
    for d in DEM:
        print(" demand %d" % d)
        for w in ("1e-06", "3e-06", "1e-05", "3e-05", "0.0001"):
            cost = "tts:1|restrict:%s" % w
            ideal = C(cost, d, D0)
            filt = C(cost, d, D0, quant=10.0, hold=60.0)
            row(PS, ideal, d, label="   %-22s ideal" % cost)
            row(PS, filt, d, label="   %-22s filtered" % cost)
            a, b = paired(PS, ideal, C("nocontrol", d, D0), PRIMARY), \
                paired(PS, filt, C("nocontrol", d, D0), PRIMARY)
            if a and b:
                print("      degradation under the real sign: %+.1f%% -> %+.1f%%"
                      % (a["pct"], b["pct"]))
        for cost in ("tts",):
            a = paired(PS, C(cost, d, D0), C("nocontrol", d, D0), PRIMARY)
            b = paired(PS, C(cost, d, D0, quant=10.0, hold=60.0),
                       C("nocontrol", d, D0), PRIMARY)
            if a and b:
                print("   [reference] plain %s: %+.1f%% ideal -> %+.1f%% filtered"
                      % (cost, a["pct"], b["pct"]))

    # -------------------------------------------------------- rounds 27 and 32
    L = load("long")
    PL, agreel, conflictl = pool(L)
    dupl = agreel
    head("ROUNDS 27 & 32 (mergerule, mergedeploy) -- THE MERGE ROAD\n"
         "%d ok long-road rows -> %d cell/seed pairs (%d exact duplicates).\n"
         "Merge metrics: front reach (km, the wave metric), road time in the\n"
         "controlled window, and accel_rms_win. There is no planted jam here."
         % (len(L), len(PL), dupl))
    ncl = C("nocontrol")
    print(" no control: front reach %.2f km, road %.1f s/veh, accel_rms_win %.4f"
          % (mean(PL, ncl, "wave_vc30_front_reach"), mean(PL, ncl, road_spv),
             mean(PL, ncl, "accel_rms_win")))

    def mrow(cell, label):
        f = paired(PL, cell, ncl, "wave_vc30_front_reach")
        a = paired(PL, cell, ncl, "accel_rms_win")
        r = paired(PL, cell, ncl, road_spv)
        q = paired(PL, cell, ncl, queue_spv)
        print("   %-30s front %s | accel %s | road %s queue %s | sign %.1f"
              % (label, fmt(f), fmt(a), dfmt(r), dfmt(q),
                 mean(PL, cell, "mean_vsl")))

    print("\n ROUND 32: the surviving merge claim under a realistic sign")
    for cost in ("tts", "tts_dvsl", "accel_tts_queue", "delay_dvsl"):
        ideal, filt = C(cost), C(cost, quant=10.0, hold=60.0)
        mrow(ideal, "%s  ideal" % cost)
        mrow(filt, "%s  quant+hold" % cost)
        print("      filtered vs ideal, paired: front %s  road %s"
              % (fmt(paired(PL, filt, ideal, "wave_vc30_front_reach")),
                 dfmt(paired(PL, filt, ideal, road_spv))))

    print("\n [determinism check] mergedeploy unfiltered vs the stored round-10")
    print(" rows for the same cells must be bit-identical; the pool() assert")
    print(" above is that check, and it passed for %d duplicate cells." % dupl)

    print("\n ROUND 27: the reactive rule on the merge road")
    for pol in sorted({r["cost"] for r in L
                       if str(r.get("cost", "")).startswith("rule_")}):
        v, n, h = pol.split("_")[1:]
        mrow(C(pol, rule=dict(v_post=float(v[1:]), n_signs=int(n[1:]),
                              hold_s=int(h[1:]), v_det=35.0, avg_s=30)), pol)


if __name__ == "__main__":
    main()
