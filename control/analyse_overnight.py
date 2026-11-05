# -*- coding: utf-8 -*-
"""
Rounds 39-41 (2026-10-05 overnight, LAPTOP). Read-only.

  39  data/sweep_vmin       is the straight-road MPC result set by the solver?
                            (never-reached v_min 20 vs 40), plus the deployable
                            `tts` on the deterministic budget on the laptop
  40  data/sweep_objretest  7 objectives x {ideal, realistic sign} x {obey, 1.5x}
  41  data/sweep_bmjam      the MPC on PARAMS_BM_JAM (calibrated on jam
                            clearance): does the sign now matter, and does it
                            stop depending on the never-reached bound?

MPC rows here are paired ONLY with each other (all laptop) or with no-control
/ rule rows, which reproduce exactly on any machine (ledger 2026-09-29). They
are pooled under an "L" prefix so they can never collide with desktop rows of
the same cell. Primary metric, pairing and test as analyse_deploy.
"""

import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from control.analyse_deploy import (C, D0, PRIMARY, SUSPEND_S, cell_of,  # noqa: E402
                                    dfmt, failpct, fmt, head, load, mean,
                                    paired, pool, queue_spv, spv)

SEEDS = range(1, 11)
DIRS = ("sweep_vmin", "sweep_objretest", "sweep_bmjam")


def rule30():
    return dict(v_post=40.0, n_signs=9, hold_s=120, v_det=30.0, avg_s=30)


def L(cost, d, cfg="ideal", comp=None, model="bounded", vmin=None, dvmax=None,
      solver=False):
    interval, quant = {"ideal": (30, None), "block": (60, None),
                       "deploy": (60, 10.0)}[cfg]
    # `solver` = the improved failure handling of 2026-10-06 (adaptive IPOPT,
    # 1000 iterations, cold retry, shift fallback). Kept in the key so those
    # rows can never be pooled with the default-solver rows of the same cell.
    return (("L",) + C(cost, d, D0, Np=7, model=model, interval=interval,
                       quant=quant, comp=comp, budget="det")
            + (vmin, dvmax) + (("solverfix",) if solver else ()))


def nc(d):
    return C("nocontrol", d, D0)


def rl(d, comp=None):
    return C("rule_v40_n9_h120", d, D0, comp=comp, rule=rule30())


def build_pool():
    S = load("straight")
    P, _, _ = pool(S)
    n = {}
    for dname in DIRS:
        rows = []
        for f in glob.glob(os.path.join(_ROOT, "data", dname, "shard*.json")):
            rows += json.load(open(f))
        # instrumented (SSM) rows are kept OUT: they share a cell key with the
        # plain rows but are not bit-identical to them (the SSM device shifts
        # SUMO's results slightly), and safety is analysed from its own pool
        ok = [r for r in rows if r.get("ok") and not r.get("instrumented")
              and (r.get("wall_s") or 0.0) <= SUSPEND_S]
        n[dname] = (len(ok), len(rows) - len(ok))
        for r in ok:
            k = (("L",) + cell_of(r) + (r.get("vmin_req"), r.get("dvmax_req"))
                 + (("solverfix",) if r.get("solver_req") else ())
                 + (r["seed"],))
            P[k] = r
    return P, n


def line(P, pol, d, label, base=None):
    base = base or nc(d)
    a = paired(P, pol, base, PRIMARY, SEEDS)
    dl = paired(P, pol, base, spv, SEEDS)
    q = paired(P, pol, base, queue_spv, SEEDS)
    print("   %-34s jam %s | delay %s queue %s | sign %5.1f | fail %3.0f%%"
          % (label, fmt(a), dfmt(dl), dfmt(q), mean(P, pol, "mean_vsl", SEEDS),
             mean(P, pol, failpct, SEEDS)))
    return a


def vs(P, a, b, label):
    j = paired(P, a, b, PRIMARY, SEEDS)
    dl = paired(P, a, b, spv, SEEDS)
    print("      %-46s jam %s  delay %s" % (label, fmt(j), dfmt(dl)))
    return j


def main():
    P, n = build_pool()
    print("rows ok/errors: " + ", ".join("%s %d/%d" % (k, v[0], v[1])
                                          for k, v in n.items()))

    head("ROUND 39 -- IS THE RESULT SET BY THE SOLVER? A bound the posted sign\n"
         "never reaches (v_min 20 -> 40). If the objective drove the sign, the\n"
         "two arms would be identical. PARAMS_BM, laptop, 10 seeds.")
    for d, costs in ((4600, ("tts", "tts:1|terminal:5", "dens_rate:1|tts:1")),
                     (4800, ("tts",))):
        print(" demand %d" % d)
        for c in costs:
            line(P, L(c, d), d, "%s, v_min 20 (as used)" % c)
            line(P, L(c, d, vmin=40.0), d, "%s, v_min 40" % c)
            vs(P, L(c, d, vmin=40.0), L(c, d), "v_min 40 vs 20 (paired)")

    head("ROUND 39 -- REALISTIC-SIGN `tts`: SOLVER BUDGET ON ONE MACHINE\n"
         "laptop det budget (this round) vs laptop wall-clock (round 29)")
    for d in (4600, 4800):
        wall = C("tts", d, D0, Np=7, interval=60, quant=10.0)
        print(" demand %d" % d)
        line(P, wall, d, "deployable, 15 s wall clock")
        line(P, L("tts", d, "deploy"), d, "deployable, det budget")
        vs(P, L("tts", d, "deploy"), wall, "det vs wall clock, same machine")
        line(P, rl(d), d, "rule v_det 30 (deployable by design)")
        vs(P, L("tts", d, "deploy"), rl(d), "deployable MPC vs rule (paired)")

    head("ROUND 40 -- OBJECTIVES RE-RANKED UNDER REALISTIC CONDITIONS\n"
         "4600 veh/h, laptop. Closed-loop behaviour of each controller -- under\n"
         "PARAMS_BM the TTS family's objective is flat in the sign (ledger).")
    from control.sweep import RETEST_OBJECTIVES
    for cname, cfg, comp in (("ideal sign, drivers obey", "ideal", None),
                             ("REALISTIC sign, drivers obey", "deploy", None),
                             ("ideal sign, drivers 1.5x", "ideal", 1.5),
                             ("REALISTIC sign, drivers 1.5x", "deploy", 1.5)):
        print(" -- %s" % cname)
        res = []
        for c in RETEST_OBJECTIVES:
            a = line(P, L(c, 4600, cfg, comp=comp), 4600, c)
            if a:
                res.append((a["pct"], c))
        line(P, rl(4600, comp), 4600, "[rule v_det 30, same drivers]")
        if res:
            res.sort()
            print("      ranking (best first): " +
                  ", ".join("%s %+.0f%%" % (c, v) for v, c in res))

    head("ROUND 41 -- A JAM OBJECTIVE ON A MODEL THAT CAN CLEAR A JAM\n"
         "(PARAMS_BM_JAM, term_jam, per-move rate limit lifted). Success = the\n"
         "jam falls AND a never-reached bound does not move the answer.")
    J = "tts:1|jam:5"
    B = "bm_jam_dis"          # fit B: discharge-matched jam-clearance model
    for d in (4600, 4800):
        print(" demand %d" % d)
        main_ = L(J, d, model=B, dvmax=60.0, vmin=40.0)
        line(P, main_, d, "jam:5 ideal, floor 40 (fit B)")
        dep = L(J, d, "deploy", model=B, dvmax=60.0, vmin=40.0)
        line(P, dep, d, "jam:5 DEPLOYABLE, floor 40")
        line(P, rl(d), d, "[rule v_det 30]")
        vs(P, main_, rl(d), "jam:5 ideal MPC vs rule (paired)")
        vs(P, dep, rl(d), "jam:5 deployable MPC vs rule (paired)")
        comp = L(J, d, comp=1.5, model=B, dvmax=60.0, vmin=40.0)
        line(P, comp, d, "jam:5 ideal, drivers 1.5x")
        line(P, rl(d, 1.5), d, "[rule v_det 30, drivers 1.5x]")
        vs(P, comp, rl(d, 1.5), "jam:5 MPC vs rule at 1.5x (paired)")
        line(P, L("tts", d, model=B), d, "tts ideal (fit B, defaults)")
        vs(P, L("tts", d, model=B), L("tts", d),
           "tts: fit B vs old model (paired)")
        line(P, L("tts:1|jam:20", d, model=B, dvmax=60.0, vmin=40.0), d,
             "jam:20 ideal, floor 40")
        line(P, L(J, d, model=B, dvmax=60.0), d, "jam:5 ideal, floor 20")
        old = L(J, d, dvmax=60.0, vmin=40.0)
        line(P, old, d, "jam:5 ideal on OLD model (PARAMS_BM)")
        vs(P, main_, old, "fit B vs old model (paired)")
        od = L(J, d, "deploy", dvmax=60.0, vmin=40.0)
        line(P, od, d, "jam:5 DEPLOYABLE on OLD model")
        vs(P, dep, od, "fit B vs old model, deployable (paired)")
        oc = L(J, d, comp=1.5, dvmax=60.0, vmin=40.0)
        line(P, oc, d, "jam:5 drivers 1.5x on OLD model")
        vs(P, comp, oc, "fit B vs old model, drivers 1.5x (paired)")
        fa = L(J, d, model="bm_jam", dvmax=60.0, vmin=40.0)
        line(P, fa, d, "jam:5 ideal on fit A")
        vs(P, main_, fa, "fit B vs fit A (paired)")


def solver_section(P):
    """Default vs improved failure handling on the new MPC (2026-10-06)."""
    head("SOLVER FIX -- new MPC (jam:5, fit B), default IPOPT vs adaptive + "
         "1000 iterations + cold retry + shift fallback. Same seeds, laptop.")
    J, B = "tts:1|jam:5", "bm_jam_dis"
    for d in (4600, 4800):
        print(" demand %d" % d)
        for cfg in ("ideal", "deploy"):
            a = L(J, d, cfg, model=B, dvmax=60.0, vmin=40.0)
            b = L(J, d, cfg, model=B, dvmax=60.0, vmin=40.0, solver=True)
            line(P, a, d, "%s, default solver" % cfg)
            line(P, b, d, "%s, IMPROVED solver" % cfg)
            vs(P, b, a, "improved vs default (paired)")
            line(P, rl(d), d, "[rule v_det 30]")


if __name__ == "__main__":
    main()
    solver_section(build_pool()[0])
