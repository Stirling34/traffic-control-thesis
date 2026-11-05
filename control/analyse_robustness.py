# -*- coding: utf-8 -*-
"""
Final report, Table 9 (Section 7.4): the MPC (`tts:1|jam:5`, i.e. TTS + jam
at twice its matched weight) and the reactive rule under different sign and
driver conditions, straight road. Read-only.

Rows, all vs no control on the same 10 seeds (analyse_final.paired):
  MPC, idealised sign              round 41, data/sweep_bmjam (30 s moves,
                                   continuous limits)
  MPC, realistic sign              round 41 (60 s moves, 10 km/h steps)
  MPC, realistic sign, drivers 1.5x  round 41 (`comp_req` 1.5)
  Reactive rule                    v_det 30, v_post 40, 9 signs, 120 s hold,
  Reactive rule, drivers 1.5x      data/sweep_straight (solver-free rows)
All MPC rows: PARAMS_BM_JAM_DIS, v_min 40, dv_max 60, default solver, plain
(not SSM-instrumented). No control: round 42 (data/sweep_final), solver-free,
so it pairs with rows from any machine.

Columns: change in congestion (wave_vc30_cong_cells, the report's primary
measure, %) and mean change in total time per vehicle incl. the entrance
queue (s).

usage: python control/analyse_robustness.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "control"))

import analyse_final as af                                        # noqa: E402
from control.analyse_deploy import spv                            # noqa: E402
from control.sweep import STRAIGHT_DIST                           # noqa: E402

DEMANDS = (4600, 4800)


def rows(pattern):
    out = []
    for f in glob.glob(os.path.join(ROOT, pattern)):
        out += [r for r in json.load(open(f)) if r.get("ok")]
    return out


def pool():
    P = {}
    for r in rows("data/sweep_final/shard*.json"):
        if (r.get("exp") == "final_straight" and r["cost"] == "nocontrol"
                and r.get("rule_req") is None):
            P[("NO CONTROL", r["demand_req"], r["seed"])] = r
    for r in rows("data/sweep_bmjam/shard*.json"):
        if (r["cost"] == "tts:1|jam:5" and r.get("model_name") == "bm_jam_dis"
                and r.get("vmin_req") == 40.0 and not r.get("instrumented")
                and not r.get("solver_req")):
            lab = ("MPC, realistic sign" if r.get("quant_req")
                   else "MPC, idealised sign")
            if r.get("comp_req") == 1.5:
                lab += ", drivers 1.5x"
            P[(lab, r["demand_req"], r["seed"])] = r
    dist = dict(STRAIGHT_DIST)
    for r in rows("data/sweep_straight/shard*.json"):
        rule = r.get("rule_req")
        if (r.get("net") == "straight" and not r.get("instrumented")
                and r.get("dist_req") == dist and not r.get("quant_req")
                and not r.get("lag_req") and rule
                and rule.get("v_det") == 30.0 and rule.get("v_post") == 40.0
                and rule.get("n_signs") == 9 and rule.get("hold_s") == 120):
            lab = "Reactive rule"
            if r.get("comp_req") == 1.5:
                lab += ", drivers 1.5x"
            elif r.get("comp_req") is not None:
                continue
            P.setdefault((lab, r["demand_req"], r["seed"]), r)
    return P


def main():
    P = pool()
    pols = ["MPC, idealised sign", "MPC, realistic sign",
            "MPC, realistic sign, drivers 1.5x", "Reactive rule",
            "Reactive rule, drivers 1.5x"]
    print("%-36s %-26s %-26s" % ("policy", "congestion 4600 / 4800",
                                 "time s/veh 4600 / 4800"))
    for pol in pols:
        cong, time = [], []
        for d in DEMANDS:
            c = af.paired(P, (pol, d), ("NO CONTROL", d), "wave_vc30_cong_cells")
            t = af.paired(P, (pol, d), ("NO CONTROL", d), spv)
            cong.append("n/a" if c is None else
                        "%+.1f%%%s" % (c["pct"], "*" if c["p"] < 0.05 else ""))
            time.append("n/a" if t is None else "%+.1f" % t["dmean"])
        print("%-36s %-26s %-26s" % (pol, " / ".join(cong), " / ".join(time)))
    print("* p < 0.05, Wilcoxon signed-rank, paired on seed (n = 10)")


if __name__ == "__main__":
    main()
