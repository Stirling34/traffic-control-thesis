# -*- coding: utf-8 -*-
"""
Round 42b (data/sweep_final_ssm): SAFETY of the headline straight-road
objectives, realistic sign, final configuration. Read-only.

Instrumented rows only (they are not bit-identical to plain rows, ledger
2026-10-06). Baselines: no control and the rule (v_det 30, 40 km/h, 9 signs,
120 s) from the instrumented straight-road pool -- solver-free, so they pool
across machines; `tts:1|jam:5` realistic from round 41 group E (laptop, same
machine as round 42b). Paired on seed within demand, Wilcoxon signed-rank.

usage: python control/analyse_final_ssm.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, wilcoxon

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from control.sweep import STRAIGHT_DIST                       # noqa: E402

SEEDS = range(1, 11)
METRICS = [("n_drac_events", "hard braking"), ("TIT_s", "TIT"),
           ("TET_s", "TET"), ("jam_vc30_veh_h", "jam veh-h")]


def rows(pattern):
    out = []
    for f in glob.glob(os.path.join(_ROOT, pattern)):
        out += [r for r in json.load(open(f)) if r.get("ok")]
    return out


def pool():
    P = {}
    dist = dict(STRAIGHT_DIST)
    base = [r for r in rows("data/sweep_straight/shard*.json")
            if r.get("instrumented") and r.get("net") == "straight"
            and r.get("dist_req") == dist and not r.get("quant_req")
            and not r.get("comp_req") and not r.get("lag_req")]
    for r in base:
        rule = r.get("rule_req")
        if r["cost"] == "nocontrol" and rule is None:
            lab = "NO CONTROL"
        elif (rule and rule.get("v_det") == 30.0 and rule.get("v_post") == 40.0
              and rule.get("n_signs") == 9 and rule.get("hold_s") == 120):
            lab = "RULE"
        else:
            continue
        P.setdefault((lab, r["demand_req"], r["seed"]), r)
    for r in rows("data/sweep_bmjam/shard*.json"):
        if (r.get("instrumented") and r["cost"] == "tts:1|jam:5"
                and r.get("quant_req") and r.get("model_name") == "bm_jam_dis"
                and not r.get("solver_req")):
            P[("tts:1|jam:5 (round 41)", r["demand_req"], r["seed"])] = r
    for r in rows("data/sweep_final_ssm/shard*.json"):
        P[(r["cost"], r["demand_req"], r["seed"])] = r
    return P


def paired(P, a, b, m, d):
    xs, ys = [], []
    for s in SEEDS:
        ra, rb = P.get((a, d, s)), P.get((b, d, s))
        if ra is None or rb is None or ra.get(m) is None or rb.get(m) is None:
            continue
        xs.append(ra[m])
        ys.append(rb[m])
    if len(xs) < 6:
        return None
    xs, ys = np.array(xs, float), np.array(ys, float)
    d_ = xs - ys
    p = wilcoxon(xs, ys).pvalue if np.any(d_ != 0) else 1.0
    return dict(n=len(xs), pct=100 * d_.sum() / ys.sum(), p=p)


def fmt(r):
    if r is None:
        return "      n/a       "
    return "%+6.1f%%%s p=%.3f" % (r["pct"], "*" if r["p"] < 0.05 else " ", r["p"])


def main():
    P = pool()
    pols = ["RULE", "tts:1|jam:5 (round 41)", "tts:1|jam:10",
            "tts:1|speed_var:1.2", "tts:1|conflict_risk:60", "tts:1|dvsl:0.1"]
    for d in (4600, 4800):
        n_nc = sum(1 for s in SEEDS if ("NO CONTROL", d, s) in P)
        print("\n=== %d veh/h (no-control seeds: %d)" % (d, n_nc))
        print("%-26s " % "policy" + " ".join("%-18s" % lab for _, lab in METRICS))
        dj, dt = [], []
        for pol in pols:
            res = [paired(P, pol, "NO CONTROL", m, d) for m, _ in METRICS]
            print("%-26s " % pol + " ".join("%-18s" % fmt(r) for r in res))
            if res[3] and res[1]:
                dj.append(res[3]["pct"])
                dt.append(res[1]["pct"])
        if len(dj) >= 4:
            r, p = pearsonr(dj, dt)
            print("r(jam change, TIT change) across %d policies = %+.2f (p=%.3f)"
                  % (len(dj), r, p))


if __name__ == "__main__":
    main()
