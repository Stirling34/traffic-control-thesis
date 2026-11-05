# -*- coding: utf-8 -*-
"""
Round 42 (data/sweep_final): every cost function, realistic sign, final
configuration, both roads. Read-only.

Primary metric per road (pre-registered and trajectory-validated):
  straight  jam vehicle-hours below 30 km/h  (jam_vc30_veh_h)
  merge     congestion front reach            (wave_vc30_front_reach)
Guards, always alongside: total time spent incl. the entrance queue (% and the
MEDIAN paired delay, s/veh -- delay is heavy-tailed, ledger 2026-10-05), the
entrance-queue component on its own (the metering channel), mean posted sign,
solver failure rate, and mean achieved speed (separates smoother traffic from
merely slower traffic, ledger 2026-09-02). accel_rms is kept only as the old
metric, for the side panel.

Every MPC row here was produced on the LAPTOP, and is paired only with
laptop MPC rows or with the solver-free no-control / rule rows, which
reproduce exactly on any machine (ledger 2026-09-29). Pairing is on seed,
strictly within road and demand (Hegyi 2004 sec. 9.1: comparisons across
demand scenarios are not meaningful).

Seven seed-3 rows have wall times > 3000 s from one laptop suspend; on the
deterministic iteration budget a suspend changes the clock, not the result,
so they are kept (flagged in the output).

usage: python control/analyse_final.py      -> table + data/sweep_final/summary.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from control.analyse_deploy import queue_spv, spv          # noqa: E402

DIR = os.path.join(_ROOT, "data", "sweep_final")
SEEDS = range(1, 11)
PRIMARY = {"final_straight": ("jam_vc30_veh_h", "jam veh-h"),
           "final_merge": ("wave_vc30_front_reach", "front reach")}


def label(r):
    if r.get("rule_req") is not None:
        return "RULE"
    return r["cost"] if r["cost"] != "nocontrol" else "NOCONTROL"


def load():
    P, slow = {}, []
    for f in glob.glob(os.path.join(DIR, "shard*.json")):
        for r in json.load(open(f)):
            if not r.get("ok"):
                continue
            k = (r["exp"], label(r), r.get("demand_req"), r["seed"])
            if k in P:
                raise ValueError("duplicate row %s" % (k,))
            P[k] = r
            if r["wall_s"] > 3000:
                slow.append(k)
    return P, slow


def speed(r):
    return r["veh_km"] / r["TTS_road_h"]


def fail(r):
    return 100.0 * (r.get("n_failed") or 0) / max(r.get("n_solves") or 1, 1)


def paired(P, a, b, f):
    xs, ys = [], []
    for s in SEEDS:
        ra, rb = P.get(a + (s,)), P.get(b + (s,))
        if ra is None or rb is None:
            continue
        x = f(ra) if callable(f) else ra.get(f)
        y = f(rb) if callable(f) else rb.get(f)
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs, float), np.array(ys, float)
    if len(xs) < 6:
        return None
    d = xs - ys
    try:
        p = wilcoxon(xs, ys).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(n=len(xs), pct=100.0 * d.sum() / ys.sum() if ys.sum() else np.nan,
                dmean=float(d.mean()), dmed=float(np.median(d)), p=float(p),
                val=float(xs.mean()), base=float(ys.mean()))


def mean_of(P, a, f):
    v = [f(P[a + (s,)]) if callable(f) else P[a + (s,)].get(f)
         for s in SEEDS if a + (s,) in P]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def star(res):
    if res is None:
        return "   n/a        "
    sig = "*" if res["p"] < 0.05 else " "
    return "%+7.1f%%%s p=%.3f" % (res["pct"], sig, res["p"])


def main():
    P, slow = load()
    print("rows %d; suspend-inflated wall time (kept): %s" % (len(P), slow))
    summary = []
    for exp, demands in (("final_merge", [None]),
                         ("final_straight", [4600, 4800])):
        prim, pname = PRIMARY[exp]
        for d in demands:
            nc, rl = (exp, "NOCONTROL", d), (exp, "RULE", d)
            pols = sorted({k[1] for k in P if k[0] == exp and k[2] == d})
            rows = []
            for pol in pols:
                a = (exp, pol, d)
                if pol == "NOCONTROL":
                    continue
                res = dict(
                    road=exp.replace("final_", ""), demand=d, policy=pol,
                    prim=paired(P, a, nc, prim),
                    prim_vs_rule=(paired(P, a, rl, prim) if pol != "RULE"
                                  else None),
                    tts=paired(P, a, nc, "TTS_h"),
                    delay=paired(P, a, nc, spv),
                    queue=paired(P, a, nc, queue_spv),
                    speed=paired(P, a, nc, speed),
                    accel=paired(P, a, nc, "accel_rms"),
                    reach=(paired(P, a, nc, "jam_vc30_reach_km")
                           if exp == "final_straight" else None),
                    dur=paired(P, a, nc, "jam_vc30_dur_s" if exp ==
                               "final_straight" else "wave_vc30_front_dur"),
                    sign=mean_of(P, a, "mean_vsl"),
                    fail=(mean_of(P, a, fail) if pol != "RULE" else 0.0))
                rows.append(res)
                summary.append(res)
            rows.sort(key=lambda r: r["prim"]["pct"] if r["prim"] else 1e9)
            print("\n" + "=" * 150)
            print("%s%s   primary = %s vs no control (no-control mean %.2f)"
                  % (exp, "" if d is None else " @ %d veh/h" % d, pname,
                     mean_of(P, nc, prim)))
            print("%-40s %-24s %-24s %-16s %9s %8s %8s %-16s %5s %5s"
                  % ("policy", pname + " vs NC", "vs RULE", "TTS",
                     "med dly", "queue", "speed", "accel(old)", "sign",
                     "fail"))
            for r in rows:
                print("%-40s %-24s %-24s %-16s %+8.1fs %+7.1fs %+7.1f%% %-16s %5.1f %4.0f%%"
                      % (r["policy"][:40], star(r["prim"]), star(r["prim_vs_rule"]),
                         star(r["tts"])[:16],
                         r["delay"]["dmed"] if r["delay"] else np.nan,
                         r["queue"]["dmean"] if r["queue"] else np.nan,
                         r["speed"]["pct"] if r["speed"] else np.nan,
                         star(r["accel"])[:16], r["sign"], r["fail"]))
    out = os.path.join(DIR, "summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
