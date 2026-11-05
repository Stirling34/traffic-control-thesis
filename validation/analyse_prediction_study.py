# -*- coding: utf-8 -*-
"""
Summarise data/prediction_study/*.json (validation/prediction_study.py).

A. Forecast skill: speed RMSE of the model vs persistence at each lead, over
   start times whose next 600 s contain a jam (any SUMO cell < 30 km/h) --
   where control matters -- and over all start times. Plus the error in the
   two wave features the controller acts on (600 s veh-h and jam veh-h).
B. Sign response: for every branch (moment x plan), SUMO's change in jam
   veh-h and veh-h vs the model's; sign agreement, correlation, and whether
   the model ranks the four plans in SUMO's order at each moment.
usage: python validation/analyse_prediction_study.py
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "prediction_study")


def rmse(lst):
    a = np.array([x for row in lst for x in row], float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DIR, "*.json")))]
    for net in ("straight", "long"):
        R = [r for r in runs if r["net"] == net]
        if not R:
            continue
        params, leads = R[0]["params"], [str(L) for L in R[0]["leads"]]
        print("\n" + "=" * 100)
        print("%s: %d runs (%s)" % (net, len(R), ", ".join(
            "d%s s%d" % (r["demand"], r["seed"]) for r in R)))
        print("all branches replayed bit-identically before t0:",
              all(b["replay_identical"] for r in R for b in r["B"]))
        # ---- A
        for subset in ("jam-relevant", "all"):
            print("\nA. speed RMSE km/h, %s start times (model | persistence)" % subset)
            print("%-20s " % "params" + " ".join("%11s" % ("%ss" % L) for L in leads))
            pers_line = None
            for p in params:
                cells = {L: [] for L in leads}
                pcells = {L: [] for L in leads}
                for r in R:
                    a = r["A"][p]
                    act = r["A_actual"]
                    for i, t0 in enumerate(a["t0"]):
                        if subset == "jam-relevant" and act["jamh"][i] <= 0.05:
                            continue
                        for L in leads:
                            cells[L].append(a["err"][L][i])
                            pcells[L].append(a["pers"][L][i])
                print("%-20s " % p.replace("PARAMS_", "") +
                      " ".join("%11.1f" % rmse(cells[L]) for L in leads))
                pers_line = " ".join("%11.1f" % rmse(pcells[L]) for L in leads)
            print("%-20s %s" % ("persistence", pers_line))
        print("\nA. 600 s wave features, jam-relevant start times: median |pred - actual|"
              " (actual median)")
        for p in params:
            ev, ej, av, aj = [], [], [], []
            for r in R:
                a, act = r["A"][p], r["A_actual"]
                for i in range(len(a["t0"])):
                    if act["jamh"][i] <= 0.05:
                        continue
                    ev.append(abs(a["pred_vehh"][i] - act["vehh"][i]))
                    ej.append(abs(a["pred_jamh"][i] - act["jamh"][i]))
                    av.append(act["vehh"][i])
                    aj.append(act["jamh"][i])
            print("  %-18s veh-h %6.2f (%6.2f)   jam veh-h %6.2f (%6.2f)  n=%d"
                  % (p.replace("PARAMS_", ""), np.median(ev), np.median(av),
                     np.median(ej), np.median(aj), len(ev)))
        # ---- B
        print("\nB. SIGN RESPONSE (change vs no sign, 900 s): SUMO vs model")
        for metric, key in (("jam veh-h", "d_jamh"), ("veh-h", "d_vehh")):
            sk = "sumo_" + key
            print("  %s:" % metric)
            for p in params:
                s = np.array([b[sk] for r in R for b in r["B"]])
                m = np.array([b["model"][p][key] for r in R for b in r["B"]])
                big = np.abs(s) > 0.5
                agree = np.mean(np.sign(s[big]) == np.sign(m[big])) if big.any() else np.nan
                rho = spearmanr(s, m).correlation if np.std(m) > 0 else np.nan
                # plan ORDER at each moment
                taus = []
                for r in R:
                    for t0 in sorted({b["t0"] for b in r["B"]}):
                        bb = [b for b in r["B"] if b["t0"] == t0]
                        ss = [b[sk] for b in bb]
                        mm = [b["model"][p][key] for b in bb]
                        if np.std(ss) > 0 and np.std(mm) > 0:
                            taus.append(spearmanr(ss, mm).correlation)
                print("    %-18s sign agrees %4.0f%% (|SUMO|>0.5, n=%d)  spearman %+.2f"
                      "  plan-order rho median %+.2f (n=%d)  SUMO mean %+.2f model mean %+.2f"
                      % (p.replace("PARAMS_", ""), 100 * agree, big.sum(), rho,
                         np.median(taus) if taus else np.nan, len(taus),
                         s.mean(), m.mean()))
        print("\n  per moment (SUMO d_jam | each model d_jam), mean over runs and plans:")
        for t0 in sorted({b["t0"] for r in R for b in r["B"]}):
            for kmh, dur in sorted({(b["kmh"], b["dur"]) for r in R for b in r["B"]}):
                bb = [b for r in R for b in r["B"] if b["t0"] == t0
                      and b["kmh"] == kmh and b["dur"] == dur]
                print("    t0 %4d  %2.0f km/h %3d s  SUMO %+6.2f | %s"
                      % (t0, kmh, dur, np.mean([b["sumo_d_jamh"] for b in bb]),
                         " ".join("%s %+6.2f" % (p.replace("PARAMS_", ""),
                                                 np.mean([b["model"][p]["d_jamh"]
                                                          for b in bb]))
                                  for p in params)))


if __name__ == "__main__":
    main()
