# -*- coding: utf-8 -*-
"""
The figure behind the audit: prediction skill against horizon.

The controller optimises across 210 s. This shows what the model it optimises
through actually knows at that horizon, against two references: assuming
nothing changes, and the best calibrated model the project already produced in
the bake-off and left in experiments/.

The crossing point is the thing to read. Below about 45 s the model is useful.
Beyond that it falls under persistence, and by the controller's own horizon it
is close to uncorrelated with what SUMO does.

Usage:  python validation/plot_prediction_skill.py
Output: docs/figures/prediction_skill.png
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

FIG = os.path.join(_ROOT, "docs", "figures")
HZ = [30, 60, 120, 210]


def series(d, key):
    return [d["prediction"]["%ds" % h].get(key, np.nan) for h in HZ]


def main():
    ref = json.load(open(os.path.join(_ROOT, "experiments", "ref_metanet",
                                      "result.json")))
    best, bcorr = None, -9
    others = []
    for f in glob.glob(os.path.join(_ROOT, "experiments", "*", "res_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("VALID"):
            continue
        e = d.get("prediction", {}).get("210s", {})
        if e.get("corr") is None:
            continue
        others.append(d)
        if e["corr"] > bcorr:
            best, bcorr = d, e["corr"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    for ax, key, pkey, lbl in [(axs[0], "corr", "pers_corr",
                                "correlation with SUMO speed"),
                               (axs[1], "mae", "pers_mae",
                                "mean absolute speed error (km/h)")]:
        for d in others:
            ax.plot(HZ, series(d, key), color="0.8", lw=0.8, zorder=1)
        ax.plot(HZ, series(ref, pkey), "--", color="k", lw=2.2, zorder=4,
                label="assume nothing changes (persistence)")
        ax.plot(HZ, series(best, key), "-o", color="C2", lw=2.2, ms=6, zorder=5,
                label="best calibrated model already on disk")
        ax.plot(HZ, series(ref, key), "-o", color="C3", lw=2.6, ms=7, zorder=6,
                label="METANET, as used by the NMPC")
        ax.axvline(210, color="C0", lw=1.4, ls=":")
        ax.annotate("the controller's\nown horizon", (210, ax.get_ylim()[1]),
                    color="C0", fontsize=9, ha="right", va="top",
                    xytext=(-6, -6), textcoords="offset points")
        ax.set_xlabel("prediction horizon (s)")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.25)
        ax.set_xticks(HZ)
    axs[0].set_title("Skill falls below persistence at about 45 s.\n"
                     "The MPC optimises across 210 s.")
    axs[1].set_title("Grey lines are the other 80+ models from the bake-off,\n"
                     "all calibrated on the same data")
    axs[0].legend(fontsize=9, loc="lower left")

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "prediction_skill.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)
    print("  METANET (NMPC)   210 s: corr %.3f, MAE %.1f km/h"
          % (ref["prediction"]["210s"]["corr"], ref["prediction"]["210s"]["mae"]))
    print("  persistence      210 s: corr %.3f, MAE %.1f km/h"
          % (ref["prediction"]["210s"]["pers_corr"],
             ref["prediction"]["210s"]["pers_mae"]))
    print("  best on disk     210 s: corr %.3f, MAE %.1f km/h  (%s)"
          % (best["prediction"]["210s"]["corr"],
             best["prediction"]["210s"]["mae"], best["name"]))


if __name__ == "__main__":
    main()
