# -*- coding: utf-8 -*-
"""
Final-report figure for Section 7.5: does the prediction model know what the
sign does?

  fig_sign_response.png  For every prediction-study branch (moment x fixed sign
                         plan: 40/60 km/h for 120/300 s), SUMO's change in jam
                         vehicle-hours over 900 s against the change the
                         controller's model predicts from the same starting
                         state. Left: straight road, PARAMS_BM_JAM_DIS (the
                         final straight-road model). Right: merge road,
                         PARAMS_BM. Dashed line = perfect prediction.

Reads data/prediction_study/*.json (validation/prediction_study.py); the
summary statistics quoted in the report come from
validation/analyse_prediction_study.py.
usage: python control/plot_sign_response.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "prediction_study")
OUT = os.path.join(ROOT, "docs", "final_report", "figures")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e6e6", "#ffffff"
BLUE = "#2a78d6"

PANELS = [("straight", "PARAMS_BM_JAM_DIS", "Straight road"),
          ("long", "PARAMS_BM", "Merge road")]


def branches(net, params):
    xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(SRC, "%s_*.json" % net))):
        r = json.load(open(f))
        for b in r["B"]:
            if params in b["model"]:
                xs.append(b["sumo_d_jamh"])
                ys.append(b["model"][params]["d_jamh"])
    return np.array(xs), np.array(ys)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), facecolor=SURF)
    for ax, (net, params, title) in zip(axs, PANELS):
        x, y = branches(net, params)
        lim = 1.08 * max(np.abs(x).max(), np.abs(y).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.axhline(0, color=INK, lw=0.8, zorder=1)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.plot([-lim, lim], [-lim, lim], ls="--", color=INK2, lw=1.0,
                zorder=1)
        ax.text(0.97 * lim, 0.97 * lim, "perfect\nprediction", ha="right",
                va="top", fontsize=8, color=INK2)
        ax.scatter(x, y, s=34, c=BLUE, edgecolors=INK2, linewidths=0.6,
                   alpha=0.85, zorder=3)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("SUMO: change in jam (veh-h)", fontsize=9.5, color=INK)
        ax.tick_params(labelsize=9, colors=INK)
        print(net, params, "n", len(x))
    axs[0].set_ylabel("model prediction: change in jam (veh-h)", fontsize=9.5,
                      color=INK)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_sign_response.png")
    fig.savefig(p, dpi=200, facecolor=SURF)
    print("wrote", p)


if __name__ == "__main__":
    main()
