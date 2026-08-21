# -*- coding: utf-8 -*-
"""
The Phase D figure: the shockwave/mobility trade-off.

One point per cost function, positioned by how much it reduces the shockwave
(vertical) against what that costs in delay (horizontal), both as a percentage
change against no control on the same seeds. The bottom-left quadrant is the
interesting one: better on both at once.

Usage:  python control/plot_phase_d.py
Output: docs/figures/phase_d_tradeoff.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

RESULTS = os.path.join(_ROOT, "data", "phase_d.json")
FIG = os.path.join(_ROOT, "docs", "figures")


def paired(by, cost, key):
    base = by.get("nocontrol", {})
    out = []
    for seed, r in by.get(cost, {}).items():
        b = base.get(seed)
        if b and np.isfinite(r.get(key, np.nan)) and b.get(key):
            out.append(100.0 * (r[key] / b[key] - 1.0))
    return np.array(out)


def main():
    rows = json.load(open(RESULTS))
    by = {}
    for r in rows:
        by.setdefault(r["cost"], {})[r["seed"]] = r
    costs = [c for c in by if c != "nocontrol"]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)

    ax = axs[0]
    for c in costs:
        a, t = paired(by, c, "accel_rms"), paired(by, c, "TTS_h")
        if not len(a) or not len(t):
            continue
        n = min(len(a), len(t))
        ma, mt = np.median(a), np.median(t)
        # spread across seeds, so a difference can be read against the noise
        ax.errorbar(mt, ma,
                    xerr=[[mt - np.percentile(t, 25)], [np.percentile(t, 75) - mt]],
                    yerr=[[ma - np.percentile(a, 25)], [np.percentile(a, 75) - ma]],
                    fmt="o", ms=7, capsize=3, lw=1.2, alpha=0.85)
        ax.annotate("%s (n=%d)" % (c, n), (mt, ma), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("change in total time spent (%)   <- better")
    ax.set_ylabel("change in shockwave acceleration (%)   <- better")
    ax.set_title("The trade-off\nbottom-left = better on both; bars are the "
                 "inter-quartile range across seeds")
    ax.grid(alpha=0.25)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([xl[0], 0], yl[0], 0, color="green", alpha=0.06)
    ax.set_xlim(xl); ax.set_ylim(yl)

    ax = axs[1]
    order = sorted(costs, key=lambda c: np.median(paired(by, c, "accel_rms"))
                   if len(paired(by, c, "accel_rms")) else 9e9)
    y = np.arange(len(order))
    for i, key, lbl, col in [(0, "accel_rms", "shockwave", "C0"),
                             (1, "TTS_h", "delay", "C3"),
                             (2, "cvs", "speed variation", "C2")]:
        vals = [np.median(paired(by, c, key)) if len(paired(by, c, key)) else np.nan
                for c in order]
        ax.barh(y + (i - 1) * 0.26, vals, height=0.25, label=lbl, color=col,
                alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("change vs no control (%)   <- better")
    ax.set_title("By metric")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="x")

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "phase_d_tradeoff.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
