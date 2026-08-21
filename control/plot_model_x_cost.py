# -*- coding: utf-8 -*-
"""
The figure for the central result: which cost function is best depends on the model.

Left panel is the whole Phase D comparison, one line per cost function, joining
its score on the shipped prediction model to its score on the ported one. If
cost-function performance were a property of the cost function, the lines would
be flat. They cross instead.

Right panel is the direct crossover: each model run with both its own best cost
and the other model's best cost, on the same ten seeds. Each model's own best
wins on that model.

The horizontal band is the reference that matters - a constant speed limit with
no controller at all, restricted to the window in which the controller is
active. Anything not clear of that band is not earning its complexity.

Usage:  python control/plot_model_x_cost.py
Output: docs/figures/model_x_cost.png
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

DATA = os.path.join(_ROOT, "data")
FIG = os.path.join(_ROOT, "docs", "figures")
MAX_FAIL = 0.15          # drop cost functions whose solver was unhealthy


def paired(fname, key="accel_rms"):
    rows = json.load(open(os.path.join(DATA, fname)))
    by = {}
    for r in rows:
        by.setdefault(r["cost"], {})[r["seed"]] = r
    base = by["nocontrol"]
    out = {}
    for c, seeds in by.items():
        if c == "nocontrol":
            continue
        rs = [r for s, r in seeds.items() if s in base]
        if len(rs) < 10:
            continue
        if sum(r["n_failed"] for r in rs) / max(sum(r["n_solves"] for r in rs), 1) > MAX_FAIL:
            continue
        out[c] = np.median([100 * (r[key] / base[r["seed"]][key] - 1) for r in rs])
    return out


def main():
    A = paired("phase_d.json")
    B = paired("phase_d_s14.json")
    common = sorted(set(A) & set(B), key=lambda c: B[c])

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), constrained_layout=True)

    ax = axs[0]
    lbl = []
    for c in common:
        up = B[c] < A[c]
        ax.plot([0, 1], [A[c], B[c]], "-o", ms=5, lw=1.6, alpha=0.85,
                color="C2" if up else "C3")
        lbl.append((B[c], c))
    # spread labels that would otherwise print on top of each other
    lbl.sort()
    last = -1e9
    for y, c in lbl:
        yy = max(y, last + 0.16)
        ax.annotate(c, (1, y), fontsize=7.5, va="center",
                    xytext=(6, (yy - y) * 12), textcoords="offset points")
        last = yy
    ax.axhspan(-5.2, -5.2, color="k")
    ax.axhline(-5.2, color="k", ls="--", lw=1.6)
    ax.annotate("best constant sign, no controller (-5.2%)", (0.02, -5.2),
                fontsize=8.5, va="bottom")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["shipped METANET\n210 s speed R$^2$ = +0.07",
                        "ported bake-off winner\n210 s speed R$^2$ = +0.57"])
    ax.set_xlim(-0.15, 1.45)
    ax.set_ylabel("change in shockwave acceleration (%)      <- better")
    ax.set_title("Every cost function, both models, same 10 seeds\n"
                 "green = improves with the better model, red = gets worse")
    ax.grid(alpha=0.25, axis="y")

    ax = axs[1]
    pl = json.load(open(os.path.join(DATA, "fix_plateau.json")))

    def cell(fix, cost):
        rs = [r for r in pl if r["fix"] == fix
              and r.get("cost", "speedvar_tts") == cost]
        return (np.median([r["d_accel"] for r in rs]) if len(rs) >= 10
                else np.nan)

    labels = ["speedvar_tts\n(best on shipped)", "accel_delay\n(best on ported)"]
    shipped = [cell("baseline", "speedvar_tts"), cell("baseline", "accel_delay")]
    ported = [cell("s14", "speedvar_tts"), cell("s14", "accel_delay")]
    x = np.arange(2)
    # deliberately NOT red/green: those mean "direction of change" on the left
    # panel, and reusing them here for "which model" would read as the same code
    ax.bar(x - 0.19, shipped, 0.36, label="shipped model", color="#7b6ca8",
           alpha=0.9)
    ax.bar(x + 0.19, ported, 0.36, label="ported model", color="#3f7fa6",
           alpha=0.9)
    for xi, v in zip(x - 0.19, shipped):
        ax.annotate("%.1f%%" % v, (xi, v), ha="center", va="top",
                    fontsize=9, xytext=(0, -3), textcoords="offset points")
    for xi, v in zip(x + 0.19, ported):
        ax.annotate("%.1f%%" % v, (xi, v), ha="center", va="top",
                    fontsize=9, xytext=(0, -3), textcoords="offset points")
    ax.axhline(-5.2, color="k", ls="--", lw=1.6)
    ax.annotate("best constant sign, no controller", (1.42, -5.2), fontsize=8.5,
                ha="right", va="bottom")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("change in shockwave acceleration (%)      <- better")
    ax.set_title("The crossover, isolated\n"
                 "each model's own best cost wins on that model")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, axis="y")

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "model_x_cost.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)
    moved = sum(1 for c in common if abs(A[c] - B[c]) > 1.0)
    print("  %d of %d cost functions move by more than 1 percentage point "
          "between models" % (moved, len(common)))
    print("  best on shipped: %s (%.1f%%);  best on ported: %s (%.1f%%)"
          % (min(A, key=A.get), min(A.values()),
             min(B, key=B.get), min(B.values())))


if __name__ == "__main__":
    main()
