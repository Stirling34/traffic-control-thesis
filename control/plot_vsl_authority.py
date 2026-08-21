# -*- coding: utf-8 -*-
"""
The figure that puts the controller in context.

Three things on one pair of axes, all paired against the same no-control runs
on the same seeds, so they are directly comparable:

  circles   the 18 MPC cost functions (Phase D)
  line      a plain constant speed limit at each rate b, no controller at all
  crosses   the four attempted fixes for the inert-VSL plateau

Left panel is the trade-off plane: shockwave reduction against what it costs in
delay. Down and left is better. Two things to read off it. The constant limit
sits in the same place as the best of the 18 cost functions, so on the original
model the controller was not earning its complexity. The one point well below
the cloud is `carlson`, the same cost function on a model where the VSL
reshapes the fundamental diagram instead of capping it.

Right panel is why: it shows what a speed limit does to the SUMO fundamental
diagram, measured by calibration/vsl_fd.py. The critical density really does
move right as the limit comes down, and per-lane capacity really does peak below
the legal limit. So the mechanism exists; it is just worth only a few percent.

Usage:  python control/plot_vsl_authority.py
Output: docs/figures/vsl_authority.png
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


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else []


def paired_phase_d():
    rows = load("phase_d.json")
    by = {}
    for r in rows:
        by.setdefault(r["cost"], {})[r["seed"]] = r
    base = by.get("nocontrol", {})
    out = {}
    for cost, seeds in by.items():
        if cost == "nocontrol":
            continue
        da, dt = [], []
        for s, r in seeds.items():
            b = base.get(s)
            if not b:
                continue
            da.append(100 * (r["accel_rms"] / b["accel_rms"] - 1))
            dt.append(100 * (r["TTS_h"] / b["TTS_h"] - 1))
        if da:
            out[cost] = (np.median(dt), np.median(da), len(da))
    return out


def med(rows, key, field):
    """Median of `field` grouped by `key`, returned in key order."""
    ks = sorted({r[key] for r in rows})
    return ks, [np.median([r[field] for r in rows if r[key] == k]) for k in ks]


def main():
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.6), constrained_layout=True)

    ax = axs[0]
    pd = paired_phase_d()
    if pd:
        xs = [v[0] for v in pd.values()]
        ys = [v[1] for v in pd.values()]
        ax.scatter(xs, ys, s=42, c="C0", alpha=0.7, zorder=3,
                   label="MPC cost functions (18)")
        best = min(pd.items(), key=lambda kv: kv[1][1])
        ax.annotate("%s\n%.1f%%" % (best[0], best[1][1]),
                    (best[1][0], best[1][1]), fontsize=8, fontweight="bold",
                    xytext=(8, -14), textcoords="offset points", color="C0")

    fx = load("fixed_vsl.json")
    if fx:
        bs, ya = med(fx, "b", "d_accel")
        _, xa = med(fx, "b", "d_TTS")
        ax.plot(xa, ya, "-o", color="C3", ms=6, lw=1.8, zorder=4,
                label="constant limit, no controller")
        for b, x, y in zip(bs, xa, ya):
            ax.annotate("b=%.2f" % b, (x, y), fontsize=7.5, color="C3",
                        xytext=(5, 5), textcoords="offset points")

    pl = load("fix_plateau.json")
    if pl:
        fixes = sorted({r["fix"] for r in pl})
        for f in fixes:
            rs = [r for r in pl if r["fix"] == f]
            ax.scatter(np.median([r["d_TTS"] for r in rs]),
                       np.median([r["d_accel"] for r in rs]),
                       marker="X", s=110, zorder=5,
                       label="plateau fix: %s" % f)

    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([xl[0], 0], yl[0], 0, color="green", alpha=0.06)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xlabel("change in total time spent (%)      <- better")
    ax.set_ylabel("change in shockwave acceleration (%)      <- better")
    ax.set_title("Everything, paired against no control on the same seeds\n"
                 "a constant sign matches every cost function; only the "
                 "reformulated model beats it")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axs[1]
    coef = os.path.join(DATA, "vsl_fd_coef.json")
    if os.path.exists(coef):
        c = json.load(open(coef))
        b = np.array(c["b"])
        ax.plot(b, c["q_cap_per_lane"], "-o", color="C2", lw=2,
                label="capacity (veh/h/lane)")
        ax.set_xlabel("speed limit as a fraction of the legal limit, b")
        ax.set_ylabel("capacity per lane (veh/h)", color="C2")
        ax.tick_params(axis="y", labelcolor="C2")
        i = int(np.argmax(c["q_cap_per_lane"]))
        ax.annotate("peak %.0f at b=%.2f" % (c["q_cap_per_lane"][i], b[i]),
                    (b[i], c["q_cap_per_lane"][i]), fontsize=9, color="C2",
                    xytext=(-10, -24), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="C2", lw=1))
        ax2 = ax.twinx()
        ax2.plot(b, c["rho_crit"], "-s", color="C4", lw=2,
                 label="critical density")
        ax2.set_ylabel("critical density (veh/km/lane)", color="C4")
        ax2.tick_params(axis="y", labelcolor="C4")
        ax.set_title("What a speed limit does to the SUMO fundamental diagram\n"
                     "A = %+.2f, E = %+.2f fitted to these points" %
                     (c["A"], c["E"]))
        ax.grid(alpha=0.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower center")

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "vsl_authority.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
