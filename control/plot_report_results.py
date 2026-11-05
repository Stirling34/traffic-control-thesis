# -*- coding: utf-8 -*-
"""
Final-report figures for Section 7 (results). Writes to
docs/final_report/figures/.

  fig_merge_tradeoff.png          merge road, every cost function
  fig_straight_tradeoff_4600.png  straight road at 4600 veh/h
  fig_straight_tradeoff_4800.png  straight road at 4800 veh/h

Drawn in the style of the Assessment-2 trade-off figure the user prefers
(docs/figures/report_tradeoff.png): time on x, shockwave on y, circles
coloured by cost-function family, shaded "improves both" corner, one marker
size. Every direct label has a leader line to its point; label positions are
set by hand in DATA coordinates, routed through empty space so no line
crosses a point or another line.

ONE shockwave measure for both roads (the user's rule: the same graph for both
situations): change in CONGESTION -- the number of 200 m x 8 s cells anywhere
on the road with mean speed below 30 km/h (`wave_vc30_cong_cells`,
validation/wave_metrics.py, identical code on both roads). It counts all
congestion, including any the controller creates itself; on the straight road
it tracks all post-disturbance congestion vehicle-hours exactly
(`jam_vc30_post_veh_h`, r = 1.00 across objectives), whereas the
disturbance-only `jam_vc30_veh_h` does not (r = -0.24). The merge rows store
no vehicle counts, so a vehicle-weighted version needs re-runs there.

x = change in total time per vehicle INCLUDING the entrance queue
(analyse_deploy.spv). Both vs no control, paired on seed (analyse_final.paired:
y is the change of the 10-seed total, x the mean per-seed difference).

Palette: validated categorical slots blue / red / aqua / yellow
(dataviz validate_palette.js, all pairs, light: all checks pass; worst CVD
pair dE 6.9 is legal only with secondary encoding, so key points carry direct
labels and the legend is always present).

Reads data/sweep_final/shard*.json (realistic sign, final configuration).
usage: python control/plot_report_results.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Rectangle                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "control"))
import analyse_final as af                                        # noqa: E402
from control.analyse_deploy import spv                            # noqa: E402

OUT = os.path.join(ROOT, "docs", "final_report", "figures")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e6e6", "#ffffff"
GOOD = "#e8f3ea"
YKEY = "wave_vc30_cong_cells"
XLABEL = "change in total time per vehicle, incl. entrance queue (s)      <- better"
YLABEL = "change in congestion, road below 30 km/h (%)      <- better"
SIZE = 55.0

# (name, colour, terms) -- checked in this order, first match wins, so a
# combination is filed under its most distinctive term
FAMILIES = [
    ("acceleration", "#e34948", ("accel", "accel_temporal", "accel_dens")),
    ("speed variance / safety", "#eda100",
     ("speed_var", "speed_grad", "conflict_risk")),
    ("jam / wave", "#2a78d6", ("jam", "dens_rate", "upstream_queue")),
    ("time / density only", "#1baf7a", ()),
]

# Each figure: labels as policy -> (text, x, y, ha) with (x, y) the label's
# anchor in DATA coordinates; optional fixed axis limits; legend corner.
FIGS = {
    "fig_merge_tradeoff.png": dict(
        exp="final_merge", demand=None, xlim=None, ylim=None,
        legend="upper left",
        labels={
            "RULE": ("reactive rule", 4.0, -3.0, "left"),
            "tts": ("TTS only", 9.4, 19.6, "center"),
            "tts:1|jam:1.3": ("jam", 34.0, 10.0, "left"),
            "tts:1|speed_var:0.6": ("speed variance", 49.0, 12.5, "left"),
            "tts:1|bneck_density:40": ("bottleneck density", 70.0, 52.0, "right"),
            "density_track:1|dvsl:0.1": ("density tracking", 88.9, 46.5, "right"),
        }),
    "fig_straight_tradeoff_4600.png": dict(
        exp="final_straight", demand=4600, xlim=(-12, 175), ylim=(-100, 232),
        legend="upper left",
        labels={
            "RULE": ("reactive rule", 2.0, -84.0, "left"),
            "tts:1|jam:10": ("jam ×4", -5.0, -35.0, "center"),
            "tts:1|speed_var:1.2": ("speed variance", 38.0, -91.0, "left"),
            "tts:1|conflict_risk:60": ("conflict risk", 62.0, -64.0, "left"),
            "tts": ("TTS only", 52.0, 25.0, "left"),
            "accel:1": ("acceleration only", 95.0, 140.0, "right"),
        }),
    "fig_straight_tradeoff_4800.png": dict(
        exp="final_straight", demand=4800, xlim=(-12, 215), ylim=(-100, 197),
        legend="upper left",
        labels={
            "RULE": ("reactive rule", 12.0, -84.0, "left"),
            "tts:1|jam:10": ("jam ×4", 50.0, -92.0, "left"),
            "tts:1|speed_var:1.2": ("speed variance", 15.0, -15.0, "center"),
            "tts:1|conflict_risk:60": ("conflict risk", 88.0, -84.0, "left"),
            "tts": ("TTS only", 82.0, 12.0, "left"),
            "accel:1": ("acceleration only", 190.0, 45.0, "right"),
        }),
    # merge road, sharpest speed drop: at each moment the largest speed
    # difference between neighbouring 200 m segments, averaged over the run
    # (`max_jump`, validation/control_authority.shockwave_metrics -- a legacy
    # cell-based measure, not trajectory-validated). It shows the edge of the
    # queue getting gentler; read with fig_merge_tradeoff.png, which shows the
    # same objectives adding congestion and time.
    "fig_merge_speed_drop.png": dict(
        exp="final_merge", demand=None, xlim=None, ylim=(-14.5, 3.5),
        ykey="max_jump",
        ylabel="change in sharpest speed drop between segments (%)      <- better",
        legend="lower right",
        labels={
            "RULE": ("reactive rule", -1.5, 1.2, "left"),
            "tts": ("TTS only", 6.0, -1.3, "center"),
            "tts:1|jam:1.3": ("jam", 20.0, -1.0, "left"),
            "tts:1|conflict_risk:80": ("conflict risk", 8.0, -9.3, "center"),
            "tts:1|speed_var:0.6": ("speed variance", 50.0, 1.2, "left"),
            "tts:1|accel:1.2": ("acceleration", 34.0, -13.6, "left"),
            "tts:1|jam:5.2": ("jam ×4", 4.0, -11.0, "left"),
            "density_track:1|dvsl:0.1": ("density tracking", 88.5, 2.8, "right"),
        }),
}
OFF_NAMES = {"tts:1|dens_rate:120000": "density rate ×4",
             "tts:1|dens_rate:30000": "density rate"}
LEADER = dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=2, shrinkB=4.5)


def family(policy):
    terms = [t.split(":")[0] for t in policy.split("|")]
    for i, (_, _, keys) in enumerate(FAMILIES[:-1]):
        if any(t in keys for t in terms):
            return i
    return len(FAMILIES) - 1


def points(P, exp, d, ykey=YKEY):
    nc = (exp, "NOCONTROL", d)
    out = []
    for pol in sorted({k[1] for k in P if k[0] == exp and k[2] == d}):
        if pol == "NOCONTROL":
            continue
        a = (exp, pol, d)
        y, x = af.paired(P, a, nc, ykey), af.paired(P, a, nc, spv)
        if x and y:
            out.append(dict(policy=pol, x=x["dmean"], y=y["pct"], p=y["p"]))
    return out


def auto_limits(pts):
    xs = [r["x"] for r in pts] + [0.0]
    ys = [r["y"] for r in pts] + [0.0]
    xp, yp = 0.06 * (max(xs) - min(xs)), 0.06 * (max(ys) - min(ys))
    return (min(xs) - xp, max(xs) + xp), (min(ys) - yp, max(ys) + yp)


def legend(ax, loc):
    h = [Line2D([], [], marker="o", ls="", markerfacecolor=c,
                markeredgecolor=INK2, markersize=8, label=n)
         for n, c, _ in FAMILIES]
    h.append(Line2D([], [], marker="*", ls="", color=INK, markersize=11,
                    label="reactive rule (benchmark)"))
    ax.legend(handles=h, title="cost function family", loc=loc,
              fontsize=8, title_fontsize=8.5, framealpha=0.95)


def figure(P, name, spec):
    pts = points(P, spec["exp"], spec["demand"], spec.get("ykey", YKEY))
    ax_x, ax_y = auto_limits(pts)
    (x0, x1) = spec["xlim"] or ax_x
    (y0, y1) = spec["ylim"] or ax_y
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor=SURF)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.add_patch(Rectangle((x0, y0), -x0, -y0, color=GOOD, zorder=0.5, lw=0))
    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    off = []
    for r in pts:
        if not (x0 <= r["x"] <= x1 and y0 <= r["y"] <= y1):
            off.append("%s: %+.0f s, %+.0f%%" % (
                OFF_NAMES.get(r["policy"], r["policy"]), r["x"], r["y"]))
            continue
        if r["policy"] == "RULE":
            ax.scatter(r["x"], r["y"], s=230, marker="*", c=INK,
                       edgecolors=SURF, linewidths=1.0, zorder=4)
        else:
            _, col, _ = FAMILIES[family(r["policy"])]
            ax.scatter(r["x"], r["y"], s=SIZE, c=col, edgecolors=INK2,
                       linewidths=0.7, alpha=0.9, zorder=3)
        lab = spec["labels"].get(r["policy"])
        if lab:
            text, tx, ty, ha = lab
            ax.annotate(text, (r["x"], r["y"]), xytext=(tx, ty),
                        textcoords="data", fontsize=9, color=INK,
                        weight="bold", ha=ha, va="center", zorder=6,
                        arrowprops=LEADER)
    if off:
        ax.text(0.98, 0.98, "off scale (not shown):\n" + "\n".join(off),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=INK2, bbox=dict(boxstyle="round,pad=0.35", fc=SURF,
                                      ec=GRID))
    ax.set_xlabel(XLABEL, fontsize=9.5, color=INK)
    ax.set_ylabel(spec.get("ylabel", YLABEL), fontsize=9.5, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    legend(ax, spec["legend"])
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor=SURF)
    plt.close(fig)
    print("wrote", p)


def main():
    os.makedirs(OUT, exist_ok=True)
    P, _ = af.load()
    for name, spec in FIGS.items():
        figure(P, name, spec)


if __name__ == "__main__":
    main()
