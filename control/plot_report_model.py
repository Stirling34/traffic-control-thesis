# -*- coding: utf-8 -*-
"""
Final-report figures for Section 4 (prediction model). Writes to
docs/final_report/figures/.

  fig_model_merge.png  SUMO vs standard METANET vs Bounded-METANET, merge road,
                       open loop (drawn here by control/plot_report_part1.py's
                       fig_models with PARAMS_BM, then copied from
                       docs/figures/report_models.png).
  fig_model_jam.png    straight road, held-out run (4600 veh/h, seed 2): SUMO's
                       speed field after the slow vehicle releases (840 s) vs
                       the jam-calibrated model (PARAMS_BM_JAM_DIS) started from
                       SUMO's measured state at 840 s with the controller's
                       inflow forecast, plus the number of vehicles in the jam
                       (segments below 30 km/h) over time.

usage: python control/plot_report_model.py
"""
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
sys.path.insert(0, os.path.join(ROOT, "sumo"))

import model.metanet_model as mm                                  # noqa: E402
from control import plot_report_part1 as part1                    # noqa: E402
from flat_objective_check import setup                            # noqa: E402
from prediction_study import Road, rollout, sumo                  # noqa: E402

OUT = os.path.join(ROOT, "docs", "final_report", "figures")
T0, H, V_C = 840, 600, 30.0
INK, INK2 = "#0b0b0b", "#52514e"


def main():
    os.makedirs(OUT, exist_ok=True)
    # Draw it rather than rely on an earlier run of plot_report_part1.py, whose
    # other (pre-final-report) figures need caches that are not shared.
    os.makedirs(part1.FIG, exist_ok=True)
    part1.fig_models()
    shutil.copyfile(os.path.join(ROOT, "docs", "figures", "report_models.png"),
                    os.path.join(OUT, "fig_model_merge.png"))

    env = setup("straight", 4600, mm.PARAMS_BM_JAM_DIS)
    road = Road(env, "straight")
    model = env["model"]
    V, C, _ = sumo(env, 2, tag="fig_model_jam")
    rho0, v0, q_in = road.state(V, C, T0)
    R, Vm = rollout(model, rho0, v0, q_in, road, T0, H)
    Tm = model.p.T_sec

    seg = road.seg_idx
    n = len(seg)
    # SUMO speed per segment, 5 s means, over [T0, T0+H]
    ts = np.arange(T0 + Tm, T0 + H + 1e-9, Tm).astype(int)
    Vs = np.array([np.nanmean(V[t - int(Tm):t][:, seg], axis=0) for t in ts])
    Cs = np.array([np.mean(C[t - int(Tm):t][:, seg], axis=0) for t in ts])
    # vehicles in the jam
    veh_m = R * np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
    jam_m = np.where(Vm < V_C, veh_m, 0.0).sum(axis=1)
    jam_s = np.where(np.isfinite(Vs) & (Vs < V_C), Cs, 0.0).sum(axis=1)

    def clear_time(j):
        idx = np.nonzero(j > 0.5)[0]
        return (ts[idx[-1]] - T0) if len(idx) else 0

    cs_, cm_ = clear_time(jam_s), clear_time(jam_m)
    print("clear (s after %d): SUMO %d, model %d" % (T0, cs_, cm_))

    x = np.arange(n + 1) * 0.2
    t_rel = np.concatenate([[0], ts - T0])
    fig, ax5 = plt.subplots(1, 5, figsize=(15.5, 4.8),
                            gridspec_kw=dict(width_ratios=[1, 1, 0.045, 0.22,
                                                           1.15]))
    ax5[3].axis("off")                          # spacer
    axs = [ax5[0], ax5[1], ax5[4]]
    for ax, F, title in ((axs[0], Vs, "SUMO (ground truth)"),
                         (axs[1], Vm, "Bounded-METANET prediction")):
        im = ax.pcolormesh(x, t_rel, F, cmap="coolwarm_r", vmin=0, vmax=90,
                           shading="flat")
        ax.axvline(1.8, color=INK, lw=1, ls="--")
        ax.text(0.9, H * 0.97, "signed section", ha="center", va="top",
                fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("distance from first sign (km)", fontsize=9, color=INK2)
    axs[0].set_ylabel("time after the slow vehicle releases (s)", fontsize=9,
                      color=INK2)
    axs[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=ax5[2])
    cb.set_label("speed (km/h)", fontsize=9)
    ax = axs[2]
    ax.plot(ts - T0, jam_s, color="#2a78d6", lw=2, label="SUMO")
    ax.plot(ts - T0, jam_m, color="#eb6834", lw=2, label="model prediction")
    ax.set_xlabel("time after the slow vehicle releases (s)", fontsize=9,
                  color=INK2)
    ax.set_ylabel("vehicles in the jam (segments below 30 km/h)", fontsize=9,
                  color=INK2)
    ax.set_title("Jam size", fontsize=10, loc="left", color=INK)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    p = os.path.join(OUT, "fig_model_jam.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
