# -*- coding: utf-8 -*-
"""
Where does the disagreement actually show up, in the free-running comparison?

This uses the SAME open-loop run as the original side-by-side figure: METANET
started on an empty road, given only the inflow, left alone for 2500 s with no
feedback from SUMO. Nothing here is the 210 s prediction test.

The original figure looks like a good match. Its actual cell-by-cell correlation
with SUMO is only 0.21. These plots show where that goes, using views the heat
map cannot give:

  1. SHARED COLOUR SCALE. The original scales each density panel to its own
     range, so METANET's peak of 80 and SUMO's 185 both render as deep red.
     Put on one scale, the difference is visible.
  2. DIFFERENCE MAP. METANET minus SUMO, so the disagreement is the thing being
     plotted rather than something the eye has to infer.
  3. CORRELATION OVER TIME. At each instant, how well does the shape of
     METANET's road match the shape of SUMO's road? Shows when agreement fails.
  4. PROFILE SNAPSHOTS. Speed against position at single moments. Lines resolve
     45 vs 75 km/h; colours do not.

Usage:  python open_loop_diagnosis.py
Output: docs/figures/openloop_shared_scale.png
        docs/figures/openloop_correlation.png
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.metanet_model import MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, demand_declared

FIG = os.path.join(_ROOT, "docs", "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    geom = load_geometry()
    p = MetanetParams()
    m = MetanetModel(geom, p)
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    T = p.T_sec

    # the free-running run: empty road, inflow only, no feedback
    r = m.simulate(demand_declared(int(n_sec / T), T), check_demand=False)

    v_s = np.where(sumo["speed"] < 0, np.nan, sumo["speed"])
    rho_s = sumo["rho"]
    idx = np.clip((np.arange(n_sec) / T).astype(int), 0, r["v"].shape[0] - 1)
    v_m = r["v"][:, :10][idx]
    rho_m = r["rho"][:, :10][idx]

    xb = np.concatenate([[0.0], np.cumsum(geom.lengths_m)])
    tb = np.arange(n_sec + 1)
    xc = np.cumsum(geom.lengths_m) - np.array(geom.lengths_m) / 2

    # ---- figure 1: shared colour scale + difference ----------------------
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for j, (fs, fm, lbl, cm, lo, hi, unit) in enumerate([
            (v_s, v_m, "speed", "coolwarm_r", 0, 100, "km/h"),
            (rho_s, rho_m, "density", "coolwarm", 0, 185, "veh/km/lane")]):
        me = axs[j, 0].pcolormesh(xb, tb, fs, cmap=cm, vmin=lo, vmax=hi, shading="flat")
        axs[j, 0].set_title(f"SUMO {lbl}")
        fig.colorbar(me, ax=axs[j, 0], label=unit)
        me = axs[j, 1].pcolormesh(xb, tb, fm, cmap=cm, vmin=lo, vmax=hi, shading="flat")
        axs[j, 1].set_title(f"METANET {lbl}, left alone\nSAME colour scale")
        fig.colorbar(me, ax=axs[j, 1], label=unit)
        d = fm - fs
        lim = float(np.nanpercentile(np.abs(d), 99))
        me = axs[j, 2].pcolormesh(xb, tb, d, cmap="RdBu_r", vmin=-lim, vmax=lim,
                                  shading="flat")
        axs[j, 2].set_title(f"METANET minus SUMO ({lbl})")
        fig.colorbar(me, ax=axs[j, 2], label=unit)
    for ax in axs.ravel():
        ax.set_xlabel("position on road (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("time (s)")
    path = os.path.join(FIG, "openloop_shared_scale.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"Wrote {path}")

    # ---- figure 2: correlation over time + snapshots ---------------------
    win = 30
    corr, occ = [], []
    for k in range(n_sec):
        msk = (rho_s[k] > 0.5) & ~np.isnan(v_s[k])
        occ.append(msk.sum())
        if msk.sum() >= 4 and np.std(v_s[k][msk]) > 1 and np.std(v_m[k][msk]) > 1:
            corr.append(np.corrcoef(v_m[k][msk], v_s[k][msk])[0, 1])
        else:
            corr.append(np.nan)
    corr = np.array(corr)
    sm = np.convolve(np.nan_to_num(corr), np.ones(win) / win, mode="same")
    sm[np.isnan(corr)] = np.nan

    times = [400, 600, 800, 1000, 1300, 1600]
    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(np.arange(n_sec), corr, color="C0", lw=0.6, alpha=0.35)
    ax.plot(np.arange(n_sec), sm, color="C0", lw=2, label="30 s smoothed")
    ax.axhline(0, color="k", lw=0.8)
    for tt in times:
        ax.axvline(tt, color="grey", ls=":", lw=1)
    ax.set_xlim(0, n_sec); ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("spatial correlation\nwith SUMO")
    ax.set_xlabel("time (s)")
    ax.set_title("How well METANET's road shape matches SUMO's, at each instant "
                 "(free-running, no feedback).  1.0 = identical shape, "
                 "0 = unrelated")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25)

    for i, tt in enumerate(times):
        a = fig.add_subplot(gs[1 + i // 3, i % 3])
        a.plot(xc, v_s[tt], "k-o", ms=4, lw=2, label="SUMO")
        a.plot(xc, v_m[tt], "C3-s", ms=4, lw=1.8, label="METANET, left alone")
        c = corr[tt]
        a.set_title(f"t = {tt} s" + ("" if np.isnan(c) else f"   correlation {c:+.2f}"),
                    fontsize=10)
        a.set_ylim(-3, 108); a.grid(alpha=0.25)
        a.set_xlabel("position (m)")
        if i % 3 == 0:
            a.set_ylabel("speed (km/h)")
        if i == 0:
            a.legend(fontsize=8, loc="lower left")
    path = os.path.join(FIG, "openloop_correlation.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"Wrote {path}")

    ok = (rho_s > 0.5) & ~np.isnan(v_s)
    print(f"\nFree-running open loop, whole episode:")
    print(f"  pooled correlation {np.corrcoef(v_m[ok], v_s[ok])[0,1]:.2f}   "
          f"speed MAE {np.abs(v_m-v_s)[ok].mean():.1f} km/h   "
          f"bias {(v_m-v_s)[ok].mean():+.1f}")
    for lo, hi in [(0, 400), (400, 800), (800, 1200), (1200, 1800)]:
        w = np.nanmean(corr[lo:hi])
        print(f"  mean instantaneous correlation, t={lo}-{hi}s: {w:+.2f}")


if __name__ == "__main__":
    main()
