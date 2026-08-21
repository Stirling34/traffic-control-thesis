# -*- coding: utf-8 -*-
"""
Visual validation of the 210 s prediction metric.

The space-time heat maps make METANET look like a close match while the numbers
say the 210 s correlation is zero. Both are true, and the reason is that a heat
map hides exactly the three things the metric measures:
  * each density panel is scaled to its own range, hiding the 2x density gap;
  * a 150-200 s timing offset is invisible on a 2500 s tall plot;
  * colour cannot distinguish 45 km/h from 75 km/h.

Three figures that show what the metric actually sees:

  1. SCATTER      predicted vs actual speed, one point per cell per start time,
                  against the 1:1 line. Correlation IS the shape of this cloud,
                  so a round blob is the zero correlation made visible.
                  Benchmarked against the do-nothing predictor on the same axes.
  2. PROFILES     speed against position at a few moments: what SUMO actually
                  did, versus what METANET predicted 210 s earlier. Lines are
                  precise where colours are not.
  3. FORECAST MAP the 210 s-ahead forecasts assembled back into a space-time
                  field and placed beside the truth on a SHARED colour scale,
                  plus the signed error. This is the honest visual analogue of
                  the heat map everyone has been reading.

Usage:  python prediction_visuals.py
Output: docs/figures/prediction_scatter.png
        docs/figures/prediction_profiles.png
        docs/figures/prediction_forecast_map.png
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

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH
from validation.metanet_prediction_error import rollout, sumo_state

FIG = os.path.join(_ROOT, "docs", "figures")
LEAD = 210
SHORT = 60
STRIDE = 5


def build():
    geom = load_geometry()
    p = MetanetParams()
    model = MetanetModel(geom, p)
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    T = p.T_sec
    n_h = int(LEAD / T)
    n_run = int(n_sec / T)
    d_all = np.where(np.arange(n_run) * T < DEMAND_END_S, DEMAND_VPH, 0.0)
    u = model.v_ctrl_ub
    keep = list(range(10))                      # 10 distinct SUMO edges

    t0s = np.arange(60, n_sec - LEAD, STRIDE)
    out = {k: [] for k in ("p210", "p60", "pers", "act", "rho", "t")}
    for t0 in t0s:
        rho0, v0 = sumo_state(sumo, t0, geom, model, limits)
        x0 = model.pack_x(rho0, v0, float(sumo["pending"][t0]))
        k0 = int(t0 / T)
        w = d_all[k0:k0 + n_h]
        if len(w) < n_h:
            w = np.concatenate([w, np.zeros(n_h - len(w))])
        _, V = rollout(model, x0, u, w)
        r_t, v_t = sumo_state(sumo, t0 + LEAD, geom, model, limits)
        out["p210"].append(V[int(LEAD / T) - 1][keep])
        out["p60"].append(V[int(SHORT / T) - 1][keep])
        out["pers"].append(v0[keep])            # do-nothing: state never changes
        out["act"].append(v_t[keep])
        out["rho"].append(r_t[keep])
        out["t"].append(t0 + LEAD)
    for k in out:
        out[k] = np.array(out[k])
    # actual at t0+60 for the short-lead panel
    a60 = []
    for t0 in t0s:
        _, v_t = sumo_state(sumo, t0 + SHORT, geom, model, limits)
        a60.append(v_t[keep])
    out["act60"] = np.array(a60)
    return geom, model, p, sumo, out, t0s, keep, limits, d_all, n_h, u


def fig_scatter(out):
    m = out["rho"] > 0.5
    panels = [
        (out["p210"], out["act"], f"METANET, {LEAD} s ahead"),
        (out["pers"], out["act"], f"Do-nothing guess, {LEAD} s ahead"),
        (out["p60"], out["act60"], f"METANET, {SHORT} s ahead"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (p, a, title) in zip(axs, panels):
        pp, aa = p[m], a[m]
        ax.plot([0, 105], [0, 105], "k-", lw=1.2, zorder=3, label="perfect prediction")
        ax.scatter(aa, pp, s=3, alpha=0.12, color="C0", edgecolors="none")
        c = np.corrcoef(pp, aa)[0, 1]
        bias = (pp - aa).mean()
        ax.set_title(f"{title}\ncorrelation {c:.2f}   bias {bias:+.1f} km/h")
        ax.set_xlabel("what SUMO actually did (km/h)")
        ax.set_xlim(0, 105); ax.set_ylim(0, 105); ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("what METANET predicted (km/h)")
    axs[0].legend(loc="lower right", fontsize=9)
    path = os.path.join(FIG, "prediction_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")


def fig_profiles(geom, out):
    x = np.cumsum(geom.lengths_m) - np.array(geom.lengths_m) / 2
    times = [500, 700, 900, 1300]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for ax, tt in zip(axs.ravel(), times):
        i = int(np.argmin(np.abs(out["t"] - tt)))
        ax.plot(x, out["act"][i], "k-o", ms=4, lw=2, label="SUMO actual")
        ax.plot(x, out["p210"][i], "C3-s", ms=4, lw=1.8,
                label=f"METANET, predicted {LEAD} s earlier")
        ax.plot(x, out["pers"][i], "C7--", lw=1.4, label="do-nothing guess")
        ax.set_title(f"t = {out['t'][i]:.0f} s")
        ax.set_xlabel("position on road (m)"); ax.set_ylim(-3, 108)
        ax.grid(alpha=0.25)
    axs[0, 0].set_ylabel("speed (km/h)"); axs[1, 0].set_ylabel("speed (km/h)")
    axs[0, 0].legend(fontsize=9, loc="lower left")
    path = os.path.join(FIG, "prediction_profiles.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")


def fig_forecast_map(geom, out):
    xb = np.concatenate([[0.0], np.cumsum(geom.lengths_m)])
    t = out["t"]
    tb = np.concatenate([t - STRIDE / 2, [t[-1] + STRIDE / 2]])
    act, pred = out["act"], out["p210"]
    err = pred - act

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    for ax, f, ttl, cm, vmin, vmax in [
        (axs[0], act, "SUMO actual", "coolwarm_r", 0, 100),
        (axs[1], pred, f"METANET forecast, made {LEAD} s earlier", "coolwarm_r", 0, 100),
    ]:
        mesh = ax.pcolormesh(xb, tb, f, cmap=cm, vmin=vmin, vmax=vmax, shading="flat")
        ax.set_title(f"{ttl}\nspeed, SHARED scale")
        fig.colorbar(mesh, ax=ax, label="speed (km/h)")
    lim = 60
    mesh = axs[2].pcolormesh(xb, tb, err, cmap="RdBu_r", vmin=-lim, vmax=lim,
                             shading="flat")
    axs[2].set_title("forecast minus actual\nblue = predicted too slow")
    fig.colorbar(mesh, ax=axs[2], label="error (km/h)")
    for ax in axs:
        ax.set_xlabel("position on road (m)")
    axs[0].set_ylabel("time (s)")
    path = os.path.join(FIG, "prediction_forecast_map.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    geom, model, p, sumo, out, *_ = build()
    fig_scatter(out)
    fig_profiles(geom, out)
    fig_forecast_map(geom, out)
    m = out["rho"] > 0.5
    print(f"\n{m.sum()} scored points from {len(out['t'])} start times")
    for lbl, pr, ac in [("METANET 210 s", out["p210"], out["act"]),
                        ("do-nothing   ", out["pers"], out["act"]),
                        ("METANET  60 s", out["p60"], out["act60"])]:
        print(f"  {lbl}: corr {np.corrcoef(pr[m], ac[m])[0,1]:5.2f}   "
              f"MAE {np.abs(pr[m]-ac[m]).mean():5.1f}   "
              f"bias {(pr[m]-ac[m]).mean():+6.1f} km/h")
