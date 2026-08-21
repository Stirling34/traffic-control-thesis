# -*- coding: utf-8 -*-
"""
The density plot: no control against the controller, same seed, same colour scale.

This is the picture the shockwave metric is a number for. Time runs left to
right, road runs top to bottom with the merge at the bottom, and colour is
density. A shockwave is a dark band running diagonally UP the picture - a jam
forming at the merge and travelling backwards against the traffic.

Both panels use one colour scale, so "less red" means less red. The third panel
is the difference, which is where the effect is actually legible: blue is
density the controller removed, red is density it added.

Usage:  python control/plot_density_compare.py --fix bm_scaled --seed 5
Output: docs/figures/density_<fix>_seed<n>.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MS_TO_KMH, load_geometry              # noqa: E402
from control.nmpc_metanet import NmpcVslController                    # noqa: E402
from control.fix_plateau import FIXES, _model_for, STEPS              # noqa: E402
from validation.control_authority import shockwave_metrics            # noqa: E402
from validation.sumo_metrics import CONFIG                            # noqa: E402

FIG = os.path.join(_ROOT, "docs", "figures")


def run(fix, seed, geom, cost, controlled=True):
    """One run. `controlled=False` gives the no-control baseline."""
    import traci
    kw = dict(FIXES[fix])
    kw.pop("model", None)
    kw.pop("coef", None)
    kw.pop("preset", None)
    kw.pop("kink_init", None)
    cost = kw.pop("cost", cost)
    model = _model_for(fix, geom)
    ctrl = NmpcVslController(model, cost=cost, **kw) if controlled else None
    p = model.p

    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    applied = limits[: model.n_ctrl].copy()

    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    dens, spd, vsl = [], [], []
    for step in range(STEPS):
        traci.simulationStep()
        t = step + 1
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        vraw = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        v_edges = np.where(c > 0, vraw, np.append(applied, limits[-1]))
        dens.append(c / (Lkm * lanes))
        spd.append(np.where(c > 0, vraw, np.nan))
        vsl.append(applied.copy())
        if ctrl is None or t % 30:
            continue
        rho_seg = dens[-1][model.seg_edge_index]
        if rho_seg.min() <= 1.0:
            continue
        d_hat = np.where(t + np.arange(ctrl.Np * ctrl.M) * p.T_sec < 1000.0,
                         4000.0, 0.0)
        u, info = ctrl.solve(rho_seg, v_edges[model.seg_edge_index],
                             float(len(traci.simulation.getPendingVehicles())),
                             d_hat, applied)
        if info["success"]:
            applied = np.asarray(u, float)
            for e, vv in zip(edges[: model.n_ctrl], applied):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
    traci.close()
    dens, spd, vsl = np.array(dens), np.array(spd), np.array(vsl)
    m = shockwave_metrics(dens * 0 + spd, dens, 1.0, Lkm, lanes)
    return dens, spd, vsl, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", default="bm_scaled", choices=list(FIXES))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--cost", default="accel_delay")
    ap.add_argument("--rerun", action="store_true",
                    help="ignore the cached fields and run SUMO again")
    a = ap.parse_args()
    geom = load_geometry()

    cache = os.path.join(_ROOT, "data", "fields_%s_seed%d.npz" % (a.fix, a.seed))
    if os.path.exists(cache) and not a.rerun:
        z = np.load(cache)
        d0, s0, d1, s1, vsl = z["d0"], z["s0"], z["d1"], z["s1"], z["vsl"]
        m0 = shockwave_metrics(s0, d0, 1.0, np.array(geom.lengths_m) / 1000.0,
                               np.array(geom.lanes, float))
        m1 = shockwave_metrics(s1, d1, 1.0, np.array(geom.lengths_m) / 1000.0,
                               np.array(geom.lanes, float))
    else:
        d0, s0, _, m0 = run(a.fix, a.seed, geom, a.cost, controlled=False)
        d1, s1, vsl, m1 = run(a.fix, a.seed, geom, a.cost, controlled=True)
        np.savez_compressed(cache, d0=d0, s0=s0, d1=d1, s1=s1, vsl=vsl)
    n_ctrl = vsl.shape[1]
    d_acc = 100 * (m1["accel_rms"] / m0["accel_rms"] - 1)

    extent = [0, len(d0), n_ctrl, 0]
    vmax = float(np.nanpercentile(np.concatenate([d0[:, :n_ctrl],
                                                  d1[:, :n_ctrl]]), 99.5))

    fig, axs = plt.subplots(4, 1, figsize=(13, 12.5), sharex=True,
                            constrained_layout=True)

    for ax, dd, ttl in ((axs[0], d0, "No control"),
                        (axs[1], d1, "Bounded-METANET NMPC (accel_delay, scaled VSL cap)")):
        im = ax.imshow(dd[:, :n_ctrl].T, aspect="auto", origin="upper",
                       extent=extent, cmap="inferno_r", vmin=0, vmax=vmax,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label="density (veh/km/lane)", pad=0.01)
        ax.set_ylabel("segment\n(merge at bottom)")
        ax.set_title(ttl)

    ax = axs[2]
    diff = d1[:, :n_ctrl] - d0[:, :n_ctrl]
    lim = float(np.nanpercentile(np.abs(diff), 99))
    im = ax.imshow(diff.T, aspect="auto", origin="upper", extent=extent,
                   cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="density change (veh/km/lane)", pad=0.01)
    ax.set_ylabel("segment\n(merge at bottom)")
    ax.set_title("Difference: BLUE = density the controller removed, "
                 "RED = density it added")

    ax = axs[3]
    im = ax.imshow(vsl.T, aspect="auto", origin="upper", extent=extent,
                   cmap="viridis", vmin=20, vmax=float(np.max(vsl)),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="posted limit (km/h)", pad=0.01)
    ax.set_ylabel("segment\n(merge at bottom)")
    ax.set_xlabel("time (s)")
    ax.set_title("What the controller posted")

    fig.suptitle("Seed %d: shockwave acceleration %+.1f%%  "
                 "(peak density %.0f -> %.0f veh/km/lane)"
                 % (a.seed, d_acc, np.nanmax(d0[:, :n_ctrl]),
                    np.nanmax(d1[:, :n_ctrl])), fontsize=13)

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "density_%s_seed%d.png" % (a.fix, a.seed))
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)
    print("  shockwave accel RMS %.4f -> %.4f  (%+.1f%%)"
          % (m0["accel_rms"], m1["accel_rms"], d_acc))
    print("  peak density        %.1f -> %.1f veh/km/lane"
          % (np.nanmax(d0[:, :n_ctrl]), np.nanmax(d1[:, :n_ctrl])))
    hi = lambda d: 100.0 * np.mean(d[:, :n_ctrl] > 60)
    print("  share of the picture above 60 veh/km/lane: %.1f%% -> %.1f%%"
          % (hi(d0), hi(d1)))
    accel_figure(a, geom, d0, s0, d1, s1, m0, m1, n_ctrl)


def accel_figure(a, geom, d0, s0, d1, s1, m0, m1, n_ctrl):
    """Where the acceleration reduction actually lives.

    The density plot shows how much traffic is on the road; it says nothing
    about how violently that traffic changes speed. The metric is the second
    thing. This plots the per-vehicle acceleration magnitude the metric is an
    RMS of, so the improvement can be seen rather than taken on trust.
    """
    lam = np.array(geom.lanes, float)
    Lkm = np.array(geom.lengths_m) / 1000.0

    def field(v, rho):
        """Vehicle-weighted |acceleration| per cell, Zegeye eqs. (16)-(19)."""
        T_h = 1.0 / 3600.0
        q = rho * v * lam
        with np.errstate(invalid="ignore"):
            a_temp = np.abs(v[1:] - v[:-1]) / 1.0 / 3.6
            n_temp = np.maximum(Lkm * lam * rho[:-1] - T_h * q[:-1], 0.0)
            a_spat = np.abs(v[1:, 1:] - v[:-1, :-1]) / 1.0 / 3.6
            n_spat = T_h * q[:-1, :-1]
            out = np.zeros_like(a_temp)
            wt = np.zeros_like(a_temp)
            ok = np.isfinite(a_temp) & (n_temp > 0)
            out[ok] += n_temp[ok] * a_temp[ok]
            wt[ok] += n_temp[ok]
            ok2 = np.isfinite(a_spat) & (n_spat > 0)
            out[:, :-1][ok2] += n_spat[ok2] * a_spat[ok2]
            wt[:, :-1][ok2] += n_spat[ok2]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(wt > 0, out / np.maximum(wt, 1e-9), np.nan)

    f0, f1 = field(s0, d0), field(s1, d1)
    extent = [0, len(f0), n_ctrl, 0]
    vmax = float(np.nanpercentile(np.concatenate(
        [f0[:, :n_ctrl], f1[:, :n_ctrl]]), 99.0))

    fig, axs = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True,
                            constrained_layout=True)
    for ax, ff, ttl in ((axs[0], f0, "No control"),
                        (axs[1], f1, "Bounded-METANET NMPC")):
        im = ax.imshow(ff[:, :n_ctrl].T, aspect="auto", origin="upper",
                       extent=extent, cmap="magma_r", vmin=0, vmax=vmax,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label="|acceleration| (m/s^2)", pad=0.01)
        ax.set_ylabel("segment\n(merge at bottom)")
        ax.set_title(ttl)

    ax = axs[2]
    diff = f1[:, :n_ctrl] - f0[:, :n_ctrl]
    lim = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99))
    im = ax.imshow(diff.T, aspect="auto", origin="upper", extent=extent,
                   cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="change (m/s^2)", pad=0.01)
    ax.set_ylabel("segment\n(merge at bottom)")
    ax.set_xlabel("time (s)")
    ax.set_title("Difference: BLUE = gentler than no control, RED = harsher")

    fig.suptitle("Seed %d: this is what the -%.1f%% is measuring "
                 "(vehicle-weighted acceleration RMS %.3f -> %.3f m/s^2)"
                 % (a.seed, abs(100 * (m1["accel_rms"] / m0["accel_rms"] - 1)),
                    m0["accel_rms"], m1["accel_rms"]), fontsize=13)
    p = os.path.join(FIG, "accel_%s_seed%d.png" % (a.fix, a.seed))
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)
    for nm, ff in (("no control", f0), ("controlled", f1)):
        v = ff[:, :n_ctrl]
        v = v[np.isfinite(v)]
        print("  %-11s mean |accel| %.3f   share above 1.0 m/s^2 %.1f%%"
              % (nm, v.mean(), 100.0 * np.mean(v > 1.0)))


if __name__ == "__main__":
    main()
