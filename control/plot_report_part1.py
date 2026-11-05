# -*- coding: utf-8 -*-
"""
Figures for Part 1 of the report: the three defects and what fixing them did.

  report_models.png    SUMO against the two prediction models, side by side, as
                       density over time and space. This is the figure that
                       shows why the original model could not support control:
                       it smears the queue into a uniform blur, so the jam it
                       predicts is not the jam SUMO produces.

  report_skill.png     the same comparison as a number: forecast accuracy at
                       the 210 s horizon the controller actually plans over,
                       against a "nothing changes" reference.

  report_actuator.png  the speed limit's effect on predicted equilibrium speed,
                       before and after correcting the compliance factor. The
                       flat line is the defect: the model's prediction does not
                       respond to the sign at all.

  report_metric.png    the acceleration distribution at 1 s and 8 s sampling,
                       with the physically achievable range marked.

Usage:  python control/plot_report_part1.py
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

from model.metanet_model import (PARAMS_BM, MetanetModel,          # noqa: E402
                                 MetanetParams, load_geometry)

DATA = os.path.join(_ROOT, "data")
FIG = os.path.join(_ROOT, "docs", "figures")
STEPS = 2500


def model_fields():
    """Open-loop density and speed from both prediction models, on their own
    time grids (the models step at T s, SUMO at 1 s)."""
    geom = load_geometry()
    out = {}
    for tag, kw in (("original METANET", {}),
                    ("Bounded-METANET", PARAMS_BM)):
        m = MetanetModel(geom, MetanetParams(**kw))
        T = m.p.T_sec
        d = np.where(np.arange(int(STEPS / T)) * T < 1000.0, 4000.0, 0.0)
        sim = m.simulate(d, check_demand=False)
        xb = np.concatenate([[0.0], np.cumsum(m.seg_lengths_km) * 1000.0])
        out[tag] = (xb, T, sim["rho"], sim["v"])
    return out


def fig_models(window_s=1000.0):
    """SUMO against both prediction models.

    Conventions follow validation/metanet_vs_sumo.py, which is what the earlier
    figures used: position along the road on the x-axis, time running up the
    y-axis, `coolwarm` colouring, and cropped to the demand period. Density and
    speed are both shown because a shockwave is visible in each.

    Density shares one scale across the three panels here, unlike the earlier
    per-panel scaling. The point of this figure is precisely that the models
    disagree with SUMO about how dense the queue gets, and per-panel scaling
    would hide it.
    """
    geom = load_geometry()
    z = np.load(os.path.join(DATA, "sumo_baseline.npz"))
    n_s = min(int(window_s), z["rho"].shape[0])
    x_edges = np.concatenate([[0.0], np.cumsum(geom.lengths_m)])
    t_sumo = np.arange(n_s + 1) * 1.0
    v_sumo = np.where(z["speed"][:n_s] < 0, np.nan, z["speed"][:n_s])

    cols = [("SUMO (ground truth)", x_edges, t_sumo, z["rho"][:n_s], v_sumo)]
    for tag, (xb, T, rho, v) in model_fields().items():
        n_m = int(window_s / T)
        cols.append((tag, xb, np.arange(min(n_m, len(rho)) + 1) * T,
                     rho[:n_m], v[:n_m]))

    rho_top = max(float(np.nanmax(c[3])) for c in cols)
    v_top = max(float(np.nanmax(c[4])) for c in cols)

    fig, axs = plt.subplots(2, len(cols), figsize=(15.5, 9.4),
                            constrained_layout=True)
    for j, (title, xb, tb, rho, v) in enumerate(cols):
        mrho = axs[0, j].pcolormesh(xb, tb, rho, cmap="coolwarm",
                                    vmin=0, vmax=rho_top, shading="flat")
        axs[0, j].set_title("%s\ndensity (peak %.0f veh/km/lane)"
                            % (title, float(np.nanmax(rho))), fontsize=11)
        mv = axs[1, j].pcolormesh(xb, tb, v, cmap="coolwarm_r",
                                  vmin=0, vmax=v_top, shading="flat")
        axs[1, j].set_title("%s\nspeed" % title, fontsize=11)
        for ax in (axs[0, j], axs[1, j]):
            ax.set_xlabel("position along road (m)")
    fig.colorbar(mrho, ax=axs[0, :], label="density (veh/km/lane)",
                 fraction=0.028, pad=0.012)
    fig.colorbar(mv, ax=axs[1, :], label="speed (km/h)",
                 fraction=0.028, pad=0.012)
    for ax in axs[:, 0]:
        ax.set_ylabel("time (s)")
    p = os.path.join(FIG, "report_models.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)
    for title, xb, tb, rho, v in cols[1:]:
        print("   %-18s peak density %5.1f   (SUMO %5.1f)"
              % (title, float(np.nanmax(rho)), float(np.nanmax(cols[0][3]))))


def fig_skill():
    got = []
    for tag, lbl in (("shipped", "original\nMETANET"),
                     ("s14", "asymmetric\nanticipation"),
                     ("bounded", "Bounded-\nMETANET")):
        p = os.path.join(DATA, "prediction_skill_%s.json" % tag)
        if os.path.exists(p):
            d = json.load(open(p))
            got.append((lbl,
                        np.nanmean([r["r2_rho_model"] for r in d]),
                        np.nanmean([r["r2_v_model"] for r in d]),
                        np.nanmean([r["r2_rho_persist"] for r in d]),
                        np.nanmean([r["r2_v_persist"] for r in d])))
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    x = np.arange(len(got))
    ax.bar(x - 0.19, [g[1] for g in got], 0.36, label="density", color="C0")
    ax.bar(x + 0.19, [g[2] for g in got], 0.36, label="speed", color="C4")
    ax.axhline(np.mean([g[3] for g in got]), color="C0", ls="--", lw=1.5,
               label="assume nothing changes (density)")
    ax.axhline(np.mean([g[4] for g in got]), color="C4", ls=":", lw=1.8,
               label="assume nothing changes (speed)")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in got], fontsize=10)
    ax.set_ylabel("forecast accuracy at 210 s ($R^2$ against SUMO)")
    ax.set_title("A model below the dashed lines is worse than assuming\n"
                 "the traffic never changes", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    p = os.path.join(FIG, "report_skill.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("wrote", p)


def fig_actuator():
    """How predicted equilibrium speed responds to the posted limit."""
    import casadi as cs
    import sym_metanet as sm
    geom = load_geometry()
    eng = sm.engines.get_current_engine()
    us = np.linspace(30.0, 95.0, 60)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True,
                            constrained_layout=True)
    for ax, (scaled, title) in zip(axs, (
            (False, "Before: assumed drivers exceed the sign by 10%"),
            (True, "After: measured compliance (11% below the sign)"))):
        m = MetanetModel(geom, MetanetParams(**PARAMS_BM, vsl_scaled=scaled))
        L = m.L1
        rho = cs.SX.sym("rho", L.N)
        L.actions["v_ctrl"] = cs.SX.sym("u", len(L.vsl))
        # Reproduce what the Bounded-METANET step actually computes. Calling
        # _get_equilibrium_speed here would fall through to the PARENT class,
        # which knows nothing about vsl_scaled and would make both panels
        # identical -- the cap lives in _vsl_cap, applied inside the step.
        V_fd = eng.links.Veq(cs.fmax(rho, 0.0), L.v_free, L.rho_crit, L.a)
        V = cs.fmin(V_fd, L._vsl_cap())
        f = cs.Function("f", [rho, L.actions["v_ctrl"]], [V])
        for rr, col in ((10.0, "C0"), (30.0, "C2"), (60.0, "C3")):
            v = [float(np.array(f(np.full(L.N, rr), np.full(9, u)))[4, 0])
                 for u in us]
            ax.plot(us, v, color=col, lw=2.2,
                    label="density %.0f veh/km/lane" % rr)
        ax.axvline(float(np.mean(geom.main_limits_kmh)), color="k", ls=":",
                   lw=1.4)
        ax.text(float(np.mean(geom.main_limits_kmh)) - 1, 8, "legal limit",
                rotation=90, fontsize=9, ha="right", color="dimgray")
        ax.set_xlabel("posted speed limit (km/h)")
        ax.set_title(title, fontsize=11.5)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("predicted equilibrium speed (km/h)")
    axs[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("What the model thinks the sign does. A flat line means the "
                 "prediction ignores the sign entirely,\nso the optimiser has "
                 "no reason to change it.", fontsize=12)
    p = os.path.join(FIG, "report_actuator.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("wrote", p)


def fig_metric():
    from model.metanet_model import load_geometry as lg
    geom = lg()
    lam = np.array(geom.lanes, float)
    Lkm = np.array(geom.lengths_m) / 1000.0
    z = np.load(os.path.join(DATA, "fields_bm_scaled_seed5.npz"))
    v, rho = z["s0"], z["d0"]

    def accel(step):
        n = (len(v) // step) * step
        if step > 1:
            with np.errstate(invalid="ignore"):
                vv = np.nanmean(v[:n].reshape(-1, step, v.shape[1]), axis=1)
                rr = np.nanmean(rho[:n].reshape(-1, step, rho.shape[1]), axis=1)
        else:
            vv, rr = v, rho
        Th = step / 3600.0
        q = rr * vv * lam
        with np.errstate(invalid="ignore"):
            at = np.abs(vv[1:] - vv[:-1]) / step / 3.6
            nt = np.maximum(Lkm * lam * rr[:-1] - Th * q[:-1], 0.0)
            asp = np.abs(vv[1:, 1:] - vv[:-1, :-1]) / step / 3.6
            ns = Th * q[:-1, :-1]
        a = np.concatenate([at.ravel(), asp.ravel()])
        n_ = np.concatenate([nt.ravel(), ns.ravel()])
        m = np.isfinite(a) & np.isfinite(n_) & (n_ > 0)
        return a[m], n_[m]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    bins = np.linspace(0, 26, 70)
    for step, col, lbl in ((1, "C3", "sampled at 1 s (as originally used)"),
                           (8, "C0", "aggregated to 8 s (corrected)")):
        a, n_ = accel(step)
        ax.hist(a, bins=bins, weights=n_ / n_.sum(), histtype="step", lw=2.2,
                color=col, label="%s   peak %.1f m/s$^2$" % (lbl, a.max()))
    ax.axvspan(0, 4.5, color="green", alpha=0.09)
    ax.text(2.2, ax.get_ylim()[1] * 0.62, "physically achievable\n(vehicle "
            "braking limit 4.5 m/s$^2$)", fontsize=9.5, ha="center",
            color="darkgreen")
    ax.set_yscale("log")
    ax.set_xlabel("acceleration attributed to vehicles (m/s$^2$)")
    ax.set_ylabel("share of vehicle-seconds (log scale)")
    ax.set_title("The metric before and after correcting its time step\n"
                 "82% of the original metric came from values outside the "
                 "shaded region", fontsize=12)
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.25)
    p = os.path.join(FIG, "report_metric.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("wrote", p)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_models()
    fig_skill()
    fig_actuator()
    fig_metric()


if __name__ == "__main__":
    main()
