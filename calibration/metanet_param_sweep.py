# -*- coding: utf-8 -*-
"""
Is METANET's free-flow prediction error tunable, or structural?

`metanet_prediction_error.py` showed METANET models congestion well (speed MAE
plateaus ~14 km/h at 210 s, beating persistence) and free flow badly (MAE grows
to ~47 km/h). This sweeps the parameters that could plausibly cause that and
measures the effect on the SAME 210 s-ahead objective, so the question is
settled with evidence rather than argument.

The prime suspect is the shape of the fundamental diagram. METANET's
equilibrium speed is

    Veq(rho) = v_free * exp( -(1/a) * (rho/rho_crit)^a )

so at capacity (rho = rho_crit) it predicts v_crit = v_free * exp(-1/a),
regardless of anything else. With the calibrated a = 1.867 that is 59 km/h.
SUMO's bottleneck actually discharges at **81.4 km/h at density 29.6** — a
22 km/h error built into the FD, sitting exactly in the regime where the
prediction error is worst.

`a` is swept with `rho_crit` re-derived each time to hold bottleneck capacity
at the measured SUMO discharge, so every candidate carries the same flow and
only the FD *shape* changes. Then tau (speed relaxation) and eta
(anticipation) are swept around the best `a`.

NOTE ON DEFENSIBILITY: the project's ledger cites a literature range of
a in [1.4, 2.3]. Values above that are outside it and need a citation check
before going in the thesis. This is NOT the ledger's rejected approach — that
one pushed a DOWN to ~1.0 to chase queue density. This pushes the other way and
trades queue density (already ~2x low) for free-flow fidelity.

Usage:  python metanet_param_sweep.py
Output: data/param_sweep.npz, printed tables
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH
from validation.metanet_prediction_error import (CONGESTED_KMH, HORIZON_S, rollout,
                                      sumo_state)

T0_STRIDE = 20           # coarser than the main test — this is a relative comparison
Q_TARGET = 2325.0        # SUMO bottleneck discharge, t=300-1000s, corrected window
OUT_NPZ = os.path.join(_ROOT, "data", "param_sweep.npz")


def evaluate(geom, sumo, params, t0s, limits_kmh):
    """210 s-ahead error for one parameter set. Returns a metrics dict."""
    model = MetanetModel(geom, params)
    T = params.T_sec
    n_h = int(HORIZON_S / T)
    n_run = int(sumo["rho"].shape[0] / T)
    t_grid = np.arange(n_run) * T
    d_all = np.where(t_grid < DEMAND_END_S, DEMAND_VPH, 0.0)
    u = model.v_ctrl_ub

    pv, prho, tv, trho = [], [], [], []
    for t0 in t0s:
        rho0, v0 = sumo_state(sumo, t0, geom, model, limits_kmh)
        x0 = model.pack_x(rho0, v0, float(sumo["pending"][t0]))
        k0 = int(t0 / T)
        d_win = d_all[k0:k0 + n_h]
        if len(d_win) < n_h:
            d_win = np.concatenate([d_win, np.zeros(n_h - len(d_win))])
        R, V = rollout(model, x0, u, d_win)
        pv.append(V[-1]); prho.append(R[-1])
        r_t, v_t = sumo_state(sumo, t0 + HORIZON_S, geom, model, limits_kmh)
        tv.append(v_t); trho.append(r_t)

    pv, prho = np.array(pv), np.array(prho)
    tv, trho = np.array(tv), np.array(trho)

    loaded = trho > 0.5
    cong = (tv < CONGESTED_KMH) & loaded
    free = (tv >= CONGESTED_KMH) & loaded

    def mae(mask):
        e = np.where(mask, pv - tv, np.nan)
        return float(np.nanmean(np.abs(e)))

    seg_veh = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
    veh_p = (prho * seg_veh).sum(axis=1)
    veh_t = (trho * seg_veh).sum(axis=1)
    ok = veh_t > 1.0

    v_crit = params.v_free_bneck * np.exp(-1.0 / params.a)
    return {
        "free_mae": mae(free), "cong_mae": mae(cong), "all_mae": mae(loaded),
        "veh_mae": float(np.mean(np.abs(veh_p - veh_t)[ok])),
        "veh_bias": float(np.mean((veh_p - veh_t)[ok])),
        "v_crit": float(v_crit), "rho_crit": float(params.rho_crit),
    }


def rho_crit_for(a, v_free_bneck, q_target=Q_TARGET):
    """rho_crit that holds 1-lane capacity at the measured SUMO discharge."""
    return q_target / (v_free_bneck * np.exp(-1.0 / a))


def show(title, rows):
    print(f"\n{title}")
    print(f"{'config':>26} {'rho_crit':>9} {'v_crit':>7} | "
          f"{'FREE mae':>9} {'cong mae':>9} {'all mae':>8} | "
          f"{'veh mae':>8} {'veh bias':>9}")
    print("-" * 106)
    for label, m in rows:
        print(f"{label:>26} {m['rho_crit']:9.1f} {m['v_crit']:7.1f} | "
              f"{m['free_mae']:9.2f} {m['cong_mae']:9.2f} {m['all_mae']:8.2f} | "
              f"{m['veh_mae']:8.1f} {m['veh_bias']:+9.1f}")


def main():
    geom = load_geometry()
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    limits_kmh = np.array(geom.limits_ms) * MS_TO_KMH
    t0s = np.arange(60, n_sec - HORIZON_S, T0_STRIDE)

    base = MetanetParams()
    v_fb = geom.limits_ms[-1] * MS_TO_KMH
    print(f"{len(t0s)} rollouts per config, {HORIZON_S} s ahead, no control.")
    print(f"SUMO truth at the bottleneck while free-flowing: "
          f"81.4 km/h at density 29.6 veh/km/lane.")

    results = {}

    # ---- stage 1: FD shape ------------------------------------------------
    rows = []
    for a in [1.867, 2.3, 3.0, 3.5, 4.0, 5.0]:
        p = MetanetParams(a=a, rho_crit=rho_crit_for(a, v_fb))
        m = evaluate(geom, sumo, p, t0s, limits_kmh)
        label = f"a={a}" + ("  (current)" if a == base.a else "")
        rows.append((label, m))
        results[f"a{a}"] = m
    show("STAGE 1 - fundamental diagram shape (capacity held at SUMO's 2325 veh/h)",
         rows)
    best_a = min(rows, key=lambda r: r[1]["free_mae"])
    a_star = float(best_a[0].split("=")[1].split()[0])
    print(f"\n  best free-flow MAE at a = {a_star}")

    # ---- stage 2: dynamics, around the best FD ----------------------------
    rows = []
    for tau in [12.0, 18.0, 30.0]:
        for eta in [15.0, 30.0, 60.0]:
            p = MetanetParams(a=a_star, rho_crit=rho_crit_for(a_star, v_fb),
                              tau_sec=tau, eta=eta)
            m = evaluate(geom, sumo, p, t0s, limits_kmh)
            rows.append((f"a={a_star} tau={tau:.0f} eta={eta:.0f}", m))
            results[f"a{a_star}_tau{tau}_eta{eta}"] = m
    show("STAGE 2 - relaxation & anticipation, at the best FD", rows)

    # ---- reference: the shipped configuration -----------------------------
    m0 = evaluate(geom, sumo, MetanetParams(), t0s, limits_kmh)
    show("REFERENCE - configuration as shipped", [("shipped", m0)])
    best = min(rows, key=lambda r: r[1]["free_mae"])
    print(f"\nBest overall: {best[0]}")
    print(f"  free-flow speed MAE {m0['free_mae']:.2f} -> {best[1]['free_mae']:.2f} km/h "
          f"({100*(1-best[1]['free_mae']/m0['free_mae']):+.0f}%)")
    print(f"  congested speed MAE {m0['cong_mae']:.2f} -> {best[1]['cong_mae']:.2f} km/h "
          f"({100*(1-best[1]['cong_mae']/m0['cong_mae']):+.0f}%)")
    print(f"  vehicles-on-road MAE {m0['veh_mae']:.1f} -> {best[1]['veh_mae']:.1f} veh "
          f"({100*(1-best[1]['veh_mae']/m0['veh_mae']):+.0f}%)")
    print("\nNOTE: a change here alters the open-loop calibration too — re-run"
          "\n`python metanet_vs_sumo.py` before adopting anything.")

    os.makedirs("data", exist_ok=True)
    np.savez(OUT_NPZ, **{f"{k}__{kk}": v for k, m in results.items()
                         for kk, v in m.items()})
    print(f"Saved {OUT_NPZ}")


if __name__ == "__main__":
    main()
