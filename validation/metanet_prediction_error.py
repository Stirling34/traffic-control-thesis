# -*- coding: utf-8 -*-
"""
How accurate is METANET at the horizon the MPC actually uses?

Every accuracy number the project had before this file was a 2500 s open-loop
match (`metanet_vs_sumo.py`). The controller never does that. It re-reads the
state from SUMO every 30 s and predicts only Np*M*T = 210 s ahead
(`nmpc_metanet.py`). So the relevant question is:

    starting from a SUMO-measured state, how wrong is METANET after 210 s?

Method
------
For each start time t0 across the SUMO baseline:
  1. read SUMO's per-edge density/speed and insertion queue at t0
     (exactly the mapping main_metanet_v1.py uses for state feedback),
  2. roll METANET forward 210 s at the legal speed limits (the baseline is
     no-control, so that is the correct input),
  3. compare against what SUMO actually did at t0 + lead.

Benchmarked against PERSISTENCE — "assume nothing changes for 210 s". This is
the benchmark that matters: a prediction model that cannot beat persistence is
adding nothing, however good its open-loop heat map looks.

Two demand forecasts are tested:
  declared — 4000 veh/h until t = 1000 s then 0 (known a priori, and
             control-independent, so this is the honest option for Phase D)
  realised — SUMO's measured injections replayed (not knowable a priori;
             included as an upper bound on what a perfect demand forecast buys)

Outputs: data/prediction_error.npz, docs/figures/prediction_error.png
Usage:   python metanet_prediction_error.py
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import (BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH,
                             demand_realised)

FIG_DIR = os.path.join(_ROOT, "docs", "figures")
OUT_NPZ = os.path.join(_ROOT, "data", "prediction_error.npz")

HORIZON_S = 210          # Np(7) * M(6) * T(5s) — must match nmpc_metanet.py
LEADS_S = [30, 60, 90, 120, 150, 180, 210]
T0_STRIDE = 10           # start a rollout every 10 s
CONGESTED_KMH = 50.0     # SUMO speed below this = "congested" regime


def sumo_state(sumo, t, geom, model, limits_kmh):
    """SUMO state at second `t`, mapped onto METANET segments.

    Same convention as main_metanet_v1.py: an empty edge reports -1 m/s, which
    is both meaningless and (at exactly v_free) a numerical landmine, so empty
    edges fall back to the posted limit.
    """
    rho_edges = sumo["rho"][t]
    v_raw = sumo["speed"][t]
    v_edges = np.where(v_raw > 0, v_raw, limits_kmh)
    return rho_edges[model.seg_edge_index], v_edges[model.seg_edge_index]


def rollout(model, x0, u, d_window):
    """Roll METANET forward, returning (rho, v) at each step."""
    x = np.asarray(x0, dtype=float)
    RHO, V = [], []
    for d in d_window:
        x = np.asarray(model.F(x, u, float(d))[0]).ravel()
        rho, v, _ = model.unpack_x(x)
        RHO.append(rho)
        V.append(v)
    return np.array(RHO), np.array(V)


def main():
    geom = load_geometry()
    params = MetanetParams()
    model = MetanetModel(geom, params)

    if not os.path.exists(BASELINE_NPZ):
        raise SystemExit(f"{BASELINE_NPZ} missing — run `python metanet_vs_sumo.py` first")
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    if "departed" not in sumo:
        raise SystemExit("Baseline predates the departed-flow fix — "
                         "run `python metanet_vs_sumo.py --rerun-sumo`")

    T = params.T_sec
    n_steps_h = int(HORIZON_S / T)
    limits_kmh = np.array(geom.limits_ms) * MS_TO_KMH
    u_nocontrol = model.v_ctrl_ub                      # legal limits = no control

    # demand forecasts, one value per model step over the whole run
    n_run = int(n_sec / T)
    t_grid = np.arange(n_run) * T
    d_declared = np.where(t_grid < DEMAND_END_S, DEMAND_VPH, 0.0)
    d_realised = demand_realised(sumo["departed"], n_run, T)

    t0s = np.arange(60, n_sec - HORIZON_S, T0_STRIDE)
    lead_idx = [int(l / T) - 1 for l in LEADS_S]

    # err[model][metric] -> (n_t0, n_leads, n_seg)
    variants = {"declared": d_declared, "realised": d_realised}
    pred_v = {k: np.full((len(t0s), len(LEADS_S), model.n_seg), np.nan)
              for k in list(variants) + ["persistence"]}
    pred_rho = {k: np.full((len(t0s), len(LEADS_S), model.n_seg), np.nan)
                for k in list(variants) + ["persistence"]}
    true_v = np.full((len(t0s), len(LEADS_S), model.n_seg), np.nan)
    true_rho = np.full((len(t0s), len(LEADS_S), model.n_seg), np.nan)

    for i, t0 in enumerate(t0s):
        rho0, v0 = sumo_state(sumo, t0, geom, model, limits_kmh)
        w0 = float(sumo["pending"][t0])
        x0 = model.pack_x(rho0, v0, w0)
        k0 = int(t0 / T)

        for name, d_all in variants.items():
            d_win = d_all[k0:k0 + n_steps_h]
            if len(d_win) < n_steps_h:                 # pad past the end
                d_win = np.concatenate([d_win, np.zeros(n_steps_h - len(d_win))])
            R, V = rollout(model, x0, u_nocontrol, d_win)
            pred_rho[name][i] = R[lead_idx]
            pred_v[name][i] = V[lead_idx]

        # persistence: the state simply does not change
        pred_rho["persistence"][i] = np.tile(rho0, (len(LEADS_S), 1))
        pred_v["persistence"][i] = np.tile(v0, (len(LEADS_S), 1))

        for j, lead in enumerate(LEADS_S):
            r_t, v_t = sumo_state(sumo, t0 + lead, geom, model, limits_kmh)
            true_rho[i, j] = r_t
            true_v[i, j] = v_t

    # ---- error metrics --------------------------------------------------
    # Only score segments that actually carry traffic. On an empty edge SUMO
    # reports -1 m/s and we substitute the posted limit, while METANET relaxes
    # to its own scalar v_free (91.8 km/h, the mean of the 84.6-99 km/h edge
    # limits) — so an empty road produces several km/h of "error" that is pure
    # bookkeeping convention. The road is clear from t ~ 1773 s, which is ~22%
    # of the rollouts, and including them inflates every number.
    has_traffic = true_rho > 0.5
    congested = (true_v < CONGESTED_KMH) & has_traffic    # regime from the TRUTH
    free_flow = (true_v >= CONGESTED_KMH) & has_traffic

    def stats(pred, truth, mask):
        e = np.where(mask, pred - truth, np.nan)
        if np.all(np.isnan(e)):
            return float("nan"), float("nan")
        return (float(np.nanmean(np.abs(e))),
                float(np.sqrt(np.nanmean(e ** 2))))

    print(f"\nScored points: {int(has_traffic.sum())} of {has_traffic.size} "
          f"({100 * has_traffic.mean():.0f}% carry traffic; the rest are empty "
          f"road and are excluded)")
    print(f"  of those, congested (<{CONGESTED_KMH:.0f} km/h): "
          f"{100 * congested.sum() / has_traffic.sum():.0f}%")

    names = ["persistence", "declared", "realised"]
    print("=" * 78)
    print(f"METANET {HORIZON_S} s-ahead prediction error vs SUMO")
    print(f"{len(t0s)} rollouts, t0 = {t0s[0]}..{t0s[-1]} s, stride {T0_STRIDE} s")
    print("=" * 78)

    print("\n--- SPEED error (km/h) — the directly comparable quantity ---")
    print(f"{'lead':>6} | " + " | ".join(f"{n:>22}" for n in names))
    print(f"{'':>6} | " + " | ".join(f"{'MAE':>10} {'RMSE':>11}" for _ in names))
    speed_mae = {n: [] for n in names}
    for j, lead in enumerate(LEADS_S):
        row = []
        for n in names:
            mae, rmse = stats(pred_v[n][:, j], true_v[:, j], has_traffic[:, j])
            speed_mae[n].append(mae)
            row.append(f"{mae:10.2f} {rmse:11.2f}")
        print(f"{lead:5d}s | " + " | ".join(row))

    print("\n--- SPEED error, CONGESTED segments only (SUMO < "
          f"{CONGESTED_KMH:.0f} km/h) ---")
    speed_mae_c = {n: [] for n in names}
    for j, lead in enumerate(LEADS_S):
        row = []
        for n in names:
            mae, rmse = stats(pred_v[n][:, j], true_v[:, j], congested[:, j])
            speed_mae_c[n].append(mae)
            row.append(f"{mae:10.2f} {rmse:11.2f}")
        print(f"{lead:5d}s | " + " | ".join(row))

    print("\n--- SPEED error, FREE-FLOW segments only (SUMO >= "
          f"{CONGESTED_KMH:.0f} km/h) ---")
    speed_mae_f = {n: [] for n in names}
    for j, lead in enumerate(LEADS_S):
        row = []
        for n in names:
            mae, rmse = stats(pred_v[n][:, j], true_v[:, j], free_flow[:, j])
            speed_mae_f[n].append(mae)
            row.append(f"{mae:10.2f} {rmse:11.2f}")
        print(f"{lead:5d}s | " + " | ".join(row))

    print("\n--- DENSITY error (veh/km/lane) ---")
    print("  NOTE: METANET's congested branch carries SUMO's discharge at ~77")
    print("  veh/km/lane where SUMO sits at 150-185. That offset is a known,")
    print("  accepted calibration trade-off (Do Not Retry ledger), so density")
    print("  error in congestion is expected and is NOT a prediction failure.")
    dens_mae = {n: [] for n in names}
    for j, lead in enumerate(LEADS_S):
        row = []
        for n in names:
            mae, rmse = stats(pred_rho[n][:, j], true_rho[:, j], has_traffic[:, j])
            dens_mae[n].append(mae)
            row.append(f"{mae:10.2f} {rmse:11.2f}")
        print(f"{lead:5d}s | " + " | ".join(row))

    # ---- skill score vs persistence -------------------------------------
    print("\n--- SKILL vs persistence (speed MAE; >0 means METANET is better) ---")
    for n in ["declared", "realised"]:
        sk = [100 * (1 - a / b) for a, b in zip(speed_mae[n], speed_mae["persistence"])]
        print(f"  {n:>10}: " + " ".join(f"{lead}s:{s:+6.1f}%"
                                        for lead, s in zip(LEADS_S, sk)))

    # ---- vehicles on road (TTS-relevant aggregate) -----------------------
    seg_veh = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, dtype=float)
    print("\n--- Vehicles on road at t0+210s (drives TTS in the cost function) ---")
    j = len(LEADS_S) - 1
    tv = (true_rho[:, j] * seg_veh).sum(axis=1)
    loaded = tv > 1.0                      # ignore rollouts that land on empty road
    for n in names:
        pv = (pred_rho[n][:, j] * seg_veh).sum(axis=1)
        e = (pv - tv)[loaded]
        print(f"  {n:>12}: MAE {np.mean(np.abs(e)):6.1f} veh   "
              f"bias {np.mean(e):+6.1f} veh   "
              f"({100 * np.mean(np.abs(e)) / np.mean(tv[loaded]):.0f}% of the "
              f"SUMO mean, {np.mean(tv[loaded]):.1f} veh)")

    # ---- save + plot ------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    np.savez(OUT_NPZ, t0s=t0s, leads=np.array(LEADS_S),
             true_v=true_v, true_rho=true_rho,
             **{f"pred_v_{n}": pred_v[n] for n in names},
             **{f"pred_rho_{n}": pred_rho[n] for n in names})
    print(f"\nSaved raw predictions + truth to {OUT_NPZ}")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    style = {"persistence": ("k", "--"), "declared": ("C0", "-"), "realised": ("C1", "-")}
    for n in names:
        c, ls = style[n]
        axs[0].plot(LEADS_S, speed_mae[n], ls, color=c, marker="o", label=n)
        axs[1].plot(LEADS_S, speed_mae_c[n], ls, color=c, marker="o", label=n)
        axs[2].plot(LEADS_S, speed_mae_f[n], ls, color=c, marker="o", label=n)
        axs[3].plot(LEADS_S, dens_mae[n], ls, color=c, marker="o", label=n)
    axs[0].set_title("Speed MAE, all loaded segments")
    axs[1].set_title(f"Speed MAE, congested (<{CONGESTED_KMH:.0f} km/h)")
    axs[2].set_title(f"Speed MAE, free flow (>={CONGESTED_KMH:.0f} km/h)")
    axs[3].set_title("Density MAE (offset is a known FD trade-off)")
    for ax in axs[:3]:
        ax.set_ylabel("MAE (km/h)")
    axs[3].set_ylabel("MAE (veh/km/lane)")
    for ax in axs:
        ax.set_xlabel("Prediction lead time (s)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, "prediction_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
