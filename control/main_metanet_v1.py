# -*- coding: utf-8 -*-
"""
Closed-loop SUMO + nonlinear MPC (METANET prediction model, CasADi/IPOPT).

This is the METANET/NMPC successor to mpc_v1.py (which used the linearised
LWR model + OSQP). Control architecture is unchanged:

  every CONTROL_INTERVAL_SEC:
      read per-edge densities & speeds from SUMO (TraCI)
      -> solve the NMPC over the METANET model
      -> apply the first VSL move to the SUMO edges (setMaxSpeed)

Run:  python main_metanet_v1.py            (headless)
      set USE_GUI = True for sumo-gui.

Outputs: results npz in data/, figures in docs/figures/.
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
import traci

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from control.nmpc_metanet import NmpcVslController

# -----------------------------
# Config
# -----------------------------
USE_GUI = False
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sumo", "simulation2.sumocfg")
# Must match metanet_vs_sumo.SIM_STEPS or the baseline comparison below is
# meaningless. 2500 s covers the whole episode: the declared demand stops at
# t=1000 s, SUMO's insertion backlog keeps injecting until t=1440 s, and the
# road is not clear until t~1773 s. The old 1000 s window ended at the instant
# the demand stopped, so it contained no drain at all.
SIM_STEPS = 2500                # SUMO steps of 1 s
CONTROL_INTERVAL_SEC = 30       # MPC cadence (matches the LWR MPC)
CONTROL_SCHEME = 1              # 0 = no control (baseline), 1 = NMPC
DEMAND_VPH = 4000.0             # inflow declared in routes/route.rou2.xml
DEMAND_END_S = 1000.0           # flows stop at t=1000 s
# KNOWN DEFECT, not yet fixed (see docs/research/2026-08-02-drain-mismatch.md):
# the d_hat forecast below assumes DEMAND_VPH until DEMAND_END_S then zero.
# SUMO's *realised* inflow is nothing like that — it collapses to ~200 veh/h
# around t=850 s when the entry jams, then returns to ~3600 veh/h over
# t=1200-1440 s as the backlog flushes. The controller is therefore given a
# forecast that is wrong in both magnitude and trend exactly when it matters.
ACTIVATE_RHO_MIN = 1.0          # controller sleeps until every segment has traffic

FIG_DIR = os.path.join(_ROOT, "docs", "figures")
OUT_NPZ = os.path.join(_ROOT, "data", "nmpc_closed_loop.npz")
BASELINE_NPZ = os.path.join(_ROOT, "data", "sumo_baseline.npz")


def main():
    geom = load_geometry()
    params = MetanetParams()
    model = MetanetModel(geom, params)
    ctrl = NmpcVslController(model)

    T = params.T_sec                        # METANET step (5 s)
    M = ctrl.M                              # model steps per control interval
    assert M * T == CONTROL_INTERVAL_SEC, (
        f"controller cadence M*T = {M * T}s != CONTROL_INTERVAL_SEC"
    )
    horizon_steps = ctrl.Np * M

    edge_ids = geom.edge_ids
    ctrl_edge_ids = edge_ids[: model.n_ctrl]
    lengths_km = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, dtype=float)
    limits_kmh = np.array(geom.limits_ms) * MS_TO_KMH

    os.makedirs("data", exist_ok=True)
    sumo_bin = "sumo-gui" if USE_GUI else "sumo"
    # fcd output redirected so repeated runs don't clobber fcd_output2.xml
    # --end overrides the 1000 s in the .sumocfg so the run covers the drain
    traci.start([sumo_bin, "-c", CONFIG_FILE,
                 "--end", str(SIM_STEPS), "--no-warnings"])

    # book-keeping (per SUMO second)
    rho_edges_t = []            # (t, 10) veh/km/lane
    vmean_edges_t = []          # (t, 10) km/h
    applied_kmh_t = []          # (t, 9) applied VSL
    pending_t, arrived_t = [], []
    arrived_total = 0
    n_solves = n_fail = 0
    controller_active = False

    applied_kmh = limits_kmh[: model.n_ctrl].copy()   # start at legal limits

    for step in range(SIM_STEPS):
        traci.simulationStep()
        t_now = step + 1

        counts = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edge_ids])
        vmean = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edge_ids])
        rho_edges = counts / (lengths_km * lanes)                  # veh/km/lane
        v_edges = np.where(counts > 0, vmean * MS_TO_KMH,
                           np.append(applied_kmh, limits_kmh[-1]))
        rho_edges_t.append(rho_edges)
        vmean_edges_t.append(v_edges)
        pending_t.append(len(traci.simulation.getPendingVehicles()))
        arrived_total += traci.simulation.getArrivedNumber()
        arrived_t.append(arrived_total)
        applied_kmh_t.append(applied_kmh.copy())

        if CONTROL_SCHEME != 1 or t_now % CONTROL_INTERVAL_SEC != 0:
            continue

        # ---- assemble the METANET state from SUMO measurements ----
        rho_seg = rho_edges[model.seg_edge_index]
        v_seg = v_edges[model.seg_edge_index]
        w0 = float(pending_t[-1])

        if not controller_active and rho_seg.min() > ACTIVATE_RHO_MIN:
            controller_active = True
            print(f"[NMPC] t={t_now}s: controller activated")
        if not controller_active:
            continue

        # demand forecast over the horizon (veh/h per model step)
        t_future = t_now + np.arange(horizon_steps) * T
        d_hat = np.where(t_future < DEMAND_END_S, DEMAND_VPH, 0.0)

        u, info = ctrl.solve(rho_seg, v_seg, w0, d_hat, applied_kmh)
        n_solves += 1
        if not info["success"]:
            n_fail += 1
            print(f"[NMPC] t={t_now}s: SOLVE FAILED ({info['status']}) — "
                  "holding previous limits")
        else:
            applied_kmh = np.asarray(u, dtype=float)
            for eid, v_new in zip(ctrl_edge_ids, applied_kmh):
                traci.edge.setMaxSpeed(eid, float(v_new) / MS_TO_KMH)
            print(f"[NMPC] t={t_now}s: {info['solve_time']:.2f}s  "
                  f"VSL={np.round(applied_kmh, 0)}")

    traci.close()

    rho_edges_t = np.array(rho_edges_t)
    vmean_edges_t = np.array(vmean_edges_t)
    applied_kmh_t = np.array(applied_kmh_t)
    pending_t = np.array(pending_t, dtype=float)
    arrived_t = np.array(arrived_t, dtype=float)

    np.savez(OUT_NPZ, rho=rho_edges_t, vmean=vmean_edges_t,
             applied_kmh=applied_kmh_t, pending=pending_t, arrived=arrived_t)
    print(f"\nSaved results to {OUT_NPZ}")
    print(f"Solves: {n_solves}, failures: {n_fail}")

    # ---- summary metrics (vs cached no-control baseline) ----
    veh_on_road = (rho_edges_t * lengths_km * lanes).sum(axis=1)
    tts = float((veh_on_road + pending_t).sum()) / 3600.0    # veh·h
    print(f"TTS (road + insertion queue): {tts:.1f} veh·h")
    print(f"Throughput (arrived): {arrived_t[-1]:.0f} veh")
    if os.path.exists(BASELINE_NPZ):
        base = dict(np.load(BASELINE_NPZ))
        # TTS is a *sum over time*, so comparing runs of different lengths is
        # not a small error — it scales with the length difference. Refuse
        # rather than print a number that looks fine and is wrong.
        n_base = base["rho"].shape[0]
        if n_base != SIM_STEPS:
            print(f"\n  !! Baseline in {BASELINE_NPZ} is {n_base} s but this run "
                  f"is {SIM_STEPS} s — skipping the comparison.\n"
                  f"     Re-run `python metanet_vs_sumo.py --rerun-sumo` with a "
                  f"matching SIM_STEPS first.")
        else:
            veh_base = (base["rho"] * lengths_km * lanes).sum(axis=1)
            tts_base = float((veh_base + base["pending"]).sum()) / 3600.0
            print(f"Baseline TTS: {tts_base:.1f} veh·h   "
                  f"Baseline throughput: {base['arrived'][-1]:.0f} veh")

    # ---- figures ----
    os.makedirs(FIG_DIR, exist_ok=True)
    x_edges = np.concatenate([[0.0], np.cumsum(geom.lengths_m)])
    t_ax = np.arange(SIM_STEPS + 1)

    if os.path.exists(BASELINE_NPZ) and base["rho"].shape[0] == SIM_STEPS:
        fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True,
                                constrained_layout=True)
        vmax = max(base["rho"].max(), rho_edges_t.max())
        m0 = axs[0].pcolormesh(x_edges, t_ax, base["rho"], cmap="coolwarm",
                               vmin=0, vmax=vmax)
        axs[0].set_title("SUMO no control")
        axs[1].pcolormesh(x_edges, t_ax, rho_edges_t, cmap="coolwarm",
                          vmin=0, vmax=vmax)
        axs[1].set_title("SUMO + METANET NMPC")
        for ax in axs:
            ax.set_xlabel("Position on road (m)")
        axs[0].set_ylabel("Time (s)")
        fig.colorbar(m0, ax=axs, label="Density (veh/km/lane)", shrink=0.85)
        fig.savefig(os.path.join(FIG_DIR, "nmpc_closed_loop.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for i in range(model.n_ctrl):
        ax.step(np.arange(SIM_STEPS), applied_kmh_t[:, i], where="post",
                label=f"S{i+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Applied VSL (km/h)")
    ax.legend(ncol=5, fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "nmpc_vsl_trajectory.png"), dpi=150)
    plt.close(fig)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
