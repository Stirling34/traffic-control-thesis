# -*- coding: utf-8 -*-
"""
Phase B validation: compare the METANET model (open loop, no control) against
the SUMO no-control baseline — the same ground truth the LWR model was
calibrated against ("Figure 2").

Runs SUMO headless once and caches per-edge densities/speeds to
data/sumo_baseline.npz, then simulates METANET and plots both spatio-temporal
maps, plus bottleneck outflow and origin queue traces. Figures in docs/figures/.

Two things were wrong here before 2026-08-02 and are fixed (see
docs/research/2026-08-02-drain-mismatch.md):

  1. The window ended at t = 1000 s — the exact instant the declared demand
     stops. SUMO still had 328 vehicles queued for insertion at that point,
     which then enter over t = 1000-1440 s and cause a second congestion
     episode; the road does not clear until t ~ 1773 s. The whole drain was
     off the end of the figure, so *neither* model could be seen to disperse.
  2. METANET was fed a flat DEMAND_VPH for the whole run. SUMO's realised
     inflow is nothing like flat — it collapses to ~200 veh/h once the entry
     edge jams, then floods again. The model was being asked to reproduce
     SUMO while receiving a different boundary condition.

So this now runs long enough to see the drain, and simulates METANET under
*two* demand profiles:

  "declared" — the demand actually declared in routes/route.rou2.xml
               (4000 veh/h until t = 1000 s, then nothing). This is the honest
               like-for-like validation: same demand in, do the two agree?
               METANET's own origin queue w absorbs whatever the road can't
               accept, exactly as SUMO's insertion queue does.
  "realised" — SUMO's measured second-by-second injections, replayed exactly.
               This removes the origin-boundary difference and isolates how
               well the road dynamics themselves match.

Usage:  python metanet_vs_sumo.py [--rerun-sumo]
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sumo", "simulation2.sumocfg")
BASELINE_NPZ = os.path.join(_ROOT, "data", "sumo_baseline.npz")
FIG_DIR = os.path.join(_ROOT, "docs", "figures")

# Long enough to contain the whole story. Measured: SUMO's last vehicle enters
# at t = 1440 s and the road is empty by t ~ 1773 s. 2500 s leaves clear margin.
SIM_STEPS = 2500            # SUMO steps of 1 s
# The SIMULATION always runs the full 2500 s — that is what makes the drain
# measurable. This only controls what the MAIN FIGURE shows. It is set to
# 1000 s because the post-1000 s METANET behaviour (the origin queue emptying
# onto the road after the demand has stopped) is visually confusing and not
# what the calibration is about. The full-range version is still written to
# metanet_vs_sumo_baseline_full.png so the drain evidence is not lost.
# WARNING: do not conclude anything about draining from the 1000 s figure —
# SUMO is still holding 328 vehicles at t=1000 s and does not clear until
# t~1773 s. That mistake is what docs/research/2026-08-02-drain-mismatch.md
# is about.
PLOT_WINDOW_S = 1000
# Demand actually declared in routes/route.rou2.xml: 4 flows x 1000 veh/h,
# begin=0 end=1000. NOTE the flows STOP at t = 1000 s — that shutoff is the
# only reason anything ever drains, and the old 1000 s window cut it off.
DEMAND_VPH = 4000.0
DEMAND_END_S = 1000.0
# Above MetanetModel.origin_flow_cap (~4352 veh/h here) the road state is
# bit-identical regardless of demand; simulate() warns if that happens.


def run_sumo_baseline(geom):
    """Run SUMO headless with no control; record per-edge state each second."""
    import traci

    edge_ids = geom.edge_ids
    lengths_km = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, dtype=float)

    # Redirect fcd output to a throwaway file so we don't clobber fcd_output2.xml.
    # --end overrides the 1000 s in the .sumocfg so the drain is inside the run.
    os.makedirs("data", exist_ok=True)
    fcd_scratch = os.path.join(_ROOT, "data", "fcd_scratch.xml")
    traci.start(["sumo", "-c", CONFIG_FILE, "--fcd-output", fcd_scratch,
                 "--end", str(SIM_STEPS), "--no-warnings"])

    rho, speed, pending, arrived, departed = [], [], [], [], []
    arrived_total = departed_total = 0
    for _ in range(SIM_STEPS):
        traci.simulationStep()
        counts = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edge_ids])
        vmean = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edge_ids])
        rho.append(counts / (lengths_km * lanes))          # veh/km/lane
        speed.append(vmean * MS_TO_KMH)                    # km/h (-1*3.6 if empty)
        pending.append(len(traci.simulation.getPendingVehicles()))
        arrived_total += traci.simulation.getArrivedNumber()
        arrived.append(arrived_total)
        # departures = vehicles that actually got onto the road. This is the
        # measured origin boundary condition and is NOT the declared demand.
        departed_total += traci.simulation.getDepartedNumber()
        departed.append(departed_total)
    traci.close()

    data = {
        "rho": np.array(rho), "speed": np.array(speed),
        "pending": np.array(pending, dtype=float),
        "arrived": np.array(arrived, dtype=float),
        "departed": np.array(departed, dtype=float),
        "sim_steps": np.array(SIM_STEPS),
    }
    os.makedirs("data", exist_ok=True)
    np.savez(BASELINE_NPZ, **data)
    print(f"Saved SUMO baseline to {BASELINE_NPZ}")
    return data


def load_or_run_sumo(geom, force=False):
    """Load the cached baseline, but only if it matches what we now need.

    The pre-2026-08-02 cache is a 1000 s crop with no `departed` array. Loading
    it silently would reproduce exactly the bug this script was fixed for, so
    stale caches are rejected rather than used.
    """
    if not force and os.path.exists(BASELINE_NPZ):
        cached = dict(np.load(BASELINE_NPZ))
        n = cached["rho"].shape[0]
        if "departed" not in cached:
            print(f"Cached baseline predates the departed-flow fix — re-running SUMO.")
        elif n != SIM_STEPS:
            print(f"Cached baseline is {n} s but SIM_STEPS is {SIM_STEPS} — "
                  "re-running SUMO.")
        else:
            print(f"Loading cached SUMO baseline from {BASELINE_NPZ}")
            return cached
    return run_sumo_baseline(geom)


# ---------------------------------------------------------------------------
# Demand profiles
# ---------------------------------------------------------------------------
def demand_declared(n_steps, T_sec):
    """The demand declared in the route file: DEMAND_VPH, then nothing."""
    t = np.arange(n_steps) * T_sec
    return np.where(t < DEMAND_END_S, DEMAND_VPH, 0.0)


def demand_realised(departed, n_steps, T_sec):
    """SUMO's measured injections, replayed exactly at METANET's step rate.

    Derived from the cumulative departure count, so vehicle numbers are
    conserved exactly — no smoothing, no rescaling. Sub-step spikes above the
    origin's flow ceiling are absorbed by METANET's own origin queue w and
    released later, which is the correct behaviour for an oversaturated origin.
    """
    cum = np.concatenate([[0.0], np.asarray(departed, dtype=float)])
    idx = np.clip((np.arange(n_steps + 1) * T_sec).astype(int), 0, len(cum) - 1)
    veh_per_step = np.diff(cum[idx])
    return veh_per_step / T_sec * 3600.0


def edge_boundaries_m(geom):
    """Cumulative section boundaries (m) for pcolormesh x-axis."""
    return np.concatenate([[0.0], np.cumsum(geom.lengths_m)])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_comparison(geom, sumo, runs, model, params,
                    fname="metanet_vs_sumo_baseline.png", window_s=None):
    """SUMO vs METANET under each demand profile, as density AND speed.

    Why density gets per-panel scaling: SUMO's queue reaches ~185 veh/km/lane
    while still discharging ~2390 veh/h. METANET's congested branch carries
    that same flow at ~77 veh/km/lane, and 185 would mean a dead stop (~4
    veh/h). The two genuinely cannot agree on absolute density, and forcing
    them to would corrupt the flow behaviour the controller plans on (see the
    Do Not Retry ledger). Per-panel scaling compares the *shape* — onset, front
    propagation, drain — which is what is being calibrated.

    Speed shares one km/h scale across all panels: it is directly comparable,
    and it is what a shockwave actually consists of.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    x_edges = edge_boundaries_m(geom)
    x_segs = np.concatenate([[0.0], np.cumsum(model.seg_lengths_km) * 1000.0])

    # crop to the display window (the simulation itself is always full length)
    n_s = sumo["rho"].shape[0] if window_s is None else min(
        int(window_s), sumo["rho"].shape[0])
    n_m = None if window_s is None else int(window_s / params.T_sec)

    t_sumo = np.arange(n_s + 1) * 1.0
    # SUMO reports -1 m/s on empty edges; mask so it doesn't skew the scale
    v_sumo = np.where(sumo["speed"][:n_s] < 0, np.nan, sumo["speed"][:n_s])

    cols = [("SUMO baseline (no control)", x_edges, t_sumo,
             sumo["rho"][:n_s], v_sumo)]
    for label, res in runs.items():
        rho_m = res["rho"][:n_m] if n_m else res["rho"]
        v_m = res["v"][:n_m] if n_m else res["v"]
        t_met = np.arange(rho_m.shape[0] + 1) * params.T_sec
        cols.append((f"METANET — {label} demand", x_segs, t_met, rho_m, v_m))

    n = len(cols)
    fig, axs = plt.subplots(2, n, figsize=(6.2 * n, 10), constrained_layout=True)

    v_top = max(np.nanmax(c[4]) for c in cols)
    for j, (title, xb, tb, rho, v) in enumerate(cols):
        vmax = float(np.nanmax(rho))
        m = axs[0, j].pcolormesh(xb, tb, rho, cmap="coolwarm", vmin=0, vmax=vmax,
                                 shading="flat")
        axs[0, j].set_title(f"{title}\ndensity, own scale "
                            f"(peak {vmax:.0f} veh/km/lane)")
        fig.colorbar(m, ax=axs[0, j], label="Density (veh/km/lane)")

        m = axs[1, j].pcolormesh(xb, tb, v, cmap="coolwarm_r", vmin=0, vmax=v_top,
                                 shading="flat")
        axs[1, j].set_title(f"{title}\nspeed (shared scale, comparable)")
        fig.colorbar(m, ax=axs[1, j], label="Speed (km/h)")

    t_max = float(cols[0][2][-1])
    for ax in axs.ravel():
        ax.set_xlabel("Position on road (m)")
        if DEMAND_END_S < t_max:
            ax.axhline(DEMAND_END_S, color="k", lw=0.8, ls="--", alpha=0.6)
    for ax in axs[:, 0]:
        ax.set_ylabel("Time (s)")
    if DEMAND_END_S < t_max:
        axs[0, 0].text(20, DEMAND_END_S + 25, "demand declared to stop",
                       fontsize=8, alpha=0.8)

    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_outflow_queue(sumo, runs, params, fname="metanet_vs_sumo_outflow.png"):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)

    # -- inflow: the boundary condition each model actually received ---------
    win = 60
    dep = sumo["departed"]
    in_sumo = (dep[win:] - dep[:-win]) / win * 3600.0
    axs[0].plot(np.arange(win, len(dep)), in_sumo, "k", label="SUMO realised inflow")
    for label, res in runs.items():
        t_met = np.arange(res["q"].shape[0]) * params.T_sec
        axs[0].plot(t_met, res["q"][:, -1], label=f"METANET origin flow ({label})")
    axs[0].axhline(DEMAND_VPH, color="grey", ls=":", label="declared demand")
    axs[0].set_title("Origin inflow — the boundary condition")
    axs[0].set_ylabel("Flow (veh/h)")

    # -- bottleneck discharge ------------------------------------------------
    arr = sumo["arrived"]
    out_sumo = (arr[win:] - arr[:-win]) / win * 3600.0
    axs[1].plot(np.arange(win, len(arr)), out_sumo, "k", label="SUMO (60 s window)")
    for label, res in runs.items():
        t_met = np.arange(res["q"].shape[0]) * params.T_sec
        axs[1].plot(t_met, res["q"][:, -2], label=f"METANET ({label})")
    axs[1].set_title("Bottleneck discharge rate")
    axs[1].set_ylabel("Outflow (veh/h)")

    # -- origin queue --------------------------------------------------------
    axs[2].plot(np.arange(len(sumo["pending"])), sumo["pending"], "k",
                label="SUMO pending vehicles")
    for label, res in runs.items():
        t_met = np.arange(res["w"].shape[0]) * params.T_sec
        axs[2].plot(t_met, res["w"][:, 0], label=f"METANET w ({label})")
    axs[2].set_title("Origin queue")
    axs[2].set_ylabel("Vehicles")

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.axvline(DEMAND_END_S, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
def clear_time(veh_on_road, dt, after_s=DEMAND_END_S, thresh=1.0):
    """First time (s) the road holds < `thresh` vehicles, after `after_s`."""
    t = np.arange(len(veh_on_road)) * dt
    hit = np.where((veh_on_road < thresh) & (t > after_s))[0]
    return t[hit[0]] if len(hit) else None


def summarise(geom, sumo, runs, params, model):
    lengths_km = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, dtype=float)
    arr, dep = sumo["arrived"], sumo["departed"]
    on_sumo = (sumo["rho"] * lengths_km * lanes).sum(axis=1)

    print("\n=== SUMO ground truth ===")
    print(f"  window simulated                 : {len(arr):5d} s")
    print(f"  demand declared to stop at       : {DEMAND_END_S:5.0f} s")
    rate = np.diff(np.concatenate([[0.0], dep]))
    last_in = np.where(rate > 0)[0]
    print(f"  last vehicle actually entered at : "
          f"{last_in[-1] if len(last_in) else -1:5.0f} s "
          f"({dep[-1] - dep[int(DEMAND_END_S) - 1]:.0f} veh entered after the "
          f"declared stop, from the insertion backlog)")
    print(f"  vehicles departed / arrived      : {dep[-1]:.0f} / {arr[-1]:.0f}")
    print(f"  peak vehicles on road            : {on_sumo.max():5.0f} at "
          f"t={on_sumo.argmax():.0f} s")
    tc = clear_time(on_sumo, 1.0)
    print(f"  road clear (<1 veh) at           : "
          f"{('%.0f s' % tc) if tc is not None else 'NOT within the window'}")
    # discharge measured only while the bottleneck is actually loaded
    lo, hi = int(0.3 * DEMAND_END_S), int(DEMAND_END_S)
    q_sumo = (arr[hi] - arr[lo]) / (hi - lo) * 3600.0
    print(f"  discharge over t={lo}-{hi}s         : {q_sumo:5.0f} veh/h")

    print(f"\n=== METANET ===")
    print(f"  origin flow ceiling              : {model.origin_flow_cap:5.0f} veh/h "
          f"({'SATURATED' if DEMAND_VPH > model.origin_flow_cap else 'ok, demand responsive'})")
    seg_km = np.array(model.seg_lengths_km)
    seg_lam = np.array(model.seg_lanes, dtype=float)
    for label, res in runs.items():
        on_met = (res["rho"] * seg_km * seg_lam).sum(axis=1)
        tc_m = clear_time(on_met, params.T_sec)
        k0, k1 = int(0.3 * DEMAND_END_S / params.T_sec), int(DEMAND_END_S / params.T_sec)
        print(f"\n  -- {label} demand --")
        print(f"    total vehicles injected      : "
              f"{res['q'][:, -1].sum() * params.T_sec / 3600.0:5.0f} veh")
        print(f"    peak vehicles on road        : {on_met.max():5.0f} at "
              f"t={on_met.argmax() * params.T_sec:.0f} s")
        print(f"    road clear (<1 veh) at       : "
              f"{('%.0f s' % tc_m) if tc_m is not None else 'NOT within the window'}")
        print(f"    discharge over t={lo}-{hi}s     : "
              f"{np.mean(res['q'][k0:k1, -2]):5.0f} veh/h")
        print(f"    peak origin queue w          : {res['w'][:, 0].max():5.0f} veh "
              f"(SUMO peak pending {sumo['pending'].max():.0f})")
    print(f"\n  METANET capacity (1-lane link)   : "
          f"{params.capacity_per_lane(params.v_free_bneck):5.0f} veh/h")
    print(f"  rho_crit to match SUMO discharge : "
          f"{q_sumo / (params.v_free_bneck * np.exp(-1.0 / params.a)):5.1f} veh/km/lane "
          f"(current {params.rho_crit})")


def main():
    force = "--rerun-sumo" in sys.argv
    geom = load_geometry()
    sumo = load_or_run_sumo(geom, force=force)

    params = MetanetParams()
    model = MetanetModel(geom, params)
    n_steps = int(SIM_STEPS / params.T_sec)

    runs = {
        "declared": model.simulate(demand_declared(n_steps, params.T_sec)),
        "realised": model.simulate(
            demand_realised(sumo["departed"], n_steps, params.T_sec),
            check_demand=False),   # spikes above the cap are absorbed by w
    }

    # Main figure: cropped to PLOT_WINDOW_S for readability.
    plot_comparison(geom, sumo, runs, model, params, window_s=PLOT_WINDOW_S)
    # Full-range companion: this is the one that actually shows the drain.
    plot_comparison(geom, sumo, runs, model, params,
                    fname="metanet_vs_sumo_baseline_full.png", window_s=None)
    plot_outflow_queue(sumo, runs, params)
    summarise(geom, sumo, runs, params, model)


if __name__ == "__main__":
    main()
