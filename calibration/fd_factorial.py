# -*- coding: utf-8 -*-
"""
Which change actually helps: the fundamental diagram, the capacity drop, or both?

A clean factorial. Two fundamental-diagram families x three capacity-drop
settings, every cell scored on the same four measures:

  A. 210 s prediction error   -- free-flow MAE is the dominant defect (46.8 km/h)
  B. storage / spatial extent -- the diagnostic that identified the root cause:
                                 how much road is congested for the SAME number of
                                 vehicles stored? SUMO packs 300 veh into 0.80 km;
                                 plain METANET needs 2.29 km.
  C. open-loop fidelity       -- discharge and clear time vs SUMO
  D. control authority        -- does ANY fixed VSL policy beat no-control?

Usage:  python fd_factorial.py
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH, demand_declared
from validation.metanet_prediction_error import CONGESTED_KMH, HORIZON_S, rollout, sumo_state
from validation.control_authority import policies, run_metanet

CELLS = [(fd, a) for fd in ("exponential", "triangular")
         for a in (0.00, 0.25, 0.50)]
T0_STRIDE = 25


def params_for(fd, a):
    return MetanetParams(fd_type=fd, alpha_cd=a,
                         rho_cd_ref=(80.0 if a > 0 else None))


def pred_err(geom, sumo, params, t0s, limits):
    model = MetanetModel(geom, params)
    T = params.T_sec
    n_h = int(HORIZON_S / T)
    n_run = int(sumo["rho"].shape[0] / T)
    d_all = np.where(np.arange(n_run) * T < DEMAND_END_S, DEMAND_VPH, 0.0)
    u = model.v_ctrl_ub
    pv, tv, trho = [], [], []
    for t0 in t0s:
        rho0, v0 = sumo_state(sumo, t0, geom, model, limits)
        x0 = model.pack_x(rho0, v0, float(sumo["pending"][t0]))
        k0 = int(t0 / T)
        w = d_all[k0:k0 + n_h]
        if len(w) < n_h:
            w = np.concatenate([w, np.zeros(n_h - len(w))])
        _, V = rollout(model, x0, u, w)
        pv.append(V[-1])
        r_t, v_t = sumo_state(sumo, t0 + HORIZON_S, geom, model, limits)
        tv.append(v_t); trho.append(r_t)
    pv, tv, trho = np.array(pv), np.array(tv), np.array(trho)
    loaded = trho > 0.5
    def mae(msk):
        return float(np.nanmean(np.abs(np.where(msk, pv - tv, np.nan))))
    return (mae((tv >= CONGESTED_KMH) & loaded),
            mae((tv < CONGESTED_KMH) & loaded), mae(loaded))


def extent_at(model, res, target):
    """km of road congested (v<50) when `target` vehicles are on the road."""
    seg = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
    veh = (res["rho"] * seg).sum(axis=1)
    k = int(np.argmin(np.abs(veh[:len(veh) // 2] - target)))
    return float((np.array(model.seg_lengths_km) * (res["v"][k] < 50.0)).sum())


def main():
    geom = load_geometry()
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    t0s = np.arange(60, n_sec - HORIZON_S, T0_STRIDE)
    P = policies(np.array(geom.main_limits_kmh), 9)

    # SUMO reference for the storage test
    L = np.array(geom.lengths_m) / 1000.0
    lam = np.array(geom.lanes, float)
    v_s = np.where(sumo["speed"] < 0, np.nan, sumo["speed"])
    veh_s = (sumo["rho"] * L * lam).sum(axis=1)
    ref_ext = {}
    for tgt in (200, 300):
        k = int(np.argmin(np.abs(veh_s[:1200] - tgt)))
        ref_ext[tgt] = float((L * (v_s[k] < 50.0)).sum())

    print(f"{len(t0s)} rollouts/cell. SUMO reference: discharge 2325 veh/h, "
          f"clear 1773 s,\n  congested extent {ref_ext[200]:.2f} km @200 veh, "
          f"{ref_ext[300]:.2f} km @300 veh, front -13.0 km/h\n")

    hdr = (f"{'FD':>12} {'a_cd':>5} | {'FREE':>6} {'cong':>6} {'all':>6} | "
           f"{'ext200':>7} {'ext300':>7} | {'disch':>6} {'clear':>6} | "
           f"{'VSL beats none?':>16}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for fd, a in CELLS:
        p = params_for(fd, a)
        model = MetanetModel(geom, p)
        T = p.T_sec
        n = int(n_sec / T)

        f, c, allm = pred_err(geom, sumo, p, t0s, limits)
        res = model.simulate(demand_declared(n, T), check_demand=False)
        e200, e300 = extent_at(model, res, 200), extent_at(model, res, 300)
        lo, hi = int(300 / T), int(1000 / T)
        disch = float(np.mean(res["q"][lo:hi, -2]))
        seg = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
        on = (res["rho"] * seg).sum(axis=1) + res["w"][:, 0]
        t = np.arange(len(on)) * T
        hit = np.where((on < 1.0) & (t > DEMAND_END_S))[0]
        clear = t[hit[0]] if len(hit) else np.nan

        pol = {nm: run_metanet(v, model, p, n)["tts_road"] for nm, v in P.items()}
        best = min(pol.items(), key=lambda kv: kv[1])
        gain = 100 * (best[1] / pol["no control"] - 1)
        verdict = (f"{best[0][:11]} {gain:+.1f}%" if best[0] != "no control"
                   else "no")

        print(f"{fd:>12} {a:5.2f} | {f:6.1f} {c:6.1f} {allm:6.1f} | "
              f"{e200:7.2f} {e300:7.2f} | {disch:6.0f} {clear:6.0f} | "
              f"{verdict:>16}")
        rows.append((fd, a, f, c, e300, disch, clear, gain))

    base = rows[0]
    print(f"\nvs the current model (exponential, a_cd=0):")
    for fd, a, f, c, e300, disch, clear, gain in rows[1:]:
        print(f"  {fd:>12} a_cd={a:.2f}:  free-flow {100*(f/base[2]-1):+6.1f}%   "
              f"congested {100*(c/base[3]-1):+6.1f}%   "
              f"extent@300 {e300:.2f} km (SUMO {ref_ext[300]:.2f}, "
              f"was {base[4]:.2f})")


if __name__ == "__main__":
    main()
