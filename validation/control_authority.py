# -*- coding: utf-8 -*-
"""
Does VSL have any authority — in the PLANT, and in the MODEL?

This is the test that decides whether Phase D is viable with plain METANET.
Cost-function comparison only means something if different control actions
produce different outcomes. Two separate questions:

  (a) In SUMO (the plant): does changing the speed limits change the outcome?
      If not, VSL simply does not help this scenario and no controller or cost
      function can rescue it — that would be a SCENARIO problem.

  (b) In METANET (the prediction model): does changing the speed limits change
      the *predicted* outcome? If not, the optimiser has nothing to optimise
      and every cost function returns roughly the same VSLs — that would be a
      MODEL problem.

The diagnosis follows from the pair:

  (a) large, (b) small  -> METANET cannot see a benefit that really exists.
                           Plain METANET blocks Phase D; need capacity drop or
                           a different model.
  (a) small, (b) small  -> VSL genuinely does not help here. Fix the scenario
                           (geometry / demand), not the model.
  (a) large, (b) large  -> fine, proceed to Phase D.

Open loop with FIXED speed limits throughout — deliberately not the MPC, so
the result cannot be blamed on the controller, the cost function or the solver.

Usage:  python control_authority.py
Output: data/control_authority.npz, printed table
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import (CONFIG_FILE, DEMAND_END_S, DEMAND_VPH, SIM_STEPS,
                             demand_declared)

OUT_NPZ = os.path.join(_ROOT, "data", "control_authority.npz")


def policies(limits_kmh, n_ctrl):
    """Fixed VSL policies to compare. None = the legal limits (no control)."""
    legal = limits_kmh[:n_ctrl].copy()
    P = {"no control": legal}
    for v in (80, 70, 60, 50, 40):
        P[f"uniform {v}"] = np.minimum(legal, float(v))
    # metering upstream only, bottleneck approach left alone
    up = legal.copy(); up[:5] = np.minimum(up[:5], 50.0)
    P["upstream 50 (segs 1-5)"] = up
    up2 = legal.copy(); up2[:3] = np.minimum(up2[:3], 40.0)
    P["upstream 40 (segs 1-3)"] = up2
    return P


def run_sumo(vsl_kmh, geom):
    """SUMO with fixed speed limits applied from t=0. Returns metrics."""
    import traci

    edge_ids = geom.edge_ids
    lengths_km = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, dtype=float)
    traci.start(["sumo", "-c", CONFIG_FILE,
                 "--end", str(SIM_STEPS), "--no-warnings",
                 ])   # no --fcd-output: nothing here reads trajectories
    for eid, v in zip(edge_ids[:len(vsl_kmh)], vsl_kmh):
        traci.edge.setMaxSpeed(eid, float(v) / MS_TO_KMH)

    on_road, pend, spd, occ, dens = [], [], [], [], []
    arrived = 0
    for _ in range(SIM_STEPS):
        traci.simulationStep()
        counts = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edge_ids])
        v = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edge_ids]) * MS_TO_KMH
        on_road.append(counts.sum())
        spd.append(np.where(counts > 0, v, np.nan))   # NaN where no traffic
        dens.append(counts / (lengths_km * lanes))   # veh/km/lane
        occ.append(counts > 0)
        pend.append(len(traci.simulation.getPendingVehicles()))
        arrived += traci.simulation.getArrivedNumber()
    traci.close()

    on_road = np.array(on_road, float); pend = np.array(pend, float)
    out = {
        "tts_road": on_road.sum() / 3600.0,
        "tts_total": (on_road + pend).sum() / 3600.0,
        "throughput": float(arrived),
        "clear_s": float(np.argmax((on_road + pend) < 1)) if np.any((on_road + pend) < 1) else np.nan,
    }
    out.update(shockwave_metrics(np.array(spd), np.array(dens), 1.0,
                                 lengths_km, lanes))
    return out


CFL_AGG_S = 8   # see shockwave_metrics: 200 m cells / ~25 m/s free flow


def shockwave_metrics(v, rho, dt_s, lengths_km, lanes, agg_s=CFL_AGG_S):
    """See the note on `agg_s` below before changing how this is called."""
    """Vehicle-weighted acceleration measures, after Zegeye et al. (2013).

    Zegeye, S.K., De Schutter, B., Hellendoorn, J., Breunesse, E.A. & Hegyi, A.
    (2013), "A predictive traffic controller for sustainable mobility using
    parameterized control policies", Transportation Research Part C 31:158-171,
    equations (16)-(19). This is the VT-macro derivation of per-vehicle
    acceleration statistics from macroscopic (density, speed) fields.

    Two populations, because a cell's vehicles do two different things in a step:

      TEMPORAL (eq. 16-17) - vehicles that STAY in segment i:
          a_temp_i(k) = ( v_i(k+1) - v_i(k) ) / T
          n_temp_i(k) = L_i * lam_i * rho_i(k) - T * q_i(k)      [vehicles]

      SPATIAL-TEMPORAL (eq. 18-19) - vehicles CROSSING from i to i+1:
          a_spat_i(k) = ( v_{i+1}(k+1) - v_i(k) ) / T
          n_spat_i(k) = T * q_i(k)                                [vehicles]

    Why this replaces the earlier hand-rolled versions: the old "sharpness"
    compared neighbouring cells at the SAME instant, which is not any real
    vehicle's experience, and it averaged over cells rather than over vehicles,
    so empty road counted as much as a full lane. The form above follows the
    vehicle and weights by how many actually experience the acceleration. It is
    also published, so it can be cited rather than defended.

    Returns the vehicle-weighted RMS acceleration in m/s^2 (headline, and the
    term that is smooth enough to sit inside the MPC cost when squared), its two
    components, and the legacy measures for continuity.
    """
    # -- TIME STEP: this is not a free choice --------------------------------
    # Zegeye's eq. (18) is a_spat_i = ( v_{i+1}(k+1) - v_i(k) ) / T, i.e. the
    # speed change of a vehicle that CROSSES from cell i to cell i+1 in one
    # step. That is only an acceleration if a vehicle really does cross one cell
    # per step -- the CFL condition T * v_free ~ L that METANET is discretised
    # under.
    #
    # Fed raw 1 Hz SUMO output on 200 m cells it is violated by 8x: at 90 km/h a
    # vehicle covers 25 m in a second, not 200. The spatial term is then a
    # spatial GRADIENT divided by 1 s, and it is inflated by the same 8x.
    # Measured on one run: peak "acceleration" 25.0 m/s^2, which is 2.5 g and
    # not something any vehicle does, with 82% of the whole metric coming from
    # values above the vType's own 4.5 m/s^2 braking limit.
    #
    # So the field is aggregated to `agg_s` first, chosen so T * v_free ~ L.
    # At 8 s the peak becomes 3.1 m/s^2 and nothing implausible remains.
    # Measured effect of getting this wrong: absolute RMS 0.943 against a true
    # 0.331 m/s^2. The RELATIVE comparison happened to survive (-15.5% at 1 s
    # against -17.3% at 8 s), but the absolute figure is meaningless and a peak
    # of 25 m/s^2 is indefensible in a write-up.
    #
    # agg_s=None or 1 restores the raw behaviour, for reproducing old numbers.
    v = np.asarray(v, float)
    rho = np.asarray(rho, float)
    if agg_s and agg_s > 1 and len(v) >= agg_s:
        k = (len(v) // int(agg_s)) * int(agg_s)
        r = lambda x: np.nanmean(x[:k].reshape(-1, int(agg_s), x.shape[1]), axis=1)
        with np.errstate(invalid="ignore"):
            v, rho = r(v), r(rho)
        dt_s = float(dt_s) * int(agg_s)

    T_h = dt_s / 3600.0
    lam = np.asarray(lanes, float)
    Lkm = np.asarray(lengths_km, float)
    q = rho * v * lam                                   # veh/h per cell

    with np.errstate(invalid="ignore"):
        # temporal: same cell, one step later
        a_temp = (v[1:] - v[:-1]) / dt_s / 3.6                       # m/s^2
        n_temp = np.maximum(Lkm * lam * rho[:-1] - T_h * q[:-1], 0.0)
        # spatial-temporal: cell i at k -> cell i+1 at k+1
        a_spat = (v[1:, 1:] - v[:-1, :-1]) / dt_s / 3.6              # m/s^2
        n_spat = T_h * q[:-1, :-1]

        def wrms(a, n):
            m = np.isfinite(a) & np.isfinite(n) & (n > 0)
            if not m.any():
                return float("nan"), 0.0
            return float(np.sqrt((n[m] * a[m] ** 2).sum() / n[m].sum())), float(n[m].sum())

        rms_t, w_t = wrms(a_temp, n_temp)
        rms_s, w_s = wrms(a_spat, n_spat)
        tot = w_t + w_s
        rms_all = (float(np.sqrt((w_t * rms_t ** 2 + w_s * rms_s ** 2) / tot))
                   if tot > 0 else float("nan"))

        # legacy, unweighted -- kept so earlier numbers stay comparable
        dx = np.abs(np.diff(v, axis=1))
        dt = np.abs(np.diff(v, axis=0)) / dt_s * 60.0
        return {
            "accel_rms": rms_all,          # headline, m/s^2, vehicle-weighted
            "accel_rms_temporal": rms_t,
            "accel_rms_spatial": rms_s,
            "veh_weight": tot,
            "sharpness": float(np.nanmean(dx)),        # legacy
            "max_jump": float(np.nanmean(np.nanmax(dx, axis=1))),   # legacy
            "braking": float(np.nanmean(dt)),          # legacy
            "v_std": float(np.nanstd(v)),
            "cvs": float(np.nanstd(v) / max(np.nanmean(v), 1e-6)),  # Abdel-Aty
            "cong_frac": float(np.nanmean((v < 50.0).astype(float)[~np.isnan(v)])),
        }


def run_metanet(vsl_kmh, model, params, n_steps):
    res = model.simulate(demand_declared(n_steps, params.T_sec),
                         v_ctrl=vsl_kmh, check_demand=False)
    seg_veh = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
    on_road = (res["rho"] * seg_veh).sum(axis=1)
    w = res["w"][:, 0]
    dt_h = params.T_sec / 3600.0
    total = on_road + w
    clear = np.where(total < 1.0)[0]
    out = {
        "tts_road": float(on_road.sum() * dt_h),
        "tts_total": float(total.sum() * dt_h),
        "throughput": float(res["q"][:, -2].sum() * dt_h),
        "clear_s": float(clear[0] * params.T_sec) if len(clear) else np.nan,
    }
    # mask empty segments so the comparison with SUMO is like-for-like
    v = np.where(res["rho"] > 0.5, res["v"], np.nan)
    out.update(shockwave_metrics(v, res["rho"], params.T_sec,
                                 np.array(model.seg_lengths_km),
                                 np.array(model.seg_lanes, float)))
    return out


def spread(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return 100.0 * (v.max() - v.min()) / v.min() if len(v) and v.min() > 0 else np.nan


def main():
    geom = load_geometry()
    params = MetanetParams()
    model = MetanetModel(geom, params)
    limits_kmh = np.array(geom.main_limits_kmh)
    P = policies(limits_kmh, model.n_ctrl)
    n_steps = int(SIM_STEPS / params.T_sec)

    print(f"Fixed-VSL policies, open loop, {SIM_STEPS} s, demand "
          f"{DEMAND_VPH:.0f} veh/h until t={DEMAND_END_S:.0f} s.\n")

    rows = {}
    print(f"{'policy':>24} | {'SUMO TTS':>9} {'SUMO thru':>10} {'SUMO clr':>9} "
          f"| {'MET TTS':>9} {'MET thru':>10} {'MET clr':>9}")
    print("-" * 92)
    for name, vsl in P.items():
        s = run_sumo(vsl, geom)
        m = run_metanet(vsl, model, params, n_steps)
        rows[name] = {"sumo": s, "metanet": m, "vsl": vsl}
        print(f"{name:>24} | {s['tts_road']:9.1f} {s['throughput']:10.0f} "
              f"{s['clear_s']:9.0f} | {m['tts_road']:9.1f} {m['throughput']:10.0f} "
              f"{m['clear_s']:9.0f}")

    # ---- shockwave metrics: the ones this project actually cares about ----
    print(f"\n{'policy':>24} | {'SUMO sharp':>10} {'SUMO brake':>10} {'SUMO cong':>9} "
          f"| {'MET sharp':>10} {'MET brake':>10} {'MET cong':>9}")
    print("-" * 92)
    for name, r in rows.items():
        s, m = r["sumo"], r["metanet"]
        print(f"{name:>24} | {s['sharpness']:10.2f} {s['braking']:10.2f} "
              f"{s['cong_frac']:9.3f} | {m['sharpness']:10.2f} {m['braking']:10.2f} "
              f"{m['cong_frac']:9.3f}")

    print("\n=== CONTROL AUTHORITY (spread across policies, % of best) ===")
    for key, label in [("tts_road", "TTS on road"), ("throughput", "throughput"),
                       ("accel_rms", "accel RMS (Zegeye)"), ("cvs", "CVS (safety)"),
                       ("sharpness", "sharpness [legacy]"), ("max_jump", "worst front [legacy]"),
                       ("braking", "braking rate [legacy]"), ("v_std", "speed spread"),
                       ("cong_frac", "time congested")]:
        s_sp = spread([r["sumo"][key] for r in rows.values()])
        m_sp = spread([r["metanet"][key] for r in rows.values()])
        print(f"  {label:>16}:  SUMO {s_sp:7.1f}%   METANET {m_sp:7.1f}%")

    print("\n=== Can VSL REDUCE the shockwave, vs no control? "
          "(negative = improvement) ===")
    for key, label in [("accel_rms", "accel RMS (Zegeye)"), ("cvs", "CVS (safety)"),
                       ("sharpness", "sharpness [legacy]"),
                       ("braking", "braking rate [legacy]"), ("cong_frac", "time congested")]:
        b_s = rows["no control"]["sumo"][key]
        b_m = rows["no control"]["metanet"][key]
        best_s = min(rows.items(), key=lambda kv: kv[1]["sumo"][key])
        best_m = min(rows.items(), key=lambda kv: kv[1]["metanet"][key])
        print(f"  {label:>16}: SUMO best '{best_s[0]}' "
              f"{100*(best_s[1]['sumo'][key]/b_s-1):+6.1f}%   |   "
              f"METANET best '{best_m[0]}' "
              f"{100*(best_m[1]['metanet'][key]/b_m-1):+6.1f}%")

    base_s = rows["no control"]["sumo"]
    base_m = rows["no control"]["metanet"]
    print(f"\n=== Best achievable vs no control ===")
    best_s = min(rows.items(), key=lambda kv: kv[1]["sumo"]["tts_road"])
    best_m = min(rows.items(), key=lambda kv: kv[1]["metanet"]["tts_road"])
    print(f"  SUMO   : best is '{best_s[0]}' at TTS {best_s[1]['sumo']['tts_road']:.1f} "
          f"vs {base_s['tts_road']:.1f} no-control "
          f"({100*(best_s[1]['sumo']['tts_road']/base_s['tts_road']-1):+.1f}%)")
    print(f"  METANET: best is '{best_m[0]}' at TTS {best_m[1]['metanet']['tts_road']:.1f} "
          f"vs {base_m['tts_road']:.1f} no-control "
          f"({100*(best_m[1]['metanet']['tts_road']/base_m['tts_road']-1):+.1f}%)")

    print("\n  Does METANET rank policies the same way SUMO does?")
    order_s = sorted(rows, key=lambda k: rows[k]["sumo"]["tts_road"])
    order_m = sorted(rows, key=lambda k: rows[k]["metanet"]["tts_road"])
    print(f"    SUMO best->worst : {order_s}")
    print(f"    METANET best->worst: {order_m}")
    ranks_s = {k: i for i, k in enumerate(order_s)}
    ranks_m = {k: i for i, k in enumerate(order_m)}
    d = np.array([ranks_s[k] - ranks_m[k] for k in rows])
    n = len(rows)
    rho_s = 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1))
    print(f"    Spearman rank correlation: {rho_s:+.2f}  "
          f"(+1 = model ranks policies exactly as the plant does)")

    os.makedirs("data", exist_ok=True)
    np.savez(OUT_NPZ, **{f"{k}__{who}__{kk}": v
                         for k, r in rows.items() for who in ("sumo", "metanet")
                         for kk, v in r[who].items()})
    print(f"\nSaved {OUT_NPZ}")


if __name__ == "__main__":
    main()
