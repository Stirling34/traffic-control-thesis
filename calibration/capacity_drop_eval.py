# -*- coding: utf-8 -*-
"""
Does the capacity drop actually help? The payoff tests.

`test_capacity_drop.py` shows the mechanism works in isolation (hysteresis
appears, regression is clean). This asks the two questions that decide whether
it is worth keeping:

  1. PREDICTION  Does the 210 s-ahead free-flow error (the dominant defect,
                 ~47 km/h) come down, without wrecking congested accuracy?
  2. AUTHORITY   Does a fixed VSL policy now beat no-control on TTS? Plain
                 METANET says no policy ever helps, which is what makes
                 TTS-shaped cost functions unusable. If the capacity drop fixes
                 that, Phase D opens up properly.

Also re-checks the open-loop match against SUMO so we can see what the drop
costs in fidelity elsewhere.

Usage:  python capacity_drop_eval.py
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH, demand_declared
from validation.metanet_prediction_error import CONGESTED_KMH, HORIZON_S, rollout, sumo_state
from validation.control_authority import policies, run_metanet, spread

# (alpha_cd, rho_cd_ref) combinations to compare. ref = None means rho_max
# (Han-faithful) which barely bites here because our congested branch only
# reaches ~80 veh/km/lane; ref = 80 makes alpha_cd mean what Han means.
VARIANTS = [
    ("plain (off)",      0.00, None),
    ("alpha=0.15 ref80", 0.15, 80.0),
    ("alpha=0.25 ref80", 0.25, 80.0),
    ("alpha=0.35 ref80", 0.35, 80.0),
    ("alpha=0.50 ref80", 0.50, 80.0),
]
T0_STRIDE = 20


def prediction_error(geom, sumo, params, t0s, limits_kmh):
    model = MetanetModel(geom, params)
    T = params.T_sec
    n_h = int(HORIZON_S / T)
    n_run = int(sumo["rho"].shape[0] / T)
    d_all = np.where(np.arange(n_run) * T < DEMAND_END_S, DEMAND_VPH, 0.0)
    u = model.v_ctrl_ub
    pv, tv, trho = [], [], []
    for t0 in t0s:
        rho0, v0 = sumo_state(sumo, t0, geom, model, limits_kmh)
        x0 = model.pack_x(rho0, v0, float(sumo["pending"][t0]))
        k0 = int(t0 / T)
        d_win = d_all[k0:k0 + n_h]
        if len(d_win) < n_h:
            d_win = np.concatenate([d_win, np.zeros(n_h - len(d_win))])
        _, V = rollout(model, x0, u, d_win)
        pv.append(V[-1])
        r_t, v_t = sumo_state(sumo, t0 + HORIZON_S, geom, model, limits_kmh)
        tv.append(v_t); trho.append(r_t)
    pv, tv, trho = np.array(pv), np.array(tv), np.array(trho)
    loaded = trho > 0.5
    cong = (tv < CONGESTED_KMH) & loaded
    free = (tv >= CONGESTED_KMH) & loaded

    def mae(mask):
        return float(np.nanmean(np.abs(np.where(mask, pv - tv, np.nan))))
    return mae(free), mae(cong), mae(loaded)


def main():
    geom = load_geometry()
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    limits_kmh = np.array(geom.limits_ms) * MS_TO_KMH
    t0s = np.arange(60, n_sec - HORIZON_S, T0_STRIDE)
    P = policies(np.array(geom.main_limits_kmh), 9)

    print(f"{len(t0s)} rollouts/variant at {HORIZON_S} s; "
          f"{len(P)} fixed VSL policies over {n_sec} s.\n")

    print("=" * 88)
    print("1. PREDICTION ERROR at 210 s (km/h MAE; free-flow is the defect)")
    print("=" * 88)
    print(f"{'variant':>18} {'FREE-FLOW':>11} {'congested':>11} {'all':>9}")
    pred = {}
    for label, a, ref in VARIANTS:
        p = MetanetParams(alpha_cd=a, rho_cd_ref=ref)
        f, c, allm = prediction_error(geom, sumo, p, t0s, limits_kmh)
        pred[label] = (f, c, allm)
        print(f"{label:>18} {f:11.2f} {c:11.2f} {allm:9.2f}")
    base_f, base_c, _ = pred["plain (off)"]
    print("\n  change vs plain:")
    for label, _, _ in VARIANTS[1:]:
        f, c, _ = pred[label]
        print(f"    {label:>18}  free-flow {100*(f/base_f-1):+6.1f}%   "
              f"congested {100*(c/base_c-1):+6.1f}%")

    print("\n" + "=" * 88)
    print("2. CONTROL AUTHORITY — does any VSL policy now beat no-control?")
    print("=" * 88)
    for label, a, ref in VARIANTS:
        params = MetanetParams(alpha_cd=a, rho_cd_ref=ref)
        model = MetanetModel(geom, params)
        n_steps = int(n_sec / params.T_sec)
        res = {name: run_metanet(vsl, model, params, n_steps)
               for name, vsl in P.items()}
        base = res["no control"]
        best = min(res.items(), key=lambda kv: kv[1]["tts_road"])
        best_sh = min(res.items(), key=lambda kv: kv[1]["sharpness"])
        print(f"\n  {label}")
        print(f"    TTS       : best '{best[0]}' {best[1]['tts_road']:7.1f} vs "
              f"no-control {base['tts_road']:7.1f}  "
              f"({100*(best[1]['tts_road']/base['tts_road']-1):+5.1f}%)"
              f"{'   <-- CONTROL HELPS' if best[0] != 'no control' else ''}")
        print(f"    sharpness : best '{best_sh[0]}' "
              f"({100*(best_sh[1]['sharpness']/base['sharpness']-1):+5.1f}%)")
        print(f"    spread across policies: TTS "
              f"{spread([r['tts_road'] for r in res.values()]):5.1f}%   "
              f"sharpness {spread([r['sharpness'] for r in res.values()]):5.1f}%")

    print("\n" + "=" * 88)
    print("3. OPEN-LOOP FIDELITY — what does the drop cost elsewhere?")
    print("  SUMO reference: discharge 2325 veh/h, clear 1773 s, 1112 vehicles")
    print("=" * 88)
    print(f"{'variant':>18} {'discharge':>10} {'clear s':>9} {'vehicles':>9}")
    for label, a, ref in VARIANTS:
        params = MetanetParams(alpha_cd=a, rho_cd_ref=ref)
        model = MetanetModel(geom, params)
        T = params.T_sec
        n = int(n_sec / T)
        r = model.simulate(demand_declared(n, T), check_demand=False)
        lo, hi = int(300 / T), int(1000 / T)
        seg = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
        on = (r["rho"] * seg).sum(axis=1) + r["w"][:, 0]
        t = np.arange(len(on)) * T
        hit = np.where((on < 1.0) & (t > DEMAND_END_S))[0]
        print(f"{label:>18} {np.mean(r['q'][lo:hi, -2]):10.0f} "
              f"{(t[hit[0]] if len(hit) else np.nan):9.0f} "
              f"{r['q'][:, -1].sum() * T / 3600.0:9.0f}")


def demand_sweep():
    """The decisive test: is the blocker the model, or the demand level?

    VSL can only ever help by PREVENTING breakdown. Our scenario runs 4000 veh/h
    into a bottleneck whose capacity is ~2389 — 68% oversaturated — so breakdown
    is unavoidable no matter what the signs say, and a capacity drop just makes
    the inevitable queue worse. If that is the real blocker, then at a demand
    close to capacity the capacity drop should suddenly make VSL worth using.
    """
    geom = load_geometry()
    P = policies(np.array(geom.main_limits_kmh), 9)
    print("\n" + "=" * 88)
    print("4. DEMAND SWEEP — at what demand does VSL start to pay off?")
    print("   (1-lane bottleneck capacity ~2389 veh/h)")
    print("=" * 88)
    print(f"{'demand':>8} {'alpha_cd':>9} {'best policy':>24} {'TTS gain':>9}")
    for D in (2200.0, 2400.0, 2600.0, 2800.0, 3200.0, 4000.0):
        for a, ref in ((0.00, None), (0.25, 80.0), (0.50, 80.0)):
            params = MetanetParams(alpha_cd=a, rho_cd_ref=ref)
            model = MetanetModel(geom, params)
            T = params.T_sec
            n = int(2500 / T)
            d = np.where(np.arange(n) * T < DEMAND_END_S, D, 0.0)
            seg = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
            out = {}
            for name, vsl in P.items():
                r = model.simulate(d, v_ctrl=vsl, check_demand=False)
                on = (r["rho"] * seg).sum(axis=1) + r["w"][:, 0]
                out[name] = float(on.sum() * T / 3600.0)
            base = out["no control"]
            best = min(out.items(), key=lambda kv: kv[1])
            gain = 100 * (best[1] / base - 1)
            flag = "   <-- CONTROL HELPS" if best[0] != "no control" else ""
            print(f"{D:8.0f} {a:9.2f} {best[0]:>24} {gain:+8.2f}%{flag}")
        print()


def timed_vsl_sweep():
    """Time-WINDOWED VSL — the test the fixed-policy sweep could not do.

    A fixed speed limit held for the whole run pays a delay penalty during the
    free-flowing period in order to prevent a breakdown that only threatens
    later, so it can essentially never win. That is a property of the *policy
    class*, not of the model, and it means "no fixed policy beats no control"
    does NOT establish that no controller can help. Real VSL — and MPC — acts
    only when breakdown is imminent and then releases.

    Here the limit is applied to the upstream segments only between t_on and
    t_off and the legal limits are restored afterwards.
    """
    geom = load_geometry()
    limits = np.array(geom.main_limits_kmh)
    print("\n" + "=" * 88)
    print("5. TIME-WINDOWED VSL — does control help when it acts only when needed?")
    print("=" * 88)

    D = 2600.0                      # just above the ~2389 veh/h bottleneck capacity
    for a, ref in ((0.00, None), (0.25, 80.0), (0.50, 80.0)):
        params = MetanetParams(alpha_cd=a, rho_cd_ref=ref)
        model = MetanetModel(geom, params)
        T = params.T_sec
        n = int(2500 / T)
        d = np.where(np.arange(n) * T < DEMAND_END_S, D, 0.0)
        seg = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)

        def tts(vsl_of_step):
            x = model.x0_empty()
            total = 0.0
            for k in range(n):
                xn, _ = model.F(x, vsl_of_step(k), float(d[k]))
                x = np.asarray(xn).ravel()
                rho, _, w = model.unpack_x(x)
                total += (rho * seg).sum() + w[0]
            return total * T / 3600.0

        base = tts(lambda k: limits)
        best = (None, base)
        for vlim in (40.0, 50.0, 60.0, 70.0):
            for t_on in (0.0, 100.0, 200.0, 300.0):
                for t_off in (400.0, 600.0, 800.0, 1000.0):
                    if t_off <= t_on:
                        continue
                    u_on = limits.copy()
                    u_on[:5] = np.minimum(u_on[:5], vlim)

                    def pol(k, u_on=u_on, t_on=t_on, t_off=t_off):
                        t = k * T
                        return u_on if (t_on <= t < t_off) else limits

                    val = tts(pol)
                    if val < best[1]:
                        best = (f"{vlim:.0f} km/h on segs 1-5, "
                                f"t={t_on:.0f}-{t_off:.0f}s", val)
        gain = 100 * (best[1] / base - 1)
        print(f"  alpha_cd={a:.2f}  no-control TTS {base:7.2f}  |  "
              f"best {best[1]:7.2f} ({gain:+5.2f}%)")
        print(f"      {best[0] if best[0] else 'nothing beat no-control'}"
              f"{'   <-- CONTROL HELPS' if best[0] else ''}")


if __name__ == "__main__":
    main()
    demand_sweep()
    timed_vsl_sweep()
