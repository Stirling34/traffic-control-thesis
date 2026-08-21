# -*- coding: utf-8 -*-
"""
Acceptance tests for the Han-style capacity drop grafted onto METANET.

Run:  python test_capacity_drop.py

Tests, in the order they must pass:

  1. REGRESSION   alpha_cd = 0 must reproduce the plain model's documented
                  numbers exactly. If this fails, the refactor broke something
                  and nothing below means anything.
  2. JAM WAVE     Inject a jam into an UNDERSATURATED road and watch it. Plain
                  METANET is expected to disperse it; with the capacity drop it
                  should propagate upstream as a coherent band leaving free flow
                  behind. Run on alpha_cd = 0 too, because if plain METANET
                  already propagates jam waves then our recorded limitation is
                  narrower than written and docs must be corrected.
  3. HYSTERESIS   The 2026-07-27 test that plain METANET failed. Two runs at the
                  same final demand: (A) never congested, (B) forced into
                  breakdown then relieved. A real capacity drop means B settles
                  strictly lower than A and never recovers.
  4. CALIBRATION  What drop is actually realised for a given alpha_cd? Our
                  congested branch only reaches ~77-80 veh/km/lane, so
                  normalising by rho_max = 182 wastes most of alpha_cd.
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from model.metanet_model import MetanetModel, MetanetParams, load_geometry

np.set_printoptions(linewidth=200, suppress=True)

# Reference numbers from docs/model-accuracy.md, produced by the plain model
# on the corrected 2500 s window.
REF_DISCHARGE = 2418.0     # veh/h, t = 300-1000 s
REF_CLEAR_S = 1755.0       # s, road below 1 vehicle
REF_VEHICLES = 1111.0      # total injected

DEMAND_VPH, DEMAND_END_S, SIM_S = 4000.0, 1000.0, 2500.0


def build(alpha_cd=0.0, rho_cd_ref=None, **kw):
    return MetanetModel(load_geometry(),
                        MetanetParams(alpha_cd=alpha_cd, rho_cd_ref=rho_cd_ref, **kw))


def declared(n, T):
    return np.where(np.arange(n) * T < DEMAND_END_S, DEMAND_VPH, 0.0)


def roll(model, d_seq, x0=None, u=None, inject=None):
    """Step the model manually so a jam can be injected mid-run.

    inject : (step_index, segment_index, rho_value, v_value) or None
    """
    p = model.p
    x = model.x0_empty() if x0 is None else np.asarray(x0, float)
    u = model.v_ctrl_ub if u is None else np.asarray(u, float)
    RHO, V, Q = [], [], []
    for k, d in enumerate(d_seq):
        if inject is not None and k == inject[0]:
            rho, v, w = model.unpack_x(x)
            rho = rho.copy(); v = v.copy()
            rho[inject[1]] = inject[2]
            v[inject[1]] = inject[3]
            x = model.pack_x(rho, v, w)
        xn, q = model.F(x, u, float(d))
        x = np.asarray(xn).ravel()
        rho, v, _ = model.unpack_x(x)
        RHO.append(rho); V.append(v); Q.append(np.asarray(q).ravel())
    return np.array(RHO), np.array(V), np.array(Q)


def veh_on_road(model, rho):
    seg = np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
    return (rho * seg).sum(axis=1)


# ---------------------------------------------------------------------------
def test_regression():
    print("=" * 76)
    print("TEST 1 - REGRESSION: alpha_cd = 0 must reproduce the plain model")
    print("=" * 76)
    m = build(0.0)
    T = m.p.T_sec
    n = int(SIM_S / T)
    rho, v, q = roll(m, declared(n, T))

    lo, hi = int(300 / T), int(1000 / T)
    dis = float(np.mean(q[lo:hi, -2]))
    on = veh_on_road(m, rho)
    t = np.arange(len(on)) * T
    hit = np.where((on < 1.0) & (t > DEMAND_END_S))[0]
    clear = t[hit[0]] if len(hit) else np.nan
    veh = float(q[:, -1].sum() * T / 3600.0)

    ok = True
    for name, got, ref, tol in [("discharge veh/h", dis, REF_DISCHARGE, 1.0),
                                ("clear time s", clear, REF_CLEAR_S, 5.0),
                                ("vehicles", veh, REF_VEHICLES, 1.0)]:
        good = abs(got - ref) <= tol
        ok &= good
        print(f"  {name:>18}: {got:8.1f}  (reference {ref:7.1f})  "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"\n  => {'PASS' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------------------
JAM_DEMAND = 1800.0     # MUST be below the 1-lane bottleneck capacity (~2389 veh/h),
                        # otherwise a standing queue forms regardless and the test
                        # measures nothing. 3000 veh/h was the original mistake here.


def jam_wave(model, label):
    """Genuinely undersaturated road, then a jam is injected. Does it travel?

    A *detached* jam wave means: a narrow low-speed band that marches upstream
    (towards lower segment indices) while the road DOWNSTREAM of it recovers to
    free flow. If instead the whole road stays slow, that is a standing queue,
    not a wave.
    """
    T = model.p.T_sec
    n = int(2400 / T)
    d = np.full(n, JAM_DEMAND)
    k_inj = int(600 / T)
    seg = model.n_ctrl - 2                  # inside the 2-lane stretch
    rho, v, q = roll(model, d, inject=(k_inj, seg, 150.0, 5.0))

    main = slice(0, model.n_ctrl)
    post = v[k_inj:, main]
    pos = np.argmin(post, axis=1)
    vmin = post.min(axis=1)
    slow = post < 40.0                       # what counts as "in the jam"

    print(f"  -- {label} --")
    print(f"     {'t':>6} {'jam seg':>8} {'v_jam':>7} {'v upstream':>11} "
          f"{'v downstream':>13} {'width':>6}")
    for tt in (0, 60, 120, 240, 360, 600, 900, 1200):
        k = int(tt / T)
        if k >= len(post):
            break
        up = post[k, max(0, pos[k] - 2)]                       # traffic arriving
        dn = post[k, min(model.n_ctrl - 1, pos[k] + 2)]        # traffic that left
        print(f"     {tt:6d} {pos[k]:8d} {vmin[k]:7.1f} {up:11.1f} {dn:13.1f} "
              f"{int(slow[k].sum()):6d}")

    k1, k2 = int(120 / T), int(900 / T)
    k2 = min(k2, len(post) - 1)
    alive = bool(vmin[k2] < 40.0)
    moved = int(pos[k1] - pos[k2])                     # +ve = travelled upstream
    # detached == recovered downstream of the jam while the jam is still alive
    dn_free = float(np.mean([post[k, min(model.n_ctrl - 1, pos[k] + 2)]
                             for k in range(k1, k2)]))
    width = float(np.mean(slow[k1:k2].sum(axis=1)))
    detached = alive and dn_free > 60.0 and width < model.n_ctrl - 1
    print(f"     alive at 900 s: {alive} | travelled upstream: {moved} seg | "
          f"mean downstream speed: {dn_free:.1f} km/h | mean width: {width:.1f} seg")
    print(f"     => {'DETACHED JAM WAVE' if detached else 'standing queue / dispersed'}")
    return detached, moved


def test_jam_wave():
    print("=" * 76)
    print("TEST 2 - JAM WAVE on an undersaturated road")
    print("=" * 76)
    s0, _ = jam_wave(build(0.0), "alpha_cd = 0 (plain METANET)")
    print()
    s1, _ = jam_wave(build(0.25, rho_cd_ref=80.0), "alpha_cd = 0.25, ref = 80")
    print()
    s2, _ = jam_wave(build(0.50, rho_cd_ref=80.0), "alpha_cd = 0.50, ref = 80")
    print(f"\n  => plain METANET: {'detached wave' if s0 else 'NO detached wave'};  "
          f"alpha_cd=0.25: {'detached wave' if s1 else 'no'};  "
          f"alpha_cd=0.50: {'detached wave' if s2 else 'no'}")
    if s0:
        print("     NOTE: plain METANET sustaining a DETACHED wave would mean our")
        print("     recorded limitation is narrower than written - correct the docs.")
    print()
    return s0, s1


# ---------------------------------------------------------------------------
def hysteresis(model, label, relief=2350.0):
    """(A) never congested vs (B) broken down then relieved, same final demand."""
    T = model.p.T_sec
    n = int(3000 / T)
    tail = slice(int(2500 / T), n)

    dA = np.full(n, relief)                                   # gentle throughout
    dB = np.concatenate([np.full(int(600 / T), 4200.0),
                         np.full(n - int(600 / T), relief)])  # overload, then relieve
    _, _, qA = roll(model, dA)
    _, _, qB = roll(model, dB)
    a, b = float(np.mean(qA[tail, -2])), float(np.mean(qB[tail, -2]))
    print(f"  {label:>34}:  A {a:7.1f}   B {b:7.1f}   B-A {b - a:+7.1f} veh/h"
          f"   {'HYSTERESIS' if b < a - 5 else 'none'}")
    return a, b


def test_hysteresis():
    print("=" * 76)
    print("TEST 3 - HYSTERESIS (the 2026-07-27 test plain METANET failed)")
    print("  A = never congested, B = forced breakdown then relieved to the same")
    print("  demand. A real capacity drop means B settles strictly BELOW A.")
    print("=" * 76)
    hysteresis(build(0.0), "alpha_cd = 0 (plain)")
    for a in (0.15, 0.35, 0.66):
        hysteresis(build(a, rho_cd_ref=80.0), f"alpha_cd = {a}, ref = 80")
    for a in (0.35, 0.66):
        hysteresis(build(a), f"alpha_cd = {a}, ref = rho_max (Han)")
    print()


# ---------------------------------------------------------------------------
def test_calibration():
    print("=" * 76)
    print("TEST 4 - realised capacity drop vs the nominal alpha_cd")
    print("=" * 76)
    m0 = build(0.0)
    cap = m0.p.capacity_per_lane(m0.p.v_free_bneck)
    print(f"  plain 1-lane FD capacity: {cap:.0f} veh/h")
    print(f"  {'alpha_cd':>9} {'ref rho':>9} {'discharge':>10} {'realised drop':>14}")
    T = m0.p.T_sec
    n = int(2500 / T)
    d = np.concatenate([np.full(int(1500 / T), 4200.0),
                        np.full(n - int(1500 / T), 2350.0)])
    for a in (0.0, 0.15, 0.35, 0.66):
        for ref in (None, 80.0):
            m = build(a, rho_cd_ref=ref)
            _, _, q = roll(m, d)
            dis = float(np.mean(q[int(1000 / T):int(1500 / T), -2]))
            print(f"  {a:9.2f} {str(ref if ref else 'rho_max'):>9} {dis:10.1f} "
                  f"{100 * (dis / cap - 1):13.1f}%")
    print()


if __name__ == "__main__":
    ok = test_regression()
    if not ok:
        print("Regression failed - stopping, everything below would be meaningless.")
        raise SystemExit(1)
    test_jam_wave()
    test_hysteresis()
    test_calibration()
