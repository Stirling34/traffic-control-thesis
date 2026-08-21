# -*- coding: utf-8 -*-
"""
Is our prediction-error metric actually fair? An audit of the measurement itself.

Everything we have concluded rests on one number: the 210 s-ahead speed error.
Before drawing any more conclusions from it, check the measurement is not
punishing the model for the wrong things. Five specific worries:

  1. DOUBLE COUNTING. `seg_edge_index` maps BOTH bottleneck segments onto the
     single SUMO edge n9end, so that one edge supplies 2 of 11 comparison cells
     and is weighted twice as heavily as any other. Fixed here by de-duplicating.

  2. INITIALISATION SHOCK vs REAL DRIFT. We drop SUMO's (density, speed) pair
     into METANET, but that exact pair is not one METANET would ever produce
     itself, so it lurches in the first step or two. That is a fair cost of
     using it as a predictor, but it is NOT the same thing as the model
     drifting away over 210 s, and the two should be reported separately.

  3. BIAS vs SCATTER. A model that is consistently 20 km/h optimistic is a very
     different (and much more fixable) problem from one that is randomly wrong
     by 20 km/h. MAE alone cannot tell them apart.

  4. IS IT JUST RUNNING EARLY? If METANET gets the whole pattern right but ~100 s
     ahead of SUMO, pointwise error looks terrible while the model is actually
     good. Measured by finding the time shift that best lines the two up.

  5. CONDITIONING. Splitting by "free-flow" and "congested" uses SUMO's speed at
     the TARGET time, so the free-flow group is by construction the cells where
     SUMO stayed fast — which is exactly where a slightly-early model looks
     worst. Reported alongside the unconditioned number so the split cannot
     flatter or damn the model on its own.

Usage:  python metric_audit.py
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH
from validation.metanet_prediction_error import HORIZON_S, rollout, sumo_state

LEADS = [5, 15, 30, 60, 90, 120, 150, 180, 210]
STRIDE = 10


def main():
    geom = load_geometry()
    params = MetanetParams()
    model = MetanetModel(geom, params)
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    T = params.T_sec

    # ---- worry 1: de-duplicate the bottleneck ----------------------------
    seg_edge = np.array(model.seg_edge_index)
    _, first = np.unique(seg_edge, return_index=True)
    uniq = np.sort(first)                       # one cell per distinct SUMO edge
    print(f"Comparison cells: {model.n_seg} segments -> {len(uniq)} distinct SUMO "
          f"edges.\n  segment->edge map {list(seg_edge)}; keeping segments "
          f"{list(uniq)}\n")

    t0s = np.arange(60, n_sec - HORIZON_S - max(LEADS), STRIDE)
    n_h = int(HORIZON_S / T)
    n_run = int(n_sec / T)
    d_all = np.where(np.arange(n_run) * T < DEMAND_END_S, DEMAND_VPH, 0.0)
    u = model.v_ctrl_ub

    P = np.full((len(t0s), len(LEADS), model.n_seg), np.nan)   # predicted speed
    A = np.full((len(t0s), len(LEADS), model.n_seg), np.nan)   # actual speed
    R = np.full((len(t0s), len(LEADS), model.n_seg), np.nan)   # actual density
    for i, t0 in enumerate(t0s):
        rho0, v0 = sumo_state(sumo, t0, geom, model, limits)
        x0 = model.pack_x(rho0, v0, float(sumo["pending"][t0]))
        k0 = int(t0 / T)
        w = d_all[k0:k0 + n_h]
        if len(w) < n_h:
            w = np.concatenate([w, np.zeros(n_h - len(w))])
        _, V = rollout(model, x0, u, w)
        for j, ld in enumerate(LEADS):
            P[i, j] = V[int(ld / T) - 1]
            r_t, v_t = sumo_state(sumo, t0 + ld, geom, model, limits)
            A[i, j] = v_t
            R[i, j] = r_t

    P, A, R = P[:, :, uniq], A[:, :, uniq], R[:, :, uniq]
    loaded = R > 0.5

    # ---- worries 2 + 3 ----------------------------------------------------
    print("=" * 74)
    print("ALL loaded cells — no free-flow/congested split, so nothing is")
    print("selected on the answer. Bias = mean(pred - actual): negative means")
    print("METANET is pessimistic (predicts slower than reality).")
    print("=" * 74)
    print(f"{'lead':>6} {'MAE':>7} {'bias':>7} {'scatter':>8} {'corr':>6}  "
          f"{'what it means':>28}")
    for j, ld in enumerate(LEADS):
        m = loaded[:, j]
        e = (P[:, j] - A[:, j])[m]
        p, a = P[:, j][m], A[:, j][m]
        corr = float(np.corrcoef(p, a)[0, 1])
        note = "initialisation shock" if ld <= 15 else ""
        print(f"{ld:5d}s {np.abs(e).mean():7.2f} {e.mean():+7.2f} "
              f"{e.std():8.2f} {corr:6.2f}  {note:>28}")

    j0, j1 = 0, len(LEADS) - 1
    e0 = np.abs((P[:, j0] - A[:, j0])[loaded[:, j0]]).mean()
    e1 = np.abs((P[:, j1] - A[:, j1])[loaded[:, j1]]).mean()
    print(f"\n  Of the {e1:.1f} km/h error at 210 s, {e0:.1f} km/h is already "
          f"there after ONE 5 s step.")
    print(f"  So {100*e0/e1:.0f}% is the jolt of being handed a state METANET "
          f"would not itself produce,\n  and only {100*(e1-e0)/e1:.0f}% is the "
          f"model actually drifting over 210 s.")

    # ---- worry 4: is it just running early? -------------------------------
    print("\n" + "=" * 74)
    print("Is METANET simply RUNNING EARLY? Compare its 210 s prediction against")
    print("SUMO at 210 s + shift, and find the shift that fits best.")
    print("=" * 74)
    best = (None, 1e9)
    print(f"{'shift':>8} {'MAE':>8}")
    for sh in range(-180, 181, 30):
        errs = []
        for i, t0 in enumerate(t0s):
            tt = t0 + HORIZON_S + sh
            if not (0 <= tt < n_sec):
                continue
            _, v_t = sumo_state(sumo, tt, geom, model, limits)
            r_t, _ = sumo_state(sumo, tt, geom, model, limits)
            v_t = v_t[uniq]
            m = R[i, j1] > 0.5
            errs.append(np.abs(P[i, j1][m] - v_t[m]))
        mae = float(np.concatenate(errs).mean())
        mark = ""
        if mae < best[1]:
            best = (sh, mae)
        print(f"{sh:+7d}s {mae:8.2f}{mark}")
    print(f"\n  best fit at shift {best[0]:+d} s  (MAE {best[1]:.2f} vs "
          f"{e1:.2f} unshifted)")
    if best[0] < 0:
        print(f"  => METANET's 210 s picture matches SUMO {abs(best[0])} s EARLIER."
              f"\n     It runs ahead of reality rather than being randomly wrong.")
    elif best[0] == 0:
        print("  => no timing offset; the error is genuine, not a shift.")

    # ---- worry 5: what the split is really selecting ----------------------
    print("\n" + "=" * 74)
    print("What the free-flow / congested split actually selects")
    print("=" * 74)
    m = loaded[:, j1]
    a, p = A[:, j1][m], P[:, j1][m]
    ff, cg = a >= 50.0, a < 50.0
    print(f"  free-flow cells: {ff.sum():5d} ({100*ff.mean():.0f}%)   "
          f"SUMO mean {a[ff].mean():5.1f} km/h, METANET mean {p[ff].mean():5.1f}")
    print(f"  congested cells: {cg.sum():5d} ({100*cg.mean():.0f}%)   "
          f"SUMO mean {a[cg].mean():5.1f} km/h, METANET mean {p[cg].mean():5.1f}")
    print(f"\n  Unconditioned MAE over everything: {np.abs(p - a).mean():.2f} km/h")
    print(f"  Mean speed, SUMO {a.mean():.1f} vs METANET {p.mean():.1f} km/h "
          f"({p.mean()-a.mean():+.1f})")


if __name__ == "__main__":
    main()
