# -*- coding: utf-8 -*-
"""
Parity test for the Bounded-METANET port, and a check that the VSL is alive.

`model/metanet_model.py` now implements Bounded-METANET (Zhao, Roncoli &
Yildirimoglu 2025, TR-C) as CasADi link classes so the NMPC can differentiate
through it. The original in `experiments/expB_bounded/bmetanet.py` is numpy.
Two independent implementations of the same equations are worth nothing unless
they agree, so this asserts they do.

It also reports whether the posted speed limit actually constrains anything,
because on both good models it did not. Under the shipped cap
`min(V_fd(rho), (1+alpha)*u)` with alpha = +0.1, the model believes drivers
travel at 1.10x the sign; measured in SUMO they travel at 0.89x. With a
calibrated free-flow speed of 74 km/h (Bounded-METANET) or 81.5 (expA) and legal
limits of 84.6-99, the FD term always wins and the sign never binds.

`vsl_scaled=True` replaces the cap with `(u / u_legal) * v_free`: the posted
limit scales the free-flow speed in proportion. At the legal limit the cap
equals v_free exactly, so no control still means no change; below it the cap
binds immediately at every segment.

Usage:  python validation/test_bounded_parity.py
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import (PARAMS_BM, PARAMS_S14, MetanetModel,   # noqa: E402
                                 MetanetParams, load_geometry)

STEPS = 400


def numpy_reference(uniform_lengths=True):
    sys.path.insert(0, os.path.join(_ROOT, "experiments", "expB_bounded"))
    import bmetanet
    if not hasattr(bmetanet, "_ORIG_EDGE_LEN_M"):
        bmetanet._ORIG_EDGE_LEN_M = np.array(bmetanet.EDGE_LEN_M)
    bmetanet.EDGE_LEN_M = np.array(bmetanet._ORIG_EDGE_LEN_M)   # restore first
    if uniform_lengths:
        # The original models the road as one 11-segment array and can give the
        # ninth mainline edge its true 196 m. sym-metanet cannot: its origin and
        # node equations assume SCALAR link parameters, so a link carries one
        # length for all its segments (this is the 2026-07-21 ledger entry). The
        # 196 m edge is therefore modelled at the 199.56 m link mean. Held
        # equal, the two implementations agree to machine precision; left
        # unequal the difference is 1.7 veh/km/lane, i.e. that geometry
        # approximation alone.
        bmetanet.EDGE_LEN_M = np.array([199.5555555555556] * 9 + [496.0])
    from bmetanet import BoundedMetanet
    prm = json.load(open(os.path.join(
        _ROOT, "experiments", "expB_bounded", "params_pub_pred.json")))
    bm = BoundedMetanet(**prm)
    rho, v, w = bm.x0_empty()
    R, V = [], []
    for k in range(STEPS):
        rho, v, w, _ = bm.step(rho, v, w,
                               4000.0 if k * 5.0 < 1000.0 else 0.0)
        R.append(np.asarray(rho).copy())
        V.append(np.asarray(v).copy())
    return np.array(R), np.array(V)


def main():
    geom = load_geometry()
    d = np.where(np.arange(STEPS) * 5.0 < 1000.0, 4000.0, 0.0)
    mine = MetanetModel(geom, MetanetParams(**PARAMS_BM)).simulate(
        d, check_demand=False)
    Rn, Vn = numpy_reference(uniform_lengths=True)
    Rp, Vp = numpy_reference(uniform_lengths=False)

    n = min(mine["rho"].shape[1], Rn.shape[1])
    er = np.abs(mine["rho"][:, :n] - Rn[:, :n]).max()
    ev = np.abs(mine["v"][:, :n] - Vn[:, :n]).max()
    ok = max(er, ev) < 1e-9
    print("PARITY, CasADi port vs the numpy original (%d steps, uncontrolled)"
          % STEPS)
    print("  max|d rho| = %.3e    max|d v| = %.3e     %s"
          % (er, ev, "PASS" if ok else "MISMATCH"))
    print("  peak rho  port %6.1f  numpy %6.1f      min v  port %5.1f  numpy %5.1f"
          % (mine["rho"].max(), Rn.max(), mine["v"].min(), Vn.min()))
    m = min(mine["rho"].shape[1], Rp.shape[1])
    print("  against the original's per-edge lengths (196 m ninth edge), which")
    print("  sym-metanet cannot represent: max|d rho| = %.3e   max|d v| = %.3e"
          % (np.abs(mine["rho"][:, :m] - Rp[:, :m]).max(),
             np.abs(mine["v"][:, :m] - Vp[:, :m]).max()))

    print("\nACTUATOR: does the posted limit constrain anything?")
    lim = np.array(geom.main_limits_kmh)
    for tag, kw in (("bounded, shipped cap", PARAMS_BM),
                    ("bounded, scaled cap", dict(PARAMS_BM, vsl_scaled=True)),
                    ("expA,    shipped cap", PARAMS_S14),
                    ("expA,    scaled cap", dict(PARAMS_S14, vsl_scaled=True))):
        m = MetanetModel(geom, MetanetParams(**kw))
        L = m.L1
        cap = (L.v_free * np.ones_like(lim) if m.p.vsl_scaled
               else (1.0 + L.alpha) * lim)
        binds = cap <= L.v_free + 1e-9
        print("  %-21s cap at the legal limit %5.1f-%5.1f  vs v_free %5.1f"
              "  ->  binds on %d/%d segments"
              % (tag, cap.min(), cap.max(), L.v_free, binds.sum(), len(lim)))

    # and the thing that actually matters: is there a dead zone to descend?
    print("\n  dead zone = how far BELOW the legal limit the sign must go before")
    print("  it changes anything at zero density:")
    for tag, kw in (("bounded, shipped cap", PARAMS_BM),
                    ("bounded, scaled cap", dict(PARAMS_BM, vsl_scaled=True)),
                    ("expA,    scaled cap", dict(PARAMS_S14, vsl_scaled=True))):
        m = MetanetModel(geom, MetanetParams(**kw))
        L = m.L1
        u_bind = (lim if m.p.vsl_scaled
                  else np.full_like(lim, L.v_free / (1.0 + L.alpha)))
        print("  %-21s %+5.1f km/h" % (tag, float(np.mean(u_bind - lim))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
