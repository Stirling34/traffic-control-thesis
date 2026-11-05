# -*- coding: utf-8 -*-
"""
Parity: the numpy Bounded-METANET on an EXPLICIT geometry vs the CasADi model
used by the controller, on the straight jam-wave road.

Why this exists. The straight road is to be recalibrated, and the fit runs on
the fast numpy implementation (`experiments/expB_bounded/bmetanet.py`, now
accepting `lengths_m`/`lanes`) while the controller runs the CasADi port in
`model/metanet_model.py`. Fitting one and controlling with the other is only
legitimate if they are the same model, so this asserts it to ~1e-12 -- the same
standard the merge port was held to (validation/test_bounded_parity.py, 5.8e-13).

`vsl_scaled=False` here on purpose: with it on, the CasADi cap is
smooth_min(Veq, v_free, eps=0.25), which sits ~0.1 km/h below Veq even at the
legal limit. That smoothing is a controller-side choice, not part of the model
being fitted, and it would mask a real parity failure.

Usage:  python validation/test_bounded_parity_straight.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sumo"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "expB_bounded"))

import build_straight_net as bsn                                   # noqa: E402
from bmetanet import BoundedMetanet                                # noqa: E402
from model.metanet_model import (PARAMS_BM, PARAMS_BM_STRAIGHT, PARAMS_BM_JAM, PARAMS_BM_JAM_DIS,  # noqa: E402
                                 MetanetModel, MetanetParams,
                                 load_geometry, straight_model_geometry)

N_STEPS = 400
SETS = {"bm": PARAMS_BM, "straight": PARAMS_BM_STRAIGHT, "jam": PARAMS_BM_JAM, "jam_dis": PARAMS_BM_JAM_DIS}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "bm"
    geom = load_geometry(os.path.join(_ROOT, "sumo", "net", "plain.net4.xml"),
                         allow_straight=True)
    sub, _ = straight_model_geometry(geom, bsn.N_UP)
    print("parameter set: %s" % which)
    mp = MetanetParams(**SETS[which])        # vsl_scaled False -- see docstring
    cas = MetanetModel(sub, mp, n_ctrl=bsn.N_CTRL)

    npy = BoundedMetanet(
        v_free_main=mp.v_free_main, v_free_bneck=mp.v_free_main,
        rho_crit=mp.rho_crit, a=mp.a, tau_sec=mp.tau_sec, eta_t=mp.eta_t,
        kap_t=mp.kap_t, rho_max=mp.rho_max, T_sec=mp.T_sec, phi_t=0.0,
        down_bc=mp.down_bc, origin_throttle=True, C_o=mp.C_o,
        lengths_m=np.asarray(sub.lengths_m, float),
        lanes=np.asarray(sub.lanes, float))

    assert npy.n == cas.n_seg, "segment counts differ: %d vs %d" % (npy.n, cas.n_seg)
    d = np.concatenate([np.full(N_STEPS // 2, 4200.0), np.full(N_STEPS // 2, 1500.0)])

    x = cas.x0_empty()
    u = cas.v_ctrl_ub.copy()
    rho, v, w = npy.x0_empty()
    dr = dv = dw = 0.0
    for k in range(N_STEPS):
        x = np.asarray(cas.F(x, u, float(d[k]))[0]).ravel()
        rho, v, w, _ = npy.step(rho, v, w, float(d[k]))
        r_c, v_c, w_c = cas.unpack_x(x)
        dr = max(dr, float(np.max(np.abs(r_c - rho))))
        dv = max(dv, float(np.max(np.abs(v_c - v))))
        dw = max(dw, abs(float(w_c[0]) - w))

    print("PARITY, numpy (explicit geometry) vs CasADi, straight road, %d steps"
          % N_STEPS)
    print("  segments %d, lanes %s, lengths %s km"
          % (npy.n, sorted(set(npy.lam.tolist())), sorted(set(npy.L.tolist()))))
    print("  max|d rho| = %.3e   max|d v| = %.3e   max|d w| = %.3e"
          % (dr, dv, dw))
    ok = max(dr, dv, dw) < 1e-9
    print("  %s" % ("PASS" if ok else "FAIL -- do not calibrate on the numpy model"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
