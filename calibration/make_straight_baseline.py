# -*- coding: utf-8 -*-
"""
Ground truth for calibrating the prediction model to the STRAIGHT jam-wave road.

The merge scenario has `data/sumo_baseline.npz`, fixed and never regenerated, and
every model in the bake-off was scored against it. The straight road has no such
file, so every straight-road result so far runs a model calibrated for a
different road: PARAMS_BM was fitted where the mainline never passed ~1750
veh/h/lane, while this road carries ~2350, and its jam discharge is roughly half
what SUMO does here (ledger 2026-09-11).

This writes the equivalent file: per-second speed and vehicle count on every
edge, for no-control runs at several loadings, with the round-11/12 disturbance.
Fit on some, hold out the others -- `calibrate_straight.py` does that split.

Usage:  python calibration/make_straight_baseline.py
Writes: data/straight_baseline.npz
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sumo"))

import build_straight_net as bsn                                    # noqa: E402
from control.phase_d import run_closed_loop                         # noqa: E402
from control.sweep import STRAIGHT_DIST, STRAIGHT_GATE_AVG_S        # noqa: E402
from model.metanet_model import (PARAMS_BM, MetanetModel, MetanetParams,  # noqa: E402
                                 load_geometry, straight_model_geometry)

OUT = os.path.join(_ROOT, "data", "straight_baseline.npz")
CASES = [(4000, 1), (4600, 1), (4600, 2), (4800, 1)]
STEPS = 2700


def main():
    geom = load_geometry(os.path.join(_ROOT, "sumo", "net", "plain.net4.xml"),
                         allow_straight=True)
    sub, off = straight_model_geometry(geom, bsn.N_UP)
    # a model is needed only to drive run_closed_loop's plumbing; cost=None means
    # it is never solved, so its parameters do not touch the data
    model = MetanetModel(sub, MetanetParams(**PARAMS_BM, vsl_scaled=True),
                         n_ctrl=bsn.N_CTRL)
    out = {}
    for demand, seed in CASES:
        tag = "d%d_s%d" % (demand, seed)
        r = run_closed_loop(None, seed, steps=STEPS, model=model, geom=geom,
                            offset=off, config=bsn.cfg_path(demand),
                            demand_end_s=bsn.DEMAND_END_S,
                            disturbance=dict(STRAIGHT_DIST),
                            gate_avg_s=STRAIGHT_GATE_AVG_S,
                            return_fields=True, tag="base_" + tag)
        out[tag + "_v"] = np.asarray(r["_field_v"], np.float32)
        out[tag + "_cnt"] = np.asarray(r["_field_cnt"], np.float32)
        print("%s: %d s x %d edges, throughput %.0f veh/h, jam %.2f veh-h"
              % (tag, out[tag + "_v"].shape[0], out[tag + "_v"].shape[1],
                 r["throughput"] * 3600.0 / bsn.DEMAND_END_S,
                 r["jam_vc30_veh_h"]))
    np.savez_compressed(
        OUT, lengths_m=np.asarray(geom.lengths_m, float),
        lanes=np.asarray(geom.lanes, float), offset=off,
        n_ctrl=bsn.N_CTRL, cases=np.array(["d%d_s%d" % c for c in CASES]),
        demands=np.array([c[0] for c in CASES]),
        dist_t=STRAIGHT_DIST["t"], dist_hold=STRAIGHT_DIST["hold_s"],
        dist_frac=STRAIGHT_DIST["v_frac"], **out)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
