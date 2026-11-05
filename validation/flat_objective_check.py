# -*- coding: utf-8 -*-
"""
DOES THE MPC'S OBJECTIVE ACTUALLY DEPEND ON THE SIGN? (added 2026-10-05)

Ledger 2026-09-29: on the straight road under PARAMS_BM the `tts` objective is
flat in the posted value, so IPOPT returns a sign set by its barrier centring
and bounds, not by the forecast -- shown by moving a lower bound the sign
never reaches (20 -> 40 km/h) and watching the answer move at identical cost.

This script applies that test to any road and objective:
  1. one closed-loop run (deterministic budget), logging every NMPC solve's
     inputs (measured state, previous sign, demand forecast);
  2. offline, every `stride`-th logged state is re-solved from a COLD start
     with v_min = 20 and v_min = 40;
  3. a state is FLAT if the first move shifts by > 1 km/h while the optimal
     cost changes by < 1e-4 (relative) -- the bound moved the answer, the
     objective did not notice.

Usage:
  python validation/flat_objective_check.py long tts 1
  python validation/flat_objective_check.py straight tts 1 4600
Appends one JSON line per run to data/flat_check.jsonl.
"""

import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sumo"))

import build_straight_net as bsn                                       # noqa: E402
import control.nmpc_metanet as nm                                      # noqa: E402
from control.phase_d import run_closed_loop                            # noqa: E402
from control.sweep import (DET_MAX_ITER, DET_MAX_WALL, LONG_CFG,      # noqa: E402
                           LONG_NET, LONG_STEPS, STRAIGHT_DIST,
                           STRAIGHT_GATE_AVG_S, STRAIGHT_STEPS)
from model.metanet_model import (MetanetModel, MetanetParams, PARAMS_BM,  # noqa: E402
                                 load_geometry, model_geometry,
                                 straight_model_geometry)

OUT = os.path.join(_ROOT, "data", "flat_check.jsonl")


def setup(net, demand, pars=PARAMS_BM):
    if net == "long":
        geom = load_geometry(os.path.join(_ROOT, LONG_NET))
        mgeom, off = model_geometry(geom, n_ctrl=9)
        model = MetanetModel(mgeom, MetanetParams(**pars, vsl_scaled=True))
        return dict(geom=geom, model=model, offset=off,
                    config=os.path.join(_ROOT, LONG_CFG), steps=LONG_STEPS)
    geom = load_geometry(bsn.net_path(""), allow_straight=True)
    mgeom, off = straight_model_geometry(geom, bsn.N_UP,
                                         n_seg=bsn.N_CTRL + bsn.N_DOWN)
    model = MetanetModel(mgeom, MetanetParams(**pars, vsl_scaled=True),
                         n_ctrl=bsn.N_CTRL)
    return dict(geom=geom, model=model, offset=off,
                config=bsn.cfg_path(demand, ""), steps=STRAIGHT_STEPS,
                demand_end_s=bsn.DEMAND_END_S, disturbance=dict(STRAIGHT_DIST),
                gate_avg_s=STRAIGHT_GATE_AVG_S)


def main():
    net, cost, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    demand = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    # optional 5th argument: a parameter-set NAME in model.metanet_model
    # (e.g. PARAMS_BM_JAM); default PARAMS_BM, the set every result used
    pname = sys.argv[5] if len(sys.argv) > 5 else "PARAMS_BM"
    import model.metanet_model as mm
    stride = 3
    env = setup(net, demand, getattr(mm, pname))
    model = env.pop("model")

    log = []
    orig = nm.NmpcVslController.solve

    def logged(self, rho0, v0, w0, d, v_last, v_enf0=None):
        u, info = orig(self, rho0, v0, w0, d, v_last, v_enf0=v_enf0)
        log.append(dict(rho0=np.array(rho0), v0=np.array(v0),
                        w0=np.array(w0), d=np.array(d),
                        v_last=np.array(v_last), u=np.array(u),
                        ok=bool(info["success"])))
        return u, info

    nm.NmpcVslController.solve = logged
    t0 = time.time()
    r = run_closed_loop(cost, seed, model=model, Np=None,
                        max_wall=DET_MAX_WALL, max_iter=DET_MAX_ITER,
                        tag="flat_%s_%s_s%d" % (net, cost, seed), **env)
    nm.NmpcVslController.solve = orig

    ctrls = {vm: nm.NmpcVslController(model, cost=cost,
                                      ipopt_max_iter=DET_MAX_ITER,
                                      ipopt_max_wall=DET_MAX_WALL,
                                      v_min_kmh=vm) for vm in (20.0, 40.0)}
    shifts, dcost, flat, n = [], [], 0, 0
    for rec in log[::stride]:
        if not rec["ok"]:
            continue
        res = {}
        for vm, c in ctrls.items():
            c._sol_prev = None
            u, info = c.solve(rec["rho0"], rec["v0"], rec["w0"], rec["d"],
                              rec["v_last"])
            res[vm] = (u, info)
        if not (res[20.0][1]["success"] and res[40.0][1]["success"]):
            continue
        n += 1
        sh = float(np.mean(res[40.0][0]) - np.mean(res[20.0][0]))
        c20, c40 = res[20.0][1]["cost"], res[40.0][1]["cost"]
        rel = abs(c40 - c20) / max(abs(c20), 1e-12)
        shifts.append(sh)
        dcost.append(rel)
        # the 40 bound is inactive at this state if neither solution touches it
        inactive = min(np.min(res[20.0][0]), np.min(res[40.0][0])) > 40.5
        if inactive and abs(sh) > 1.0 and rel < 1e-4:
            flat += 1

    out = dict(net=net, cost=cost, seed=seed, demand=demand, params=pname,
               jam_or_accel=r.get("jam_vc30_veh_h", r.get("accel_rms")),
               mean_vsl=r.get("mean_vsl"), n_solves=len(log),
               n_tested=n, n_flat=flat,
               median_shift_kmh=float(np.median(shifts)) if shifts else None,
               mean_abs_shift_kmh=float(np.mean(np.abs(shifts))) if shifts else None,
               median_rel_cost_change=float(np.median(dcost)) if dcost else None,
               wall_s=round(time.time() - t0, 1))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
