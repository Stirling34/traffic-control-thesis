# -*- coding: utf-8 -*-
"""
DOES THIS OBJECTIVE'S COST ACTUALLY DEPEND ON THE SIGN? (round 42, 2026-10-12)

The bound-shift test of flat_objective_check.py (v_min 20 vs 40) cannot be used
on the final configuration: v_min IS 40 and the floor is reached. Two tests
instead, both on the realistic-sign NLP (M=12, Np=7, v_min 40, dv_max 60,
fit B), from the states of one logged closed-loop run (4600 veh/h, seed 1):

  1. START-POINT TEST. Re-solve each logged state twice, changing ONLY the
     initial guess (vals0): a model rollout holding the legal limit, and one
     holding 40 km/h. Every parameter, v_ctrl_last included, is byte-identical
     (ledger 2026-08-18: varying v_ctrl_last changes the feasible set and fakes
     multimodality). A state is FLAT if the two first moves differ by > 3 km/h
     while the optimal costs agree to < 1e-4 relative -- the starting point,
     not the objective, chose the sign.
  2. PREFERENCE. The objective's cost evaluated numerically (model rollout +
     the controller's own cost terms) under three constant plans held for the
     horizon -- legal, 60, 40 km/h -- reported as relative spread and as which
     plan it prefers. A cost that barely changes across 40-84.6 km/h cannot
     be what is choosing the sign.

One JSON line per objective is APPENDED to data/sign_check.jsonl as soon as it
finishes (user may need to shut the laptop).
usage: python validation/sign_check.py "<cost>" [seed] [demand]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
sys.path.insert(0, os.path.join(ROOT, "sumo"))

import control.nmpc_metanet as nm                         # noqa: E402
import model.metanet_model as mm                          # noqa: E402
from control.cost_functions import build_cost             # noqa: E402
from control.phase_d import run_closed_loop               # noqa: E402
from control.sweep import DET_MAX_ITER, DET_MAX_WALL      # noqa: E402
from flat_objective_check import setup                    # noqa: E402
from term_scale import bundle                             # noqa: E402

OUT = os.path.join(ROOT, "data", "sign_check.jsonl")
NP, M, NC = 7, 12, 5


def numeric_cost(model, cost, rho0, v0, w0, d, plan_kmh, v_last):
    N = NP * M
    x = np.concatenate([rho0, v0, [w0]]).reshape(-1, 1)
    xs = [x.ravel()]
    u = np.minimum(np.full(model.n_ctrl, plan_kmh), model.v_ctrl_ub)
    for k in range(N):
        x = np.array(model.F(x, u, float(d[min(k, len(d) - 1)]))[0]).reshape(-1, 1)
        xs.append(x.ravel())
    X = np.array(xs).T
    n = model.n_seg
    plan = np.tile(u.reshape(-1, 1), (1, NC))
    S = bundle(model, X[:n], X[n:2 * n], X[2 * n:2 * n + 1], plan, v_last)
    return float(build_cost(cost, S))


def main():
    cost = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    demand = int(sys.argv[3]) if len(sys.argv) > 3 else 4600
    t_start = time.time()
    env = setup("straight", demand, mm.PARAMS_BM_JAM_DIS)
    model = env.pop("model")
    log = []
    orig = nm.NmpcVslController.solve

    def logged(self, rho0, v0, w0, d, v_last, v_enf0=None):
        u, info = orig(self, rho0, v0, w0, d, v_last, v_enf0=v_enf0)
        log.append((np.array(rho0, float), np.array(v0, float),
                    float(np.ravel(w0)[0]), np.array(d, float).ravel(),
                    np.array(v_last, float).ravel(), np.array(u, float).ravel(),
                    bool(info["success"])))
        return u, info

    nm.NmpcVslController.solve = logged
    r = run_closed_loop(cost, seed, model=model, Np=None, max_wall=DET_MAX_WALL,
                        max_iter=DET_MAX_ITER, ctrl_interval=60,
                        sign_quant_kmh=10.0, v_min_kmh=40.0, dv_max_kmh=60.0,
                        tag="signcheck_%s_s%d" % (cost, seed), **env)
    nm.NmpcVslController.solve = orig
    t_loop = time.time() - t_start

    ctrl = nm.NmpcVslController(model, cost=cost, M=M, v_min_kmh=40.0,
                                dv_max_kmh=60.0, ipopt_max_iter=DET_MAX_ITER,
                                ipopt_max_wall=DET_MAX_WALL)
    legal = float(np.max(model.v_ctrl_ub))
    states = [x for x in log if x[6]]
    n = flat = both_ok = multi = worse_than_legal = 0
    shifts, rel_dc, spread, prefers, jam_flags = [], [], [], [], []
    for rho0, v0, w0, d, vl, u_cl, _ in states:
        n += 1
        jam = bool(np.min(v0) < 30.0)
        jam_flags.append(jam)
        # 2. preference across constant plans
        J = {p: numeric_cost(model, cost, rho0, v0, w0, d, p, vl)
             for p in (legal, 60.0, 40.0)}
        jmin, jmax = min(J.values()), max(J.values())
        spread.append((jmax - jmin) / max(abs(jmin), 1e-12))
        prefers.append(min(J, key=J.get))
        # 1. start-point test: ONLY vals0 differs
        res = {}
        for g in (legal, 40.0):
            guess_anchor = np.minimum(np.full_like(vl, g), model.v_ctrl_ub)
            ctrl._sol_prev = ctrl._rollout_guess(
                np.asarray(rho0, float),
                np.minimum(np.asarray(v0, float), 0.995 * np.asarray(
                    model.seg_v_free, float)),
                np.atleast_1d(w0), guess_anchor.reshape(-1, 1),
                np.asarray(d, float).reshape(1, -1))
            u, info = ctrl.solve(rho0, v0, w0, d, vl)
            res[g] = (u, info)
        if res[legal][1]["success"] and res[40.0][1]["success"]:
            both_ok += 1
            sh = float(np.mean(np.abs(res[legal][0] - res[40.0][0])))
            c1, c2 = res[legal][1]["cost"], res[40.0][1]["cost"]
            rel = abs(c1 - c2) / max(abs(c1), 1e-12)
            shifts.append(sh)
            rel_dc.append(rel)
            if sh > 3.0 and rel < 1e-4:
                flat += 1
            elif sh > 3.0:
                multi += 1          # start-dependent AND different cost
            # the better of the two IPOPT answers costs MORE than simply
            # holding the legal limit (a feasible plan): the solver is not
            # finding the optimum of its own objective (0.5% margin covers
            # the NLP's 0.995 v_free clip on v0)
            if min(c1, c2) > J[legal] * 1.005:
                worse_than_legal += 1
    jam_flags = np.array(jam_flags)
    spread = np.array(spread)
    out = dict(cost=cost, seed=seed, demand=demand,
               loop_jam=r.get("jam_vc30_veh_h"), loop_TTS_h=r.get("TTS_h"),
               loop_mean_vsl=r.get("mean_vsl"), n_states=n, n_both_ok=both_ok,
               n_flat=flat, n_multi_optima=multi,
               n_worse_than_holding_legal=worse_than_legal,
               median_start_shift_kmh=float(np.median(shifts)) if shifts else None,
               max_start_shift_kmh=float(np.max(shifts)) if shifts else None,
               median_rel_cost_diff=float(np.median(rel_dc)) if rel_dc else None,
               median_spread_all=float(np.median(spread)),
               median_spread_jam=(float(np.median(spread[jam_flags]))
                                  if jam_flags.any() else None),
               prefers_all={str(p): int(sum(1 for q in prefers if q == p))
                            for p in (legal, 60.0, 40.0)},
               prefers_jam={str(p): int(sum(1 for q, j in zip(prefers, jam_flags)
                                            if q == p and j))
                            for p in (legal, 60.0, 40.0)},
               loop_s=round(t_loop, 1), wall_s=round(time.time() - t_start, 1))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
