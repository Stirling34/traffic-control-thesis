# -*- coding: utf-8 -*-
"""
Calibrate Bounded-METANET so that it CLEARS A JAM THE WAY SUMO DOES.

WHY (ledger 2026-09-22 / 2026-09-29). The two parameter sets the project has
bracket the plant on the one behaviour the controller depends on:
  * PARAMS_BM (fitted on the merge): capacity 3714 veh/h < the 4600 arriving,
    tau 48.8 s -> a released jam NEVER clears in the model (SUMO: ~282 s);
  * PARAMS_BM_STRAIGHT (fitted here on open-loop RRMSE): capacity right but
    rho_crit 91 > the real jam density ~78 -> the jam is free flow in the
    model and "clears" in 10-20 s.
And because the model cannot represent a jam that drains when its inflow is
cut, restricting the sign never pays in the MPC's objective: it is flat or
rising in the posted value, and the posted sign is set by the solver.

Neither earlier objective scored jam clearance. calibrate_straight.py has a
jam WINDOW, but its open loop is driven by measured inflow only -- the slow
vehicle is not an input -- so that window compares a model with no
disturbance against a plant with one and adds noise, not gradient (tau was
free down to 6 s and the optimiser chose 46.6).

WHAT THIS DOES. The objective of calibrate_straight.score (whole-episode
RRMSE, 210 s prediction, jam window, physical-validity and capacity-band
penalties) PLUS a jam-clearance term: the model is started from SUMO's own
measured state at t = 840 / 900 / 960 s -- after the disturbance has released,
so nothing the model cannot know is still acting -- fed the measured inflow,
and rolled 600 s. Scored against SUMO over the same window on
  * time until no modelled cell is below 30 km/h (the project's v_c), and
  * jam cell-seconds below 30 km/h,
each as |log ratio| (with a 30 s / 30 cell-s floor so tiny jams do not blow
up). Fit on d4000_s1 + d4600_s1, held out on d4600_s2 + d4800_s1 -- the same
split as calibrate_straight, so the two fits are directly comparable.

tau_sec's lower bound is 15 s (ledger 2026-08-17: below ~3x the 5 s step the
relaxation is numerically marginal).

Usage:
    python calibration/calibrate_jam.py --iters 80 --pop 12 --workers 2 --seed 0
Writes data/params_jam_s<seed>.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "calibration"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "expB_bounded"))

import calibrate_straight as cs                                      # noqa: E402
from model.metanet_model import PARAMS_BM, PARAMS_BM_STRAIGHT        # noqa: E402

V_C = 30.0
SEED_T = (840, 900, 960)      # disturbance fires at 600 s, held 240 s
WIN_S = 600
W_CLEAR = 2.0                 # weight of the clearance term vs the old objective
# OPTIONAL DISCHARGE TERM (variant B). The clearance-only fit (seed 0) matched
# clearance with tau 78 s and capacity on its 4700 floor: a jam that drains
# slowly because capacity barely exceeds inflow, with drivers leaving the queue
# at 3586 veh/h peak against SUMO's ~4700. That is clearance for the wrong
# reason, and restriction works through exactly that inflow-vs-discharge
# balance. W_DIS > 0 adds |log(peak discharge / 4700)|. Read from the
# environment because scipy's Windows workers re-import this module and would
# not see a value set in main().
W_DIS = float(os.environ.get("JAM_W_DIS", "0"))
Q_DIS_SUMO = 4700.0           # SUMO jam outflow on this road (ledger 2026-09-11)
NAMES = cs.NAMES
BOUNDS = dict(cs.BOUNDS, tau_sec=(15.0, 80.0))


def sumo_clearance(c, t0):
    """SUMO's own answer from t0: (clear time s, jam cell-seconds)."""
    v = c["v"][t0:t0 + WIN_S]
    jam = np.isfinite(v) & (v < V_C)
    ct = WIN_S
    for k in range(jam.shape[0]):
        if not jam[k].any():
            ct = k + 1
            break
    return float(ct), float(jam.sum())


def model_clearance(m, c, t0, v_free):
    """The model started from SUMO's state at t0, measured inflow, 600 s."""
    rho = np.nan_to_num(c["rho"][t0 - 1]).astype(float)
    v = c["v"][t0 - 1].astype(float)
    v = np.where(np.isfinite(v), v, v_free)    # empty cell = free flow
    w = 0.0
    T = m.T_sec
    n = int(WIN_S / T)
    ct, cells = float(WIN_S), 0.0
    found = False
    for k in range(n):
        j = min(t0 + int(k * T), len(c["q_in"]) - 1)
        rho, v, w, _ = m.step(rho, v, w, float(c["q_in"][j]))
        jam = v < V_C
        cells += float(jam.sum()) * T
        if not found and not jam.any():
            ct, found = (k + 1) * T, True
    return ct, cells


def clearance_term(p, cases, tags, L_m, lanes, detail=False):
    m = cs.build(p, L_m, lanes)
    tot, rows = 0.0, []
    for tag in tags:
        c = cases[tag]
        for t0 in SEED_T:
            cs_t, cs_c = sumo_clearance(c, t0)
            cm_t, cm_c = model_clearance(m, c, t0, p["v_free_main"])
            e = (abs(np.log((cm_t + 30.0) / (cs_t + 30.0)))
                 + abs(np.log((cm_c + 30.0) / (cs_c + 30.0))))
            tot += e
            rows.append((tag, t0, cs_t, cm_t, cs_c, cm_c))
    tot /= (len(tags) * len(SEED_T))
    return (tot, rows) if detail else tot


def objective(p, tags):
    base, _ = cs.score(p, cs._CASES, cs._L_M, cs._LANES, tags)
    out = base + W_CLEAR * clearance_term(p, cs._CASES, tags, cs._L_M,
                                          cs._LANES)
    if W_DIS > 0.0:
        out += W_DIS * abs(np.log(max(discharge(p)[1], 1.0) / Q_DIS_SUMO))
    return out


def _obj(x):
    try:
        return objective(dict(zip(NAMES, x)), cs.FIT_CASES)
    except Exception:
        return 1e6


def discharge(p):
    """Pure-discharge probe: 5 cells at rho 90, v 0, zero inflow. Flow is taken
    out of the LAST JAMMED CELL, not the road exit -- the exit measurement
    convolves discharge with propagation time (ledger 2026-09-22, which is why
    calibrate_straight.jam_discharge under-reports)."""
    m = cs.build(p, cs._L_M, cs._LANES)
    rho, v, w = m.x0_empty()
    rho[2:7] = 90.0
    v[2:7] = 0.0
    q = []
    for _ in range(24):                       # 120 s
        rho, v, w, _ = m.step(rho, v, w, 0.0)
        q.append(float(rho[6] * v[6] * cs._LANES[6]))
    return float(np.mean(q)), float(np.max(q))


def capacity(p):
    return 2.0 * p["rho_crit"] * p["v_free_main"] * np.exp(-1.0 / p["a"])


def report(label, p):
    print("\n" + label)
    print("  params: " + ", ".join("%s=%.4g" % (k, p[k]) for k in NAMES))
    out = {}
    for split, tags in (("fit", cs.FIT_CASES), ("held-out", cs.HELD_CASES)):
        base, diag = cs.score(p, cs._CASES, cs._L_M, cs._LANES, tags)
        ct, rows = clearance_term(p, cs._CASES, tags, cs._L_M, cs._LANES,
                                  detail=True)
        print("  %-9s old objective %.3f | clearance error %.3f | total %.3f"
              % (split, base, ct, base + W_CLEAR * ct))
        for tag, d in diag.items():
            print("     %-9s RRMSE v %.3f rho %.3f pred210 %.3f penalty %.2f"
                  % (tag, d["rrmse_v"], d["rrmse_rho"], d["rrmse_pred210"],
                     d["penalty"]))
        for tag, t0, cs_t, cm_t, cs_c, cm_c in rows:
            print("     %-9s from %4d s: clears SUMO %4.0f s  model %4s s | "
                  "jam cell-s SUMO %5.0f model %5.0f"
                  % (tag, t0, cs_t, "%.0f" % cm_t if cm_t < WIN_S else ">600",
                     cs_c, cm_c))
        out[split] = dict(base=base, clearance=ct, rows=rows)
    mq, pq = discharge(p)
    print("  2-lane capacity %.0f veh/h | pure-discharge flow %.0f mean / %.0f "
          "peak veh/h (SUMO ~4700)" % (capacity(p), mq, pq))
    out.update(capacity=capacity(p), discharge_mean=mq, discharge_peak=pq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    def pick(P):
        return {k: P.get(k, P["v_free_main"]) for k in NAMES}

    report("PARAMS_BM (merge-fitted, used for every result)", pick(PARAMS_BM))
    report("PARAMS_BM_STRAIGHT (straight-road open-loop fit)",
           pick(PARAMS_BM_STRAIGHT))
    res = differential_evolution(
        _obj, [BOUNDS[n] for n in NAMES], maxiter=a.iters, popsize=a.pop,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7, seed=a.seed,
        polish=True, workers=a.workers, updating="deferred", disp=True)
    fit = dict(zip(NAMES, [float(x) for x in res.x]))
    out = report("JAM-CLEARANCE FIT (seed %d)" % a.seed, fit)
    path = os.path.join(_ROOT, "data", "params_jam%s_s%d.json"
                        % ("_dis" if W_DIS > 0 else "", a.seed))
    json.dump({"params": fit, "objective": float(res.fun),
               "fit_cases": list(cs.FIT_CASES),
               "held_cases": list(cs.HELD_CASES),
               "capacity": out["capacity"],
               "discharge_mean": out["discharge_mean"],
               "held_clearance_error": out["held-out"]["clearance"],
               "held_old_objective": out["held-out"]["base"],
               "fit_clearance_error": out["fit"]["clearance"]},
              open(path, "w"), indent=1)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
