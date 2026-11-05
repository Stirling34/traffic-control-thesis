# -*- coding: utf-8 -*-
"""
Calibrate Bounded-METANET to the STRAIGHT jam-wave road.

WHY. Every straight-road result so far runs `PARAMS_BM`, fitted on the merge
where the mainline never passed ~1750 veh/h/lane and where queue discharge is
set by the 1-lane bottleneck. On this road SUMO carries ~2350 veh/h/lane and
drains a jam at ~4700 veh/h, while the model's own capacity is 3714 veh/h and
its jam discharge under 2500 -- so the controller has been planning against a
road that cannot exist (ledger 2026-09-11).

PROTOCOL -- the same one `experiments/expB_bounded/calibrate_bm.py` used, which
follows the Bounded-METANET paper's Sec. 4:
  * open loop, whole episode, measured SUMO inflow as the upstream boundary
  * objective = RRMSE(speed) + RRMSE(density), plus a 210 s PREDICTION term
    (the horizon the MPC actually uses; the merge fit found adding it took
    held-out skill 0.701 -> 0.784)
  * physical-validity penalties, because the project has been burnt by
    optimising fit alone (log 2026-08-11, "skill sweep")
  * scipy differential_evolution
  * FIT on some runs, HOLD OUT others, and report both

The fit runs on the numpy model, the controller runs the CasADi one; they are
parity-checked to 2e-14 by validation/test_bounded_parity_straight.py.

Usage:
    python calibration/calibrate_straight.py --iters 60 --pop 12 --workers 6
Writes: data/params_straight.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "expB_bounded"))

from bmetanet import BoundedMetanet                                  # noqa: E402
from model.metanet_model import PARAMS_BM                            # noqa: E402

BASE = os.path.join(_ROOT, "data", "straight_baseline.npz")
OUT = os.path.join(_ROOT, "data", "params_straight.json")
SPLIT_S = 1500          # fit on t < SPLIT, report t >= SPLIT held out
RHO_JAM_PHYS = 185.0    # 5 m vehicle + 0.5 m gap
PRED_LEAD = 210         # the MPC horizon
JAM_T0 = 600            # the disturbance fires here (control/sweep.STRAIGHT_DIST)
JAM_WINDOW_S = 600      # scored separately -- see `score`
# THE MODEL'S CAPACITY MUST BE THE ROAD'S CAPACITY. Measured: SUMO carries
# 4782 veh/h here with zero congested cells, so capacity is at least ~4800; the
# fleet's headway (minGap 0.5 m, tau 1 s) puts the ceiling near 5800. Without
# this the fit buys accuracy with an impossible road -- widening rho_crit's
# bound took capacity to 6729 veh/h for a held-out score no better than the
# constrained fit (1.4673 vs 1.4647), the same "degenerate model scores well"
# trap as the 2026-08-11 skill sweep.
CAP_LO, CAP_HI = 4700.0, 5900.0
FIT_CASES = ("d4000_s1", "d4600_s1")
HELD_CASES = ("d4600_s2", "d4800_s1")

NAMES = ["v_free_main", "rho_crit", "a", "tau_sec", "eta_t", "kap_t",
         "rho_max", "C_o"]
BOUNDS = {"v_free_main": (70., 110.),   # SUMO free flow here is ~84.6 posted
          # rho_crit and rho_max widened 2026-09-15: the first fit pinned
          # rho_crit at 70, and a parameter sitting on its bound is not a
          # converged answer. This fleet packs tightly (minGap 0.5 m, tau 1 s,
          # ~2900 veh/h/lane), so a high critical density is physical here, and
          # SUMO's jam reaches ~180 veh/km/lane.
          "rho_crit": (15., 120.),
          "a": (0.5, 3.5),
          "tau_sec": (6., 80.),         # paper [5,80]; must be >= T = 5 s
          "eta_t": (0.0, 1.0),
          "kap_t": (1., 2000.),
          "rho_max": (100., 260.),
          "C_o": (3000., 6500.)}        # this road inserts up to ~4750 veh/h


def load():
    z = np.load(BASE)
    L_m = z["lengths_m"]
    lanes = z["lanes"]
    off = int(z["offset"])
    cases = {}
    for tag in [str(x) for x in z["cases"]]:
        v = np.asarray(z[tag + "_v"], float)
        cnt = np.asarray(z[tag + "_cnt"], float)
        Lkm = L_m / 1000.0
        rho = cnt / (Lkm * lanes)
        # inflow into the modelled stretch = flow on the edge feeding the cut
        q_in = np.nan_to_num(cnt[:, off - 1] / Lkm[off - 1] * np.nan_to_num(v[:, off - 1]))
        cases[tag] = dict(v=v[:, off:], rho=rho[:, off:], q_in=q_in)
    return cases, L_m[off:], lanes[off:]


def build(p, L_m, lanes):
    return BoundedMetanet(
        v_free_main=p["v_free_main"], v_free_bneck=p["v_free_main"],
        rho_crit=p["rho_crit"], a=p["a"], tau_sec=p["tau_sec"],
        eta_t=p["eta_t"], kap_t=p["kap_t"], rho_max=p["rho_max"],
        C_o=p["C_o"], T_sec=5.0, phi_t=0.0, down_bc="zero",
        origin_throttle=True, lengths_m=L_m, lanes=lanes)


def _rrmse(pred, obs, mask):
    d = pred[mask] - obs[mask]
    den = np.sqrt(np.mean(obs[mask] ** 2))
    return float(np.sqrt(np.mean(d ** 2)) / max(den, 1e-6))


def score(p, cases, L_m, lanes, tags, with_pred=True):
    """Objective (lower is better) plus diagnostics."""
    m = build(p, L_m, lanes)
    tot, diag = 0.0, {}
    for tag in tags:
        c = cases[tag]
        n_sec = len(c["q_in"])
        ol = m.open_loop(c["q_in"])
        vp, rp = ol["v"], ol["rho"]
        k = min(SPLIT_S, n_sec)
        occ = c["rho"][:k] > 0.5              # score where SUMO has traffic
        if not occ.any():
            return 1e6, {}
        e_v = _rrmse(vp[:k], np.nan_to_num(c["v"][:k]), occ)
        e_r = _rrmse(rp[:k], c["rho"][:k], occ)
        obj = e_v + e_r
        # prediction at the MPC horizon, from measured states inside the window
        e_p = np.nan
        if with_pred:
            errs = []
            for t0 in range(60, k - PRED_LEAD, 40):
                out = m.predict(c["rho"][t0], np.nan_to_num(c["v"][t0]), 0.0,
                                c["q_in"][t0:], PRED_LEAD)
                t1 = t0 + PRED_LEAD
                msk = c["rho"][t1] > 0.5
                if msk.any():
                    errs.append(_rrmse(out["v"], np.nan_to_num(c["v"][t1]), msk))
            e_p = float(np.mean(errs)) if errs else 1.0
            obj += e_p
        # THE JAM WINDOW, weighted in on its own. The whole point of this
        # recalibration is that the model drains a jam ~6x too slowly, and a
        # whole-episode RRMSE barely notices a 10-minute window. Scoring the
        # post-disturbance window separately puts the defect in the objective.
        t0j = int(JAM_T0)
        t1j = min(t0j + JAM_WINDOW_S, k)
        if t1j > t0j + 60:
            mj = c["rho"][t0j:t1j] > 0.5
            if mj.any():
                obj += (_rrmse(vp[t0j:t1j], np.nan_to_num(c["v"][t0j:t1j]), mj)
                        + _rrmse(rp[t0j:t1j], c["rho"][t0j:t1j], mj))

        # physical validity
        pen = 0.0
        if rp.max() > RHO_JAM_PHYS:
            pen += 2.0 * (rp.max() / RHO_JAM_PHYS - 1.0)
        # NOT "the road must clear": on this road it does not, and neither does
        # SUMO -- demand runs to 2400 s and the cut is still taking ~2800 veh/h
        # at t = 2700, with SUMO's own final density 42.5 veh/km/lane. The merge
        # scenario cleared by 1773 s and that assumption was carried over here,
        # where it rewarded models that drain FASTER than the plant -- the exact
        # error this recalibration exists to fix. Compare to SUMO instead.
        if rp[-1].max() > 3.0 * max(c["rho"][-1].max(), 1.0) + 5.0:
            pen += 1.0
        cap = 2.0 * p["rho_crit"] * p["v_free_main"] * np.exp(-1.0 / p["a"])
        if cap < CAP_LO:
            pen += 2.0 * (CAP_LO / cap - 1.0)
        elif cap > CAP_HI:
            pen += 2.0 * (cap / CAP_HI - 1.0)
        q_model = float(np.mean(rp[:k] * vp[:k] * lanes))
        q_sumo = float(np.mean(c["rho"][:k] * np.nan_to_num(c["v"][:k]) * lanes))
        if q_sumo > 0 and abs(q_model / q_sumo - 1.0) > 0.10:
            pen += abs(q_model / q_sumo - 1.0)
        tot += obj + pen
        diag[tag] = dict(rrmse_v=e_v, rrmse_rho=e_r, rrmse_pred210=e_p,
                         penalty=pen, peak_rho=float(rp.max()))
    return tot / len(tags), diag


def jam_discharge(p, L_m, lanes):
    """Flow out of a standing jam on empty road -- the defect that motivated this."""
    m = build(p, L_m, lanes)
    rho, v, w = m.x0_empty()
    rho[2:7] = 90.0
    v[2:7] = 0.0
    q = []
    for _ in range(24):                       # 120 s
        rho, v, w, _ = m.step(rho, v, w, 0.0)
        q.append(float(rho[-1] * v[-1] * lanes[-1]))
    return float(np.mean(q)), float(np.max(q))


def report(label, p, cases, L_m, lanes):
    f, df = score(p, cases, L_m, lanes, FIT_CASES)
    h, dh = score(p, cases, L_m, lanes, HELD_CASES)
    mean_q, peak_q = jam_discharge(p, L_m, lanes)
    cap = 2.0 * p["rho_crit"] * p["v_free_main"] * np.exp(-1.0 / p["a"])
    print("\n%s" % label)
    print("  params: " + ", ".join("%s=%.4g" % (k, p[k]) for k in NAMES))
    print("  objective  fit %.4f   held-out %.4f" % (f, h))
    for tag, d in list(df.items()) + list(dh.items()):
        print("    %-10s RRMSE v %.3f  rho %.3f  pred210 %.3f  penalty %.2f  peak rho %.0f"
              % (tag, d["rrmse_v"], d["rrmse_rho"], d["rrmse_pred210"],
                 d["penalty"], d["peak_rho"]))
    print("  2-lane capacity %.0f veh/h (SUMO carries ~4750 without congesting)" % cap)
    print("  jam discharge   %.0f veh/h mean over 120 s, peak %.0f "
          "(SUMO ~4700)" % (mean_q, peak_q))
    return f, h


# Loaded at IMPORT, not inside main(), and the objective is a module-level
# function: scipy's `workers` pickles the function by reference and Windows
# spawns fresh interpreters, so a closure over main()'s locals cannot be sent
# ("Can't pickle local object 'main.<locals>.obj'"). Each worker re-imports this
# module and re-reads the npz, which is cheap.
_CASES, _L_M, _LANES = load()


def _obj(x):
    p = dict(zip(NAMES, x))
    try:
        return score(p, _CASES, _L_M, _LANES, FIT_CASES)[0]
    except Exception:
        return 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    cases, L_m, lanes = _CASES, _L_M, _LANES
    print("fit on %s; held out %s; %d modelled edges"
          % (", ".join(FIT_CASES), ", ".join(HELD_CASES), len(L_m)))
    cur = {k: PARAMS_BM.get(k, {"v_free_main": PARAMS_BM["v_free_main"]}.get(k))
           for k in NAMES}
    report("CURRENT (PARAMS_BM, fitted on the merge)", cur, cases, L_m, lanes)
    if a.report_only:
        return

    res = differential_evolution(
        _obj, [BOUNDS[n] for n in NAMES], maxiter=a.iters, popsize=a.pop,
        tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, seed=0, polish=True,
        workers=a.workers, updating="deferred", disp=True)
    fit = dict(zip(NAMES, res.x))
    report("FITTED (this road)", fit, cases, L_m, lanes)
    json.dump({"params": fit, "objective": float(res.fun),
               "fit_cases": list(FIT_CASES), "held_cases": list(HELD_CASES)},
              open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
