# -*- coding: utf-8 -*-
"""How big is each cost term, and how strongly does it respond to the sign,
RELATIVE TO `tts`, on the final configuration's realistic-sign horizon?

For every logged solve state (stride), roll the prediction model forward
Np*M steps (Np=7, M=12 -> 84 x 5 s = 420 s, the realistic-sign NLP) under
  plan A: hold the currently applied sign, and
  plan B: all signs at 40 km/h (the floor; the rule's posted value),
evaluate every term numerically with the controller's own symbol bundle, and
report per term:
  mag   median |term(A)|
  sens  median |term(B) - term(A)|   (the part that can steer the optimiser)
  w*    median |dtts| / median |dterm|  = weight at which the term's response to
        the sign matches tts's response (a weight of 1.0 on tts)
usage: python term_scale.py <states.npz> <PARAMS_NAME> <net long|straight>
"""
import os
import sys

import casadi as cs
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
sys.path.insert(0, os.path.join(ROOT, "sumo"))

import model.metanet_model as mm                         # noqa: E402
from control.cost_functions import _TERMS                # noqa: E402
from flat_objective_check import setup                   # noqa: E402

NP, M, NC = 7, 12, 5
TERMS = ["tts", "delay", "terminal", "jam", "accel", "accel_temporal",
         "accel_dens", "dens_rate", "upstream_queue", "speed_var", "speed_grad",
         "conflict_risk", "density_sq", "density_track", "queue", "dvsl",
         "ttd_reward"]
BNECK = ["bneck_density", "bneck_flow"]


def rollout(model, rho0, v0, w0, d, plan):
    """plan: (n_ctrl, NC) km/h, each move held M steps, last move held after."""
    N = NP * M
    x = np.concatenate([rho0, v0, [w0]]).reshape(-1, 1)
    xs = [x.ravel()]
    for k in range(N):
        mv = min(k // M, NC - 1)
        dk = d[min(k, len(d) - 1)]
        x = np.array(model.F(x, plan[:, mv], dk)[0]).reshape(-1, 1)
        xs.append(x.ravel())
    X = np.array(xs).T
    n = model.n_seg
    return X[:n], X[n:2 * n], X[2 * n:2 * n + 1]


def bundle(model, rho, v, w, plan, v_last):
    p = model.p
    v_scale = model.v_ctrl_ub.reshape(-1, 1)
    dv = np.diff(np.hstack([v_last.reshape(-1, 1), plan]), axis=1)
    return {"rho": cs.DM(rho), "v": cs.DM(v), "w": cs.DM(w),
            "v_ctrl": cs.DM(plan), "dv": cs.DM(dv),
            "Lkm": np.array(model.seg_lengths_km).reshape(-1, 1),
            "lam": np.array(model.seg_lanes, float).reshape(-1, 1),
            "T_h": p.T, "T_s": p.T_sec, "p": p, "v_free": p.v_free_main,
            "v_scale": v_scale, "rho_crit": p.rho_crit,
            "rho_crit_bneck": (p.rho_crit if p.rho_crit_bneck is None
                               else p.rho_crit_bneck),
            "rho_target_frac": 1.2 * p.rho_crit / p.rho_max,
            "tts_scale": 100.0, "accel_scale": 2000.0}


def main():
    path, pname, net = sys.argv[1:4]
    z = np.load(path)
    env = setup(net, 4600 if net == "straight" else None, getattr(mm, pname))
    model = env["model"]
    terms = TERMS + (BNECK if net == "long" else [])
    idx = [i for i in range(len(z["ok"])) if z["ok"][i]][::2]
    A, B = {t: [] for t in terms}, {t: [] for t in terms}
    mean_v = []
    for i in idx:
        rho0, v0, w0, d, vl = (z["rho0"][i], z["v0"][i], float(z["w0"][i]),
                               z["d"][i], z["v_last"][i])
        hold = np.repeat(vl.reshape(-1, 1), NC, axis=1)
        low = np.full_like(hold, 40.0)
        for plan, store in ((hold, A), (low, B)):
            rho, v, w = rollout(model, rho0, v0, w0, d, plan)
            S = bundle(model, rho, v, w, plan, vl)
            for t in terms:
                store[t].append(float(_TERMS[t](S)))
        mean_v.append(float(np.mean(v0)))
    A = {t: np.array(a) for t, a in A.items()}
    B = {t: np.array(b) for t, b in B.items()}
    dt = np.median(np.abs(B["tts"] - A["tts"]))
    # split states into jam (any measured cell < 30 km/h) and not
    jam = np.array([np.min(z["v0"][i]) < 30.0 for i in idx])
    print("%s %s: %d states (%d with a cell < 30 km/h)" % (net, pname, len(idx),
                                                          jam.sum()))
    print("%-16s %11s %11s %11s %10s %10s  sign(dB-A) in jam"
          % ("term", "mag(A)", "sens", "sens/jam", "w*", "w*/jam"))
    dtj = np.median(np.abs(B["tts"][jam] - A["tts"][jam])) if jam.any() else np.nan
    for t in terms:
        s = np.median(np.abs(B[t] - A[t]))
        sj = np.median(np.abs(B[t][jam] - A[t][jam])) if jam.any() else np.nan
        sg = np.median(np.sign(B[t][jam] - A[t][jam])) if jam.any() else np.nan
        print("%-16s %11.4g %11.4g %11.4g %10.3g %10.3g  %+.0f"
              % (t, np.median(np.abs(A[t])), s, sj,
                 dt / s if s > 0 else np.inf, dtj / sj if sj > 0 else np.inf, sg))


if __name__ == "__main__":
    main()
