# -*- coding: utf-8 -*-
"""
CAN THE PREDICTION MODEL FORECAST EACH ROAD, AND THE EFFECT OF ITS OWN SIGN?
(added 2026-10-09, round 42 wave 2; the user's question for the final report)

Two questions, on both roads, with no MPC and no solver anywhere -- only SUMO
and the model's own dynamics, so nothing here can be blamed on an optimiser.

  A. FORECAST SKILL. One uncontrolled SUMO run. Every 30 s, hand the model the
     state the controller would have been handed (same measurement, same held
     inflow forecast as control/phase_d.run_closed_loop), roll it 600 s under
     the legal sign, and compare cell speeds with what SUMO did at leads
     30-600 s, against PERSISTENCE ("nothing changes"). Also the two wave
     features the controller acts on: vehicle-hours and JAM vehicle-hours
     (vehicles in cells below 30 km/h) over the next 600 s.

  B. SIGN RESPONSE -- the quantity the optimiser actually trades on. At four
     moments per run, apply a fixed sign plan (40 or 60 km/h on all nine signs,
     for 120 or 300 s) in SUMO and in the model from the SAME state, and
     compare the CHANGE each produces relative to no sign, over the next 900 s:
     does the model get the sign of the effect right, its size, and the order
     of the plans?

Why branch by REPLAY rather than SUMO's saveState/loadState: SUMO's state files
do not save the car-following or lane-change models' internal state (SUMO docs,
Simulation/SaveAndLoad), so a loaded state does not continue exactly. A replay
from t=0 with the same seed is bit-exact (every no-control and rule row in the
project reproduces to the last digit), and the existing fixed_b + window path
posts the plan from t0, so the branch IS the base run up to t0 by construction.

usage:
  python validation/prediction_study.py straight 1 4600
  python validation/prediction_study.py long 1
Writes data/prediction_study/<net>[_d<demand>]_s<seed>.json
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

import model.metanet_model as mm                          # noqa: E402
from control.phase_d import DEMAND_END_S as MERGE_DEMAND_END  # noqa: E402
from control.phase_d import run_closed_loop               # noqa: E402
from flat_objective_check import setup                    # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "prediction_study")
V_C = 30.0                      # the project's jam threshold, km/h
H_A, H_B = 600, 900             # forecast and branch horizons, s
LEADS = [30, 60, 120, 210, 300, 420, 600]
PARAMS = {"straight": ["PARAMS_BM", "PARAMS_BM_STRAIGHT", "PARAMS_BM_JAM",
                       "PARAMS_BM_JAM_DIS"],
          "long": ["PARAMS_BM", "PARAMS_S14"]}
BRANCH_T0 = {"straight": [540, 660, 780, 900],   # disturbance 600-840 s
             "long": [300, 500, 700, 900]}       # queue forms ~250-1000 s
PLANS = [(40.0, 120), (40.0, 300), (60.0, 120), (60.0, 300)]   # km/h, s


def sumo(env, seed, b=None, window=None, tag="ps"):
    kw = {k: v for k, v in env.items() if k != "model"}
    r = run_closed_loop(None, seed, model=env["model"], Np=None, fixed_b=b,
                        window=window, return_fields=True, tag=tag, **kw)
    return (np.asarray(r.pop("_field_v"), float),
            np.asarray(r.pop("_field_cnt"), float), r)


class Road:
    """Geometry shared by SUMO fields and the model's segments."""

    def __init__(self, env, net):
        g, m, off = env["geom"], env["model"], env["offset"]
        self.net, self.off = net, off
        self.Lkm = np.array(g.lengths_m) / 1000.0
        self.lanes = np.array(g.lanes, float)
        self.limits = np.array(g.limits_ms) * mm.MS_TO_KMH
        self.seg_idx = off + np.asarray(m.seg_edge_index)
        # unique edges (the merge model maps BOTH bottleneck segments to one
        # SUMO edge; count it once on the SUMO side)
        self.edges_u = np.unique(self.seg_idx)
        self.d_end = (float(env["demand_end_s"]) if env.get("demand_end_s")
                      else MERGE_DEMAND_END)
        self.lag_s = float(np.sum(self.Lkm[:off] / np.maximum(self.limits[:off],
                                                              1e-9)) * 3600.0)
        self.ub = np.array(m.v_ctrl_ub, float)

    def state(self, V, C, t0):
        """What run_closed_loop hands the controller at time t0 (k = t0-1)."""
        k = t0 - 1
        c, v = C[k], V[k]
        rho = c[self.seg_idx] / (self.Lkm[self.seg_idx] * self.lanes[self.seg_idx])
        vs = np.where(c[self.seg_idx] > 0, v[self.seg_idx],
                      self.limits[self.seg_idx])
        # held inflow forecast on the edge feeding the model (offset > 0 on
        # both roads), mean over the last 30 s, exactly as the controller
        k0 = max(0, k + 1 - 30)
        cu, vu = C[k0:k + 1, self.off - 1], V[k0:k + 1, self.off - 1]
        q_in = float(np.nanmean(np.where(cu > 0, cu * vu / self.Lkm[self.off - 1],
                                         0.0)))
        return rho, np.nan_to_num(vs, nan=60.0), q_in

    def sumo_window(self, V, C, t0, H):
        """Vehicle-hours and jam vehicle-hours on the modelled road, (t0, t0+H]."""
        c = C[t0:t0 + H][:, self.edges_u]
        v = V[t0:t0 + H][:, self.edges_u]
        jam = np.where(np.isfinite(v) & (v < V_C), c, 0.0)
        return float(c.sum() / 3600.0), float(jam.sum() / 3600.0)

    def sumo_speed(self, V, t):
        """Cell speeds averaged over the model step ending at t (5 s)."""
        w = V[t - 5:t][:, self.seg_idx]
        with np.errstate(all="ignore"):
            return np.nanmean(w, axis=0)


def rollout(model, rho0, v0, q_in, road, t0, H, plan=None):
    """Roll H s from (rho0, v0) with the held inflow forecast; plan = (kmh, s)
    posted on every sign from t0, legal sign otherwise. Returns per-step
    (rho, v) for steps 1..N."""
    T = model.p.T_sec
    N = int(H // T)
    x = np.concatenate([rho0, v0, [0.0]]).reshape(-1, 1)
    n = model.n_seg
    R, Vv = [], []
    for k in range(N):
        t = t0 + k * T
        d = q_in if t < road.d_end + road.lag_s else 0.0
        if plan is not None and k * T < plan[1]:
            u = np.minimum(np.full_like(road.ub, plan[0]), road.ub)
        else:
            u = road.ub
        x = np.array(model.F(x, u, float(d))[0]).reshape(-1, 1)
        R.append(x[:n, 0].copy())
        Vv.append(x[n:2 * n, 0].copy())
    return np.array(R), np.array(Vv)


def model_window(model, R, Vv):
    T = model.p.T_sec
    veh = R * np.array(model.seg_lengths_km) * np.array(model.seg_lanes, float)
    jam = np.where(Vv < V_C, veh, 0.0)
    return float(veh.sum() * T / 3600.0), float(jam.sum() * T / 3600.0)


def main():
    net, seed = sys.argv[1], int(sys.argv[2])
    demand = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "%s%s_s%d" % (net, "_d%d" % demand if demand else "", seed)
    t_start = time.time()
    envs = {p: setup(net, demand, getattr(mm, p)) for p in PARAMS[net]}
    env0 = envs[PARAMS[net][0]]
    road = Road(env0, net)
    V, C, base = sumo(env0, seed, tag="ps_%s_base" % name)
    steps = V.shape[0]
    out = dict(net=net, seed=seed, demand=demand, params=PARAMS[net],
               leads=LEADS, base_TTS_h=base.get("TTS_h"),
               base_jam=base.get("jam_vc30_veh_h"),
               base_reach=base.get("wave_vc30_front_reach"))

    # ---- A. forecast skill ------------------------------------------------
    A = {p: dict(t0=[], pred_vehh=[], pred_jamh=[], err={L: [] for L in LEADS},
                 pers={L: [] for L in LEADS}, truth={L: [] for L in LEADS})
         for p in PARAMS[net]}
    act = dict(t0=[], vehh=[], jamh=[])
    for t0 in range(300, steps - H_A, 30):
        vehh, jamh = road.sumo_window(V, C, t0, H_A)
        act["t0"].append(t0)
        act["vehh"].append(vehh)
        act["jamh"].append(jamh)
        for p, env in envs.items():
            m = env["model"]
            rho0, v0, q_in = road.state(V, C, t0)
            R, Vv = rollout(m, rho0, v0, q_in, road, t0, H_A)
            pv, pj = model_window(m, R, Vv)
            a = A[p]
            a["t0"].append(t0)
            a["pred_vehh"].append(pv)
            a["pred_jamh"].append(pj)
            for L in LEADS:
                truth = road.sumo_speed(V, t0 + L)
                a["truth"][L].append(truth.tolist())
                a["err"][L].append((Vv[L // int(m.p.T_sec) - 1] - truth).tolist())
                a["pers"][L].append((v0 - truth).tolist())
    out["A_actual"] = act
    out["A"] = {p: dict(t0=a["t0"], pred_vehh=a["pred_vehh"],
                        pred_jamh=a["pred_jamh"],
                        err={str(L): a["err"][L] for L in LEADS},
                        pers={str(L): a["pers"][L] for L in LEADS},
                        truth={str(L): a["truth"][L] for L in LEADS})
                for p, a in A.items()}
    print("[%s] A done, %.0f s" % (name, time.time() - t_start), flush=True)

    # ---- B. sign response ---------------------------------------------------
    B = []
    for t0 in BRANCH_T0[net]:
        s_none = road.sumo_window(V, C, t0, H_B)
        rho0, v0, q_in = road.state(V, C, t0)
        m_none = {p: model_window(e["model"], *rollout(e["model"], rho0, v0,
                                                       q_in, road, t0, H_B))
                  for p, e in envs.items()}
        for kmh, dur in PLANS:
            b = kmh / float(np.max(road.ub))
            Vb, Cb, rb = sumo(env0, seed, b=b, window=[t0, t0 + dur],
                              tag="ps_%s_%d_%d_%d" % (name, t0, kmh, dur))
            # replay check: the branch must equal the base run before t0
            same = bool(np.array_equal(Cb[:t0 - 1], C[:t0 - 1]))
            s_plan = road.sumo_window(Vb, Cb, t0, H_B)
            row = dict(t0=t0, kmh=kmh, dur=dur, b=b, replay_identical=same,
                       sumo_d_vehh=s_plan[0] - s_none[0],
                       sumo_d_jamh=s_plan[1] - s_none[1],
                       sumo_none=s_none, sumo_TTS_h=rb.get("TTS_h"),
                       sumo_jam=rb.get("jam_vc30_veh_h"),
                       sumo_reach=rb.get("wave_vc30_front_reach"), model={})
            for p, e in envs.items():
                mp_ = model_window(e["model"], *rollout(
                    e["model"], rho0, v0, q_in, road, t0, H_B, plan=(kmh, dur)))
                row["model"][p] = dict(d_vehh=mp_[0] - m_none[p][0],
                                       d_jamh=mp_[1] - m_none[p][1],
                                       none=m_none[p])
            B.append(row)
            print("[%s] B t0=%d %g km/h %d s: SUMO d_jam %+.2f d_veh %+.2f | %s"
                  % (name, t0, kmh, dur, row["sumo_d_jamh"], row["sumo_d_vehh"],
                     " ".join("%s %+.2f/%+.2f" % (p.replace("PARAMS_", ""),
                                                  v["d_jamh"], v["d_vehh"])
                              for p, v in row["model"].items())), flush=True)
    out["B"] = B
    out["wall_s"] = round(time.time() - t_start, 1)
    path = os.path.join(OUT_DIR, name + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    print("[%s] wrote %s in %.0f s" % (name, path, out["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
