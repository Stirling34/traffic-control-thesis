# -*- coding: utf-8 -*-
"""
Does the prediction model have any skill over the horizon the MPC actually uses?

Everything the controller does rests on one assumption: that rolling the model
forward 210 s from the current measured state tells you something true about
what SUMO will do. If it does not, then the optimiser is choosing control
actions by minimising a cost over a trajectory that will not happen, and no
cost function can rescue that.

Method. Run SUMO, record the full state every second. At many start times,
hand the model SUMO's measured state and the true future demand, roll it
forward 210 s, and compare against what SUMO actually did. Score with R^2
against two references:

  persistence   the state never changes from t0. This is the honest bar: a
                model that cannot beat "assume nothing happens" is not
                contributing information.
  mean          the trajectory equals its own time-average. R^2 is defined
                against this, so R^2 < 0 means worse than a constant.

Start times are restricted to those where a control decision would actually be
taken and matter, so the score is not inflated by easy empty-road intervals.

  --model shipped   the model as configured in model/metanet_model.py
  --model s14       the ported bake-off winner (PARAMS_S14)

Usage:  python validation/prediction_skill.py --seeds 3 --model s14
Output: data/prediction_skill_<model>.json
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import (MS_TO_KMH, PARAMS_BM, PARAMS_S14,      # noqa: E402
                                 MetanetModel, MetanetParams, load_geometry)
from validation.sumo_metrics import CONFIG                              # noqa: E402

STEPS = 2500
HORIZON_S = 210          # Np * M * T = 7 * 6 * 5 s, the MPC's own horizon


def sumo_trace(seed, geom):
    """Record SUMO's state every second, plus the realised origin inflow."""
    import traci
    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    rho, spd, w, dep = [], [], [], []
    for _ in range(STEPS):
        traci.simulationStep()
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        v = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        rho.append(c / (Lkm * lanes))
        spd.append(np.where(c > 0, v, np.nan))
        w.append(len(traci.simulation.getPendingVehicles()))
        dep.append(traci.simulation.getDepartedNumber())
    traci.close()
    return np.array(rho), np.array(spd), np.array(w, float), np.array(dep, float)


def r2(pred, truth):
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 10:
        return np.nan
    p, t = pred[ok], truth[ok]
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stride", type=int, default=30, help="spacing of start times")
    ap.add_argument("--model", default="shipped",
                    choices=("shipped", "s14", "bounded"),
                    help="'s14' = asymmetric anticipation, 'bounded' = the "
                         "published Bounded-METANET")
    a = ap.parse_args()

    geom = load_geometry()
    mp = MetanetParams(**{"s14": PARAMS_S14, "bounded": PARAMS_BM}[a.model])         if a.model in ("s14", "bounded") else MetanetParams()
    model = MetanetModel(geom, mp)
    p = model.p
    T = int(p.T_sec)
    nH = HORIZON_S // T
    idx = model.seg_edge_index
    out_path = os.path.join(_ROOT, "data", "prediction_skill_%s.json" % a.model)

    rows = []
    for seed in range(1, a.seeds + 1):
        rho_s, spd_s, w_s, dep_s = sumo_trace(seed, geom)
        # realised inflow, block-averaged onto the model step (the ledger records
        # that sub-sampling instead injects 5.7% too many vehicles)
        n = (STEPS // T) * T
        inflow = dep_s[:n].reshape(-1, T).sum(axis=1) * 3600.0 / T

        got = {k: [] for k in ("rho_model", "rho_persist", "v_model", "v_persist")}
        for t0 in range(120, STEPS - HORIZON_S, a.stride):
            r0 = rho_s[t0][idx]
            v0 = np.where(np.isfinite(spd_s[t0][idx]), spd_s[t0][idx], p.v_free_main)
            if r0.min() <= 1.0:               # controller would be gated off
                continue
            k0 = t0 // T
            d = inflow[k0:k0 + nH]
            if len(d) < nH:
                continue
            try:
                sim = model.simulate(d, x0=model.pack_x(r0, v0, [w_s[t0]]),
                                     check_demand=False)
            except Exception:
                continue
            truth_r = rho_s[t0 + 1:t0 + 1 + HORIZON_S:T][:, idx]
            truth_v = spd_s[t0 + 1:t0 + 1 + HORIZON_S:T][:, idx]
            pr, pv = sim["rho"], sim["v"]
            m = min(len(pr), len(truth_r))
            if m < 5:
                continue
            got["rho_model"].append((pr[:m], truth_r[:m]))
            got["rho_persist"].append((np.tile(r0, (m, 1)), truth_r[:m]))
            got["v_model"].append((pv[:m], truth_v[:m]))
            got["v_persist"].append((np.tile(v0, (m, 1)), truth_v[:m]))

        res = {"seed": seed, "model": a.model, "n_windows": len(got["rho_model"])}
        for k, pairs in got.items():
            if not pairs:
                continue
            pred = np.concatenate([x[0].ravel() for x in pairs])
            truth = np.concatenate([x[1].ravel() for x in pairs])
            res["r2_" + k] = r2(pred, truth)
            res["mae_" + k] = float(np.nanmean(np.abs(pred - truth)))
        rows.append(res)
        print("  seed %d  n=%3d windows   R2 rho: model %+.3f  persistence %+.3f"
              "   |  R2 v: model %+.3f  persistence %+.3f"
              % (seed, res["n_windows"], res.get("r2_rho_model", np.nan),
                 res.get("r2_rho_persist", np.nan), res.get("r2_v_model", np.nan),
                 res.get("r2_v_persist", np.nan)), flush=True)

    json.dump(rows, open(out_path, "w"), indent=1)
    f = lambda k: np.nanmean([r.get(k, np.nan) for r in rows])
    print("\n  PREDICTION SKILL AT THE MPC HORIZON (%d s), %d seeds, model=%s"
          % (HORIZON_S, len(rows), a.model))
    print("                       R^2 density   R^2 speed   MAE density   MAE speed")
    print("    prediction model   %+8.3f    %+8.3f   %8.1f      %6.1f"
          % (f("r2_rho_model"), f("r2_v_model"),
             f("mae_rho_model"), f("mae_v_model")))
    print("    persistence        %+8.3f    %+8.3f   %8.1f      %6.1f"
          % (f("r2_rho_persist"), f("r2_v_persist"),
             f("mae_rho_persist"), f("mae_v_persist")))
    print("\n  => the model %s persistence on density, %s on speed."
          % ("BEATS" if f("r2_rho_model") > f("r2_rho_persist") else "DOES NOT BEAT",
             "BEATS" if f("r2_v_model") > f("r2_v_persist") else "DOES NOT BEAT"))
    print("     wrote", out_path)


if __name__ == "__main__":
    main()
