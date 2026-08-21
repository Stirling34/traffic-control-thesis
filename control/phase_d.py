# -*- coding: utf-8 -*-
"""
Phase D: run every candidate cost function in closed loop and score it.

Each run is SUMO + the NMPC talking over TraCI, exactly as `main_metanet_v1.py`
does, but with the objective swapped for one of the entries in
`control/cost_functions.py`. Scoring is always on SUMO output, never on the
model, because the model is what chose the control and cannot also be the judge.

Two passes:
  --screen    fast. TTS, shockwave acceleration, speed variation, emissions.
              About 40 s a run, so all candidates by several seeds is minutes.
  (default)   adds the SSM safety device for TET/TIT. ~28x slower in SUMO, so
              reserve it for the shortlist.

Common random numbers: every cost function sees the same seed set, so the
comparison is paired and the seed-to-seed variation cancels. That matters here,
because TET/TIT have a ~23% coefficient of variation against 3.5% for delay.

Usage:
    python control/phase_d.py --screen --seeds 3
    python control/phase_d.py --costs accel_tts,tts_dvsl --seeds 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import (MS_TO_KMH, PARAMS_S14, MetanetModel,   # noqa: E402
                                 MetanetParams, load_geometry)
from control.nmpc_metanet import NmpcVslController                    # noqa: E402
from control.cost_functions import COSTS                              # noqa: E402
from validation.control_authority import shockwave_metrics            # noqa: E402
from validation.sumo_metrics import (CONFIG, OUT, SUMO_DIR, _ssm_args,
                                     parse_emissions, parse_safety)   # noqa: E402

SIM_STEPS = 2500
CONTROL_INTERVAL = 30
DEMAND_VPH = 4000.0
DEMAND_END_S = 1000.0
RESULTS = os.path.join(_ROOT, "data", "phase_d.json")
# Results from the ported bake-off model go to their own file. They are NOT
# comparable with data/phase_d.json run-for-run: same seeds and same scoring,
# but a different prediction model, which is the whole point of the comparison.
RESULTS_S14 = os.path.join(_ROOT, "data", "phase_d_s14.json")
# Instrumented (SSM) runs go to their own file: they are the same experiment
# with TET/TIT added, but SUMO is far slower with the device on, so they are run
# for a shortlist only and must not be mixed with the screening rows.
RESULTS_S14_SSM = os.path.join(_ROOT, "data", "phase_d_s14_ssm.json")


def run_closed_loop(cost, seed, instrument=False, steps=SIM_STEPS, model=None,
                    geom=None):
    """One SUMO + NMPC run. `cost=None` means no control (the baseline)."""
    import traci
    geom = geom or load_geometry()
    model = model or MetanetModel(geom, MetanetParams())
    ctrl = None if cost is None else NmpcVslController(model, cost=cost)

    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    applied = limits[: model.n_ctrl].copy()
    tag = "%s_s%d" % (cost or "nocontrol", seed)

    # Each run needs its OWN emissions file. All runs previously wrote and read
    # the same sumo/add/edge_emissions.xml, so a parse could land on a file
    # another run was still writing -> "not well-formed (invalid token)".
    em_path = os.path.join(OUT, "em_%s.xml" % tag)
    add_path = os.path.join(OUT, "em_%s.add.xml" % tag)
    with open(add_path, "w") as fh:
        fh.write(
            "<additional>"
            '<edgeData id="em" type="emissions" period="60" '
            'file="%s" excludeEmpty="true"/>' % os.path.basename(em_path)
            + "</additional>")
    args = ["sumo", "-c", CONFIG, "--end", str(steps), "--no-warnings",
            "--seed", str(seed), "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--device.emissions.probability", "1.0",
            "--additional-files", add_path]
    if instrument:
        args += _ssm_args(tag)
    os.makedirs(OUT, exist_ok=True)
    traci.start(args)

    spd, dens, on_road, pend, vsl_hist = [], [], [], [], []
    arrived = n_solve = n_fail = 0
    active = False
    t_solve = 0.0
    for step in range(steps):
        traci.simulationStep()
        t_now = step + 1
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        vraw = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        v_edges = np.where(c > 0, vraw, np.append(applied, limits[-1]))
        spd.append(np.where(c > 0, vraw, np.nan))
        dens.append(c / (Lkm * lanes))
        on_road.append(c.sum())
        pend.append(len(traci.simulation.getPendingVehicles()))
        arrived += traci.simulation.getArrivedNumber()
        vsl_hist.append(applied.copy())

        if ctrl is None or t_now % CONTROL_INTERVAL != 0:
            continue
        rho_seg = dens[-1][model.seg_edge_index]
        v_seg = v_edges[model.seg_edge_index]
        # Gate the controller on traffic being present, BOTH ways. The ledger
        # records that solving on a partly empty road grinds: the density
        # target is unreachable and there is nothing to control. The original
        # gate only ever switched ON, so once the road drained the controller
        # kept solving into an empty network - which is where 35 of 81 failures
        # in a 2500 s run were coming from, all after t ~ 1700 s.
        active = rho_seg.min() > 1.0
        if not active:
            continue
        t_future = t_now + np.arange(ctrl.Np * ctrl.M) * model.p.T_sec
        d_hat = np.where(t_future < DEMAND_END_S, DEMAND_VPH, 0.0)
        t0 = time.time()
        u, info = ctrl.solve(rho_seg, v_seg, float(pend[-1]), d_hat, applied)
        t_solve += time.time() - t0
        n_solve += 1
        if info["success"]:
            applied = np.asarray(u, float)
            for e, vv in zip(edges[: model.n_ctrl], applied):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
        else:
            n_fail += 1
    traci.close()

    on_road = np.array(on_road, float)
    pend = np.array(pend, float)
    out = {
        "cost": cost or "nocontrol", "seed": seed,
        "TTS_h": float((on_road + pend).sum() / 3600.0),
        "TTS_road_h": float(on_road.sum() / 3600.0),
        "throughput": float(arrived),
        # solver health reported with every number: the field does not do this
        "n_solves": n_solve, "n_failed": n_fail,
        "solve_time_s": round(t_solve, 1),
        "mean_vsl": float(np.mean(vsl_hist)),
        "min_vsl": float(np.min(vsl_hist)),
    }
    out.update(shockwave_metrics(np.array(spd), np.array(dens), 1.0, Lkm, lanes))
    em = em_path
    if os.path.exists(em):
        out.update(parse_emissions(em, {e: L for e, L in zip(edges, Lkm)}))
    for tmp in (add_path, em_path if instrument else None):
        pass
    if instrument:
        out.update(parse_safety(os.path.join(OUT, "ssm_%s.xml" % tag)))
        try:
            os.remove(os.path.join(OUT, "ssm_%s.xml" % tag))   # ~90 MB each
        except OSError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="skip the SSM device")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--costs", type=str, default="")
    ap.add_argument("--steps", type=int, default=SIM_STEPS)
    ap.add_argument("--model", default="shipped", choices=("shipped", "s14"),
                    help="'s14' = the ported bake-off winner; results go to "
                         "data/phase_d_s14.json")
    a = ap.parse_args()

    names = [c for c in a.costs.split(",") if c] or list(COSTS)
    seeds = list(range(1, a.seeds + 1))
    geom = load_geometry()
    mp = MetanetParams(**PARAMS_S14) if a.model == "s14" else MetanetParams()
    model = MetanetModel(geom, mp)
    if a.model == "s14":
        results = RESULTS_S14 if a.screen else RESULTS_S14_SSM
    else:
        results = RESULTS

    rows = []
    if os.path.exists(results):
        rows = json.load(open(results))
    done = {(r["cost"], r["seed"], r.get("instrumented", False)) for r in rows}

    todo = [(None, s) for s in seeds] + [(c, s) for c in names for s in seeds]
    print("%d runs, instrument=%s" % (len(todo), not a.screen))
    for cost, seed in todo:
        key = (cost or "nocontrol", seed, not a.screen)
        if key in done:
            print("  skip %-18s seed %d (done)" % (key[0], seed))
            continue
        t0 = time.time()
        try:
            r = run_closed_loop(cost, seed, instrument=not a.screen,
                                steps=a.steps, model=model, geom=geom)
        except Exception as e:
            print("  FAIL %-18s seed %d: %s" % (cost, seed, str(e)[:70]))
            continue
        r["instrumented"] = not a.screen
        rows.append(r)
        json.dump(rows, open(results, "w"), indent=1)
        print("  %-18s seed %d  TTS %7.1f  accel %6.3f  cvs %5.3f  "
              "fails %d/%d  [%.0fs]"
              % (r["cost"], seed, r["TTS_h"], r["accel_rms"], r["cvs"],
                 r["n_failed"], r["n_solves"], time.time() - t0))
    print("\nwrote %s (%d rows)" % (results, len(rows)))


if __name__ == "__main__":
    main()
