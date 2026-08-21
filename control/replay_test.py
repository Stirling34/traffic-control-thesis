# -*- coding: utf-8 -*-
"""
Is the controller's advantage prediction, or is it feedback?

An MPC does two things at once. It predicts a trajectory and optimises across
it, and it re-solves every 30 s against a fresh measurement. Those are separable,
and it matters enormously which one is doing the work here, because the
prediction model has been measured to have no skill at the 210 s horizon it
optimises across (validation/prediction_skill.py) while the closed loop still
delivers a real advantage over a constant sign at the same average restriction.

The test: record the speed-limit trajectory the controller produced on seed A,
then REPLAY that exact trajectory, open loop with no solving at all, on seed B.

  - If the advantage survives the replay, it was never feedback. The controller
    is just discovering a good fixed schedule, and a lookup table would do.
  - If the advantage collapses, the advantage is feedback: reacting to the
    particular queue in front of it, which a schedule cannot do and which does
    not need an accurate long-horizon prediction to work.

Seeds are rotated (A = B + 1) so the replayed trajectory is always from a
different realisation of the same scenario. Same demand, same geometry, same
everything else.

Usage:  python control/replay_test.py --fix carlson_refit --seeds 10
Output: data/replay_test.json
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MS_TO_KMH, load_geometry          # noqa: E402
from control.fix_plateau import FIXES, STEPS                      # noqa: E402
from control.plot_vsl_field import record                         # noqa: E402
from validation.control_authority import shockwave_metrics        # noqa: E402
from validation.sumo_metrics import CONFIG                        # noqa: E402

OUT = os.path.join(_ROOT, "data", "replay_test.json")


def replay(vsl, seed, geom):
    """Apply a pre-recorded per-second VSL trajectory, with no controller."""
    import traci
    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    n_ctrl = vsl.shape[1]

    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    spd, dens, on_road, pend = [], [], [], []
    arrived = 0
    last = None
    for t in range(STEPS):
        v = vsl[min(t, len(vsl) - 1)]
        if last is None or not np.allclose(v, last):
            for e, vv in zip(edges[:n_ctrl], v):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
            last = v
        traci.simulationStep()
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        vr = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        spd.append(np.where(c > 0, vr, np.nan))
        dens.append(c / (Lkm * lanes))
        on_road.append(c.sum())
        pend.append(len(traci.simulation.getPendingVehicles()))
        arrived += traci.simulation.getArrivedNumber()
    traci.close()
    on_road = np.array(on_road, float); pend = np.array(pend, float)
    out = {"TTS_h": float((on_road + pend).sum() / 3600.0),
           "throughput": float(arrived),
           "mean_vsl": float(np.mean(vsl))}
    out.update(shockwave_metrics(np.array(spd), np.array(dens), 1.0, Lkm, lanes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", default="carlson_refit", choices=list(FIXES))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    geom = load_geometry()
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["fix"], r["seed"]) for r in rows}

    pd = json.load(open(os.path.join(_ROOT, "data", "phase_d.json")))
    base = {r["seed"]: r for r in pd if r["cost"] == "nocontrol"}
    pl = json.load(open(os.path.join(_ROOT, "data", "fix_plateau.json")))
    closed = {r["seed"]: r for r in pl if r["fix"] == a.fix}

    cache = {}
    for seed in range(1, a.seeds + 1):
        if (a.fix, seed) in done:
            continue
        src = seed % a.seeds + 1               # a DIFFERENT realisation
        if src not in cache:
            _, vsl, _ = record(a.fix, src, geom)
            cache[src] = vsl
        r = replay(cache[src], seed, geom)
        b = base[seed]
        r.update(fix=a.fix, seed=seed, src_seed=src,
                 d_accel=100 * (r["accel_rms"] / b["accel_rms"] - 1),
                 d_TTS=100 * (r["TTS_h"] / b["TTS_h"] - 1))
        rows.append(r)
        json.dump(rows, open(OUT, "w"), indent=1)
        c = closed.get(seed)
        print("  seed %2d (schedule from seed %2d)  replay %+6.1f%% accel "
              "%+6.1f%% TTS   |  closed loop %+6.1f%% / %+6.1f%%"
              % (seed, src, r["d_accel"], r["d_TTS"],
                 c["d_accel"] if c else np.nan, c["d_TTS"] if c else np.nan),
              flush=True)

    rs = [r for r in rows if r["fix"] == a.fix]
    ra = np.median([r["d_accel"] for r in rs])
    rt = np.median([r["d_TTS"] for r in rs])
    ca = np.median([closed[r["seed"]]["d_accel"] for r in rs if r["seed"] in closed])
    ct = np.median([closed[r["seed"]]["d_TTS"] for r in rs if r["seed"] in closed])
    print("\n  %-28s %9s %9s %9s" % ("", "d_accel", "d_TTS", "mean VSL"))
    print("  " + "-" * 58)
    print("  %-28s %8.1f%% %8.1f%% %9.1f"
          % ("closed loop (re-solving)", ca, ct,
             np.mean([closed[r["seed"]]["mean_vsl"] for r in rs if r["seed"] in closed])))
    print("  %-28s %8.1f%% %8.1f%% %9.1f"
          % ("replayed schedule (no solve)", ra, rt,
             np.mean([r["mean_vsl"] for r in rs])))
    print("\n  The two runs post almost the same limits on average. If the "
          "replay is much\n  worse, the advantage is FEEDBACK - reacting to this "
          "particular queue - and\n  not the long-horizon prediction, which has "
          "been measured to have no skill.")


if __name__ == "__main__":
    main()
