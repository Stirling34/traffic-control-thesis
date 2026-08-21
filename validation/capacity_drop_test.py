# -*- coding: utf-8 -*-
"""
Does this bottleneck actually drop capacity when it breaks down?

This is the question the whole project rests on and it had never been measured
directly. A variable speed limit cannot invent capacity. Everything it is
supposed to buy comes from one mechanism: if a bottleneck discharges LESS once a
queue has formed than it did just before, then holding the approach flow back so
the queue never forms preserves that difference. That gap is called the capacity
drop, and it is the entire prize.

If the drop is zero, a VSL can only smooth the flow. It cannot reduce delay, it
cannot raise throughput, and no cost function can make it. Every result so far
is consistent with that, so it needs measuring rather than assuming.

Method, following the standard empirical definition (Cassidy & Bertini 1999;
Chung, Rudjanakanoknad & Cassidy 2007): take the discharge measured at the
bottleneck exit, split it into the interval BEFORE a queue exists upstream and
the interval while a queue is present, and compare. The queue is detected on
upstream density rather than on flow, so the classification does not use the
quantity being compared.

Usage:  python validation/capacity_drop_test.py --seeds 5
Output: data/capacity_drop.json
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MS_TO_KMH, load_geometry      # noqa: E402
from validation.sumo_metrics import CONFIG                    # noqa: E402

STEPS = 2500
AGG = 60                  # discharge averaging window (s)
QUEUE_RHO = 55.0          # veh/km/lane upstream of the merge that counts as queued
OUT = os.path.join(_ROOT, "data", "capacity_drop.json")


def run(seed, geom, b=1.0):
    import traci
    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms)

    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    if b != 1.0:
        for e, v in zip(edges[:-1], limits[:-1]):
            traci.edge.setMaxSpeed(e, float(b * v))

    exit_edge = edges[-1]          # the 1-lane bottleneck
    up_edges = edges[-3:-1]        # the two mainline segments feeding it
    prev = {}
    disch = np.zeros(STEPS)
    rho_up = np.zeros(STEPS)
    for t in range(STEPS):
        traci.simulationStep()
        for v in traci.vehicle.getIDList():
            e = traci.vehicle.getRoadID(v)
            if e == exit_edge and prev.get(v) != exit_edge:
                disch[t] += 1
            if not e.startswith(":"):
                prev[v] = e
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in up_edges])
        L = np.array([Lkm[edges.index(e)] for e in up_edges])
        n = np.array([lanes[edges.index(e)] for e in up_edges])
        rho_up[t] = float(np.mean(c / (L * n)))
    traci.close()

    # per-minute discharge, and whether a queue stood upstream during that minute
    n = (STEPS // AGG) * AGG
    q = disch[:n].reshape(-1, AGG).sum(axis=1) * 3600.0 / AGG
    rq = rho_up[:n].reshape(-1, AGG).mean(axis=1)
    busy = q > 300.0                       # ignore the empty tail after the drain
    free = busy & (rq < QUEUE_RHO)
    queued = busy & (rq >= QUEUE_RHO)
    return dict(seed=seed, b=b,
                q_free=float(np.mean(q[free])) if free.any() else float("nan"),
                q_free_max=float(np.max(q[free])) if free.any() else float("nan"),
                q_queued=float(np.mean(q[queued])) if queued.any() else float("nan"),
                n_free=int(free.sum()), n_queued=int(queued.sum()),
                q_series=[float(x) for x in q], rho_series=[float(x) for x in rq])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--b", type=float, default=1.0)
    a = ap.parse_args()
    geom = load_geometry()
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["seed"], r["b"]) for r in rows}
    for s in range(1, a.seeds + 1):
        if (s, a.b) in done:
            continue
        r = run(s, geom, a.b)
        rows.append(r)
        json.dump(rows, open(OUT, "w"), indent=1)
        print("  seed %d b=%.2f  free-flow %6.0f (peak %6.0f, n=%d)  "
              "queued %6.0f (n=%d)  drop %+5.1f%%"
              % (s, a.b, r["q_free"], r["q_free_max"], r["n_free"],
                 r["q_queued"], r["n_queued"],
                 100 * (r["q_queued"] / r["q_free"] - 1)), flush=True)

    rs = [r for r in rows if r["b"] == a.b]
    qf = np.array([r["q_free"] for r in rs])
    qp = np.array([r["q_free_max"] for r in rs])
    qq = np.array([r["q_queued"] for r in rs])
    drop_mean = 100 * (np.mean(qq) / np.mean(qf) - 1)
    drop_peak = 100 * (np.mean(qq) / np.mean(qp) - 1)
    print("\n  discharge before a queue exists : %6.0f veh/h  (peak minute %6.0f)"
          % (qf.mean(), qp.mean()))
    print("  discharge while queued          : %6.0f veh/h" % qq.mean())
    print("  CAPACITY DROP                   : %+5.1f%% on the mean, "
          "%+5.1f%% on the peak" % (drop_mean, drop_peak))
    print("\n  Literature reports 5-20%% at real merges (Cassidy & Bertini 1999;\n"
          "  Chung et al. 2007; Oh & Yeo 2012 meta-analysis).")
    if drop_peak > -3.0:
        print("\n  => Essentially NO capacity drop in this scenario. Preventing "
              "breakdown\n     therefore preserves nothing, so a VSL cannot "
              "reduce delay here by any\n     cost function. Its only remaining "
              "lever is smoothing the flow, which\n     is what every measured "
              "result has shown.")
    else:
        print("\n  => A real drop exists, so there IS a prize for preventing "
              "breakdown and\n     a VSL should in principle be able to claim "
              "part of it.")


if __name__ == "__main__":
    main()
