# -*- coding: utf-8 -*-
"""
Is the controller acting like a VSL should, or is it just blanketing the road?

A variable speed limit is a one-way actuator with a specific job: hold back the
approach flow BEFORE it reaches a bottleneck, so that it arrives at a rate the
bottleneck can absorb and never breaks down. It cannot make anyone go faster,
and it can do nothing at all for a queue that has already formed.

So a controller that has understood its job should show, on these panels:
  - signs coming down UPSTREAM of the merge and BEFORE the queue appears,
  - signs going back up once the queue has formed, because by then the sign is
    powerless and holding it low only adds delay,
  - a spatial pattern, not one number applied to all nine segments at once.

A controller that has not will show a flat blanket: every segment the same, held
down for the whole run, which is a fixed speed limit with extra steps.

Top panel is density in SUMO, so the queue is visible. Bottom panel is what the
controller actually posted. Read them together.

Usage:  python control/plot_vsl_field.py --fix carlson_oneway --seed 1
Output: docs/figures/vsl_field_<fix>.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MS_TO_KMH, load_geometry          # noqa: E402
from control.nmpc_metanet import NmpcVslController                # noqa: E402
from control.fix_plateau import FIXES, _model_for, COST, STEPS    # noqa: E402
from validation.sumo_metrics import CONFIG                        # noqa: E402

FIG = os.path.join(_ROOT, "docs", "figures")


def record(fix, seed, geom):
    """One closed-loop run, keeping the whole VSL and density fields."""
    import traci
    kw = dict(FIXES[fix])
    kw.pop("model", None)
    kw.pop("kink_init", None)
    cost = kw.pop("cost", COST)
    model = _model_for(fix, geom)
    ctrl = None if fix == "nocontrol" else NmpcVslController(model, cost=cost, **kw)
    p = model.p

    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    applied = limits[: model.n_ctrl].copy()

    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    dens, vsl = [], []
    for step in range(STEPS):
        traci.simulationStep()
        t = step + 1
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        vraw = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        v_edges = np.where(c > 0, vraw, np.append(applied, limits[-1]))
        dens.append(c / (Lkm * lanes))
        vsl.append(applied.copy())
        if ctrl is None or t % 30:
            continue
        rho_seg = dens[-1][model.seg_edge_index]
        if rho_seg.min() <= 1.0:
            continue
        d_hat = np.where(t + np.arange(ctrl.Np * ctrl.M) * p.T_sec < 1000.0,
                         4000.0, 0.0)
        u, info = ctrl.solve(rho_seg, v_edges[model.seg_edge_index],
                             float(len(traci.simulation.getPendingVehicles())),
                             d_hat, applied)
        if info["success"]:
            applied = np.asarray(u, float)
            for e, vv in zip(edges[: model.n_ctrl], applied):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
    traci.close()
    return np.array(dens), np.array(vsl), limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", default="carlson_oneway", choices=list(FIXES))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    geom = load_geometry()
    dens, vsl, limits = record(a.fix, a.seed, geom)

    n_ctrl = vsl.shape[1]
    # segment 0 is the far upstream end; the merge is at the downstream end of
    # the mainline, so the bottleneck sits at the bottom of both panels.
    extent = [0, len(dens), n_ctrl, 0]

    fig, axs = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                            constrained_layout=True)

    ax = axs[0]
    im = ax.imshow(dens[:, :n_ctrl].T, aspect="auto", origin="upper",
                   extent=extent, cmap="inferno_r", vmin=0, vmax=120,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="density (veh/km/lane)")
    ax.set_ylabel("segment\n(merge at the bottom)")
    ax.set_title("What the traffic did (SUMO) - %s, seed %d" % (a.fix, a.seed))

    ax = axs[1]
    im = ax.imshow(vsl.T, aspect="auto", origin="upper", extent=extent,
                   cmap="viridis", vmin=20, vmax=float(limits.max()),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="posted speed limit (km/h)")
    ax.set_ylabel("segment\n(merge at the bottom)")
    ax.set_xlabel("time (s)")
    ax.set_title("What the controller posted\n"
                 "preemptive control = dark bands upstream and EARLY, before "
                 "the queue above appears")

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "vsl_field_%s.png" % a.fix)
    fig.savefig(p, dpi=140)
    plt.close(fig)

    # a number to go with the picture: how much of the variation is spatial
    # (targeted) versus a single blanket level moving in time
    active = vsl < 0.97 * limits[:n_ctrl]
    across = float(np.mean(np.std(vsl, axis=1)))
    print("wrote", p)
    print("mean spread ACROSS segments at a given time: %.1f km/h "
          "(0 = one blanket number for the whole road)" % across)
    print("share of segment-steps with the sign lowered at all: %.1f%%"
          % (100 * active.mean()))
    onset = np.argmax(dens[:, :n_ctrl].max(axis=1) > 60) or len(dens)
    first = np.argmax(active.any(axis=1)) if active.any() else len(vsl)
    print("queue first appears at t=%ds; sign first comes down at t=%ds  -> %s"
          % (onset, first,
             "PREEMPTIVE" if first < onset else "reactive, after the fact"))


if __name__ == "__main__":
    main()
