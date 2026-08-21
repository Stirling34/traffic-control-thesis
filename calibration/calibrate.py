# -*- coding: utf-8 -*-
"""
Proper multi-parameter calibration of METANET against the full SUMO field.

Until now exactly ONE parameter had ever been fitted: rho_crit=40.5, solved from
the measured discharge (a single scalar). a=1.867, tau=18, kappa=40 were textbook
defaults never fitted to anything, and eta=30 was hand-tuned until the stop-go
striping stopped. Unsurprisingly the model scores ~37 km/h speed RMSE against
SUMO where published METANET-vs-SUMO work reports 15.35 (Zhao, Roncoli &
Yildirimoglu 2025, Bounded-METANET, Table 3 -- verified directly; note their
segments are 1 km / 0.5 km against our 200 m, and theirs is an on-ramp merge not
a lane drop, so it is a target rather than a like-for-like benchmark).

Protocol matches the literature (and that paper specifically):
  * open loop, whole episode, measured inflow as the upstream boundary
  * objective = RRMSE(speed) + RRMSE(density), the same objective they used
  * differential evolution, the same global optimiser they used

Two things they do not do, added because we know we need them:
  * PHYSICAL VALIDITY CONSTRAINTS. Optimising fit alone previously selected a
    model with peak density 387 veh/km/lane (jam is 182) that discharged
    823 veh/h and never cleared. Fit alone is not enough.
  * TRAIN/TEST SPLIT IN TIME. Fit on the loading phase, score on the unloading
    phase, so we are not just reporting how well we memorised one trajectory.

Usage:  python calibrate.py [--iters 200]
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import numpy as np
from scipy.optimize import differential_evolution

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, demand_realised

SPLIT_S = 1250          # fit on t < SPLIT_S, validate on t >= SPLIT_S
RHO_JAM_PHYS = 185.0    # 5 m vehicle + 0.5 m gap
Q_SUMO = 2325.0         # measured bottleneck discharge

#            rho_crit      a       tau_s     eta      kappa    v_f main   v_f bneck
# v_free is anchored to the OBSERVED free-flow speed, not the posted limit.
# SUMO's drivers do not drive at the sign: measured mean speed is 86.1 km/h at
# density 0-10 and 80.1 at 10-20, against posted limits of 84.6-99. METANET's
# v_free is a traffic property, not a legal one, so the window is set around the
# observation. (Run 1 used 70-110 and pinned at 70 -- too low. Run 2 used 84-100
# and pinned at 84 -- too high a floor. This is the corrected window.)
BOUNDS = [(15.0, 70.0), (1.0, 4.0), (2., 90.), (2., 200.), (5., 400.),
          (74., 95.), (90., 115.)]
NAMES = ["rho_crit", "a", "tau_sec", "eta", "kappa", "v_free_main", "v_free_bneck"]

_G = {}


def _setup():
    if _G:
        return _G
    geom = load_geometry()
    sumo = dict(np.load(BASELINE_NPZ))
    n_sec = sumo["rho"].shape[0]
    v_s = np.where(sumo["speed"] < 0, np.nan, sumo["speed"])
    _G.update(geom=geom, sumo=sumo, n_sec=n_sec, v_s=v_s, rho_s=sumo["rho"])
    return _G


def params_from(x):
    return MetanetParams(rho_crit=x[0], a=x[1], tau_sec=x[2], eta=x[3],
                         kappa=x[4], v_free_main=x[5], v_free_bneck=x[6])


def run(x):
    """Open-loop whole-episode run with measured inflow. Returns model fields."""
    g = _setup()
    p = params_from(x)
    m = MetanetModel(g["geom"], p)
    T = p.T_sec
    n = int(g["n_sec"] / T)
    d = demand_realised(g["sumo"]["departed"], n, T)
    r = m.simulate(d, check_demand=False)
    idx = np.clip((np.arange(g["n_sec"]) / T).astype(int), 0, r["v"].shape[0] - 1)
    return m, p, r, r["v"][:, :10][idx], r["rho"][:, :10][idx]


def rrmse(pred, obs, mask):
    e = (pred - obs)[mask]
    o = obs[mask]
    return float(np.sqrt((e ** 2).sum() / max((o ** 2).sum(), 1e-9)))


def score(x, window):
    """window: 'fit', 'val' or 'all'. Returns (objective, diagnostics)."""
    g = _setup()
    try:
        m, p, r, vm, rm = run(x)
    except Exception:
        return 1e6, None
    if not np.all(np.isfinite(vm)) or not np.all(np.isfinite(rm)):
        return 1e6, None

    t = np.arange(g["n_sec"])
    sel = {"fit": t < SPLIT_S, "val": t >= SPLIT_S, "all": t >= 0}[window]
    mask = (g["rho_s"] > 0.5) & ~np.isnan(g["v_s"]) & sel[:, None]
    if mask.sum() < 100:
        return 1e6, None

    obj = rrmse(vm, g["v_s"], mask) + rrmse(rm, g["rho_s"], mask)

    # ---- physical validity ------------------------------------------------
    T = p.T_sec
    seg = np.array(m.seg_lengths_km) * np.array(m.seg_lanes, float)
    on = (r["rho"] * seg).sum(axis=1) + r["w"][:, 0]
    tt = np.arange(len(on)) * T
    peak = float(r["rho"].max())
    disch = float(np.mean(r["q"][int(300 / T):int(1000 / T), -2]))
    cleared = bool(np.any((on < 1.0) & (tt > 1000.0)))

    pen = 0.0
    if peak > RHO_JAM_PHYS:                       # denser than physically possible
        pen += 2.0 * (peak / RHO_JAM_PHYS - 1.0)
    rel = abs(disch - Q_SUMO) / Q_SUMO
    if rel > 0.10:                                # discharge must be right
        pen += 5.0 * (rel - 0.10)
    if not cleared:                               # road must empty
        pen += 2.0

    diag = dict(peak=peak, disch=disch, cleared=cleared, pen=pen,
                v_rmse=float(np.sqrt((((vm - g["v_s"]) ** 2)[mask]).mean())),
                rho_rmse=float(np.sqrt((((rm - g["rho_s"]) ** 2)[mask]).mean())))
    return obj + pen, diag


def objective(x):
    return score(x, "fit")[0]


def report(label, x):
    print(f"\n{label}")
    print("  " + "  ".join(f"{n}={v:.3g}" for n, v in zip(NAMES, x)))
    for w in ("fit", "val", "all"):
        o, d = score(x, w)
        if d is None:
            print(f"    {w:>4}: INVALID"); continue
        print(f"    {w:>4}: objective {o:6.3f}   speed RMSE {d['v_rmse']:6.2f} km/h"
              f"   density RMSE {d['rho_rmse']:6.2f}"
              + (f"   [penalty {d['pen']:.2f}]" if d["pen"] > 1e-9 else ""))
    _, d = score(x, "all")
    if d:
        print(f"    physics: peak rho {d['peak']:.0f} (max 185)   "
              f"discharge {d['disch']:.0f} (SUMO {Q_SUMO:.0f})   "
              f"clears: {d['cleared']}")


def main():
    iters = 200
    if "--iters" in sys.argv:
        iters = int(sys.argv[sys.argv.index("--iters") + 1])
    _setup()

    cur = MetanetParams()
    m0 = MetanetModel(_setup()["geom"], cur)
    x0 = [cur.rho_crit, cur.a, cur.tau_sec, cur.eta, cur.kappa,
          m0.p.v_free_main, m0.p.v_free_bneck]
    report("CURRENT (only rho_crit was ever fitted)", x0)

    print(f"\nCalibrating: differential evolution, {iters} iterations, "
          f"7 parameters, fit window t < {SPLIT_S} s ...")
    res = differential_evolution(
        objective, BOUNDS, maxiter=iters, popsize=18, tol=1e-6,
        mutation=(0.5, 1.0), recombination=0.7, seed=0, polish=True,
        workers=-1, updating="deferred", disp=False)
    print(f"  done: {res.nfev} evaluations, best fit objective {res.fun:.4f}")
    report("CALIBRATED", res.x)

    print("\n" + "=" * 72)
    print("Target: published METANET vs SUMO = 15.35 km/h speed, 9.35 density")
    print("        (1 km segments, on-ramp merge -- easier geometry than ours)")
    print("=" * 72)
    np.save("data/calibrated_params.npy", res.x)
    print("Saved data/calibrated_params.npy")
    print("MetanetParams(" + ", ".join(f"{n}={v:.4g}"
                                       for n, v in zip(NAMES, res.x)) + ")")


if __name__ == "__main__":
    main()
