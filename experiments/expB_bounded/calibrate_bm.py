# -*- coding: utf-8 -*-
"""
Calibrate Bounded-METANET against the fixed SUMO baseline.

Protocol follows the project's `calibrate.py`, which in turn copies the
Bounded-METANET paper's own protocol (their Sec. 4):

  * open loop, whole episode, measured SUMO inflow as the upstream boundary
  * objective = RRMSE(speed) + RRMSE(density)
  * scipy differential_evolution (the paper uses DE too)
  * FITTED ON t < 1250 s ONLY.  bench.py's `prediction_heldout` block uses
    t0 >= 1250 s and is the number that counts, so it must stay untouched.
  * PLUS physical-validity penalties (peak density <= 185, discharge within
    10% of 2325 veh/h, road must clear) -- the project has been burnt once by
    optimising fit alone (log 2026-08-11, "skill sweep").

Some configs additionally carry a PREDICTION term in the objective.  It is
computed only from start times t0 in [60, 1040) so that t0 + 210 s < 1250 s,
i.e. it still touches nothing outside the calibration window.

Usage:  python calibrate_bm.py <config> [--iters N] [--pop N]
Writes params_<config>.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bmetanet import BoundedMetanet  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "..", "..", "data", "sumo_baseline.npz")

SPLIT_S = 1250
RHO_JAM_PHYS = 185.0
Q_SUMO = 2325.0
LANES = np.array([2.] * 9 + [1.])
LENGTHS_M = np.array([200.] * 8 + [196., 496.])

# Prediction term: start times entirely inside the calibration window.
PRED_T0 = np.arange(60, SPLIT_S - 210, 20)
PRED_LEADS = (60, 210)

FULL_BOUNDS = {
    # v_free is anchored to the OBSERVED free-flow speed (SUMO measures
    # 84.6 km/h on edge 0 rising to 100.7 on edge 9 at rho < 8), not to the
    # posted limit.  The paper's own bound [110, 200] is for a German autobahn.
    "v_free_main":  (74., 105.),
    "v_free_bneck": (85., 125.),
    "rho_crit":     (15., 70.),
    "a":            (0.8, 3.0),
    "a_cong":       (0.3, 3.0),     # fd="exp2" congested-branch exponent
    "tau_sec":      (6., 80.),      # paper [5,80]; must be >= T = 5 s
    "eta_t":        (0.0, 1.0),     # paper's stated range
    "kap_t":        (1., 2000.),    # paper [1,300]; widened -- kap_t -> inf is the
                                    # clean limit rho_hat = rho + eta(rho_dn - rho)
    "rho_max":      (90., 185.),    # paper [110,150]; widened to physical jam
    "C_o":          (2200., 6000.), # origin capacity, Eq. (9); free so it does not
                                    # ride on v_free_main via the FD
    "phi_t":        (0.0, 1.0),     # paper's stated range (never calibrated there)
    "convection":   (0.0, 2.0),     # ablation only
    "down_bc_x":    (0.0, 2.999),   # 0=crit 1=neumann 2=zero
}
BC_NAMES = ["crit", "neumann", "zero"]

# --------------------------------------------------------------------------
# Variants.  `free` = fitted parameters; `fixed` = held constant;
# `pred_w` = weight on the in-window prediction term (0 = paper protocol).
# --------------------------------------------------------------------------
_BASE_FREE = ["v_free_main", "v_free_bneck", "rho_crit", "a", "tau_sec",
              "eta_t", "kap_t", "rho_max", "phi_t", "C_o", "down_bc_x"]

CONFIGS = {
    # pub: Bounded-METANET exactly as published -- Eqs. 13 + 14 + the lane-drop
    #      virtual density (Eq. 16) applied across the link that loses a lane,
    #      exponential FD (Eq. 7).  Paper's own fitting objective.
    "pub": dict(free=_BASE_FREE,
                fixed=dict(fd="exp", lanedrop_scope="link", convection=0.0,
                           n_bneck=2),
                pred_w=0.0),

    # nold: ablation -- delete Eq. 16 entirely (phi_t = 0), i.e. base
    #       Bounded-METANET (Eqs. 13 + 14 only).  Tests whether the lane-drop
    #       term the paper never calibrated is doing any work.
    "nold": dict(free=[p for p in _BASE_FREE if p != "phi_t"],
                 fixed=dict(fd="exp", phi_t=0.0, lanedrop_scope="link",
                            convection=0.0, n_bneck=2),
                 pred_w=0.0),

    # ldlast: Eq. 16 read narrowly -- the weaving term only on the segment
    #         immediately before the lane drop (where METANET's own Eq. 11 sits).
    "ldlast": dict(free=_BASE_FREE,
                   fixed=dict(fd="exp", lanedrop_scope="last", convection=0.0,
                              n_bneck=2),
                   pred_w=0.0),

    # conv: ablation -- put METANET's convection term back on top of the
    #       bounded speed equation.  Tests whether removing it was the right call.
    "conv": dict(free=_BASE_FREE + ["convection"],
                 fixed=dict(fd="exp", lanedrop_scope="link", n_bneck=2),
                 pred_w=0.0),

    # fd2: pub with a TWO-BRANCH exponential FD (separate exponent above and
    #      below rho_crit, continuous at the join).  Dominguez Frejo, Camacho &
    #      Horowitz 2012 report it halves METANET's FD fitting error.  Here it
    #      also attacks the ratchet: a slowly-decaying congested branch keeps
    #      the bottleneck discharging after an overshoot.
    "fd2": dict(free=_BASE_FREE + ["a_cong"],
                fixed=dict(fd="exp2", lanedrop_scope="link", convection=0.0,
                           n_bneck=2),
                pred_w=0.0),

    # fdtri: pub with a triangular (Newell-Daganzo) FD, jam density tied to
    #        rho_max.  NOTE the do-not-retry ledger (2026-08-11) rules out
    #        changing the FD family as a fix for METANET's QUEUE-DENSITY
    #        mismatch, with a proof.  That is not what this is for: here the FD
    #        family is being tested as a fix for Bounded-METANET's ratchet /
    #        numerical-deadlock failure mode, which is a different mechanism.
    #        The ledger's conclusion is expected to hold for the queue density
    #        and is reported honestly either way.
    "fdtri": dict(free=[p for p in _BASE_FREE if p != "a"],
                  fixed=dict(fd="tri", a=1.8, lanedrop_scope="link",
                             convection=0.0, n_bneck=2),
                  tie_jam=True,
                  pred_w=0.0),

    # vfobs: `pub` with v_free CONSTRAINED to the free-flow speed SUMO actually
    #        exhibits (84.6 km/h on edge 0 rising to 100.7 on edge 9 at rho < 8).
    #        Every unconstrained fit pins v_free_main at its lower bound, well
    #        below the observed free-flow speed, because the RRMSE objective is
    #        dominated by the long congested phase.  This variant measures what
    #        that costs, so the trade-off is stated rather than hidden.
    "vfobs": dict(free=_BASE_FREE,
                  fixed=dict(fd="exp", lanedrop_scope="link", convection=0.0,
                             n_bneck=2),
                  bounds=dict(v_free_main=(84., 100.), v_free_bneck=(95., 110.)),
                  pred_w=0.0),

    # pubzero: `pub` with the downstream boundary FIXED to "zero" instead of
    #          fitted.  Reading decision C is a modelling assumption the paper
    #          never states, and post-hoc swaps show it moves held-out 210 s
    #          correlation by up to 0.16 -- more than any other single choice.
    #          "zero" is the physically honest reading for our network: the last
    #          SUMO edge discharges into a free exit with no queue behind it,
    #          whereas "crit" pins the phantom downstream cell AT critical
    #          density, i.e. more congested than reality.  Fixing it and
    #          re-fitting everything else is the clean test.  NOTE: this choice
    #          is argued from the geometry, NOT selected on the held-out score --
    #          selecting a structural option on the held-out block would destroy
    #          the meaning of that block.
    "pubzero": dict(free=[p for p in _BASE_FREE if p != "down_bc_x"],
                    fixed=dict(fd="exp", lanedrop_scope="link", convection=0.0,
                               n_bneck=2, down_bc="zero"),
                    pred_w=0.0),

    # pub_pred: identical model to `pub`, but the objective also rewards 60 s
    #           and 210 s prediction skill measured entirely inside t < 1250 s.
    "pub_pred": dict(free=_BASE_FREE,
                     fixed=dict(fd="exp", lanedrop_scope="link", convection=0.0,
                                n_bneck=2),
                     pred_w=2.0),

    # vfobs_pred: the two things that each helped, together -- v_free held at the
    #             observed free-flow speed (which also restores VSL authority,
    #             since a limit can only bind below v_free) AND the in-window
    #             prediction term in the objective.
    "vfobs_pred": dict(free=_BASE_FREE,
                       fixed=dict(fd="exp", lanedrop_scope="link", convection=0.0,
                                  n_bneck=2),
                       bounds=dict(v_free_main=(84., 100.),
                                   v_free_bneck=(95., 110.)),
                       pred_w=2.0),

    # fd2_pred: same for the two-branch FD.
    "fd2_pred": dict(free=_BASE_FREE + ["a_cong"],
                     fixed=dict(fd="exp2", lanedrop_scope="link", convection=0.0,
                                n_bneck=2),
                     pred_w=2.0),
}

_G = {}


def truth():
    if _G:
        return _G
    d = dict(np.load(BASELINE))
    v_s = np.where(d["speed"] < 0, np.nan, d["speed"])
    cum = np.concatenate([[0.0], d["departed"].astype(float)])
    inflow = np.diff(cum) * 3600.0
    n = v_s.shape[0]
    t = np.arange(n)
    _G.update(v_s=v_s, rho_s=d["rho"], inflow=inflow, n=n,
              pending=d["pending"].astype(float),
              fit=t < SPLIT_S, val=t >= SPLIT_S, all=t >= 0)
    return _G


def kwargs_from(cfg, x):
    kw = dict(CONFIGS[cfg]["fixed"])
    for name, val in zip(CONFIGS[cfg]["free"], x):
        if name == "down_bc_x":
            kw["down_bc"] = BC_NAMES[int(val)]
        else:
            kw[name] = float(val)
    if CONFIGS[cfg].get("tie_jam"):
        kw["rho_jam_fd"] = kw["rho_max"]
    return kw


def _pred_scores(m, t0s, leads):
    """Prediction correlation per lead over the given start times."""
    g = truth()
    out = {}
    P = {l: [] for l in leads}
    A = {l: [] for l in leads}
    R = {l: [] for l in leads}
    for tt in t0s:
        rho0 = g["rho_s"][tt]
        v0 = np.where(np.isnan(g["v_s"][tt]), 90.0, g["v_s"][tt])
        for l in leads:
            r = m.predict(rho0, v0, float(g["pending"][tt]), g["inflow"][tt:], l)
            P[l].append(np.asarray(r["v"], float))
            A[l].append(g["v_s"][tt + l])
            R[l].append(g["rho_s"][tt + l])
    for l in leads:
        p, a, r = np.array(P[l]), np.array(A[l]), np.array(R[l])
        msk = (r > 0.5) & ~np.isnan(a) & np.isfinite(p)
        if msk.sum() < 20 or np.std(p[msk]) < 1e-6:
            out[l] = float("nan")
        else:
            out[l] = float(np.corrcoef(p[msk], a[msk])[0, 1])
    return out


def evaluate(cfg, x, window="fit", want_pred=None):
    g = truth()
    kw = kwargs_from(cfg, x)
    # triangular FD needs a jam density comfortably above rho_crit
    if kw.get("fd") == "tri" and kw.get("rho_jam_fd", 1e9) <= kw["rho_crit"] + 10.0:
        return 1e6, None
    try:
        m = BoundedMetanet(**kw)
        r = m.open_loop(g["inflow"])
    except Exception:
        return 1e6, None
    vm, rm = r["v"], r["rho"]
    if not (np.isfinite(vm).all() and np.isfinite(rm).all()):
        return 1e6, None

    sel = g[window]
    mask = (g["rho_s"] > 0.5) & ~np.isnan(g["v_s"]) & sel[:, None]
    if mask.sum() < 100:
        return 1e6, None

    def rrmse(p, o):
        e = (p - o)[mask]
        return float(np.sqrt((e ** 2).sum() / max((o[mask] ** 2).sum(), 1e-9)))

    obj = rrmse(vm, g["v_s"]) + rrmse(rm, g["rho_s"])

    q = rm * vm * LANES
    disch = float(np.nanmean(q[300:1000, -1]))
    veh = (rm * LENGTHS_M / 1000.0 * LANES).sum(axis=1)
    cleared = bool(np.any(veh[1100:] < 1.0))
    peak = float(np.nanmax(rm))

    pen = 0.0
    if peak > RHO_JAM_PHYS:
        pen += 2.0 * (peak / RHO_JAM_PHYS - 1.0)
    rel = abs(disch - Q_SUMO) / Q_SUMO
    if rel > 0.10:
        pen += 5.0 * (rel - 0.10)
    if not cleared:
        pen += 2.0

    pred_w = CONFIGS[cfg].get("pred_w", 0.0)
    pred_pen = 0.0
    pred = None
    if pred_w > 0.0 or want_pred is not None:
        t0s = PRED_T0 if want_pred is None else np.asarray(want_pred)
        pred = _pred_scores(m, t0s, PRED_LEADS)
        if pred_w > 0.0:
            for l in PRED_LEADS:
                c = pred[l]
                pred_pen += pred_w * (1.0 - (0.0 if np.isnan(c) else c))

    ev = (vm - g["v_s"])[mask]
    er = (rm - g["rho_s"])[mask]
    diag = dict(peak=peak, disch=disch, cleared=cleared, pen=pen,
                pred_pen=pred_pen, pred=pred,
                v_rmse=float(np.sqrt((ev ** 2).mean())),
                v_bias=float(ev.mean()),
                rho_rmse=float(np.sqrt((er ** 2).mean())),
                corr=float(np.corrcoef(vm[mask], g["v_s"][mask])[0, 1]))
    return obj + pen + pred_pen, diag


def _obj(x, cfg):
    """Module-level so multiprocessing can pickle it."""
    return evaluate(cfg, x, "fit")[0]


def report(cfg, label, x):
    print(f"\n{label}")
    kw = kwargs_from(cfg, x)
    print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in kw.items()))
    for w in ("fit", "val", "all"):
        o, d = evaluate(cfg, x, w)
        if d is None:
            print(f"    {w:>4}: INVALID")
            continue
        print(f"    {w:>4}: obj {o:6.3f}  speed RMSE {d['v_rmse']:6.2f}  "
              f"bias {d['v_bias']:7.2f}  dens RMSE {d['rho_rmse']:6.2f}  "
              f"corr {d['corr']:6.3f}"
              + (f"  [pen {d['pen']:.2f}]" if d["pen"] > 1e-9 else ""))
    _, d = evaluate(cfg, x, "all")
    if d:
        print(f"    physics: peak rho {d['peak']:.0f} (max 185)   "
              f"discharge {d['disch']:.0f} (SUMO {Q_SUMO:.0f})   "
              f"clears: {d['cleared']}")
    # informational: prediction skill in-window vs held-out
    _, d = evaluate(cfg, x, "all", want_pred=PRED_T0)
    if d and d["pred"]:
        print("    pred (t0 in calibration window): "
              + "  ".join(f"corr@{l}s {d['pred'][l]:.3f}" for l in PRED_LEADS))
    held = np.arange(SPLIT_S, 2500 - 210, 20)
    _, d = evaluate(cfg, x, "all", want_pred=held)
    if d and d["pred"]:
        print("    pred (t0 HELD OUT, t0 >= 1250): "
              + "  ".join(f"corr@{l}s {d['pred'][l]:.3f}" for l in PRED_LEADS))
    return kw


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "pub"
    iters = 120
    pop = 15
    if "--iters" in sys.argv:
        iters = int(sys.argv[sys.argv.index("--iters") + 1])
    if "--pop" in sys.argv:
        pop = int(sys.argv[sys.argv.index("--pop") + 1])
    truth()

    free = CONFIGS[cfg]["free"]
    ov = CONFIGS[cfg].get("bounds", {})
    bounds = [ov.get(n, FULL_BOUNDS[n]) for n in free]
    print(f"config {cfg}: fitting {len(free)} params {free}")
    print(f"fixed: {CONFIGS[cfg]['fixed']}   pred_w={CONFIGS[cfg].get('pred_w', 0.0)}")

    res = differential_evolution(
        _obj, bounds, args=(cfg,),
        maxiter=iters, popsize=pop, tol=1e-6, mutation=(0.5, 1.0),
        recombination=0.7, seed=0, polish=True, workers=-1,
        updating="deferred", disp=False)
    print(f"  done: {res.nfev} evaluations, best fit objective {res.fun:.4f}")

    # Write BEFORE reporting: a DE run is expensive, a report bug must not lose it.
    kw = kwargs_from(cfg, res.x)
    out = os.path.join(HERE, f"params_{cfg}.json")
    with open(out, "w") as f:
        json.dump(kw, f, indent=2)
    print(f"wrote {out}", flush=True)

    report(cfg, f"CALIBRATED [{cfg}]", res.x)


if __name__ == "__main__":
    main()
