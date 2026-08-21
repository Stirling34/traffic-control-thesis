# -*- coding: utf-8 -*-
"""
Measure what a speed limit actually does to the fundamental diagram in SUMO.

Why this exists
---------------
The model currently follows Hegyi (2005) and lets the VSL in only as an upper
bound on the equilibrium speed:

    Veq_i = min( V_fd(rho_i), (1 + alpha) * v_ctrl_i )

That form has a flat spot. Whenever the displayed limit is above
V_fd(rho)/(1+alpha) the control term drops out of the expression entirely, so
dJ/du is exactly zero and the optimiser has no reason to move. Measured in
closed loop, the limit is inert on about 99% of segment-steps.

Carlson, Papamichail and Papageorgiou use a different form. The VSL is a rate
b = displayed limit / legal limit, in [b_min, 1], and it reshapes the whole
speed-density curve (Papamichail et al., IFAC 2008, eqs. 4-7):

    V(rho, b) = v_f(b) * exp{ -(1/a(b)) * ( rho / rho_cr(b) )^a(b) }
    v_f(b)    = b * v_f*
    rho_cr(b) = rho_cr* * ( 1 + A * (1 - b) )
    a(b)      = a* * ( E - (E - 1) * b )

Here b appears in every term, so the gradient is non-zero everywhere and the
plateau cannot exist. The second and third equations are the interesting part:
they say a lower limit pushes the critical density HIGHER, which is the actual
mechanism by which a VSL is supposed to help. Under the min() form the critical
density never moves, so the model cannot represent that mechanism at all.

A and E are calibration coefficients, not universal constants. Importing values
fitted to a Dutch motorway would be assuming the answer. This script measures
them for THIS network by running SUMO at a series of fixed uniform speed limits
and fitting the speed-density curve at each one.

The result is worth reading even if the model change is rejected: if A comes out
near zero, then in this microsimulation a speed limit buys no extra critical
density, and any VSL benefit here has to come from smoothing rather than from
delaying breakdown.

Usage:  python calibration/vsl_fd.py
Output: data/vsl_fd.json
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MS_TO_KMH, load_geometry      # noqa: E402
from validation.sumo_metrics import CONFIG                    # noqa: E402

B_VALUES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
SEEDS = [1, 2, 3]
STEPS = 2500
AGG_S = 60                 # aggregation window, matched to a detector interval
OUT = os.path.join(_ROOT, "data", "vsl_fd.json")


def run(b, seed, geom):
    """One SUMO run at a fixed uniform limit. Returns (rho, v) samples."""
    import traci
    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms)          # m/s, the legal limits

    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    for e, vlim in zip(edges, limits):
        traci.edge.setMaxSpeed(e, float(b * vlim))

    rho_s, v_s = [], []
    for _ in range(STEPS):
        traci.simulationStep()
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        v = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        rho_s.append(c / (Lkm * lanes))
        v_s.append(np.where(c > 0, v, np.nan))
    traci.close()

    rho = np.array(rho_s)
    v = np.array(v_s)
    # Aggregate to AGG_S so single-vehicle noise does not dominate the fit, and
    # keep only the mainline segments upstream of the merge, which is where the
    # VSL is applied and where the fundamental diagram is meant to hold.
    n = (len(rho) // AGG_S) * AGG_S
    n_main = len(edges) - 1
    rho = rho[:n, :n_main].reshape(-1, AGG_S, n_main).mean(axis=1).ravel()
    v = np.nanmean(v[:n, :n_main].reshape(-1, AGG_S, n_main), axis=1).ravel()
    ok = np.isfinite(v) & (rho > 1.0)
    return rho[ok], v[ok]


def fit_fd(rho, v):
    """Fit V = v_f * exp(-(1/a) * (rho/rho_cr)^a) by least squares on speed."""
    def resid(p):
        vf, rc, a = p
        return vf * np.exp(-(1.0 / a) * (rho / rc) ** a) - v

    best = None
    for rc0 in (20.0, 35.0, 50.0):
        for a0 in (1.4, 2.0, 2.8):
            try:
                r = least_squares(resid, [max(v.max(), 1.0), rc0, a0],
                                  bounds=([1.0, 5.0, 0.8], [200.0, 120.0, 6.0]))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    vf, rc, a = best.x
    q_cap = rc * vf * np.exp(-1.0 / a)          # flow at the critical density
    rmse = float(np.sqrt(np.mean(best.fun ** 2)))
    return dict(v_free=float(vf), rho_crit=float(rc), a=float(a),
                q_cap_per_lane=float(q_cap), rmse_kmh=rmse, n=int(len(rho)))


def main():
    geom = load_geometry()
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["b"], r["seed"]) for r in rows}
    for b in B_VALUES:
        for seed in SEEDS:
            if (b, seed) in done:
                continue
            rho, v = run(b, seed, geom)
            r = dict(b=b, seed=seed, **fit_fd(rho, v))
            rows.append(r)
            json.dump(rows, open(OUT, "w"), indent=1)
            print("b=%.2f seed %d  v_free %6.1f  rho_crit %5.1f  a %4.2f  "
                  "q_cap %6.0f  rmse %4.1f  (n=%d)"
                  % (b, seed, r["v_free"], r["rho_crit"], r["a"],
                     r["q_cap_per_lane"], r["rmse_kmh"], r["n"]), flush=True)

    # regress the two Carlson coefficients on the per-b fits
    bs = np.array(sorted({r["b"] for r in rows}))
    rc = np.array([np.mean([r["rho_crit"] for r in rows if r["b"] == b]) for b in bs])
    aa = np.array([np.mean([r["a"] for r in rows if r["b"] == b]) for b in bs])
    vf = np.array([np.mean([r["v_free"] for r in rows if r["b"] == b]) for b in bs])
    qc = np.array([np.mean([r["q_cap_per_lane"] for r in rows if r["b"] == b]) for b in bs])
    i1 = int(np.argmin(np.abs(bs - 1.0)))
    rc_star, a_star = rc[i1], aa[i1]

    # rho_cr(b) = rho_cr* (1 + A(1-b))  ->  regress (rho_cr/rho_cr* - 1) on (1-b)
    x = 1.0 - bs
    A = float(np.dot(x, rc / rc_star - 1.0) / np.dot(x, x))
    # a(b) = a* (E - (E-1) b)  ->  a/a* = 1 + (E-1)(1-b)
    E = 1.0 + float(np.dot(x, aa / a_star - 1.0) / np.dot(x, x))

    print("\n  b     v_free  rho_crit     a   q_cap/lane")
    print("  " + "-" * 42)
    for i, b in enumerate(bs):
        print("  %.2f  %6.1f    %5.1f   %4.2f     %6.0f"
              % (b, vf[i], rc[i], aa[i], qc[i]))
    print("\nCarlson coefficients fitted to this network:")
    print("  A = %+.3f   (critical density shift; positive means a lower limit "
          "raises it)" % A)
    print("  E = %+.3f   (curve sharpness)" % E)
    print("  rho_crit* = %.1f veh/km/lane, a* = %.2f at b = 1" % (rc_star, a_star))
    if A <= 0.05:
        print("\nA is essentially zero: in this SUMO network a lower speed limit "
              "does NOT\nraise the critical density. A VSL cannot delay "
              "breakdown here, so any\nbenefit has to come from smoothing the "
              "flow rather than from extra capacity.")
    else:
        print("\nA is positive: a lower limit does raise the critical density "
              "here, which is\nthe mechanism the min() formulation cannot "
              "represent at all.")
    print("\ncapacity at b=1: %.0f veh/h/lane; best over b: %.0f at b=%.2f"
          % (qc[i1], qc.max(), bs[int(np.argmax(qc))]))
    json.dump(rows, open(OUT, "w"), indent=1)
    # The model reads the coefficients from here, so they are never retyped.
    json.dump(dict(A=A, E=E, rho_crit_star=float(rc[i1]), a_star=float(aa[i1]),
                   b=[float(v) for v in bs],
                   q_cap_per_lane=[float(v) for v in qc],
                   rho_crit=[float(v) for v in rc],
                   b_best_capacity=float(bs[int(np.argmax(qc))]),
                   source="calibration/vsl_fd.py, %d seeds per b, %d s aggregates"
                          % (len(SEEDS), AGG_S)),
              open(os.path.join(_ROOT, "data", "vsl_fd_coef.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
