# -*- coding: utf-8 -*-
"""
Re-fit the VSL-rate coefficients against the thing the controller actually uses.

`calibration/vsl_fd.py` fits A by regressing the measured critical density on
the VSL rate, which is how Carlson defines it. That is faithful to the paper and
wrong for our purpose. A and E jointly determine the model's *capacity* curve,
and capacity is what the controller weighs when it decides how far to drop a
sign. Fitting A on rho_crit alone leaves the capacity curve free to drift, and
it drifted: the model believes a 0.70 rate buys 12.4% more capacity where SUMO
measures none, so the controller drops the signs much further than the evidence
supports and pays for it in delay.

This fits (A, E) to the measured capacity curve directly, with the critical
density kept as a soft secondary term so the parameters stay interpretable and
do not wander somewhere that fits capacity by accident. Same measurements, same
functional form, different loss.

Usage:  python calibration/fit_vsl_coef.py
Output: rewrites data/vsl_fd_coef.json (keeps the vsl_fd.py fit under
        A_rho_only / E_rho_only so the two are comparable)
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MetanetParams, load_geometry     # noqa: E402

COEF = os.path.join(_ROOT, "data", "vsl_fd_coef.json")
W_RHO = 0.25          # weight on the critical-density term, relative to capacity


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shipped", choices=("shipped", "s14"),
                    help="which model's FD base to fit against")
    args = ap.parse_args()

    c = json.load(open(COEF))
    b = np.array(c["b"])
    q_meas = np.array(c["q_cap_per_lane"])
    rc_meas = np.array(c["rho_crit"])

    # A and E are RELATIVE coefficients: rho_cr(b) = rho_cr* (1 + A(1-b)) and
    # a(b) = a* (E - (E-1) b). The model applies them to ITS OWN rho_crit and a,
    # so the fit has to use the same base or the resulting capacity curve is not
    # the one that was fitted. Fitting against the measured SUMO FD base while
    # the model runs on different parameters is what made the rate formulation
    # underperform: on the ported model's base (rho_crit 35.1, a 2.68) the same
    # A, E produce a capacity curve that never rises, so the model believes
    # slowing down can only cost capacity and the controller acts on that.
    from model.metanet_model import MetanetModel, MetanetParams, PARAMS_S14
    geom = load_geometry()
    mp = MetanetParams(**PARAMS_S14) if args.model == "s14" else MetanetParams()
    p = MetanetModel(geom, mp).p
    rc0, a0, v_free = p.rho_crit, p.a, p.v_free_main
    print("fitting against the %s model's own base: rho_crit=%.2f, a=%.3f, "
          "v_free=%.1f\n" % (args.model, rc0, a0, v_free))

    def curves(A, E):
        rc = rc0 * (1.0 + A * (1.0 - b))
        a = a0 * (E - (E - 1.0) * b)
        q = rc * (b * v_free) * np.exp(-1.0 / a)
        return q, rc

    def resid(p):
        q, rc = curves(*p)
        return np.concatenate([q / q_meas - 1.0,
                               W_RHO * (rc / rc_meas - 1.0)])

    best = None
    for A0 in (0.0, 0.5, 1.0):
        for E0 in (1.0, 1.5, 2.5):
            r = least_squares(resid, [A0, E0], bounds=([-1.0, 0.5], [4.0, 6.0]))
            if best is None or r.cost < best.cost:
                best = r
    A, E = best.x
    q_new, rc_new = curves(A, E)
    q_old, rc_old = curves(c["A"], c["E"])

    print("                       A       E    max capacity error")
    print("  fitted to rho_crit  %.3f  %.3f   %+6.1f%%   (was in use)"
          % (c["A"], c["E"], 100 * np.max(np.abs(q_old / q_meas - 1))))
    print("  fitted to capacity  %.3f  %.3f   %+6.1f%%   (new)"
          % (A, E, 100 * np.max(np.abs(q_new / q_meas - 1))))
    print("\n  b   measured q    old q     new q   |   measured rc   new rc")
    for i in range(len(b)):
        print("  %.2f    %6.0f    %6.0f    %6.0f   |      %5.1f    %5.1f"
              % (b[i], q_meas[i], q_old[i], q_new[i], rc_meas[i], rc_new[i]))
    print("\n  model's best rate: %.2f (old) -> %.2f (new);  SUMO measures %.2f"
          % (b[np.argmax(q_old)], b[np.argmax(q_new)], b[np.argmax(q_meas)]))
    print("  capacity gain the model promises at its best rate: "
          "%+.1f%% (old) -> %+.1f%% (new);  SUMO measures %+.1f%%"
          % (100 * (q_old.max() / q_old[-1] - 1),
             100 * (q_new.max() / q_new[-1] - 1),
             100 * (q_meas.max() / q_meas[-1] - 1)))

    if "A_rho_only" not in c:
        c["A_rho_only"], c["E_rho_only"] = c["A"], c["E"]
    # keep a per-model entry so each model uses coefficients fitted to its own
    # base; the top-level A/E stay as the shipped model's, for compatibility
    c.setdefault("per_model", {})[args.model] = dict(
        A=float(A), E=float(E), rho_crit=float(rc0), a=float(a0),
        v_free=float(v_free))
    if args.model == "shipped":
        c["A"], c["E"] = float(A), float(E)
    c["fit"] = ("joint fit to the measured capacity curve (weight 1) and "
                "critical density (weight %.2f), against each model's OWN FD "
                "base; see calibration/fit_vsl_coef.py" % W_RHO)
    json.dump(c, open(COEF, "w"), indent=1)
    print("\nrewrote", COEF)


if __name__ == "__main__":
    main()
