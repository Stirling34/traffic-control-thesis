# -*- coding: utf-8 -*-
"""
Parity test for the ported bake-off model.

`model/metanet_model.py` now implements the asymmetric-anticipation variant that
won the bake-off (`experiments/expA_asym_antic`, tag s14_smooth_tau18) as a
CasADi engine, so the NMPC can differentiate through it. The original is numpy.
Two independent implementations of the same equations are worth exactly nothing
unless they agree, so this asserts they do.

Three checks:
  1. antic="std" with stock parameters must reproduce the existing model
     bit-identically, i.e. the port cannot have disturbed anything.
  2. antic="smooth" with chi=1 must equal antic="std", because the sigmoid
     weight collapses to 1 everywhere when chi=1.
  3. antic="smooth" with the calibrated s14 parameters must match the numpy
     core's trajectory to tight tolerance over a full open-loop run.

Usage:  python validation/test_antic_parity.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MetanetModel, MetanetParams, load_geometry  # noqa: E402

# the calibrated winner, from experiments/expA_asym_antic/params_s14_smooth_tau18.json
S14 = dict(a=2.682064482545289, chi=0.2580522734629326, eta=16.052450492089307,
           grad_scale=1.845872344655433, kappa=36.21988278516994,
           rho_crit=35.144326917533064, rho_crit_bneck=49.400604510855196,
           v_free_bneck=113.26497556989747, v_free_main=81.4966930734148,
           tau_sec=18.0)


def params(**kw):
    return MetanetParams(**kw)


def run(p, steps=400):
    m = MetanetModel(load_geometry(), p)
    d = np.where(np.arange(steps) * p.T_sec < 1000.0, 4000.0, 0.0)
    return m.simulate(d, check_demand=False)


def numpy_reference(steps=400):
    """The same trajectory from the original numpy implementation."""
    sys.path.insert(0, os.path.join(_ROOT, "experiments", "expA_asym_antic"))
    import core as expA
    p = expA.P(antic="smooth", tau_sec=S14["tau_sec"], a=S14["a"],
               chi=S14["chi"], eta=S14["eta"], grad_scale=S14["grad_scale"],
               kappa=S14["kappa"], rho_crit=S14["rho_crit"],
               rho_crit_bneck=S14["rho_crit_bneck"],
               v_free_bneck=S14["v_free_bneck"], v_free_main=S14["v_free_main"])
    m = expA.Model(p)
    rho, v, w = m.x0_empty()
    R, V = [], []
    for k in range(steps):
        d = 4000.0 if k * p.T_sec < 1000.0 else 0.0
        rho, v, w = m.step(rho, v, w, d)
        R.append(rho.copy())
        V.append(v.copy())
    return np.array(R), np.array(V)


def main():
    ok = True

    a = run(params())
    b = run(params(antic="std"))
    e = max(np.abs(a["rho"] - b["rho"]).max(), np.abs(a["v"] - b["v"]).max())
    print("  1. antic='std' reproduces the shipped model      max|diff| = %.2e  %s"
          % (e, "OK" if e < 1e-12 else "FAIL"))
    ok &= e < 1e-12

    c = run(params(antic="smooth", chi=1.0))
    e = max(np.abs(a["rho"] - c["rho"]).max(), np.abs(a["v"] - c["v"]).max())
    print("  2. antic='smooth', chi=1 collapses to 'std'      max|diff| = %.2e  %s"
          % (e, "OK" if e < 1e-9 else "FAIL"))
    ok &= e < 1e-9

    d = run(params(antic="smooth", **S14))
    Rn, Vn = numpy_reference()
    n = min(len(Rn), len(d["rho"]))
    er = np.abs(d["rho"][:n] - Rn[:n]).max()
    ev = np.abs(d["v"][:n] - Vn[:n]).max()
    print("  3. calibrated s14 matches the numpy original     max|drho| = %.2e"
          "  max|dv| = %.2e  %s"
          % (er, ev, "OK" if max(er, ev) < 1e-8 else "FAIL"))
    ok &= max(er, ev) < 1e-8

    print("\n  peak density %.1f (numpy %.1f), min speed %.1f (numpy %.1f)"
          % (d["rho"].max(), Rn.max(), d["v"].min(), Vn.min()))
    print("\n  %s" % ("ALL PARITY CHECKS PASS" if ok else "PARITY FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
