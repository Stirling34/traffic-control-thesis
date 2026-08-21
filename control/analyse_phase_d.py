# -*- coding: utf-8 -*-
"""
Phase D analysis: rank the cost functions.

Everything is PAIRED against the no-control baseline on the same seed (common
random numbers), so seed-to-seed variation cancels rather than swamping the
effect. The effects here are single-digit percentages and the seed spread is of
the same order, so an unpaired comparison would be meaningless.

Reports, per cost function:
  d_accel   change in vehicle-weighted acceleration RMS  (the shockwave metric)
  d_TTS     change in total time spent                   (the mobility cost)
  d_cvs     change in speed coefficient of variation     (safety surrogate)
  d_CO2     change in CO2 per vehicle-km

with a Wilcoxon signed-rank test where there are enough seeds for it to mean
anything. Below about 6 paired samples the test cannot reach p < 0.05 at all,
so it is reported as "n/a" rather than as a misleadingly large p-value.

Usage:  python control/analyse_phase_d.py
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

RESULTS = os.path.join(_ROOT, "data", "phase_d.json")
METRICS = [("accel_rms", "accel", True),      # (key, label, lower_is_better)
           ("TTS_h", "TTS", True),
           ("cvs", "CVS", True),
           ("CO2_g_per_vehkm", "CO2", True),
           ("TET_s", "TET", True),
           ("TIT_s", "TIT", True)]


def load():
    rows = json.load(open(RESULTS))
    by = {}
    for r in rows:
        by.setdefault(r["cost"], {})[r["seed"]] = r
    return by


def paired(by, cost, key):
    """Percentage change vs no control, per seed."""
    base = by.get("nocontrol", {})
    out = []
    for seed, r in by.get(cost, {}).items():
        b = base.get(seed)
        if b is None or key not in r or key not in b:
            continue
        if not np.isfinite(r[key]) or not np.isfinite(b[key]) or b[key] == 0:
            continue
        out.append(100.0 * (r[key] / b[key] - 1.0))
    return np.array(out)


def wilcoxon(x):
    """Two-sided signed-rank p, or None if too few samples to ever reach 0.05."""
    x = x[x != 0]
    n = len(x)
    if n < 6:
        return None
    try:
        from scipy.stats import wilcoxon as w
        return float(w(x).pvalue)
    except Exception:
        return None


def main():
    if not os.path.exists(RESULTS):
        raise SystemExit("no results yet: run control/phase_d.py first")
    by = load()
    costs = [c for c in by if c != "nocontrol"]
    n_seeds = len(by.get("nocontrol", {}))
    print("Phase D: %d cost functions, %d seeds, paired against no control.\n"
          % (len(costs), n_seeds))

    rows = []
    for c in costs:
        d_acc = paired(by, c, "accel_rms")
        d_tts = paired(by, c, "TTS_h")
        d_cvs = paired(by, c, "cvs")
        d_co2 = paired(by, c, "CO2_g_per_vehkm")
        any_r = next(iter(by[c].values()))
        fails = sum(r["n_failed"] for r in by[c].values())
        solves = sum(r["n_solves"] for r in by[c].values())
        rows.append(dict(cost=c, n=len(d_acc),
                         acc=np.median(d_acc) if len(d_acc) else np.nan,
                         tts=np.median(d_tts) if len(d_tts) else np.nan,
                         cvs=np.median(d_cvs) if len(d_cvs) else np.nan,
                         co2=np.median(d_co2) if len(d_co2) else np.nan,
                         p=wilcoxon(d_acc),
                         vsl=np.mean([r["mean_vsl"] for r in by[c].values()]),
                         fails=fails, solves=solves))

    rows.sort(key=lambda r: (np.inf if np.isnan(r["acc"]) else r["acc"]))
    print("%-20s %8s %8s %8s %8s %7s %7s %9s" %
          ("cost function", "d_accel", "d_TTS", "d_CVS", "d_CO2", "meanVSL",
           "p(acc)", "fails"))
    print("-" * 88)
    for r in rows:
        p = "n/a" if r["p"] is None else ("%.3f" % r["p"])
        print("%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f %7s %5d/%d" %
              (r["cost"], r["acc"], r["tts"], r["cvs"], r["co2"], r["vsl"],
               p, r["fails"], r["solves"]))

    print("\nNegative is better on every column. d_accel is the shockwave "
          "metric and\nthe headline; d_TTS is what it costs in delay.")
    if n_seeds < 6:
        print("\nWith %d seeds the signed-rank test cannot reach p < 0.05 for "
              "any effect,\nso significance is reported as n/a. These are "
              "directional results only." % n_seeds)

    ok = [r for r in rows if np.isfinite(r["acc"])]
    if ok:
        best = min(ok, key=lambda r: r["acc"])
        print("\nLargest shockwave reduction: %s at %.1f%%, costing %.1f%% TTS."
              % (best["cost"], best["acc"], best["tts"]))
        tr = [r for r in ok if r["acc"] < 0 and r["tts"] < 0]
        if tr:
            b = min(tr, key=lambda r: r["acc"])
            print("Improves BOTH shockwave and delay: %s (%.1f%% accel, "
                  "%.1f%% TTS)." % (b["cost"], b["acc"], b["tts"]))
        else:
            print("No cost function improved both at once: the trade-off is "
                  "real, which is the finding.")


if __name__ == "__main__":
    main()
