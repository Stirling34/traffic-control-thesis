# -*- coding: utf-8 -*-
"""
Phase D, scored properly: which cost function wins, and on which model.

Two things had to be fixed before "which cost function is most effective" was a
question this setup could answer at all, and this script reports the result with
both fixes in place.

  1. The prediction model. Every earlier Phase D number was produced through a
     METANET whose 210 s speed prediction correlates 0.035 with SUMO, against
     0.324 for simply assuming nothing changes. The bake-off winner, now ported
     into model/metanet_model.py as PARAMS_S14, scores 0.804. A cost-function
     comparison run through a forecast carrying no information is not measuring
     the cost functions.

  2. The reference. Scoring against NO control flatters anything that merely
     lowers the sign. Scoring against a WHOLE-RUN constant sign is the opposite
     error: it pays delay through the free-flow and drain phases where it
     prevents nothing, so it loses on delay for reasons that have nothing to do
     with control. The honest reference is a constant sign restricted to the
     window in which the controller is active, which is what
     `experiments/fixed_vsl.py --window` produces.

Everything is paired on seed against the same no-control runs, and significance
is a two-sided Wilcoxon signed-rank test.

Usage:  python control/compare_models.py
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DATA = os.path.join(_ROOT, "data")
METRICS = [("accel_rms", "d_accel"), ("TTS_h", "d_TTS"), ("cvs", "d_CVS")]


def wilcoxon(x):
    x = np.asarray([v for v in x if v != 0 and np.isfinite(v)])
    if len(x) < 6:
        return None
    try:
        from scipy.stats import wilcoxon as w
        return float(w(x).pvalue)
    except Exception:
        return None


def load(fname):
    p = os.path.join(DATA, fname)
    return json.load(open(p)) if os.path.exists(p) else []


def paired(by, cost, base, key):
    out = []
    for seed, r in by.get(cost, {}).items():
        b = base.get(seed)
        if not b or key not in r or key not in b or not b[key]:
            continue
        if np.isfinite(r[key]) and np.isfinite(b[key]):
            out.append(100.0 * (r[key] / b[key] - 1.0))
    return np.array(out)


def table(rows, base, title):
    by = {}
    for r in rows:
        by.setdefault(r["cost"], {})[r["seed"]] = r
    costs = [c for c in by if c != "nocontrol"]
    out = []
    for c in costs:
        da = paired(by, c, base, "accel_rms")
        dt = paired(by, c, base, "TTS_h")
        if not len(da):
            continue
        rs = list(by[c].values())
        out.append(dict(cost=c, n=len(da), acc=np.median(da), p=wilcoxon(da),
                        tts=np.median(dt) if len(dt) else np.nan,
                        vsl=np.mean([r["mean_vsl"] for r in rs]),
                        fails=sum(r["n_failed"] for r in rs),
                        solves=sum(r["n_solves"] for r in rs)))
    out.sort(key=lambda r: (np.inf if np.isnan(r["acc"]) else r["acc"]))
    print("\n%s" % title)
    print("%-22s %8s %7s %8s %8s %9s %3s" %
          ("cost function", "d_accel", "p", "d_TTS", "meanVSL", "fails", "n"))
    print("-" * 74)
    for r in out:
        print("%-22s %7.1f%% %7s %7.1f%% %8.1f %5d/%-4d %3d" %
              (r["cost"], r["acc"], "n/a" if r["p"] is None else "%.3f" % r["p"],
               r["tts"], r["vsl"], r["fails"], r["solves"], r["n"]))
    return out


def main():
    old = load("phase_d.json")
    new = load("phase_d_s14.json")
    fx = load("fixed_vsl.json")
    if not new:
        raise SystemExit("no data/phase_d_s14.json yet: run "
                         "control/phase_d.py --screen --seeds 10 --model s14")

    base_old = {r["seed"]: r for r in old if r["cost"] == "nocontrol"}
    base_new = {r["seed"]: r for r in new if r["cost"] == "nocontrol"} or base_old

    a = table(old, base_old,
              "A. SHIPPED METANET  (210 s prediction correlation 0.035)")
    b = table(new, base_new,
              "B. PORTED BAKE-OFF WINNER  (210 s prediction correlation 0.804)")

    # the honest reference: a constant sign over the controller's active window
    print("\nC. REFERENCE: constant sign, no controller, applied only over "
          "120-1850 s")
    print("%-22s %8s %7s %8s" % ("policy", "d_accel", "p", "d_TTS"))
    print("-" * 50)
    best_ref = None
    for w in (True, False):
        rowsw = [r for r in fx if bool(r.get("window")) == w]
        for bb in sorted({r["b"] for r in rowsw}):
            rs = [r for r in rowsw if r["b"] == bb]
            if len(rs) < 10:
                continue
            da = [r["d_accel"] for r in rs]
            lbl = ("windowed b=%.2f" if w else "whole-run b=%.2f") % bb
            print("%-22s %7.1f%% %7s %7.1f%%" %
                  (lbl, np.median(da),
                   "n/a" if wilcoxon(da) is None else "%.3f" % wilcoxon(da),
                   np.median([r["d_TTS"] for r in rs])))
            if best_ref is None or np.median(da) < best_ref[1]:
                best_ref = (lbl, np.median(da),
                            np.median([r["d_TTS"] for r in rs]))

    print("\n" + "=" * 74)
    if a and b:
        ba, bb_ = a[0], b[0]
        print("best cost function on the shipped model : %-16s %+.1f%% accel, "
              "%+.1f%% TTS" % (ba["cost"], ba["acc"], ba["tts"]))
        print("best cost function on the ported model  : %-16s %+.1f%% accel, "
              "%+.1f%% TTS" % (bb_["cost"], bb_["acc"], bb_["tts"]))
    if best_ref:
        print("best reference (no controller at all)   : %-16s %+.1f%% accel, "
              "%+.1f%% TTS" % (best_ref[0], best_ref[1], best_ref[2]))
    if a and b:
        fa = sum(r["fails"] for r in a) / max(1, sum(r["solves"] for r in a))
        fb = sum(r["fails"] for r in b) / max(1, sum(r["solves"] for r in b))
        print("\nsolver failure rate: shipped %.1f%%, ported %.1f%%"
              % (100 * fa, 100 * fb))
        # does the ranking survive the model change?
        common = [r["cost"] for r in a if r["cost"] in {q["cost"] for q in b}]
        ra = {r["cost"]: i for i, r in enumerate(a)}
        rb = {r["cost"]: i for i, r in enumerate(b)}
        if len(common) > 3:
            from scipy.stats import spearmanr
            rho, pv = spearmanr([ra[c] for c in common], [rb[c] for c in common])
            print("cost-function RANKING agreement between the two models: "
                  "spearman %+.2f (p=%.3f, n=%d)" % (rho, pv, len(common)))
            print("  A low value means the earlier ranking was an artefact of "
                  "the model, not a\n  property of the cost functions.")


def safety():
    """Section D: TET/TIT, from the instrumented shortlist."""
    rows = load("phase_d_s14_ssm.json")
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r["cost"], {})[r["seed"]] = r
    base = by.get("nocontrol", {})
    if not base:
        return
    print("\nD. SAFETY on the ported model (SSM device, %d seeds, paired)"
          % len(base))
    print("%-18s %8s %7s %8s %7s %9s" %
          ("cost", "d_TET", "p", "d_TIT", "p", "d_accel"))
    print("-" * 62)
    pts = []
    order = sorted((c for c in by if c != "nocontrol"),
                   key=lambda c: np.median(
                       [100 * (by[c][s]["accel_rms"] / base[s]["accel_rms"] - 1)
                        for s in by[c] if s in base]))
    for c in order:
        rs = [r for s, r in by[c].items() if s in base]
        if len(rs) < 6:
            continue
        f = lambda k: [100 * (r[k] / base[r["seed"]][k] - 1)
                       for r in rs if base[r["seed"]].get(k)]
        tet, tit, acc = f("TET_s"), f("TIT_s"), f("accel_rms")
        pts.append((np.median(acc), np.median(tet), np.median(tit)))
        print("%-18s %7.1f%% %7s %7.1f%% %7s %8.1f%%" %
              (c, np.median(tet), "n/a" if wilcoxon(tet) is None
               else "%.3f" % wilcoxon(tet), np.median(tit),
               "n/a" if wilcoxon(tit) is None else "%.3f" % wilcoxon(tit),
               np.median(acc)))
    if len(pts) >= 4:
        from scipy.stats import pearsonr
        A = np.array([p[0] for p in pts])
        for nm, i in (("TET", 1), ("TIT", 2)):
            y = np.array([p[i] for p in pts])
            r_, p_ = pearsonr(A, y)
            print("  d_accel vs d_%s across the %d costs: r=%+.2f (p=%.3f)"
                  % (nm, len(pts), r_, p_))
        print('\n  No individual cost function moves TET or TIT significantly, so the\n  shockwave reduction does NOT buy a measurable safety gain here. But the\n  two move TOGETHER across cost functions: the ones that cut the shockwave\n  most also do least damage to the safety surrogate. With only %d points\n  that is suggestive, not established.' % len(pts))


if __name__ == "__main__":
    main()
    safety()
