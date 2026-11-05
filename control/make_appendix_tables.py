# -*- coding: utf-8 -*-
"""
Appendix A of the final report: every cost function on both roads, plus the
matched weights. Writes docs/final_report/appendix_a.tex (LaTeX, in blue as
the user's convention for new report text).

Columns, all vs no control on the same 10 seeds (analyse_final.paired):
  congestion   change in road-time below 30 km/h (wave_vc30_cong_cells), %
  p            Wilcoxon signed-rank p for the congestion change
  time         mean change in total time per vehicle incl. entrance queue, s
  sharpest     change in the sharpest speed drop between segments (max_jump), %
  mean limit   mean posted limit over the signs, km/h (summary.json `sign`)
  failures     solver failures, % of solves (summary.json `fail`)

usage: python control/make_appendix_tables.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "control"))
import analyse_final as af                                        # noqa: E402
from control.analyse_deploy import spv                            # noqa: E402
from control.sweep import FINAL_W_MERGE, FINAL_W_STRAIGHT         # noqa: E402

OUT = os.path.join(ROOT, "docs", "final_report", "appendix_a.tex")
SUMMARY = os.path.join(ROOT, "data", "sweep_final", "summary.json")

NAMES = {
    "tts": "TTS", "delay": "Delay", "terminal": "Terminal", "jam": "Jam",
    "accel": "Acceleration", "accel_temporal": "Temporal accel.",
    "accel_dens": "Density-weighted accel.", "dens_rate": "Density rate",
    "upstream_queue": "Upstream congestion", "speed_var": "Speed variance",
    "speed_grad": "Speed drop", "conflict_risk": "Conflict risk",
    "density_sq": "Squared density", "density_track": "Density tracking",
    "queue": "Entrance queue", "dvsl": "Control variation",
    "bneck_density": "Bottleneck density", "bneck_flow": "Bottleneck discharge",
}
MULT = {0.25: r"$\times\frac{1}{4}$", 4.0: r"$\times4$"}
# fixed-weight forms (literature, preliminary design, no-mobility demos) are
# named explicitly: matching their weights against W would mislabel e.g.
# density_track:1 as "x1/4" on the straight road, where W = 4
EXPLICIT = {
    "tts": "TTS only",
    "delay:1": "Delay only",
    "jam:1": "Jam only",
    "accel:1": "Acceleration only",
    "tts:1|dvsl:0.1": "TTS + Control var.\\ (standard)",
    "speed_var:1|tts:1|dvsl:0.1": "Speed var.\\ + TTS + Control var.",
    "density_track:1|dvsl:0.1": "Density tracking + Control var.",
    "accel:1|tts:1|queue:0.5|dvsl:0.1": "Accel.\\ + TTS + Queue + Control var.",
    "dens_rate:1|tts:1": "TTS + Density rate ($w=1$)",
}


def label(policy, W):
    if policy == "RULE":
        return r"\textit{Reactive rule}"
    if policy in EXPLICIT:
        return EXPLICIT[policy]
    parts = []
    for p in policy.split("|"):
        t, _, w = p.partition(":")
        w = float(w) if w else 1.0
        name = NAMES[t]
        if t == "tts" and w == 1.0:
            parts.append(name)
            continue
        tag = None
        for m in (1.0, 0.25, 4.0):
            if t in W and abs(w - W[t] * m) <= 1e-9 * max(1.0, abs(W[t] * m)):
                tag = "" if m == 1.0 else " (%s)" % MULT[m]
                break
        if tag is None:
            tag = " ($w=%g$)" % w
        parts.append(name + tag)
    return " + ".join(parts)


def fmt(x, nd=0, sign=True):
    if round(x, nd) == 0:
        return "$0$"                      # never print "-0" / "+0"
    s = ("%+." if sign else "%.") + "%df" % nd
    return "$" + (s % x) + "$"


def table(P, S, exp, d, W, title, lab):
    nc = (exp, "NOCONTROL", d)
    road = "merge" if exp == "final_merge" else "straight"
    summ = {r["policy"]: r for r in S if r["road"] == road and r["demand"] == d}
    rows = []
    for pol in sorted({k[1] for k in P if k[0] == exp and k[2] == d}):
        if pol == "NOCONTROL":
            continue
        a = (exp, pol, d)
        c = af.paired(P, a, nc, "wave_vc30_cong_cells")
        t = af.paired(P, a, nc, spv)
        m = af.paired(P, a, nc, "max_jump")
        s = summ.get(pol, {})
        rows.append((c["pct"], label(pol, W), c["pct"], c["p"], t["dmean"],
                     m["pct"], s.get("sign"), s.get("fail")))
    rows.sort(key=lambda r: (r[1].startswith(r"\textit"), r[0]))
    # two-line headers and a tighter column gap: one-line headers ran
    # ~110 pt past an A4 text width at 12 pt (checked by test compile)
    out = [r"\begin{table}[p]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\caption{%s}" % title, r"\label{%s}" % lab,
           r"\begin{tabular}{l r r r r r r}", r"\hline",
           r" & & & \textbf{Time} & \textbf{Sharpest} & \textbf{Mean} & "
           r"\textbf{Solver} \\",
           r"\textbf{Objective} & \textbf{Congestion} & \textbf{$p$} & "
           r"\textbf{(s/veh)} & \textbf{drop} & \textbf{limit} & "
           r"\textbf{failures} \\", r"\hline"]
    for _, name, cp, p, tm, mj, sign, fail in rows:
        if name.startswith(r"\textit"):
            out.append(r"\hline")
        out.append("%s & %s\\%% & %.3f & %s & %s\\%% & %s & %s \\\\" % (
            name, fmt(cp), p, fmt(tm), fmt(mj),
            "%.1f" % sign if sign is not None else "--",
            ("%.0f\\%%" % fail) if fail is not None else "--"))
    out += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def weights_table():
    keys = [k for k in NAMES if k != "tts"]
    out = [r"\begin{table}[h]", r"\centering", r"\footnotesize",
           r"\caption{Matched weight of each term on each road (TTS has weight 1).}",
           r"\label{tab:matched_weights}",
           r"\begin{tabular}{l r r}", r"\hline",
           r"\textbf{Term} & \textbf{Straight road} & \textbf{Merge road} \\",
           r"\hline"]
    for k in keys:
        if k == "delay":
            continue
        ws = FINAL_W_STRAIGHT.get(k)
        wm = FINAL_W_MERGE.get(k)
        if ws is None and wm is None:
            continue
        out.append("%s & %s & %s \\\\" % (
            NAMES[k], ("$%g$" % ws) if ws is not None else "--",
            ("$%g$" % wm) if wm is not None else "--"))
    out += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main():
    P, _ = af.load()
    S = json.load(open(SUMMARY))
    parts = [r"\section{Full Cost Function Results}", r"{\color{blue}",
             r"Tables~\ref{tab:full_merge}--\ref{tab:full_4800} list every "
             r"cost function, sorted by the change in congestion. All values "
             r"are changes relative to no control on the same 10 seeds, with "
             r"the realistic sign. $p$ is the Wilcoxon signed-rank $p$-value "
             r"for the congestion change; mean limit is the average posted "
             r"limit (km/h); failures is the percentage of solves in which "
             r"the solver did not converge. Weights are shown relative to "
             r"each term's matched weight (Table~\ref{tab:matched_weights}); "
             r"$w$ marks a fixed weight.", "",
             weights_table(),
             table(P, S, "final_merge", None, FINAL_W_MERGE,
                   "Merge road: every cost function relative to no control.",
                   "tab:full_merge"),
             table(P, S, "final_straight", 4600, FINAL_W_STRAIGHT,
                   "Straight road, 4600 veh/h: every cost function relative "
                   "to no control.", "tab:full_4600"),
             table(P, S, "final_straight", 4800, FINAL_W_STRAIGHT,
                   "Straight road, 4800 veh/h: every cost function relative "
                   "to no control.", "tab:full_4800"),
             "}"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
