# -*- coding: utf-8 -*-
"""
Validate the cell-based shockwave metrics against VEHICLE TRAJECTORIES and
against KINEMATIC-WAVE THEORY.

WHY
---
`validation/wave_metrics.py` finds the congestion front on a 200 m x 8 s grid
of edge-mean speeds. It was verified on planted synthetic waves, which shows the
arithmetic is right, but not that the grid measures the front real vehicles
experience. No published speed-threshold wave detector has been validated
against ground truth either (Ji et al. 2026 state they make no such claim), so
this has to be done here, and four independent routes are used:

  1. TRAJECTORIES. SUMO floating-car data gives every vehicle's position and
     speed each second. The jam tail at time t is the most upstream vehicle
     below v_c that has at least N_MIN-1 more slow vehicles within WIN_M
     downstream of it (so one dawdling driver is not a jam). This is the front
     as vehicles experience it, at no spatial resolution limit, and it shares
     nothing with the detector but the threshold.
  2. RESOLUTION. The same FCD re-binned at 50, 100 and 200 m. If the metric is
     an artefact of 200 m cells, reach and propagation speed will move with the
     bin size.
  3. RANKINE-HUGONIOT. A front between an upstream state (q_u, k_u) and a queue
     state (q_q, k_q) must travel at w = (q_u - q_q) / (k_u - k_q). The flows and
     densities either side of the detected front are measured independently of
     where the detector puts the front, so agreement between w and the
     detector's growth/recovery speed is a physics check on its kinematics.
  4. DENSITY vs SPEED. The detector thresholds SPEED. A controller holding
     dense traffic just above v_c would shorten the speed-front without
     shortening the queue. The same front computed on a DENSITY threshold
     exposes that, per policy.
Plus the sign check: the share of detected congestion on controlled cells that
sits under a posted limit below v_c + 20 km/h, i.e. that the sign itself could
have produced.

Scenarios:
  --scenario long      the merge road every headline result is measured on;
                       no control, constant b=0.95, `tts`, and `accel_dens_tts`
                       (the policy with the LONGEST front and most slowing)
  --scenario straight  the jam-wave road; also checks jam_metrics (tail and
                       HEAD speed, reach from the disturbance) the same way

Rows go to data/wave_validation.json (single writer, the parent). FCD files go
to %TEMP% and are deleted after parsing.

Usage:
    python validation/validate_wave_trajectories.py --scenario long --seeds 3
    python validation/validate_wave_trajectories.py --scenario straight \
        --demand 3200 --v 0 --hold 60 --seeds 3
    python validation/validate_wave_trajectories.py --plot
"""

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sumo"))

SCRATCH = os.path.join(os.environ.get("TEMP", _ROOT), "egh400_wave_validation")
OUT = os.path.join(_ROOT, "data", "wave_validation.json")
FIG = os.path.join(_ROOT, "docs", "figures", "wave_validation.png")
VC = 30.0
AGG = 8
N_MIN = 3
WIN_M = 200.0
BINS_M = (50.0, 100.0, 200.0)
K_C = (50.0, 70.0)          # density thresholds, veh/km/lane
LONG_POLICIES = [("nocontrol", None, None), ("constant b=0.95", None, 0.95),
                 ("tts", "tts", None), ("accel_dens_tts", "accel_dens_tts", None)]
STRAIGHT_POLICIES = [("nocontrol", None, None), ("tts", "tts", None)]

_T = re.compile(r'<timestep time="([0-9.]+)"')
_V = re.compile(r'x="([-0-9.]+)" speed="([-0-9.]+)"')
_G = {}
nan = float("nan")


# ---------------------------------------------------------------------------
def _init(scenario, demand):
    from model.metanet_model import (PARAMS_BM, MetanetModel, MetanetParams,
                                     load_geometry, model_geometry,
                                     straight_model_geometry)
    mp = MetanetParams(**PARAMS_BM, vsl_scaled=True)
    if scenario == "long":
        geom = load_geometry(os.path.join(_ROOT, "sumo", "net", "plain.net3.xml"))
        sub, off = model_geometry(geom, n_ctrl=9)
        model = MetanetModel(sub, mp)
        _G.update(cfg=os.path.join(_ROOT, "sumo", "simulation3.sumocfg"),
                  steps=3500, dend=None)
    else:
        import build_straight_net as bsn
        geom = load_geometry(os.path.join(_ROOT, "sumo", "net", "plain.net4.xml"),
                             allow_straight=True)
        sub, off = straight_model_geometry(geom, bsn.N_UP)
        model = MetanetModel(sub, mp, n_ctrl=bsn.N_CTRL)
        _G.update(cfg=bsn.cfg_path(demand), steps=2700, dend=bsn.DEMAND_END_S)
    _G.update(geom=geom, model=model, off=off)


def parse_fcd(path):
    T, X, S = [], [], []
    t = 0.0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "<vehicle" in line:
                m = _V.search(line)
                if m:
                    T.append(t)
                    X.append(float(m.group(1)))
                    S.append(float(m.group(2)) * 3.6)
            elif "<timestep" in line:
                t = float(_T.search(line).group(1))
    return np.array(T), np.array(X), np.array(S)


def grid_speed(T, X, S, bin_m, road_m, n_rows):
    nb = int(np.ceil(road_m / bin_m - 1e-9))
    r = (T // AGG).astype(int)
    k = np.minimum((X // bin_m).astype(int), nb - 1)
    keep = (r < n_rows) & (k >= 0)
    sv = np.zeros((n_rows, nb))
    cn = np.zeros((n_rows, nb))
    np.add.at(sv, (r[keep], k[keep]), S[keep])
    np.add.at(cn, (r[keep], k[keep]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(cn > 0, sv / cn, np.nan)
    lengths = np.diff(np.minimum(np.arange(nb + 1) * bin_m, road_m)) / 1000.0
    return v, cn / AGG, lengths


def traj_front(T, X, S, n_rows, t_from=0.0):
    """Per 8 s row: most upstream (tail) and most downstream (head) position of
    a cluster of >= N_MIN slow vehicles within WIN_M. km."""
    slow = (S < VC) & (T >= t_from)
    Ts, Xs = T[slow], X[slow]
    o = np.argsort(Ts, kind="stable")
    Ts, Xs = Ts[o], Xs[o]
    tail = np.full(n_rows, np.nan)
    head = np.full(n_rows, np.nan)
    if not len(Ts):
        return tail, head
    ut, idx = np.unique(Ts, return_index=True)
    ends = list(idx[1:]) + [len(Ts)]
    for t, a0, a1 in zip(ut, idx, ends):
        r = int(t // AGG)
        if r >= n_rows or a1 - a0 < N_MIN:
            continue
        a = np.sort(Xs[a0:a1])
        n = np.arange(len(a))
        ok = (np.searchsorted(a, a + WIN_M, side="right") - n) >= N_MIN
        if ok.any():
            tail[r] = a[ok].min() if np.isnan(tail[r]) else min(tail[r], a[ok].min())
        ok2 = (n - np.searchsorted(a, a - WIN_M, side="left") + 1) >= N_MIN
        if ok2.any():
            head[r] = a[ok2].max() if np.isnan(head[r]) else max(head[r], a[ok2].max())
    return tail / 1000.0, head / 1000.0


def _slope(t, x):
    return float(np.polyfit(t, x, 1)[0] * 3600.0)


def front_stats(x_km, x_ref_km):
    """Same definitions as wave_metrics' congestion front."""
    t_all = np.arange(len(x_km)) * float(AGG)
    ok = np.isfinite(x_km)
    t, x = t_all[ok], x_km[ok]
    if len(t) < 3:
        return dict(reach=nan, t_peak=nan, growth=nan, recov=nan)
    i = int(np.argmin(x))
    return dict(reach=float(x_ref_km - x[i]), t_peak=float(t[i]),
                growth=_slope(t[:i + 1], x[:i + 1]) if i >= 2 else nan,
                recov=_slope(t[i:], x[i:]) if len(t) - i >= 3 else nan)


def grid_front(mask, lengths_km):
    e = np.concatenate([[0.0], np.cumsum(lengths_km)])
    xc = 0.5 * (e[:-1] + e[1:])
    f = np.full(len(mask), np.nan)
    fi = np.full(len(mask), -1)
    for r in np.where(mask.any(axis=1))[0]:
        fi[r] = int(np.nonzero(mask[r])[0].min())
        f[r] = xc[fi[r]]
    return f, fi, xc


def agg(x, how="mean"):
    x = np.asarray(x, float)
    k = (len(x) // AGG) * AGG
    with np.errstate(invalid="ignore"):
        return np.nanmean(x[:k].reshape(-1, AGG, x.shape[1]), axis=1)


def rh_speeds(v, k, fi, rows, lanes):
    """Rankine-Hugoniot speed of the detected front, per row, km/h."""
    out = []
    nx = v.shape[1]
    for r in rows:
        f = fi[r]
        if f < 3 or f + 3 > nx:
            continue
        up, qd = slice(f - 3, f - 1), slice(f + 1, f + 3)
        if np.any(lanes[f - 3:f + 3] != lanes[f]):
            continue
        ku, vu = np.nanmean(k[r, up]), np.nanmean(v[r, up])
        kq, vq = np.nanmean(k[r, qd]), np.nanmean(v[r, qd])
        if not np.all(np.isfinite([ku, vu, kq, vq])) or kq - ku < 10.0:
            continue
        out.append((ku * vu - kq * vq) / (ku - kq))
    return out


# ---------------------------------------------------------------------------
def _run(job):
    from control.phase_d import run_closed_loop
    from validation.wave_metrics import jam_metrics
    scenario, label, cost, b, seed, demand, dist = job
    geom, off = _G["geom"], _G["off"]
    tag = "val_%s_%s_s%d" % (scenario, re.sub(r"[^A-Za-z0-9]", "", label), seed)
    os.makedirs(SCRATCH, exist_ok=True)
    fcd = os.path.join(SCRATCH, tag + ".fcd.xml")
    t0 = time.time()
    try:
        r = run_closed_loop(cost, seed, steps=_G["steps"], model=_G["model"],
                            geom=geom, offset=off, config=_G["cfg"],
                            max_wall=15.0 if cost else None, fixed_b=b,
                            return_fields=True, fcd_file=fcd, tag=tag,
                            demand_end_s=_G["dend"], disturbance=dist)
        T, X, S = parse_fcd(fcd)
    finally:
        try:
            os.remove(fcd)
        except OSError:
            pass

    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    road_m = float(np.sum(geom.lengths_m))
    v200 = agg(np.array(r.pop("_field_v"), float))
    cnt200 = agg(np.array(r.pop("_field_cnt"), float))
    vsl = np.array(r.pop("_field_vsl"), float)
    r.pop("_field_merge", None)
    n_rows = v200.shape[0]
    k200 = cnt200 / (Lkm * lanes)
    e200 = np.concatenate([[0.0], np.cumsum(Lkm)])
    x_ref = float(0.5 * (e200[-2] + e200[-1]))      # == wave_metrics xc[-1]

    rec = dict(scenario=scenario, policy=label, seed=seed, demand=demand,
               accel_rms=r["accel_rms"], TTS_road_h=r["TTS_road_h"],
               throughput=r["throughput"], veh_km=r.get("veh_km", nan),
               mean_vsl=r["mean_vsl"], min_vsl=r["min_vsl"],
               det_reach=r["wave_vc30_front_reach"],
               det_growth=r["wave_vc30_front_growth_kmh"],
               det_recov=r["wave_vc30_front_recov_kmh"],
               n_failed=r.get("n_failed"), n_solves=r.get("n_solves"))

    # internal consistency: re-deriving the 200 m front here must reproduce
    # the stored metric exactly, or the comparisons below are not like-for-like
    f200, fi200, xc200 = grid_front(np.isfinite(v200) & (v200 < VC), Lkm)
    s = front_stats(f200, x_ref)
    rec["recomputed_reach_err"] = abs(s["reach"] - rec["det_reach"])

    # 1 + 2: trajectories and re-binned FCD
    t_from = float(dist["t"]) if dist else 0.0
    tail, head = traj_front(T, X, S, n_rows, t_from=0.0)
    for k_, v_ in front_stats(tail, x_ref).items():
        rec["traj_" + k_] = v_
    fronts = {"det200": f200, "traj": tail}
    for bm in BINS_M:
        vg, cg, lg = grid_speed(T, X, S, bm, road_m, n_rows)
        fg, _, _ = grid_front(np.isfinite(vg) & (vg < VC), lg)
        for k_, v_ in front_stats(fg, x_ref).items():
            rec["fcd%d_%s" % (bm, k_)] = v_
        fronts["fcd%d" % bm] = fg
        if dist and bm == 50.0:
            jm = jam_metrics(vg, cg, lg, r["dist_x_km"], dist["t"], dt_s=AGG,
                             agg_s=1, v_c_sweep=(VC,))
            for k_ in ("reach_km", "tail_kmh", "head_kmh", "dur_s", "veh_h"):
                rec["fcd50_jam_" + k_] = jm["jam_vc30_" + k_]

    # 3: Rankine-Hugoniot on the TraCI 200 m field
    rows = np.where(fi200 >= 0)[0]
    if len(rows) >= 3:
        pk = rows[int(np.argmin(f200[rows]))]
        g = rh_speeds(v200, k200, fi200, rows[rows <= pk], lanes)
        c = rh_speeds(v200, k200, fi200, rows[rows >= pk], lanes)
        rec.update(rh_growth=float(np.median(g)) if g else nan, rh_growth_n=len(g),
                   rh_recov=float(np.median(c)) if c else nan, rh_recov_n=len(c))

    # 4: density-threshold front
    for kc in K_C:
        fk, _, _ = grid_front(np.isfinite(k200) & (k200 > kc), Lkm)
        for k_, v_ in front_stats(fk, x_ref).items():
            rec["dens%d_%s" % (kc, k_)] = v_

    # sign check: congested controlled cell-time under a limit < v_c + 20
    n_ctrl = vsl.shape[1]
    vsl8 = agg(vsl)[:n_rows]
    cm = (np.isfinite(v200) & (v200 < VC))[:len(vsl8), off:off + n_ctrl]
    rec["cong_ctrl_cells"] = int(cm.sum())
    rec["cong_under_low_sign_frac"] = (float((cm & (vsl8 < VC + 20.0)).sum()
                                             / cm.sum()) if cm.sum() else 0.0)
    # What matters for REACH is not congestion anywhere under a low sign, but
    # whether the FRONT itself -- the cell that sets reach -- sits under one.
    rr = [q for q in np.where(fi200 >= 0)[0] if q < len(vsl8)]
    in_ctrl = [q for q in rr if off <= fi200[q] < off + n_ctrl]
    low = [q for q in in_ctrl if vsl8[q, fi200[q] - off] < VC + 20.0]
    rec["front_rows"] = len(rr)
    rec["front_in_ctrl_frac"] = len(in_ctrl) / len(rr) if rr else 0.0
    rec["front_under_low_sign_frac"] = len(low) / len(rr) if rr else 0.0
    if rr:
        pk_row = rr[int(np.argmin([f200[q] for q in rr]))]
        rec["peak_front_cell"] = int(fi200[pk_row])
        rec["peak_front_in_ctrl"] = bool(off <= fi200[pk_row] < off + n_ctrl)
        rec["ctrl_first_cell"] = int(off)

    # straight road: jam metrics against trajectories
    if dist:
        jt, jh = traj_front(T, X, S, n_rows, t_from=t_from)
        ok = np.where(np.isfinite(jt))[0]
        for k_ in ("found", "reach_km", "tail_kmh", "head_kmh", "dur_s", "veh_h",
                   "t_end_s", "hit_entry"):
            rec["det_jam_" + k_] = r.get("jam_vc30_" + k_, nan)
        if len(ok) >= 3:
            t8 = ok * float(AGG)
            i = int(np.argmin(jt[ok]))
            rec.update(traj_jam_reach_km=float(max(0.0, r["dist_x_km"] - jt[ok][i])),
                       traj_jam_tail_kmh=_slope(t8[:i + 1], jt[ok][:i + 1]) if i >= 2 else nan,
                       traj_jam_head_kmh=_slope(t8, jh[ok]),
                       traj_jam_dur_s=float(t8[-1] - t8[0] + AGG))
        rec["dist_n_veh"] = r.get("dist_n_veh")

    np.savez_compressed(os.path.join(SCRATCH, tag + ".npz"), v200=v200,
                        x_edges=e200, **{k_: v_ for k_, v_ in fronts.items()},
                        head=head)
    rec["tag"] = tag
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
def summarise(rows):
    print("\n%-16s %-4s | %-19s | %-19s | %-19s | %s" % (
        "policy", "seed", "reach km det/traj/50m", "growth km/h det/traj/RH",
        "recov km/h det/traj/RH", "dens70 reach | low-sign%"))
    for r in sorted(rows, key=lambda r: (r["scenario"], r["policy"], r["seed"])):
        print("%-16s %-4d | %5.2f %5.2f %5.2f   | %5.1f %5.1f %5.1f   | %5.1f %5.1f %5.1f   | %5.2f | %4.1f  (recomp err %.0e)"
              % (r["policy"][:16], r["seed"], r["det_reach"], r["traj_reach"],
                 r["fcd50_reach"], r["det_growth"], r["traj_growth"],
                 r.get("rh_growth", nan), r["det_recov"], r["traj_recov"],
                 r.get("rh_recov", nan), r["dens70_reach"],
                 100 * r["cong_under_low_sign_frac"], r["recomputed_reach_err"]))
        if "det_jam_found" in r:
            print("   jam: reach det %.2f / traj %.2f / fcd50 %.2f km | tail det %.1f / traj %.1f | head det %.1f / traj %.1f / fcd50 %.1f km/h | dur det %.0f / traj %.0f s"
                  % (r["det_jam_reach_km"], r.get("traj_jam_reach_km", nan),
                     r.get("fcd50_jam_reach_km", nan), r["det_jam_tail_kmh"],
                     r.get("traj_jam_tail_kmh", nan), r["det_jam_head_kmh"],
                     r.get("traj_jam_head_kmh", nan), r.get("fcd50_jam_head_kmh", nan),
                     r["det_jam_dur_s"], r.get("traj_jam_dur_s", nan)))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("val_long_nocontrol_s1", "Merge road, no control (seed 1)"),
              ("val_long_acceldenstts_s1", "Merge road, accel_dens_tts (seed 1)")]
    extra = sorted(f[:-4] for f in os.listdir(SCRATCH)
                   if f.startswith("val_straight_nocontrol_s1") and f.endswith(".npz"))
    if extra:
        panels.append((extra[0], "Straight road, no control (seed 1)"))
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6),
                            constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, (tag, title) in zip(axs, panels):
        p = os.path.join(SCRATCH, tag + ".npz")
        if not os.path.exists(p):
            ax.set_title(title + " (missing)")
            continue
        d = np.load(p)
        v = d["v200"]
        t_min = np.arange(len(v)) * AGG / 60.0
        # cell EDGES in both directions (flat shading), row starts for the lines
        im = ax.pcolormesh(np.arange(len(v) + 1) * AGG / 60.0, d["x_edges"], v.T,
                           cmap="RdYlGn", vmin=0, vmax=100, shading="flat")
        ax.plot(t_min, d["det200"], color="k", lw=1.6, label="detector, 200 m cells")
        ax.plot(t_min, d["fcd50"], color="#2a78d6", lw=1.0, label="detector, 50 m bins")
        ax.plot(t_min, d["traj"], ".", ms=2.2, color="#8a2be2",
                label="trajectories (>=3 slow veh / 200 m)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("distance along road (km)")
    axs[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.colorbar(im, ax=axs, label="speed (km/h)", shrink=0.85)
    fig.savefig(FIG, dpi=160)
    print("wrote", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=("long", "straight"), default="long")
    ap.add_argument("--seeds", type=int, default=3)
    # straight defaults = the round-11 disturbance (control/sweep.STRAIGHT_DIST):
    # full-section blockade ("edge"), 180 s, at the near-capacity demand
    ap.add_argument("--demand", type=int, default=3500)
    ap.add_argument("--v", type=float, default=0.0)
    ap.add_argument("--hold", type=int, default=180)
    ap.add_argument("--mode", choices=("vehicle", "edge"), default="edge")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot()
        return
    dist = None
    pols = LONG_POLICIES
    if a.scenario == "straight":
        dist = dict(t=600, edge="d1", pos=100.0, v_kmh=a.v, hold_s=a.hold,
                    mode=a.mode)
        pols = STRAIGHT_POLICIES
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["scenario"], r["policy"], r["seed"], r.get("demand")) for r in rows}
    dem = a.demand if a.scenario == "straight" else None
    jobs = [(a.scenario, lab, c, b, s, dem, dist)
            for s in range(1, a.seeds + 1) for lab, c, b in pols
            if (a.scenario, lab, s, dem) not in done]
    print("%d jobs" % len(jobs), flush=True)
    new = []
    with Pool(a.procs, initializer=_init, initargs=(a.scenario, dem)) as pool:
        for rec in pool.imap_unordered(_run, jobs):
            rows.append(rec)
            new.append(rec)
            tmp = OUT + ".tmp"
            json.dump(rows, open(tmp, "w"), indent=1)
            os.replace(tmp, OUT)
            print("  done %s  [%.0fs]" % (rec["tag"], rec["wall_s"]), flush=True)
    summarise([r for r in rows if r["scenario"] == a.scenario])


if __name__ == "__main__":
    main()
