# -*- coding: utf-8 -*-
"""
Shockwave detection on the SUMO space-time speed field.

WHY THIS EXISTS
---------------
The research question is which cost function best reduces SHOCKWAVES. Until now
the only outcome measure was `accel_rms`, a vehicle-weighted RMS acceleration.
Decomposed on the uncontrolled baseline it is **93% its SPATIAL term** -- the
speed step a vehicle drives INTO -- and only 7% the temporal term, the
oscillation a vehicle experiences in place.

At a permanently over-saturated merge the spatial term is dominated by vehicles
decelerating into a STANDING queue attached to the bottleneck. A standing queue
is congestion, but it is not a shockwave. A shockwave is a jam of finite extent
that PROPAGATES upstream, detached from the bottleneck that spawned it.

So the headline metric may not measure the thing the thesis is about, and until
now nothing in this project distinguished the two. This module does.

METHOD
------
The wave-boundary approach (Ji et al. 2026, TR-C 182:105385), adopted as panel
member 3 in `docs/research/2026-08-18-shockwave-metrics.md` and never built:
threshold the space-time speed field at a critical speed v_c, label connected
congested regions, and classify each one.

  ATTACHED -- the region touches the bottleneck cells at some time. A standing
              queue.
  DETACHED -- the region never touches them. A moving jam, i.e. a shockwave.

For each detached region we report extent, duration, depth, and the propagation
speed of its upstream front from a least-squares fit of front position on time.
A genuine jam wave propagates UPSTREAM at roughly -15 to -20 km/h; anything
moving downstream, or barely moving, is not one.

v_c is SWEPT, because the count is threshold-dependent and a single value would
be an arbitrary choice presented as a measurement.

Scoring only. Nothing here goes inside the MPC.
"""

import numpy as np

V_C_SWEEP = (20.0, 30.0, 40.0)     # km/h, the pre-registered panel
MIN_CELLS = 8                      # smaller than this is noise, not a jam
MIN_DUR_S = 30.0                   # and it must last at least this long


def _label(mask):
    """Connected components of a boolean (time, cell) grid, 4-connectivity.

    Uses scipy.ndimage.label when available; otherwise a small union-find, so
    this module never becomes the reason an overnight sweep fails to start.
    """
    try:
        from scipy.ndimage import label
        lab, n = label(mask)
        return lab, int(n)
    except Exception:
        pass
    nt, nx = mask.shape
    lab = np.zeros((nt, nx), int)
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for i in range(nt):
        for j in range(nx):
            if not mask[i, j]:
                continue
            up = lab[i - 1, j] if i and mask[i - 1, j] else 0
            lf = lab[i, j - 1] if j and mask[i, j - 1] else 0
            if up and lf:
                lab[i, j] = min(up, lf)
                union(up, lf)
            elif up or lf:
                lab[i, j] = up or lf
            else:
                lab[i, j] = nxt
                parent[nxt] = nxt
                nxt += 1
    remap, out_n = {}, 0
    for i in range(nt):
        for j in range(nx):
            if lab[i, j]:
                r = find(lab[i, j])
                if r not in remap:
                    out_n += 1
                    remap[r] = out_n
                lab[i, j] = remap[r]
    return lab, out_n


def wave_metrics(v, lengths_km, dt_s=1.0, merge_idx=None, n_bneck_cells=2,
                 agg_s=8, v_c_sweep=V_C_SWEEP):
    """Detect and characterise congested regions in a (time, cell) speed field.

    v              speed field in km/h, NaN where the cell is empty
    lengths_km     per-cell length, for extent and propagation speed
    merge_idx      index of the FIRST bottleneck cell (the 1-lane exit). A
                   region is ATTACHED if it touches any cell from
                   `merge_idx - 1` downstream.

                   THIS IS THE SUBTLE PART AND THE FIRST VERSION GOT IT WRONG.
                   The natural-looking test -- "does the region touch the
                   bottleneck cells" -- gives the wrong answer on this geometry,
                   because the 1-lane exit is the DISCHARGE and runs free at
                   ~85 km/h. It is never below v_c, so nothing ever touches it,
                   and every standing queue is misclassified as a moving jam.
                   Measured: that bug reported 96-99% "moving" and ZERO standing
                   queues on an over-saturated merge, which is impossible.
                   The queue head actually sits on the last 2-LANE cell,
                   immediately upstream of the lane drop, so the attachment zone
                   must start one cell before the merge.

    n_bneck_cells  fallback when `merge_idx` is None: that many cells from the
                   downstream end, minus one, form the attachment zone.
    agg_s          time aggregation, matched to the 8 s the acceleration metric
                   uses so both describe the same field

    Returns a flat dict prefixed `wave_`, safe to merge into a results row.
    The single most informative key is `wave_vcNN_frac_moving`: the share of all
    congested cell-time that sits in a DETACHED (moving) region.
    """
    v = np.asarray(v, float)
    if agg_s and agg_s > 1 and len(v) >= agg_s:
        k = (len(v) // int(agg_s)) * int(agg_s)
        with np.errstate(invalid="ignore"):
            v = np.nanmean(v[:k].reshape(-1, int(agg_s), v.shape[1]), axis=1)
        dt_s = float(dt_s) * int(agg_s)

    Lkm = np.asarray(lengths_km, float)
    nx = v.shape[1]
    if merge_idx is None:
        merge_idx = nx - int(n_bneck_cells)
    # start ONE CELL UPSTREAM of the lane drop: that is where the queue head
    # sits, and the 1-lane exit downstream of it never congests
    bneck_from = max(0, int(merge_idx) - 1)
    edges = np.concatenate([[0.0], np.cumsum(Lkm)])
    xc = 0.5 * (edges[:-1] + edges[1:])      # cell centres, km from upstream end

    out = {}
    for v_c in v_c_sweep:
        with np.errstate(invalid="ignore"):
            mask = np.isfinite(v) & (v < v_c)
        lab, n = _label(mask)
        moving, standing, mv_cells = [], [], 0.0
        for k in range(1, n + 1):
            ti, xi = np.nonzero(lab == k)
            if len(ti) < MIN_CELLS:
                continue
            dur = (ti.max() - ti.min() + 1) * dt_s
            if dur < MIN_DUR_S:
                continue
            attached = bool((xi >= bneck_from).any())
            # propagation speed of the UPSTREAM front, by least squares of the
            # most-upstream congested cell against time. Negative = upstream.
            speed = float("nan")
            ts = np.unique(ti)
            if len(ts) >= 3:
                fx = [xc[xi[ti == t].min()] for t in ts]
                speed = float(np.polyfit(ts * dt_s, np.array(fx), 1)[0] * 3600.0)
            rec = dict(dur=float(dur),
                       extent=float(xc[xi.max()] - xc[xi.min()] + Lkm[xi].mean()),
                       speed=speed,
                       depth=float(np.nanmean(v[ti, xi])))
            if attached:
                standing.append(rec)
            else:
                moving.append(rec)
                mv_cells += len(ti)

        tag = "wave_vc%d_" % int(v_c)
        tot = float(mask.sum())
        out[tag + "n_moving"] = len(moving)
        out[tag + "n_standing"] = len(standing)
        out[tag + "frac_moving"] = float(mv_cells / tot) if tot > 0 else float("nan")
        out[tag + "cong_cells"] = tot
        for nm, arr in (("moving", moving), ("standing", standing)):
            for key in ("dur", "extent", "depth"):
                out[tag + nm + "_" + key] = (float(np.mean([r[key] for r in arr]))
                                             if arr else float("nan"))
        sp = [r["speed"] for r in moving if np.isfinite(r["speed"])]
        out[tag + "moving_speed"] = float(np.mean(sp)) if sp else float("nan")

        # ------------------------------------------------------------------
        # THE CONGESTION FRONT. This is the measurement that matters, and the
        # first version of this file threw it away.
        #
        # Classifying regions as attached/detached and then reporting a
        # propagation speed only for the DETACHED ones answers the wrong
        # question. A queue that grows upstream from the bottleneck has an
        # upstream front that PROPAGATES, and that front is a shockwave in the
        # ordinary kinematic-wave sense -- it is the diagonal line you see on a
        # space-time speed plot. It being anchored at the merge does not make it
        # not a wave; it makes it an attached one.
        #
        # So: track the most-upstream congested cell over time, and characterise
        # the excursion. Growth rate is the backward-propagating front; recovery
        # rate is the forward-propagating release wave as the queue drains.
        # A cost function that reduces shockwaves should slow the growth front,
        # shorten its reach, and bring the recovery forward.
        rows_any = np.where(mask.any(axis=1))[0]
        if len(rows_any) >= 3:
            t_idx = rows_any
            front = np.array([xc[np.nonzero(mask[t])[0].min()] for t in t_idx])
            t_s = t_idx * dt_s
            i_peak = int(np.argmin(front))          # furthest upstream = min x
            out[tag + "front_reach"] = float(xc[-1] - front[i_peak])
            out[tag + "front_t_peak"] = float(t_s[i_peak])
            out[tag + "front_dur"] = float(t_s[-1] - t_s[0])
            # growth: onset -> furthest upstream. Negative = travelling upstream.
            if i_peak >= 2:
                out[tag + "front_growth_kmh"] = float(
                    np.polyfit(t_s[:i_peak + 1], front[:i_peak + 1], 1)[0] * 3600.0)
            else:
                out[tag + "front_growth_kmh"] = float("nan")
            # recovery: furthest upstream -> clearance. Positive = releasing.
            if len(t_s) - i_peak >= 3:
                out[tag + "front_recov_kmh"] = float(
                    np.polyfit(t_s[i_peak:], front[i_peak:], 1)[0] * 3600.0)
            else:
                out[tag + "front_recov_kmh"] = float("nan")
        else:
            for k in ("front_reach", "front_t_peak", "front_dur",
                      "front_growth_kmh", "front_recov_kmh"):
                out[tag + k] = float("nan")
    return out


def _slope_kmh(t_s, x_km):
    return float(np.polyfit(np.asarray(t_s, float), np.asarray(x_km, float), 1)[0]
                 * 3600.0)


def jam_metrics(v, cnt, lengths_km, x_dist_km, t_dist_s, dt_s=1.0, agg_s=8,
                v_c_sweep=V_C_SWEEP, seed_win_s=120.0):
    """The jam a known disturbance creates, on a road with no bottleneck.

    `wave_metrics` measures the front of a queue anchored at a merge, and its
    reach is a distance from the downstream END of the road. On the straight
    jam-wave road (sumo/build_straight_net.py) there is no anchor: congestion
    starts where the disturbance is, and the question is whether it grows,
    travels and dies. So everything here is measured relative to the
    disturbance, and the jam is identified by construction rather than by
    classification -- it is the connected congested region(s) touching the
    disturbance cell (+-1 cell) within `seed_win_s` of the disturbance time.

    v, cnt        (time, cell) speed km/h (NaN = empty) and vehicle count
    x_dist_km     disturbance position along the whole road, km
    t_dist_s      disturbance time, s

    Per v_c, prefixed `jam_vcNN_`:
      found       1 if a congested region formed at the disturbance
      dur_s       jam lifetime; t_end_s when its last congested cell cleared
      reach_km    furthest the jam's upstream TAIL got from the disturbance
      tail_kmh    tail speed from onset to furthest reach (negative = upstream)
      head_kmh    speed of the jam's downstream HEAD over its life. For a
                  detached moving jam this is the characteristic jam-wave speed
      area_kmh    space-time area of the jam, km*h
      veh_h       vehicle-hours spent inside the jam: the vehicle-weighted
                  severity, i.e. how much traffic had to drive through it
      hit_entry   1 if the jam reached the first cell of the road (the road was
                  too short to contain it, so reach/area are truncated)
    and, over the whole road from the disturbance onward, `post_area_kmh` and
    `post_veh_h`: ALL congestion, including any not connected to the jam -- a
    policy that creates slow traffic elsewhere is charged for it here.
    """
    v = np.asarray(v, float)
    cnt = np.asarray(cnt, float)
    if agg_s and agg_s > 1 and len(v) >= agg_s:
        k = (len(v) // int(agg_s)) * int(agg_s)
        with np.errstate(invalid="ignore"):
            v = np.nanmean(v[:k].reshape(-1, int(agg_s), v.shape[1]), axis=1)
        cnt = cnt[:k].reshape(-1, int(agg_s), cnt.shape[1]).mean(axis=1)
        dt_s = float(dt_s) * int(agg_s)
    Lkm = np.asarray(lengths_km, float)
    nt, nx = v.shape
    edges = np.concatenate([[0.0], np.cumsum(Lkm)])
    xc = 0.5 * (edges[:-1] + edges[1:])
    j_d = int(np.clip(np.searchsorted(edges, x_dist_km, side="right") - 1, 0, nx - 1))
    i_d = int(t_dist_s // dt_s)
    i_w = min(nt, i_d + int(np.ceil(seed_win_s / dt_s)) + 1)
    dt_h = dt_s / 3600.0
    nan = float("nan")

    out = {}
    for v_c in v_c_sweep:
        tag = "jam_vc%d_" % int(v_c)
        with np.errstate(invalid="ignore"):
            mask = np.isfinite(v) & (v < v_c)
        post = mask.copy()
        post[:i_d] = False
        out[tag + "post_area_kmh"] = float((post * Lkm).sum() * dt_h)
        out[tag + "post_veh_h"] = float((post * cnt).sum() * dt_h)
        lab, n_lab = _label(mask)
        # SEED THE JAM BY WHERE IT STARTS, NOT BY THE TRIGGER POINT.
        # The first version took components touching the trigger cell +-1 within
        # `seed_win_s`. That assumes the jam forms where the disturbance begins,
        # which is only true for an ABRUPT disturbance. A vehicle easing to its
        # target speed over 90 s travels 1-2 km first, so the jam forms well
        # downstream of the trigger and the old rule found nothing at all --
        # reporting 0.00 veh-h for a run whose cells sat at 20-27 km/h.
        # A component belongs to the disturbance if it FIRST appears after the
        # trigger time, at or downstream of the trigger cell. Congestion that
        # starts upstream of the trigger is not the planted jam (it would be the
        # controller's own doing) and is still counted in `post_*`.
        seeds = []
        for lb in range(1, n_lab + 1):
            ti_l, xi_l = np.nonzero(lab == lb)
            t_first = ti_l.min()
            if not (i_d <= t_first < i_w):
                continue
            if xi_l[ti_l == t_first].max() >= j_d - 1:
                seeds.append(lb)
        seeds = np.array(seeds, dtype=int)
        if len(seeds) == 0:
            out.update({tag + "found": 0.0, tag + "dur_s": 0.0,
                        tag + "t_end_s": nan, tag + "reach_km": 0.0,
                        tag + "tail_kmh": nan, tag + "head_kmh": nan,
                        tag + "area_kmh": 0.0, tag + "veh_h": 0.0,
                        tag + "hit_entry": 0.0})
            continue
        ti, xi = np.nonzero(np.isin(lab, seeds))
        ts = np.unique(ti)
        tail = np.array([xi[ti == t].min() for t in ts])
        head = np.array([xi[ti == t].max() for t in ts])
        t_sec = ts * dt_s
        i_pk = int(np.argmin(tail))
        out[tag + "found"] = 1.0
        out[tag + "dur_s"] = float((ti.max() - ti.min() + 1) * dt_s)
        out[tag + "t_end_s"] = float((ti.max() + 1) * dt_s)
        out[tag + "reach_km"] = float(max(0.0, xc[j_d] - xc[tail.min()]))
        out[tag + "tail_kmh"] = (_slope_kmh(t_sec[:i_pk + 1], xc[tail[:i_pk + 1]])
                                 if i_pk >= 2 else nan)
        out[tag + "head_kmh"] = (_slope_kmh(t_sec, xc[head])
                                 if len(ts) >= 3 else nan)
        out[tag + "area_kmh"] = float(Lkm[xi].sum() * dt_h)
        out[tag + "veh_h"] = float(cnt[ti, xi].sum() * dt_h)
        out[tag + "hit_entry"] = float(xi.min() == 0)
    return out
