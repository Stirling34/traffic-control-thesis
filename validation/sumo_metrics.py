# -*- coding: utf-8 -*-
"""
Phase E scoring: the four metrics a cost function is judged on, all computed
from SUMO output rather than from the prediction model.

    mobility    Total Time Spent, including the origin queue
                Hegyi, De Schutter & Hellendoorn (2005), TR-C 13:185-209, eq. 13
    shockwave   vehicle-weighted acceleration RMS (see control_authority.py)
                Zegeye et al. (2013), TR-C 31:158-171, eqs. 16-19
    safety      TIT and TET from time-to-collision, threshold 3.0 s
                Minderhoud & Bovy (2001), Accid. Anal. Prev. 33(1):89-97
    emissions   CO2 via HBEFA3/PC_G_EU4, which is acceleration-sensitive

TWO CORRECTIONS THAT ARE NOT IN THE SUMO DOCUMENTATION. Both were measured, and
without them the safety numbers are wrong rather than merely noisy.

1. THE SSM DEVICE LOGS EVERY FOLLOWING ENCOUNTER TWICE. Each pair appears once
   as encounter type 2 (ego is the follower) and again as type 3 (ego is the
   leader): the same interaction seen from both vehicles. Counting both doubles
   TET and TIT. We keep type 2 only, which has the side benefit that
   `egoVelocity` is then the FOLLOWER speed, which the next correction needs.

2. MOST OF THE RAW SAFETY SIGNAL IS THE STATIONARY QUEUE. Roughly 71% of TET
   and 81% of TIT comes from vehicles below 5 m/s sitting nose to tail at
   sub-metre gaps. Their time-to-collision is tiny but nothing is happening.
   Left in, the safety metric becomes a restatement of delay and reports the
   same effect twice. We require the follower to be doing at least V_MIN_MS.

Usage:
    from validation.sumo_metrics import run_scored
    m = run_scored(vsl_kmh=None, seed=1)      # None = no control
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import MS_TO_KMH, load_geometry          # noqa: E402
from validation.control_authority import shockwave_metrics        # noqa: E402

SUMO_DIR = os.path.join(_ROOT, "sumo")
CONFIG = os.path.join(SUMO_DIR, "simulation2.sumocfg")
OUT = os.path.join(SUMO_DIR, "out")

SIM_STEPS = 2500
TTC_STAR = 3.0        # s. Minderhoud & Bovy, and SUMO's own SSM default
V_MIN_MS = 5.0        # follower speed floor, see correction 2
FOLLOWER_TYPE = "2"   # ego is the follower, see correction 1
# DRAC: deceleration rate to avoid a crash. 3.35 m/s^2 is the conventional
# threshold (Cooper & Ferguson 1976; used throughout the SSM literature).
#
# WHY IT MATTERS HERE. TET/TIT are built on TTC = gap/dv, which diverges as
# dv -> 0, so a stopped queue at minGap manufactures "critical" TTCs that are
# not conflicts. The V_MIN_MS filter exists to remove those. But the filter is
# itself policy-dependent: a controller that keeps traffic MOVING promotes
# queue-seconds (excluded) into moving-seconds (counted), so counted TET can
# rise even when total conflicts FALL. Measured on `tts`: counted TET +19.0%
# while total conflict-seconds went -12.2%. The two readings disagree, and the
# disagreement is a property of the metric, not of the control.
#
# DRAC ~ dv^2 / (2*gap) does not diverge at dv -> 0 -- it goes to zero -- so it
# needs no speed filter and is immune to that bias. It is the tie-breaker.
DRAC_STAR = 3.35      # m/s^2


def _ssm_args(tag, out_dir=None):
    """SUMO arguments for the surrogate-safety device.

    `out_dir` lets a caller put the ~90 MB SSM file somewhere other than
    sumo/out. That matters because this repository sits inside a OneDrive
    folder, so a sweep of instrumented runs otherwise pushes tens of GB of
    short-lived scratch through the sync engine. See control/phase_d.SCRATCH.
    """
    out_dir = out_dir or OUT
    return _ssm_args_in(tag, out_dir)


def _ssm_args_in(tag, out_dir):
    return ["--device.ssm.probability", "1.0",
            "--device.ssm.deterministic",
            "--device.ssm.measures", "TTC DRAC",
            # trajectories is REQUIRED: without it SUMO writes a single minTTC
            # per encounter, not a span, and TET/TIT cannot be formed at all.
            "--device.ssm.trajectories",
            # 50 m is the default. Changing it changes which interactions are
            # observed, i.e. it changes the metric, so do not tune it.
            "--device.ssm.range", "50.0",
            "--device.ssm.file", os.path.join(out_dir, "ssm_%s.xml" % tag)]


def _span(el, name):
    node = el.find(name)
    return node.get("values", "").split() if node is not None else []


def parse_safety(path, ttc_star=TTC_STAR, v_min=V_MIN_MS, dt=1.0):
    """TET and TIT from an SSM trajectories file."""
    tet = tit = 0.0
    kept = dropped_mirror = dropped_slow = 0
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "conflict":
            continue
        types = _span(el, "typeSpan")
        ttcs = _span(el, "TTCSpan")
        vels = _span(el, "egoVelocity")
        for i, ty in enumerate(types):
            if i >= len(ttcs) or i >= len(vels):
                break
            if ty != FOLLOWER_TYPE:          # correction 1: drop the mirror
                dropped_mirror += 1
                continue
            try:
                ttc = float(ttcs[i])
            except ValueError:               # "NA" = not on a collision course
                continue
            try:
                v = abs(float(vels[i].split(",")[0]))
            except ValueError:
                continue
            if v < v_min:                    # correction 2: drop the queue
                dropped_slow += 1
                continue
            if 0.0 < ttc < ttc_star:
                tet += dt
                tit += (1.0 / ttc - 1.0 / ttc_star) * dt
                kept += 1
        el.clear()
    return {"TET_s": tet, "TIT_s": tit, "n_exposed": kept,
            "n_dropped_mirror": dropped_mirror, "n_dropped_slow": dropped_slow}


def parse_drac(path, drac_star=DRAC_STAR, dt=1.0):
    """Time exposed to DRAC above the threshold, and the integrated excess.

    The unbiased companion to `parse_safety`. No speed filter is applied and
    none is needed: DRAC vanishes rather than diverges as the speed difference
    goes to zero, so a stationary queue contributes nothing instead of
    contributing spuriously. Mirror encounters are still dropped, exactly as
    for TTC -- that correction is about double-logging, not about speed.
    """
    ted = tid = 0.0
    kept = 0
    peak = 0.0
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "conflict":
            continue
        types = _span(el, "typeSpan")
        dracs = _span(el, "DRACSpan")
        for i, ty in enumerate(types):
            if i >= len(dracs):
                break
            if ty != FOLLOWER_TYPE:
                continue
            try:
                dr = float(dracs[i])
            except ValueError:            # "NA" = not on a collision course
                continue
            peak = max(peak, dr)
            if dr > drac_star:
                ted += dt
                tid += (dr - drac_star) * dt
                kept += 1
        el.clear()
    return {"TED_s": ted, "TID_ms2": tid, "n_drac_events": kept,
            "max_DRAC": peak}


def parse_emissions(path, lengths_km=None):
    """Total CO2, fuel and vehicle-km from an edgeData emissions file.

    edgeData does NOT emit a `speed` attribute, so vehicle-km is derived as
    sampledSeconds * length / traveltime: sampledSeconds is vehicle-seconds on
    the edge and sampledSeconds/traveltime is the equivalent number of full
    traversals.

    Note CO2 and fuel are the same quantity under HBEFA3, and CO2 correlates
    with TTS at r = 0.997 on this scenario. Report CO2 per vehicle-km, and never
    as an independent second result: it would be double counting delay.
    """
    lengths_km = lengths_km or {}
    co2 = fuel = dist = 0.0

    def _acc(get):
        nonlocal co2, fuel, dist
        co2 += float(get("CO2_abs", 0.0))
        fuel += float(get("fuel_abs", 0.0))
        tt = float(get("traveltime", 0.0))
        L = lengths_km.get(get("id", ""), 0.0)
        if tt > 0 and L > 0:
            dist += float(get("sampledSeconds", 0.0)) * L / tt

    try:
        for _, el in ET.iterparse(path, events=("end",)):
            if el.tag == "edge":
                _acc(lambda k, d=0.0: el.get(k, d))
            if el.tag in ("edge", "interval"):
                el.clear()
    except ET.ParseError:
        # SUMO writes edgeData incrementally and only closes the root element at
        # the end, so an interrupted run leaves a truncated file that a strict
        # parser rejects outright. The rows already written are still valid, so
        # recover them by scanning instead of losing the whole run.
        import re
        co2 = fuel = dist = 0.0
        for m in re.finditer(r"<edge([^>]*)>", io.open(path, encoding="utf-8",
                                                        errors="replace").read()):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            _acc(lambda k, d=0.0, a=attrs: a.get(k, d))

    return {"CO2_kg": co2 / 1e6, "fuel_kg": fuel / 1e6, "veh_km": dist,
            "CO2_g_per_vehkm": (co2 / 1e3 / dist) if dist > 0 else float("nan")}


def run_scored(vsl_kmh=None, seed=1, tag=None, steps=SIM_STEPS):
    """Run SUMO fully instrumented and return every Phase E metric."""
    import traci
    os.makedirs(OUT, exist_ok=True)
    tag = tag or ("s%d" % seed)
    geom = load_geometry()
    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)

    traci.start(["sumo", "-c", CONFIG, "--end", str(steps), "--no-warnings",
                 "--seed", str(seed),
                 # a teleport silently removes a jammed vehicle, which flatters
                 # TTS and destroys the interaction TET/TIT was going to score
                 "--time-to-teleport", "-1",
                 "--collision.action", "warn",
                 "--device.emissions.probability", "1.0",
                 "--additional-files",
                 os.path.join(SUMO_DIR, "add", "emissions_meandata.xml")]
                + _ssm_args(tag))
    if vsl_kmh is not None:
        for e, v in zip(edges[:len(vsl_kmh)], vsl_kmh):
            traci.edge.setMaxSpeed(e, float(v) / MS_TO_KMH)

    spd, dens, on_road, pend = [], [], [], []
    arrived = 0
    for _ in range(steps):
        traci.simulationStep()
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        v = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        spd.append(np.where(c > 0, v, np.nan))
        dens.append(c / (Lkm * lanes))
        on_road.append(c.sum())
        pend.append(len(traci.simulation.getPendingVehicles()))
        arrived += traci.simulation.getArrivedNumber()
    traci.close()

    on_road = np.array(on_road, float)
    pend = np.array(pend, float)
    out = {
        "seed": seed,
        # Hegyi eq. 13: the origin queue is part of TTS, not an optional extra.
        # Here the insertion backlog peaks at 318 vehicles, 29% of total demand,
        # so omitting it understates delay badly. It is also the LESS noisy of
        # the two (CV 3.5% against 6.0% for road-only).
        "TTS_h": float((on_road + pend).sum() / 3600.0),
        "TTS_road_h": float(on_road.sum() / 3600.0),
        "throughput": float(arrived),
    }
    out.update(shockwave_metrics(np.array(spd), np.array(dens), 1.0, Lkm, lanes))
    out.update(parse_safety(os.path.join(OUT, "ssm_%s.xml" % tag)))
    em = os.path.join(SUMO_DIR, "add", "edge_emissions.xml")
    if os.path.exists(em):
        out.update(parse_emissions(
            em, {e: L for e, L in zip(edges, Lkm)}))
    return out


if __name__ == "__main__":
    import json
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    m = run_scored(seed=1, steps=steps)
    keys = ["TTS_h", "accel_rms", "cvs", "TET_s", "TIT_s", "CO2_g_per_vehkm",
            "n_exposed", "n_dropped_mirror", "n_dropped_slow", "throughput"]
    print(json.dumps({k: round(m[k], 4) for k in keys if k in m}, indent=2))
