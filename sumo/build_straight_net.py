# -*- coding: utf-8 -*-
"""
Build the STRAIGHT-ROAD jam-wave scenario: two lanes end to end, no merge.

WHY THIS EXISTS
---------------
Every result so far comes from a 2 -> 1 lane merge at 164% of capacity. That
bottleneck is permanent and physically impassable, so the queue behind it can
never recover while demand lasts, whatever the speed limits do. Across 39
objectives none shortened the congestion front there, but that result cannot
separate "VSL-MPC cannot reduce a shockwave" from "nothing can shorten a queue
behind a permanent bottleneck".

This scenario gives the controller a wave it can in principle remove. The road
is homogeneous, demand is below capacity, and the wave comes from a TRANSIENT
disturbance (vehicles braking hard for a short time, injected over TraCI by
`control/phase_d.run_closed_loop`). Once the disturbance clears, the jam it left
survives only while it is fed faster than it discharges. That is the setting of
Hegyi, De Schutter & Hellendoorn (2005, IEEE T-ITS 6(1)) and the SPECIALIST
algorithm: signs upstream of a jam wave cut its inflow so it dissolves.

LAYOUT (all 200 m, 2 lanes, one legal limit)
--------------------------------------------
    [ N_UP approach ][ N_CTRL controlled ][ N_DOWN downstream ]
      measured,         measured, VSL      measured, modelled,
      uncontrolled      signs              no signs; the
                                           disturbance is here

The prediction model spans the controlled AND downstream segments as one
straight 2-lane road (`MetanetModel(..., n_ctrl=N_CTRL)` on the sliced
geometry), so it sees the jam form. The approach is simulated and scored but
not modelled, for the same reason as on the long merge road: vehicles a
controller holds back stay on measured road instead of vanishing into the
origin queue.

The limit is 23.5 m/s (84.6 km/h), the approach limit of the long merge road,
and the vehicle types are copied verbatim from `sumo/routes/route.long.xml`, so
the car-following population is the one Bounded-METANET was calibrated on.

Usage:
    python sumo/build_straight_net.py --demands 2800,3000,3200
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

SEG_M = 200.0
SPEED_MS = 23.5
N_UP = 18
N_CTRL = 9
N_DOWN = 6
DEMAND_END_S = 2400        # flows run [0, DEMAND_END_S)
# HOW VEHICLES ENTER, AND WHY IT IS NOT "speedLimit".
# With departSpeed="speedLimit" (the merge scenario's setting) SUMO cannot put
# more than ~3530 veh/h onto this road whatever the demand: realised flow is
# identical at 3800, 4200, 4600 and 5000 veh/h, and the road never congests. The
# scenario was therefore stuck at ~70-75% of what two lanes can carry, which is
# why no single-vehicle disturbance could tip it, and it was not the
# "just below capacity" loading this scenario is supposed to test.
# Measured over 2400 s at demand 5000: speedLimit 3594 veh/h, max 4754, and
# free/best lane or departPos="last" add nothing beyond that. So: "max".
# NOTE this differs from the merge scenario deliberately; do not compare
# straight-road runs made before this change with ones made after (the pilot
# tags carry a marker).
DEPART_SPEED = "max"

NET = os.path.join(_HERE, "net", "plain.net4.xml")
NOD = os.path.join(_HERE, "net", "straight.nod.xml")
EDG = os.path.join(_HERE, "net", "straight.edg.xml")
# A LONGER DOWNSTREAM VARIANT, for disturbances that TRAVEL.
# With the standard 6 downstream segments there is only 0.9 km past the
# trigger point, and a vehicle easing to 15% of the limit over 90 s covers
# ~1.2 km getting there -- it drives off the end of the road before the
# disturbance exists (measured: zero cells below 30 km/h, against 413 s for the
# abrupt case). Any onset-rate or long-hold study needs this variant. The
# MODELLED stretch is unchanged (9 controlled + 6 downstream), so the NLP, the
# calibration and the controller problem are identical; the extra road is
# scored, not modelled.
N_DOWN_LONG = 24


def net_path(tag=""):
    return os.path.join(_HERE, "net", "plain.net4%s.xml" % tag)


def route_path(demand, tag=""):
    return os.path.join(_HERE, "routes", "route.straight%s.d%d.xml" % (tag, demand))


def edge_ids(n_down=N_DOWN):
    return (["a%02d" % i for i in range(N_UP)]
            + ["c%d" % i for i in range(N_CTRL)]
            + ["d%d" % i for i in range(n_down)])


def build_net(n_down=N_DOWN, tag=""):
    ids = edge_ids(n_down)
    n = len(ids)
    with open(NOD, "w") as fh:
        fh.write("<nodes>\n")
        for i in range(n + 1):
            fh.write('    <node id="p%d" x="%.1f" y="0.0" type="priority"/>\n'
                     % (i, i * SEG_M))
        fh.write("</nodes>\n")
    with open(EDG, "w") as fh:
        fh.write("<edges>\n")
        for i, eid in enumerate(ids):
            fh.write('    <edge id="%s" from="p%d" to="p%d" priority="1" '
                     'numLanes="2" speed="%.1f"/>\n' % (eid, i, i + 1, SPEED_MS))
        fh.write("</edges>\n")
    r = subprocess.run(["netconvert", "--node-files", NOD, "--edge-files", EDG,
                        "--output-file", net_path(tag), "--no-turnarounds"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("netconvert failed:\n%s\n%s" % (r.stdout, r.stderr))


def write_routes(demand, path, ids=None):
    """Vehicle types and the four-flow split copied from route.long.xml; only
    the route, the rate and the demand window differ."""
    ids = ids if ids is not None else edge_ids()
    per_flow = demand / 4.0
    with open(path, "w") as fh:
        fh.write('<routes>\n')
        fh.write('    <vType id="standard" accel="2.6" decel="4.5" length="5" '
                 'minGap="0.5" maxSpeed="33.33" sigma="0.5" color="blue" '
                 'emissionClass="HBEFA3/PC_G_EU4" type ="car" '
                 'lcLookaheadLeft="1000" lcLookaheadRight="1000"/>\n')
        fh.write('    <vType id="connected" accel="2.6" decel="4.5" length="5" '
                 'minGap="0.5" maxSpeed="33.33" sigma="0.0" color="red" '
                 'emissionClass="HBEFA3/PC_G_EU4" type ="car" '
                 'lcLookaheadLeft="1000" lcLookaheadRight="1000"/>\n\n')
        fh.write('    <route id="r1" edges="%s"/>\n\n' % " ".join(ids))
        for fid, vtype, lane in (("std_flow", "standard", "1"),
                                 ("std_flow1", "standard", "0"),
                                 ("cv_flow", "connected", "1"),
                                 ("cv_flow1", "connected", "0")):
            fh.write('    <flow id="%s" type="%s" begin="0" end="%d" '
                     'vehsPerHour="%g" route="r1" departLane="%s" '
                     'departPos="0" laneChangeMode="512" '
                     'departSpeed="%s"/>\n'
                     % (fid, vtype, DEMAND_END_S, per_flow, lane, DEPART_SPEED))
        fh.write('</routes>\n')


def write_cfg(demand, path, tag=""):
    # forward slashes: see the note in sumo/build_demand_variants.py
    with open(path, "w") as fh:
        fh.write('<configuration>\n    <input>\n'
                 '        <net-file value="net/plain.net4%s.xml"/>\n'
                 '        <route-files value="routes/route.straight%s.d%d.xml"/>\n'
                 '    </input>\n\n'
                 '    <time>\n        <begin value="0"/>\n'
                 '        <end value="%d"/>\n    </time>\n\n'
                 '    <report>\n        <verbose value="true"/>\n'
                 '        <no-step-log value="true"/>\n    </report>\n'
                 '</configuration>\n' % (tag, tag, demand, DEMAND_END_S))


def cfg_path(demand, tag=""):
    return os.path.join(_HERE, "simulation4%s.d%d.sumocfg" % (tag, int(demand)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demands", default="2800,3000,3200",
                    help="comma-separated total demands, veh/h")
    ap.add_argument("--down", type=int, default=N_DOWN,
                    help="downstream segments (N_DOWN_LONG for travelling "
                         "disturbances -- see the note on N_DOWN_LONG)")
    ap.add_argument("--tag", default="",
                    help="filename suffix, e.g. _long; empty overwrites the "
                         "standard scenario")
    a = ap.parse_args()
    build_net(a.down, a.tag)
    ids = edge_ids(a.down)
    for d in [int(x) for x in a.demands.split(",") if x]:
        write_routes(d, route_path(d, a.tag), ids)
        write_cfg(d, cfg_path(d, a.tag), a.tag)
        print("demand %d veh/h -> %s" % (d, os.path.basename(cfg_path(d, a.tag))))
    print("road: %d approach + %d controlled + %d downstream = %.1f km, "
          "2 lanes, %.1f km/h" % (N_UP, N_CTRL, a.down,
                                  (N_UP + N_CTRL + a.down) * SEG_M / 1000.0,
                                  SPEED_MS * 3.6))
    print("wrote %s" % net_path(a.tag))


if __name__ == "__main__":
    main()
