# -*- coding: utf-8 -*-
"""
Build the EXTENDED-APPROACH version of the merge scenario.

WHY THIS EXISTS
---------------
Every cost function that appeared to reduce shockwaves on the original network
was doing it by metering the entrance: holding vehicles at the mainstream
origin, where they leave the road and stop being counted by the shockwave
metric. `accel_dens_tts` added 128 s per vehicle of origin queueing against
18 s of extra driving. That is the bottleneck relocated upstream, not traffic
being smoothed, and it explains 73% of the variance across the 22-objective
leaderboard (r = -0.85).

`term_queue` penalises it, but a penalty is a band-aid: it trades one arbitrary
weight against another and the smoothing dies with the metering, leaving the
question of what was real unanswered.

The root cause is a MODELLING BOUNDARY. On a real motorway, vehicles held back
by a speed limit are still on the road upstream -- decelerating, queueing, and
contributing to exactly the accelerations the metric is supposed to count. Only
in the simulation do they vanish into a point queue outside the measured region.

So: make the observed road long enough that the queue has somewhere real to go.
The controlled section, the bottleneck, the demand and the vehicle mix are all
unchanged; the only difference is that `N_UPSTREAM` extra segments of ordinary
two-lane road sit in front of the controlled section, and are MEASURED. Any
queue the controller creates now forms on instrumented road and is counted --
in on-road travel time and in the acceleration metric alike.

The controller's authority is deliberately NOT extended. It still sets the same
nine segments. The prediction model still spans only those nine plus the
bottleneck (see `model_geometry` in model/metanet_model.py), so the NLP, its
calibration and its validated 210 s prediction skill all carry over untouched.
The ONLY thing that changes between the short and long scenarios is how much
road is watched, which is what makes the pair a controlled comparison.

LAYOUT
------
    [ N_UPSTREAM x 200 m, 2 lanes ][ 9 x ~200 m, 2 lanes ][ 496 m, 1 lane ]
     measured, uncontrolled         measured, CONTROLLED   the bottleneck

Downstream of the extension the geometry is byte-for-byte the original: the
same edge ids (`start1`, `n1n2` ... `n8n9`, `n9end`), the same lengths, and the
same speed limits, which rise 84.6 -> 99.0 km/h in 1.8 km/h steps. The
extension is held at 84.6 km/h, the limit of the segment it feeds, because
continuing the downstream ramp backwards would run the far end down to 52 km/h
and quietly make the approach a second bottleneck.

Usage:
    python sumo/build_long_net.py                  # 18 extra segments (3x road)
    python sumo/build_long_net.py --upstream 30    # longer, if the queue spills
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# The original downstream geometry, transcribed from sumo/net/lane_merge.edg2.xml.
# (id, from, to, lanes, speed m/s). Lengths come from the node spacing.
DOWNSTREAM = [
    ("start1", "start", "n1", 2, 23.5),
    ("n1n2", "n1", "n2", 2, 24.0),
    ("n2n3", "n2", "n3", 2, 24.5),
    ("n3n4", "n3", "n4", 2, 25.0),
    ("n4n5", "n4", "n5", 2, 25.5),
    ("n5n6", "n5", "n6", 2, 26.0),
    ("n6n7", "n6", "n7", 2, 26.5),
    ("n7n8", "n7", "n8", 2, 27.0),
    ("n8n9", "n8", "n9", 2, 27.5),
    ("n9end", "n9", "end", 1, 28.0),
]
DOWNSTREAM_NODES = [("start", 0.0), ("n1", 200.0), ("n2", 400.0), ("n3", 600.0),
                    ("n4", 800.0), ("n5", 1000.0), ("n6", 1200.0),
                    ("n7", 1400.0), ("n8", 1600.0), ("n9", 1800.0),
                    ("end", 2300.0)]

SEG_M = 200.0
UPSTREAM_SPEED = 23.5          # m/s, == start1, the segment the extension feeds
N_CONTROLLED = 9               # unchanged: the controller's authority


def build(n_upstream):
    """Return (nodes, edges, route_edge_ids) for a road with `n_upstream`
    extra 200 m two-lane segments prepended to the original geometry."""
    nodes, edges = [], []
    # Upstream nodes run u0 .. u{n-1} at negative x, feeding `start` at x=0, so
    # every original coordinate keeps the value it had on the short network.
    for i in range(n_upstream):
        nodes.append(("u%d" % i, -(n_upstream - i) * SEG_M))
    nodes.extend(DOWNSTREAM_NODES)
    for i in range(n_upstream):
        frm = "u%d" % i
        to = "u%d" % (i + 1) if i + 1 < n_upstream else "start"
        edges.append(("%s%s" % (frm, to), frm, to, 2, UPSTREAM_SPEED))
    edges.extend(DOWNSTREAM)
    return nodes, edges, [e[0] for e in edges]


def write_xml(nodes, edges, nod_path, edg_path):
    with open(nod_path, "w") as fh:
        fh.write("<nodes>\n")
        for nid, x in nodes:
            fh.write('    <node id="%s" x="%.1f" y="0.0" type="priority"/>\n'
                     % (nid, x))
        fh.write("</nodes>\n")
    with open(edg_path, "w") as fh:
        fh.write("<edges>\n")
        for eid, frm, to, lanes, spd in edges:
            fh.write('    <edge id="%s" from="%s" to="%s" priority="1" '
                     'numLanes="%d" speed="%.1f"/>\n' % (eid, frm, to, lanes, spd))
        fh.write("</edges>\n")


def write_routes(route_edges, path):
    """Same vehicle types, same four flows, same 4000 veh/h for 1000 s as
    sumo/routes/route.rou2.xml -- only the route's edge list is longer.

    `laneChangeMode` is carried over verbatim even though it is a TraCI setting
    that SUMO silently ignores on a <flow> (see the 2026-08-19 ledger entry).
    Removing it here would change vehicle behaviour relative to the short-road
    runs, which is the one thing this scenario must not do.
    """
    edges_attr = " ".join(route_edges)
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
        fh.write('    <route id="r1" edges="%s"/>\n\n' % edges_attr)
        for fid, vtype, lane in (("std_flow", "standard", "1"),
                                 ("std_flow1", "standard", "0"),
                                 ("cv_flow", "connected", "1"),
                                 ("cv_flow1", "connected", "0")):
            fh.write('    <flow id="%s" type="%s" begin="0" end="1000" '
                     'vehsPerHour="1000" route="r1" departLane="%s" '
                     'departPos="0" laneChangeMode="512" '
                     'departSpeed="speedLimit"/>\n' % (fid, vtype, lane))
        fh.write('</routes>\n')


def write_cfg(net_rel, route_rel, path):
    with open(path, "w") as fh:
        fh.write('<configuration>\n    <input>\n'
                 '        <net-file value="%s"/>\n'
                 '        <route-files value="%s"/>\n'
                 '    </input>\n\n'
                 '    <time>\n        <begin value="0"/>\n'
                 '        <end value="1000"/>\n    </time>\n\n'
                 '    <report>\n        <verbose value="true"/>\n'
                 '        <no-step-log value="true"/>\n    </report>\n'
                 '</configuration>\n' % (net_rel, route_rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=int, default=18,
                    help="extra 200 m two-lane segments in front of the "
                         "controlled section (18 = a 3x longer road)")
    a = ap.parse_args()

    nodes, edges, route_edges = build(a.upstream)
    nod = os.path.join(_HERE, "net", "lane_merge_long.nod.xml")
    edg = os.path.join(_HERE, "net", "lane_merge_long.edg.xml")
    net = os.path.join(_HERE, "net", "plain.net3.xml")
    rou = os.path.join(_HERE, "routes", "route.long.xml")
    cfg = os.path.join(_HERE, "simulation3.sumocfg")

    write_xml(nodes, edges, nod, edg)
    r = subprocess.run(["netconvert", "--node-files", nod, "--edge-files", edg,
                        "--output-file", net, "--no-turnarounds"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("netconvert failed:\n%s\n%s" % (r.stdout, r.stderr))
    write_routes(route_edges, rou)
    write_cfg(os.path.join("net", "plain.net3.xml"),
              os.path.join("routes", "route.long.xml"), cfg)

    mainline_km = (a.upstream + N_CONTROLLED) * SEG_M / 1000.0
    print("upstream segments : %d  (%.1f km of extra measured approach)"
          % (a.upstream, a.upstream * SEG_M / 1000.0))
    print("mainline 2-lane   : %.1f km in %d edges"
          % (mainline_km, a.upstream + N_CONTROLLED))
    print("controlled        : the last %d mainline edges (unchanged)"
          % N_CONTROLLED)
    print("wrote %s" % net)
    print("wrote %s" % rou)
    print("wrote %s" % cfg)


if __name__ == "__main__":
    main()
