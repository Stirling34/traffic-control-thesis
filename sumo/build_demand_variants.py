# -*- coding: utf-8 -*-
"""
Generate route + config variants of the extended-approach scenario at a range of
demands, so the controller can be scored as a function of demand/capacity.

WHY
---
Every result in this project was measured at ONE demand: 4000 veh/h against a
measured free-flow bottleneck capacity of 2434 veh/h, i.e. d/C = 1.64. At that
loading the queue is unavoidable, so the mechanism the VSL literature actually
relies on -- preventing breakdown, and with it the ~12% capacity drop -- is not
available at all. A null result there cannot distinguish "VSL does not work"
from "VSL does not work on a road that is already past saving".

It also matters for deployability, which is the sharper point: real demand is not
constant. A policy that helps at 120% of capacity but HURTS at 85% is not usable,
because a fixed sign programme meets both within an hour. This sweep is what
turns "does it work?" into "over what range does it work, and where does it start
doing harm?"

WHAT IS AND IS NOT VARIED
------------------------
Only `vehsPerHour` on the four flows. Same network, same nine controlled
segments, same vehicle mix, same 1000 s demand window, same routes, same vType
attributes (including the `lcLookaheadLeft/Right` values and the inert
`laneChangeMode`, both carried over verbatim -- see the ledger). Nothing that
would make these runs incomparable with the existing ones is touched, and the
existing `route.long.xml` / `simulation3.sumocfg` are NOT modified: the 4000
veh/h variant regenerated here must reproduce them.

Vehicle count scales with demand (1000 s window x D/3600), so per-vehicle metrics
must be normalised by each run's OWN `throughput`, never by the hard-coded 1112.

Usage:  python sumo/build_demand_variants.py
"""

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO = os.path.join(_ROOT, "sumo")
SRC_ROUTE = os.path.join(SUMO, "routes", "route.long.xml")

# measured free-flow bottleneck discharge, veh/h (validation/capacity_drop_test.py,
# 10 seeds) -- the denominator that makes these ratios mean anything
CAPACITY = 2434.0
DEMANDS = [2100, 2440, 2920, 3400, 4000]


def main():
    src = open(SRC_ROUTE).read()
    made = []
    for d in DEMANDS:
        per_flow = d / 4.0
        # every flow carries a quarter of the demand, exactly as the original does
        txt = re.sub(r'vehsPerHour="[0-9.]+"',
                     'vehsPerHour="%g"' % per_flow, src)
        n = len(re.findall(r'vehsPerHour="%g"' % per_flow, txt))
        if n != 4:
            raise SystemExit("expected 4 flows, rewrote %d for demand %d" % (n, d))
        rp = os.path.join(SUMO, "routes", "route.long.d%d.xml" % d)
        open(rp, "w").write(txt)
        cp = os.path.join(SUMO, "simulation3.d%d.sumocfg" % d)
        open(cp, "w").write(
            '<configuration>\n'
            '    <input>\n'
            # FORWARD slashes deliberately. SUMO accepts them on Windows, and a
            # backslash here is a Python escape waiting to happen -- the first
            # version of this file wrote "routes\route..." where the \r was
            # consumed, producing "routesoute.long.d2100.xml" and a SUMO
            # "route file is not accessible" failure.
            '        <net-file value="net/plain.net3.xml"/>\n'
            '        <route-files value="routes/route.long.d%d.xml"/>\n'
            '    </input>\n\n'
            '    <time>\n'
            '        <begin value="0"/>\n'
            '        <end value="1000"/>\n'
            '    </time>\n\n'
            '    <report>\n'
            '        <verbose value="true"/>\n'
            '        <no-step-log value="true"/>\n'
            '    </report>\n'
            '</configuration>\n' % d)
        made.append((d, d / CAPACITY, per_flow, rp, cp))

    print("demand   d/C   per-flow   files")
    for d, r, pf, rp, cp in made:
        print("%6d  %.2f   %8.1f   %s" % (d, r, pf, os.path.basename(cp)))

    # the 4000 variant must be identical to the original except for formatting
    a = open(os.path.join(SUMO, "routes", "route.long.d4000.xml")).read()
    b = open(SRC_ROUTE).read()
    print("\n4000 veh/h variant reproduces route.long.xml exactly: %s"
          % (a == b))
    if a != b:
        print("  !! it must -- otherwise these runs are not comparable")
        sys.exit(1)


if __name__ == "__main__":
    main()
