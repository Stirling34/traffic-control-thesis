# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:21:08 2025

@author: jouannin
"""

# XML parser to read SUMO .net.xml files
import xml.etree.ElementTree as ET
# (These two aren’t used below; safe to remove unless used elsewhere)
from collections import defaultdict, Counter
import numpy as np

class ExtractData():
    def __init__(self):
        """Constructor: runs when you create ExtractData(). Just prints a banner."""
        print("")
        print("=======================")
        print("Extracteur initialised")
        print("=======================")
        print("")
    
    def parse_net_file(self, filename):
        """
        Parse a SUMO network XML file and return an ordered list of edge dicts.
        Each dict summarises: id, from/to nodes, lane metadata, x-position span,
        lane count, and an average speed for display/control.
        """
        # Load and parse XML
        tree = ET.parse(filename)
        root = tree.getroot()
        
        edges_info = []
        
        # Iterate all <edge> elements in the network
        for edge in root.findall('edge'):
            # Skip “internal” edges (SUMO uses these for junction internals)
            if edge.get('function') == 'internal':
                continue
            
            edge_id = edge.get('id')
            from_node = edge.get('from')
            to_node = edge.get('to')
            
            lanes = []
            x_coords = []
            # Iterate all lanes on this edge to collect metadata
            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                speed = float(lane.get('speed'))     # m/s
                length = float(lane.get('length'))   # meters
                shape_str = lane.get('shape')        # "x1,y1 x2,y2 ..."

                # Parse lane polyline: keep x-coordinates to find edge start/end
                points = [tuple(map(float, p.split(','))) for p in shape_str.split()]
                x_coords.extend([p[0] for p in points])

                lanes.append({
                    'lane_id': lane_id,
                    'speed': speed,
                    'length': length,
                })
            
            # Edge longitudinal span: min/max x across all lane points
            pos_start = min(x_coords)
            pos_end = max(x_coords)
            
            # Build a compact summary for this edge
            edges_info.append({
                'edge_id': edge_id,
                'from': from_node,
                'to': to_node,
                'lanes': lanes,
                'num_lanes': len(lanes),
                'pos_start': pos_start,
                'pos_end': pos_end,
                # Average of lane speed limits (lanes usually share the same limit)
                'avg_speed': (sum(l['speed'] for l in lanes) / len(lanes)) if lanes else 0
            })
        
            # Keep list ordered by longitudinal position so sections are left→right
            edges_info.sort(key=lambda e: e['pos_start'])
            
        return edges_info
        
    def print_sections(self, edges):
        """
        Pretty-print each section’s key stats for quick inspection.
        Assumes at least one lane exists to read length/speed.
        """
        for i, e in enumerate(edges, start=1):
            lane_length = e['lanes'][0]['length'] if e['lanes'] else 0
            print(f"Section {i}:")
            print(f"  ID           : {e['edge_id']}")
            print(f"  Lane(s)      : {e['num_lanes']}")
            print(f"  Length (m)   : {lane_length:.1f}")
            print(f"  Position (m) : {e['pos_start']:.1f} --> {e['pos_end']:.1f}")
            print(f"  Max speed (m/s)  : {e['avg_speed']:.1f}")
            print()
    
    def extract_densities(self, vehicle_positions, edges, num_sections):
        """ 
        Compute per-section density at the current timestep.

        Parameters
        ----------
        vehicle_positions : list[float]
            X-positions for all vehicles (same coordinate frame as edge shapes).
        edges : list[dict]
            Output of parse_net_file(); each has pos_start/pos_end bounds.
        num_sections : int
            Number of sections (usually len(edges)).

        Returns
        -------
        densities : list[float]
            Density per section = vehicle_count / section_length (veh/m).
            (Ignores lane count here; adjust if you want veh per lane-km.)
        """
        # Count vehicles per section by binning each x-position into an edge span
        vehicle_counts = [0] * num_sections
        for pos in vehicle_positions:
            for i, edge in enumerate(edges):
                if edge["pos_start"] <= pos < edge["pos_end"]:
                    vehicle_counts[i] += 1
                    break  # move to next vehicle once binned

        # Convert counts to densities using section physical length
        densities = []
        for i, edge in enumerate(edges):
            section_length = edge["pos_end"] - edge["pos_start"]  # meters
            density = vehicle_counts[i] / section_length          # vehicles per meter
            densities.append(density)

        return densities
