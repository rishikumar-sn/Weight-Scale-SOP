"""Conservative experimental verifier; intentionally not imported by the app.

Thresholds are pilot values, not a validated production calibration. A missing
or incompatible spatial reference can never cause rejection. Grouping only
adds context to already background-matching regions; it cannot turn weak
evidence into a rejection.
"""
from __future__ import annotations

import math


def evaluate(regions, registration):
    decisions = []
    eligible = []
    for index, region in enumerate(regions):
        m = region.get('metrics') or {}
        d = region.get('refinement_diagnostics') or {}
        if region.get('color') != 'White/Colorless':
            decisions.append({'decision': 'keep', 'reason': 'color_out_of_scope'})
            continue
        decisions.append({'decision': 'uncertain', 'reason': 'insufficient_background_evidence'})
        if not m.get('valid'):
            continue
        if m.get('local_delta_e_median', 0) >= 12:
            decisions[index] = {'decision': 'keep', 'reason': 'surface_differs_from_background'}
        strong_local = (m.get('core_pixels', 0) >= 20
                        and m.get('local_background_pixels', 0) >= 100
                        and m.get('local_delta_e_median', math.inf) <= 6
                        and m.get('local_delta_e_p90', math.inf) <= 8
                        and m.get('raw_lightness_std', math.inf) <= 4
                        and float(d.get('metal_surround_ratio', 0)) >= .65
                        and not region.get('learned_matches'))
        strong_reference = (registration.get('valid', False)
                            and m.get('reference_background_p90', math.inf) <= 3
                            and m.get('reference_delta_e_median', math.inf) <= 6
                            and m.get('reference_delta_e_p90', math.inf) <= 8)
        if strong_local:
            decisions[index] = {'decision': 'uncertain', 'reason': 'background_like_without_verified_link_group'}
        if strong_local and strong_reference:
            eligible.append(index)

    # Require three nearby, similarly sized and shaped background-like faces.
    # Never count nonmatching regions toward the group.
    neighbors = {i: set() for i in eligible}
    for pos, i in enumerate(eligible):
        x, y, w, h = regions[i]['bbox']
        for j in eligible[pos+1:]:
            if regions[i].get('jewel_id', 1) != regions[j].get('jewel_id', 1):
                continue
            u, v, a, b = regions[j]['bbox']
            area_ratio = (w*h) / max(1, a*b)
            aspect_delta = abs(max(w, h)/max(1, min(w, h)) - max(a, b)/max(1, min(a, b)))
            if (math.hypot(x+w/2-u-a/2, y+h/2-v-b/2) <= 90
                    and 1/3 <= area_ratio <= 3 and aspect_delta <= 1):
                neighbors[i].add(j)
                neighbors[j].add(i)
    unseen = set(eligible)
    while unseen:
        group, frontier = set(), {min(unseen)}
        while frontier:
            current = frontier.pop()
            if current in group:
                continue
            group.add(current)
            frontier.update(neighbors[current] - group)
        unseen -= group
        if len(group) >= 3:
            for i in group:
                decisions[i] = {'decision': 'reject_background_hole',
                                'reason': 'reference_match_and_repeated_metal_enclosed_background',
                                'group_count': len(group)}
    return decisions
