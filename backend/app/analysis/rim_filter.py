"""Conservative, calibrated exclusion of unsupported outer tray-rim strips.

Position alone never establishes that a component is a rim. Missing or
ambiguous evidence retains the component for the existing rejection gates.
"""
from __future__ import annotations

from copy import deepcopy
import cv2
import numpy as np

from .component_presence import component_presence_check


RIM_BAND_MM = 5.0
MAX_STRIP_THICKNESS_MM = 3.0
MIN_STRIP_ASPECT = 3.0
SIGLIP_JEWELLERY_VETO_CONFIDENCE = 0.70


def rim_candidates(items: list[dict], check: dict, testbed_mask: np.ndarray | None,
                   calibration: dict | None, testbed: dict | None) -> dict:
    """Find only neutral, thin, unsupported strips at the outer polygon edge.

    Use foreground pixels, not padded crop boxes. Fill internal mask holes so
    the AprilTag and tray fixture cannot masquerade as the outer boundary.
    Require another positively supported jewel; an empty tray is never accepted.
    """
    result = {"method": "calibrated_outer_boundary_strip", "items": [],
              "candidate_indices": [], "ignored_component_indices": []}
    calibration, testbed = calibration or {}, testbed or {}
    if (check.get("passed") is not True or check.get("detections")
            or not (check.get("placement_check") or {}).get("passed", True)):
        result["reason"] = "anomaly_or_placement_block"
        return result
    try:
        testbed_confidence = float(testbed.get("confidence") or 0)
    except (TypeError, ValueError):
        testbed_confidence = 0.0
    if (not calibration.get("found") or not calibration.get("available")
            or not np.isfinite(testbed_confidence) or testbed_confidence < 0.5):
        result["reason"] = "detected_testbed_and_apriltag_required"
        return result
    try:
        scales = np.array([float(calibration["mm_per_pixel_x"]),
                           float(calibration["mm_per_pixel_y"])])
    except (KeyError, ValueError, TypeError):
        scales = np.array([0.0, 0.0])
    if not np.isfinite(scales).all() or np.min(scales) <= 0:
        result["reason"] = "invalid_calibration"
        return result
    if testbed_mask is None or testbed_mask.ndim != 2 or not np.any(testbed_mask):
        result["reason"] = "missing_testbed_polygon"
        return result
    presence = component_presence_check(check, len(items))
    if not any(entry["passed"] for entry in presence["items"]):
        result["reason"] = "no_supported_jewellery"
        return result
    outer = np.zeros(testbed_mask.shape, np.uint8)
    contours, _ = cv2.findContours((testbed_mask > 0).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        result["reason"] = "ambiguous_testbed_polygon"
        return result
    cv2.drawContours(outer, contours, -1, 1, cv2.FILLED)
    distance = cv2.distanceTransform(np.pad(outer, 1), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    # Both limits must hold: calibrated physical distance and a narrow relative
    # band. A bad/huge scale cannot turn the whole tray into its border.
    band_px = min(RIM_BAND_MM / float(scales.max()), min(outer.shape) * 0.04)
    result["band_px"] = float(band_px)
    for index, (item, support) in enumerate(zip(items, presence["items"]), 1):
        record = {"source_component_index": index, "bbox": dict(item["bbox"]),
                  "jewellery_mask_share": support["jewellery_mask_share"],
                  "candidate": False, "ignored": False}
        result["items"].append(record)
        # Even partial instance support is enough to protect a possible jewel.
        if support["jewellery_mask_share"] > 0:
            record["reason"] = "jewellery_overlap_preserved"
            continue
        mask = (item["mask"] > 0).astype(np.uint8)
        x, y, w, h = (int(item["bbox"][key]) for key in ("x", "y", "w", "h"))
        if (mask.shape != (h, w) or x < 0 or y < 0 or x+w > outer.shape[1]
                or y+h > outer.shape[0] or not mask.any()):
            record["reason"] = "invalid_component_geometry"
            continue
        values = distance[y:y+h, x:x+w][mask > 0]
        record["maximum_boundary_distance_px"] = float(values.max())
        if np.any(outer[y:y+h, x:x+w][mask > 0] == 0) or values.max() > band_px:
            record["reason"] = "interior_or_ambiguous_position"
            continue
        points = cv2.findNonZero(mask)
        _, (a, b), _ = cv2.minAreaRect(points)
        thin, long = sorted((float(a)+1.0, float(b)+1.0))
        record.update(thickness_mm=thin * float(scales.max()), aspect=long / thin)
        if thin * scales.max() > MAX_STRIP_THICKNESS_MM or long / thin < MIN_STRIP_ASPECT:
            record["reason"] = "not_a_thin_boundary_strip"
            continue
        # A pin pointing into the tray is near the edge too. A rim strip must
        # run along it: its distance variation must fit its short dimension.
        distance_span = float(values.max() - values.min())
        record["boundary_distance_span_px"] = distance_span
        if distance_span > thin * 1.5:
            record["reason"] = "does_not_follow_boundary"
            continue
        # Closed ring/chain-loop shapes are not rim strips.
        _, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0):
            record["reason"] = "closed_opening_preserved"
            continue
        raw = item["raw_crop_bgr"]
        if raw.shape[:2] != mask.shape:
            record["reason"] = "raw_crop_unavailable"
            continue
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        # Protect coloured metal, stones and thread; a rim must be neutral.
        coloured = (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 45)
        colour_share = float(np.mean(coloured[mask > 0]))
        record["coloured_foreground_share"] = colour_share
        if colour_share >= 0.02:
            record["reason"] = "coloured_foreground_preserved"
            continue
        record.update(candidate=True, reason="unsupported_neutral_outer_strip")
        result["candidate_indices"].append(index)
    return result


def remap_component_check(check: dict, retained_indices: list[int]) -> dict:
    """Retain detector evidence after exclusions without shifting ownership."""
    result = deepcopy(check)
    mapping = {source: index for index, source in enumerate(retained_indices, 1)}
    for detection in result.get("jewellery_instances", []):
        matches = []
        for match in detection.get("component_overlaps", []):
            source = match["component_index"]
            if source in mapping:
                matches.append({**match, "source_component_index": source,
                                "component_index": mapping[source]})
        detection["component_overlaps"] = matches
    result["retained_source_component_indices"] = retained_indices
    return result
