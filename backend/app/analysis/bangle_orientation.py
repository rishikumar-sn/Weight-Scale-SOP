"""Conservative, rotation-independent pose checks before measuring a bangle."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _geometry(mask: np.ndarray) -> dict[str, float]:
    contours, hierarchy = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {}
    index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[index]
    area = cv2.contourArea(outer)
    if area < 80:
        return {}
    centre, axes, _ = cv2.minAreaRect(outer)
    major = max(axes)
    holes = [contours[i] for i in range(len(contours)) if hierarchy[0, i, 3] == index]
    hole = max(holes, key=cv2.contourArea) if holes else None
    result = {"outer_axis_ratio": float(min(axes) / max(major, 1)),
              "hole_area_ratio": 0.0, "hole_axis_ratio": 0.0,
              "hole_centre_offset": 1.0}
    if hole is not None:
        hole_centre, hole_axes, _ = cv2.minAreaRect(hole)
        result.update(
            hole_area_ratio=float(cv2.contourArea(hole) / area),
            hole_axis_ratio=float(min(hole_axes) / max(max(hole_axes), 1)),
            hole_centre_offset=float(np.linalg.norm(np.subtract(centre, hole_centre)) / major),
        )
    return result


def detect_bangle_orientation(
    item_mask: np.ndarray | None, *, gold_mask: np.ndarray | None = None
) -> dict[str, Any]:
    """Require a round outer body AND central opening for flat dimensions.

    An edge-on body is standing regardless of its image rotation. Near-round
    bodies without a reliable opening remain unknown and use stone analysis.
    A gold prior can recover an opening when tray texture fills the Otsu mask;
    it is constrained to the item and never overrides an elongated silhouette.
    """
    evidence: dict[str, Any] = {"orientation": "unknown", "measurement_allowed": False,
                                "reason": "Bangle silhouette is unavailable"}
    if item_mask is None or item_mask.ndim != 2 or not item_mask.size:
        return evidence
    geometry = _geometry(item_mask)
    if not geometry:
        return evidence
    evidence.update(geometry)
    evidence["source"] = "item_mask"
    if geometry["outer_axis_ratio"] < 0.72:
        evidence.update(orientation="standing", reason="Bangle body is viewed edge-on or tilted")
        return evidence

    candidates = [("item_mask", geometry)]
    if gold_mask is not None and gold_mask.shape == item_mask.shape:
        prior = ((gold_mask > 0) & (item_mask > 0)).astype(np.uint8)
        # Bridge only small breaks caused by glare; do not fill the opening.
        prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        candidates.append(("gold_prior", _geometry(prior)))
    for source, candidate in candidates:
        if (candidate.get("outer_axis_ratio", 0) >= 0.72
                and 0.18 <= candidate.get("hole_area_ratio", 0) <= 0.92
                and candidate.get("hole_axis_ratio", 0) >= 0.65
                and candidate.get("hole_centre_offset", 1) <= 0.15):
            evidence.update(candidate, source=source, orientation="flat",
                            measurement_allowed=True,
                            reason="Near-circular body with a clear central opening")
            return evidence
    evidence["reason"] = "A round central opening could not be verified; using stone analysis"
    return evidence
