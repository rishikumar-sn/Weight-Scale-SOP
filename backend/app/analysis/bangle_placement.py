"""Flag ambiguous bangle groups without inventing an instance count."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .bangle_orientation import _geometry


PLACEMENT_MESSAGE = (
    "Jewels may be touching or overlapping. Place each jewel separately with a visible gap "
    "and recapture the image."
)


def check_bangle_placement(mask: np.ndarray | None) -> dict[str, Any]:
    """Check only a component identified as a bangle.

    Two substantial enclosed openings suggest touching loops or a loop crossed
    by another jewel. A large off-centre opening with a much wider outer body
    also makes the placement ambiguous. Neither rule estimates the true count.
    Small stone/texture holes and a single central opening are allowed. Hidden
    stacks, open contours and opaque heaps can still escape these checks.
    """
    result: dict[str, Any] = {"passed": True, "method": "bangle_opening_geometry",
                              "reason": None, "large_openings": []}
    if mask is None or mask.ndim != 2 or not mask.size:
        return result
    binary = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result
    index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    area = cv2.contourArea(contours[index])
    if area < 350:
        return result
    openings = sorted(
        (cv2.contourArea(contours[i]) / area for i in range(len(contours))
         if hierarchy[0, i, 3] == index and cv2.contourArea(contours[i]) >= 80),
        reverse=True,
    )
    large = [ratio for ratio in openings if ratio >= 0.08]
    geometry = _geometry(binary)
    result.update(geometry=geometry, large_openings=large)
    if len(large) >= 2 and sum(large) >= 0.30:
        result.update(passed=False, reason="multiple_large_openings_in_one_bangle_component")
    elif (0.12 <= geometry.get("hole_area_ratio", 0) <= 0.45
          and geometry.get("hole_centre_offset", 0) > 0.16
          and geometry.get("outer_axis_ratio", 0) > 1.8 * geometry.get("hole_axis_ratio", 1)):
        result.update(passed=False, reason="off_centre_opening_with_conflicting_body_shape")
    return result
