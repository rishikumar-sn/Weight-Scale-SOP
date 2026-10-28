"""Finger-ring sizes from the customer-supplied inner-diameter chart."""
from __future__ import annotations

import math
from typing import Any

# Latest supplied chart; millimeter columns are authoritative.
RING_SIZE_CHART = (
    ("7", 15.3, 48.0),
    ("8", 15.7, 49.3),
    ("9", 16.1, 50.6),
    ("10", 16.5, 51.9),
    ("11", 16.9, 53.1),
    ("12", 17.3, 54.4),
    ("13", 17.7, 55.7),
    ("14", 18.1, 57.0),
    ("15", 18.5, 58.3),
    ("16", 18.9, 59.5),
    ("17", 19.3, 60.8),
    ("18", 19.7, 62.1),
    ("19", 20.1, 63.4),
    ("20", 20.6, 64.6),
    ("21", 21.0, 65.9),
    ("22", 21.4, 67.2),
    ("23", 21.8, 68.5),
    ("24", 22.2, 69.7),
    ("25", 22.6, 71.0),
    ("26", 23.0, 72.3),
)


def ring_size(inner_diameter_mm: Any) -> dict[str, Any]:
    result = dict(indian=None, reference_diameter_mm=None,
                  reference_circumference_mm=None, display="Unavailable",
                  status="unavailable", method="nearest_inner_diameter", chart="customer_ring_mm_v2")
    try:
        diameter = float(inner_diameter_mm)
    except (TypeError, ValueError, OverflowError):
        return result
    if not math.isfinite(diameter) or diameter <= 0:
        return result
    if not RING_SIZE_CHART[0][1] <= diameter <= RING_SIZE_CHART[-1][1]:
        return dict(result, display="Outside supplied size chart", status="out_of_range")
    # Round differences to avoid floating-point midpoint ties changing the size.
    indian, reference, circumference = min(
        RING_SIZE_CHART, key=lambda row: round(abs(row[1] - diameter), 10))
    return dict(result, indian=indian, reference_diameter_mm=reference,
                reference_circumference_mm=circumference, status="matched",
                display=f"{indian} (approx.)")


def ring_size_rows(dimensions: dict[str, Any]) -> list[list[str]]:
    if dimensions.get("applicable") is False:
        return [["Ring measurement", str(dimensions.get("reason") or "Unavailable")]]
    size = ring_size(dimensions.get("inner_diameter_mm"))
    rows = [["Indian ring size (approx.)", size["display"]]]
    diameter = dimensions.get("inner_diameter_mm")
    if diameter is not None and math.isfinite(float(diameter)) and float(diameter) > 0:
        rows.append(["Estimated inner circumference", f"{math.pi * float(diameter):.2f} mm"])
    if size["indian"]:
        rows.append(["Chart diameter / circumference",
                     f"{size['reference_diameter_mm']:g} mm / {size['reference_circumference_mm']:g} mm"])
    rows.append(["Ring opening", str(dimensions.get("opening_description") or "Visible; ellipse fitted")])
    rows.append(["Measurement basis", str(dimensions.get("measurement_note") or "Ellipse long axis; assumes a circular band")])
    if dimensions.get("is_open_ring_estimate"):
        rows.append(["Estimate note", str(dimensions.get("estimate_note") or "Current opening only; adjustable ring fit can change")])
    return rows
