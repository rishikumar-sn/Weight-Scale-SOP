"""Bangle labels from the supplied inner-diameter chart (labels are strings)."""
from __future__ import annotations

import math
from typing import Any

# Use the customer's millimeter chart directly, without mixing reference charts.
BANGLE_SIZE_CHART = (
    ("2.2", 54.0, "Petite / Small"),
    ("2.4", 57.0, "Small / Average"),
    ("2.6", 60.0, "Average / Medium"),
    ("2.8", 63.0, "Large"),
    ("3.0", 76.0, "Extra Large / Men's"),
)


def bangle_size(inner_diameter_mm: Any) -> dict[str, Any]:
    """Nearest listed size, smaller size at a midpoint; never clamp outside chart."""
    result = {"label": None, "display": "Unavailable", "fit": None,
              "reference_diameter_mm": None, "method": "nearest_inner_diameter",
              "chart": "customer_mm_v1", "status": "unavailable"}
    try:
        diameter = float(inner_diameter_mm)
    except (TypeError, ValueError, OverflowError):
        return result
    if not math.isfinite(diameter) or diameter <= 0:
        return result
    if not BANGLE_SIZE_CHART[0][1] <= diameter <= BANGLE_SIZE_CHART[-1][1]:
        return dict(result, display="Outside supplied size chart", status="out_of_range")
    label, reference, fit = min(BANGLE_SIZE_CHART, key=lambda row: abs(row[1] - diameter))
    return dict(result, label=label, display=f"{label} (approx.)", fit=fit,
                reference_diameter_mm=reference, status="matched")


def size_for_dimensions(dimensions: dict[str, Any]) -> dict[str, Any]:
    if dimensions.get("applicable") is False:
        return bangle_size(None)
    return bangle_size(dimensions.get("inner_diameter_mm"))
