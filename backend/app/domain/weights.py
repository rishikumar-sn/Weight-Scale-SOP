"""Gram-only presentation of estimated stone deductions."""
from __future__ import annotations

import math
from typing import Any


def valid_weight(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def stone_weight_fields(measurements: dict, found: bool) -> dict:
    average = (
        valid_weight(measurements.get("estimated_total_average_g"))
        if measurements.get("success") else None
    )
    if not found:
        average = 0.0
    return {
        "estimated_weight_g": average,
        "weight_min_g": average,
        "weight_max_g": round(average + (1.0 if found else 0.0), 4) if average is not None else None,
    }


def without_stone_weight_estimates(value: Any) -> Any:
    """Remove gemstone mass/carat estimates while preserving stone analysis."""
    if isinstance(value, dict):
        cleaned = {}
        has_weight_fields = any(
            "weight" in str(key).lower()
            or str(key).lower().endswith("_ct")
            or str(key).lower().endswith("_g")
            or str(key).lower().startswith("estimated_total_")
            for key in value
        )
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                "weight" in normalized
                or normalized.endswith("_ct")
                or normalized.endswith("_g")
                or normalized.startswith("estimated_total_")
            ):
                continue
            if (
                has_weight_fields
                and normalized == "note"
                and isinstance(item, str)
                and ("weight" in item.lower() or "carat" in item.lower())
            ):
                continue
            cleaned[key] = without_stone_weight_estimates(item)
        return cleaned
    if isinstance(value, list):
        return [without_stone_weight_estimates(item) for item in value]
    return value


def weight_summary(state: dict) -> dict:
    result = state.get("result") or {}
    items = result.get("items") or []
    count = result.get("count", (state.get("classification") or {}).get("count", len(items)))
    gross = valid_weight(state.get("weight_g"))
    summary = dict(gross_g=gross, stone_g=None, stone_min_g=None, stone_max_g=None,
                   net_g=None, net_min_g=None, net_max_g=None)
    if count != 1 or len(items) > 1:
        return {
            **summary,
            "note": "Stone-weight and net-weight estimates are available only for a single-jewel capture.",
        }
    stones = (items[0] if items else result).get("stones") or {}
    average = valid_weight(stones.get("estimated_weight_g"))
    lower = valid_weight(stones.get("weight_min_g"))
    upper = valid_weight(stones.get("weight_max_g"))
    if average is None or lower is None or upper is None:
        return {**summary, "note": "Stone weight unavailable; calibrated stone analysis is required."}
    summary.update(stone_g=average, stone_min_g=lower, stone_max_g=upper)
    note = "Estimated stone range: average to average +1.00 g. Net = gross minus stones."
    if average == upper == 0:
        note = "No stones detected; estimated stone deduction is 0.00 g."
    if gross is None:
        note += " Gross scale weight unavailable."
    elif average > gross:
        note += " Stone estimate exceeds gross weight; recapture before calculating net weight."
    else:
        summary.update(net_g=round(gross - average, 4),
                       net_min_g=round(max(0.0, gross - upper), 4),
                       net_max_g=round(gross - lower, 4))
        if upper > gross:
            note += " Net range is limited to zero at its lower end."
    if stones.get("found"):
        note += " Hidden depth and material are estimated; the +1 g allowance is not a validated accuracy bound."
    if ((items[0] if items else result).get("beads") or {}).get("beads_detected"):
        note += " Bead weight is not deducted separately."
    return {**summary, "note": note}


def weight_rows(summary: dict) -> list[list[str]]:
    def grams(value):
        return f"{value:.2f} g" if value is not None else "Unavailable"

    def interval(low, high):
        if low is None or high is None:
            return "Unavailable"
        return grams(low) if low == high else f"{low:.2f} - {high:.2f} g"

    return [
        ["Weight", "Value", "Range"],
        ["Gross (scale)", grams(summary["gross_g"]), "-"],
        ["Stone (estimated)", grams(summary["stone_g"]), interval(summary["stone_min_g"], summary["stone_max_g"])],
        ["Net (estimated)", grams(summary["net_g"]), interval(summary["net_min_g"], summary["net_max_g"])],
    ]
