"""Require object evidence before a foreground fragment gets a jewellery type."""
from __future__ import annotations


MIN_COMPONENT_JEWELLERY_SHARE = 0.20
PRESENCE_MESSAGE = (
    "Some foreground regions could not be verified as jewellery. "
    "Remove loose objects, shadows or reflections where possible, "
    "keep the jewellery clearly visible, and recapture."
)


def component_presence_check(check: dict, component_count: int) -> dict:
    """Use foreground-pixel overlap, never enclosing boxes or SigLIP categories.

    Missing detections are uncertainty, not proof that a region is background.
    Reject the capture instead of silently dropping a possible genuine jewel.
    A permissive 20% support requirement tolerates partially predicted masks.
    """
    instances = check.get("jewellery_instances")
    items = []
    for index in range(1, component_count + 1):
        shares = [float(match.get("component_share", 0))
                  for instance in (instances or [])
                  for match in instance.get("component_overlaps", [])
                  if match.get("component_index") == index]
        support = max(shares, default=0.0)
        passed = support >= MIN_COMPONENT_JEWELLERY_SHARE
        items.append({"source_component_index": index, "passed": passed,
                      "jewellery_mask_share": support,
                      "reason": "supported_foreground" if passed else "unverified_foreground"})
    return {"passed": bool(items) and all(item["passed"] for item in items),
            "method": "foreground_pixels_supported_by_jewellery_instance",
            "minimum_component_share": MIN_COMPONENT_JEWELLERY_SHARE,
            "instance_evidence_available": isinstance(instances, list), "items": items}


def blocks_jewellery_labels(state: dict) -> bool:
    """Also protect legacy captures that retain classifications after rejection."""
    check = state.get("foreign_object_check") or {}
    foreign = any(str(d.get("label", "")).lower() in {"paper", "plastic_packet", "foreign_obj"}
                  for d in check.get("detections", []))
    return (foreign or check.get("passed") is False
            or (state.get("component_presence_check") or {}).get("passed") is False)


def empty_classification() -> dict:
    return {"count": 0, "items": [], "labels": [], "confirmed": False,
            "predicted_label": "", "confirmed_label": None}
