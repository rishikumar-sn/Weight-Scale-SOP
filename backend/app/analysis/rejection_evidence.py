"""Explain failed checks using their own object locations, without new inference."""
from __future__ import annotations


def source_bbox(state: dict, bbox: dict | None) -> dict:
    if not bbox:
        return {}
    origin = (state.get("rois") or {}).get("processing") or {}
    return {**bbox, "x": bbox.get("x", 0) + origin.get("x", 0),
            "y": bbox.get("y", 0) + origin.get("y", 0)}


def rejection_regions(state: dict) -> list[dict]:
    """Keep unsupported foreground distinct from a predicted foreign object.

    Coordinates always refer to the raw capture. Never fabricate a location for
    unavailable verification, or infer that a REVIEW decision proves a print.
    """
    regions = []
    check = state.get("foreign_object_check") or {}
    components = {c["source_component_index"]: c for c in state.get("foreground_components", [])}
    recognized = {}
    for detection in check.get("recognized_objects", []):
        for match in detection.get("component_overlaps", []):
            if match.get("component_share", 0) >= .5:
                recognized[match["component_index"]] = detection["label"].capitalize()

    def add(tag, label, reason, bbox, gate, annotation=None):
        region = {"tag": tag, "label": label, "reason": reason, "bbox": bbox or {},
                  "gate": gate, "annotation": annotation or f"{tag}: {label}"}
        if bbox and bbox.get("w", 0) > 0 and bbox.get("h", 0) > 0:
            tray = (state.get("rois") or {}).get("testbed_source") or {}
            cx = (bbox["x"] + bbox["w"] / 2 - tray.get("x", 0)) / max(1, tray.get("w", 0))
            cy = (bbox["y"] + bbox["h"] / 2 - tray.get("y", 0)) / max(1, tray.get("h", 0))
            region["location"] = ("top" if cy < 1/3 else "bottom" if cy > 2/3 else "middle") + " " + (
                "left" if cx < 1/3 else "right" if cx > 2/3 else "centre") if tray else "marked in image"
        else:
            region["location"] = "location unavailable"
        regions.append(region)

    for number, detection in enumerate(check.get("detections", []), 1):
        label = str(detection.get("label", "")).replace("_", " ").capitalize()
        if detection.get("label") not in {"foreign_obj", "paper", "plastic_packet"}:
            continue
        if label == "Foreign obj":
            label = "Foreign object"
        add(f"A{number}", label, f"{label} detected. Remove it and recapture.",
            detection.get("bbox"), "foreign_objects")

    for item in (state.get("component_presence_check") or {}).get("items", []):
        if item.get("passed") is not False:
            continue
        index = item.get("source_component_index")
        original = item.get("original_component_index", index)
        component = components.get(original, {})
        bbox = item.get("source_bbox") or component.get("source_bbox")
        add(f"U{index}", "Unverified object", "Could not verify this object as jewellery, a coin or a bar. Remove or reposition it and recapture.",
            bbox, "component_presence")

    for number, item in enumerate((state.get("real_print_check") or {}).get("items", []), 1):
        if item.get("passed") is True or item.get("skipped"):
            continue
        index = item.get("item_index", number)
        label = item.get("label", "REVIEW")
        kind = recognized.get(index, "")
        uncertain = label != "PRINT"
        name = "Verification uncertain" if uncertain else "Possible print"
        annotation = f"P{index} {kind}: {name}" if kind else f"P{index}: {label}"
        reason = ("Real/print check is uncertain. This does not establish whether the object is gold. Reposition and recapture."
                  if uncertain else "Real/print model flagged a possible print. Check the physical object and recapture.")
        add(f"P{index}", f"{kind}: {name}" if kind else name, reason,
            item.get("source_bbox") or source_bbox(state, item.get("bbox")), "real_print", annotation)

    for number, item in enumerate((state.get("thread_only_check") or {}).get("items", []), 1):
        if item.get("passed") is not False:
            continue
        label = "Thread without jewellery" if item.get("label") == "thread_only" else "No verified jewellery"
        add(f"T{item.get('item_index', number)}", label, "Place physical jewellery clearly on the tray and recapture.",
            source_bbox(state, item.get("bbox")), "thread_only")

    placement = check.get("placement_check") or {}
    for item in placement.get("affected_components", []):
        index = item["component_index"]
        add(f"M{index}", "Objects together", "Separate these objects and recapture.",
            components.get(index, {}).get("source_bbox"), "placement")
    for item in (state.get("placement_check") or {}).get("affected_items", []):
        add(f"M{item['item_index']}", "Placement needs checking", "Separate or lay this object flat and recapture.",
            source_bbox(state, item.get("bbox")), "placement")
    if state.get("rejection_kind") == "non_jewellery":
        for number, item in enumerate(state.get("other_objects", []), 1):
            if item.get("kind") == "non_jewellery":
                add(f"O{number}", "Object not verified as jewellery", "Remove this object and recapture.",
                    source_bbox(state, item.get("bbox")), "semantic_validation")
    if (state.get("object_assignment_check") or {}).get("passed") is False:
        failures = check.get("object_assignment_failure_indices")
        for number, detection in enumerate(check.get("recognized_objects", []), 1):
            if failures is not None and number not in failures:
                continue
            label = detection["label"].capitalize()
            add(f"U{number}", f"{label}: ambiguous boundary", "This mask could not be matched to one separate object. Separate objects and recapture.",
                detection.get("bbox"), "object_assignment")
    return regions


def rejection_summary(state: dict, regions: list[dict]) -> str:
    if not regions:
        return state.get("rejection_reason") or "Verification failed. Please recapture."
    prefix = "Other objects detected. " if any(r["gate"] == "foreign_objects" for r in regions) else ""
    return prefix + " ".join(f"{r['tag']} ({r['location']}): {r['label']}. {r['reason']}" for r in regions)
