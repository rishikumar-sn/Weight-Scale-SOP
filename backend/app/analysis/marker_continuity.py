"""Recover item ownership only where marker padding erased a visible strand.

Marker pixels and recovered bridge pixels never enter measurement masks. A path
must stay outside the tag plus its quiet zone, and inside the old marker erase
band. Unknown paths require a new placement rather than an invented item count.
"""
from __future__ import annotations

import cv2
import numpy as np

from .vision import marker_ignore_mask

MARKER_PLACEMENT_MESSAGE = "Move jewellery away from the marker and recapture."


def resolve_marker_fragments(working, items, *, original, processing_roi,
                             tag_roi, corners, valid_area_mask):
    check = {"passed": True, "method": "visible_marker_padding_continuity",
             "merged_groups": [], "affected_items": []}
    if not tag_roi or not items:
        return items, {**check, "skipped": True}
    shape = working.shape[:2]
    erased = cv2.dilate(marker_ignore_mask(shape, tag_roi), np.ones((25, 25), np.uint8)) > 0
    # Work with the actual returned masks, including existing tray-edge recovery.
    owners = np.zeros(shape, np.int32)
    for index, item in enumerate(items, 1):
        b = item["bbox"]
        local = owners[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
        local[item["mask"] > 0] = index
    adjacent = cv2.dilate(erased.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    nearby = np.unique(owners[adjacent]); nearby = nearby[nearby > 0].tolist()
    if len(nearby) < 2:
        return items, check

    def uncertain(indices, reason):
        check.update(passed=False, reason=reason,
                     affected_items=[{"item_index": i, "bbox": items[i - 1]["bbox"]} for i in indices])
        return items, check

    # Never use a differently mapped photograph as connectivity evidence.
    try:
        x, y, w, h = (int(processing_roi[k]) for k in ("x", "y", "w", "h"))
        points = np.asarray(corners, np.float32).reshape(4, 2) - (x, y)
        if (original is None or x < 0 or y < 0 or w != shape[1] or h != shape[0]
                or x + w > original.shape[1] or y + h > original.shape[0]
                or not np.isfinite(points).all() or cv2.contourArea(points.astype(np.float32)) < 16
                or valid_area_mask is None or valid_area_mask.shape[:2] != shape):
            return uncertain(nearby, "missing_or_invalid_source_geometry")
        raw = original[y:y + h, x:x + w]
        if not np.array_equal(raw[owners > 0], working[owners > 0]):
            return uncertain(nearby, "source_pixels_do_not_match")
    except (KeyError, TypeError, ValueError):
        return uncertain(nearby, "missing_or_invalid_source_geometry")

    # Protect the tag and a 15% quiet zone plus a pixel clearance. The existing
    # 35% rectangular allowance and its two extra padding layers remain in the
    # measurement mask, but must not hide visible connectivity evidence.
    centre = points.mean(axis=0)
    protected = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(protected, np.rint(centre + (points - centre) * 1.3).astype(np.int32), 1)
    edge = max(float(np.linalg.norm(points[i] - points[(i + 1) % 4])) for i in range(4))
    clearance = max(2, int(np.ceil(edge * .04)))
    protected = cv2.dilate(protected, np.ones((clearance * 2 + 1,) * 2, np.uint8))
    valid = cv2.erode((valid_area_mask > 0).astype(np.uint8), np.ones((25, 25), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    band = erased & (protected == 0) & valid
    # Connectivity needs actual object pixels. The general foreground builder
    # closes/dilates gaps for detection; using it here could join two real jewels.
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    background = cv2.medianBlur(gray, 31).astype(np.int16)
    dark = gray.astype(np.int16) < background - 12
    colour = (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 28)
    foreground = (dark | colour).astype(np.uint8)
    bridge = cv2.morphologyEx((foreground * band).astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    continuity = bridge.copy()
    continuity[owners > 0] = 1
    _, labels = cv2.connectedComponents(continuity, connectivity=8)
    groups = {}
    for index in nearby:
        ids = np.unique(labels[owners == index])
        for label in ids[ids > 0]:
            groups.setdefault(int(label), set()).add(index)
    pairs = []
    for label, members in groups.items():
        members = sorted(members)
        if len(members) > 2:
            return uncertain(members, "ambiguous_multiple_objects")
        if len(members) == 2:
            pairs.append(members)
    # A component can have disconnected restored stone interiors, but it cannot
    # participate in two independent pairing decisions.
    flattened = [i for pair in pairs for i in pair]
    if len(flattened) != len(set(flattened)):
        return uncertain(sorted(set(flattened)), "ambiguous_multiple_paths")

    # Two separate proposals running into the protected tag may be one hidden
    # strand or two objects. One already-connected proposal has no split count
    # to resolve; proximity alone must not reject that existing single item.
    boundary = (cv2.dilate(protected, np.ones((3, 3), np.uint8)) > 0) & (protected == 0)
    unresolved = []
    for label, members in groups.items():
        if not set(members).intersection(flattened) and np.any((labels == label) & boundary & (bridge > 0)):
            unresolved.extend(members)
    if len(set(unresolved)) >= 2:
        return uncertain(sorted(set(unresolved)), "strand_reaches_protected_marker")

    result = [item for i, item in enumerate(items, 1) if i not in flattened]
    for pair in pairs:
        selected = [items[i - 1] for i in pair]
        x1 = min(item["bbox"]["x"] for item in selected)
        y1 = min(item["bbox"]["y"] for item in selected)
        x2 = max(item["bbox"]["x"] + item["bbox"]["w"] for item in selected)
        y2 = max(item["bbox"]["y"] + item["bbox"]["h"] for item in selected)
        mask = np.isin(owners[y1:y2, x1:x2], pair).astype(np.uint8)
        source = working[y1:y2, x1:x2]
        crop = np.full_like(source, 255); crop[mask > 0] = source[mask > 0]
        result.append({"bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                       "area_px": int(cv2.countNonZero(mask)), "mask": mask,
                       "raw_crop_bgr": source.copy(), "crop_bgr": crop})
    result.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    check["merged_groups"] = pairs
    return result, check
