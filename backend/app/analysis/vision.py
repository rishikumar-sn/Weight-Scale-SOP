from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def roi_to_pixels(roi: dict[str, Any] | None, width: int, height: int) -> dict[str, int] | None:
    if not roi:
        return None
    x = max(0, min(width - 1, int(round(float(roi["x"]) * width))))
    y = max(0, min(height - 1, int(round(float(roi["y"]) * height))))
    w = max(1, int(round(float(roi["width"]) * width)))
    h = max(1, int(round(float(roi["height"]) * height)))
    w = min(w, width - x)
    h = min(h, height - y)
    return {"x": x, "y": y, "w": w, "h": h}


def translate_roi_to_crop(
    child: dict[str, int] | None,
    parent: dict[str, int] | None,
    crop_width: int,
    crop_height: int,
) -> dict[str, int] | None:
    if not child:
        return None
    x = child["x"] - (parent["x"] if parent else 0)
    y = child["y"] - (parent["y"] if parent else 0)
    x2 = min(crop_width, x + child["w"])
    y2 = min(crop_height, y + child["h"])
    x = max(0, x)
    y = max(0, y)
    if x2 <= x or y2 <= y:
        return None
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def crop(image: np.ndarray, roi: dict[str, int] | None) -> np.ndarray:
    if not roi:
        return image.copy()
    return image[roi["y"] : roi["y"] + roi["h"], roi["x"] : roi["x"] + roi["w"]].copy()


def detect_apriltag(
    image: np.ndarray,
    roi: dict[str, int] | None,
    marker_id: int,
    width_mm: float,
    height_mm: float,
) -> dict[str, Any]:
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError("AprilTag support is unavailable in OpenCV")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # The configured ROI is a speed hint, not a hard dependency. Camera or
    # platform movement can leave the tag visible while making that crop too
    # tight for AprilTag's required surrounding border. Retry on the complete
    # frame before declaring calibration unavailable.
    detection_roi = roi
    region = crop(image, detection_roi)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None and detection_roi is not None:
        detection_roi = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise RuntimeError("AprilTag was not found")
    ids_flat = ids.reshape(-1).tolist()
    if marker_id not in ids_flat:
        raise RuntimeError(f"AprilTag {marker_id} was not found")
    index = ids_flat.index(marker_id)
    points = corners[index][0].astype(np.float32)
    if detection_roi:
        points[:, 0] += detection_roi["x"]
        points[:, 1] += detection_roi["y"]
    top = float(np.linalg.norm(points[1] - points[0]))
    right = float(np.linalg.norm(points[2] - points[1]))
    bottom = float(np.linalg.norm(points[3] - points[2]))
    left = float(np.linalg.norm(points[0] - points[3]))
    horizontal_px = (top + bottom) / 2.0
    vertical_px = (left + right) / 2.0
    return {
        "found": True,
        "id": marker_id,
        "detection_scope": "configured_roi" if detection_roi else "full_frame_fallback",
        "corners": points.tolist(),
        "mm_per_pixel_x": width_mm / max(horizontal_px, 1e-6),
        "mm_per_pixel_y": height_mm / max(vertical_px, 1e-6),
    }


def apriltag_roi_from_detection(
    image_shape: tuple[int, ...],
    detection: dict[str, Any],
    *,
    padding_ratio: float = 0.35,
    minimum_padding_px: int = 8,
) -> dict[str, int]:
    """Build a clipped rectangular ROI around detected tag corners.

    The extra band includes the printed tag's quiet zone and a little platform
    background, ensuring thresholding cannot retain a dark fringe around it.
    """
    height, width = image_shape[:2]
    points = np.asarray(detection.get("corners"), dtype=np.float32).reshape(-1, 2)
    if len(points) != 4 or not np.isfinite(points).all():
        raise ValueError("AprilTag detection does not contain four valid corners")

    tag_width = max(
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[3])),
    )
    tag_height = max(
        float(np.linalg.norm(points[2] - points[1])),
        float(np.linalg.norm(points[3] - points[0])),
    )
    pad_x = max(int(minimum_padding_px), int(np.ceil(tag_width * max(0.0, padding_ratio))))
    pad_y = max(int(minimum_padding_px), int(np.ceil(tag_height * max(0.0, padding_ratio))))
    x1 = max(0, int(np.floor(float(points[:, 0].min()))) - pad_x)
    y1 = max(0, int(np.floor(float(points[:, 1].min()))) - pad_y)
    x2 = min(width, int(np.ceil(float(points[:, 0].max()))) + pad_x + 1)
    y2 = min(height, int(np.ceil(float(points[:, 1].max()))) + pad_y + 1)
    return {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}


def roi_to_normalized(roi: dict[str, int], width: int, height: int) -> dict[str, float]:
    return {
        "x": roi["x"] / width,
        "y": roi["y"] / height,
        "width": roi["w"] / width,
        "height": roi["h"] / height,
    }


def marker_ignore_mask(shape: tuple[int, ...], roi: dict[str, int] | None) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if roi:
        pad = max(5, int(round(max(roi["w"], roi["h"]) * 0.08)))
        x1 = max(0, roi["x"] - pad)
        y1 = max(0, roi["y"] - pad)
        x2 = min(shape[1], roi["x"] + roi["w"] + pad)
        y2 = min(shape[0], roi["y"] + roi["h"] + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def _threshold(image: np.ndarray, white_background: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if white_background:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, fixed = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        result = cv2.bitwise_or(otsu, fixed)
    else:
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    hue, saturation, value = cv2.split(hsv)
    color = ((saturation > 45) & (value > 28) & (((hue >= 5) & (hue <= 110)) | (hue >= 125)))
    color = cv2.dilate(
        color.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    result = cv2.bitwise_or(result, edges)
    result = cv2.bitwise_or(result, color)
    return cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )


def _score(mask: np.ndarray) -> float:
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    inner = []
    border = []
    h, w = binary.shape
    minimum = max(80, int(binary.size * 0.0005))
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        x, y, cw, ch = (int(stats[index, key]) for key in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
        ))
        (border if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1 else inner).append(area)
    if not inner:
        return -1_000_000.0
    ratio = float(binary.mean())
    return max(inner) + len(inner) * 250 - (max(border, default=0) * 0.75) - (binary.size * ratio if ratio > 0.45 else 0)


def build_jewellery_mask(image: np.ndarray, erase_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    corners = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]
    preferred_white = float(np.mean([item.mean() for item in corners])) > 180
    masks = [_threshold(image, preferred_white), _threshold(image, not preferred_white)]
    mask = max(masks, key=_score)
    if erase_mask is not None and np.any(erase_mask):
        padded = cv2.dilate((erase_mask > 0).astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1)
        mask[padded > 0] = 0
    mask = (mask > 0).astype(np.uint8)
    prepared = np.full_like(image, 255)
    prepared[mask > 0] = image[mask > 0]
    return prepared, mask


def _restore_enclosed_dark_regions(
    image: np.ndarray,
    component_mask: np.ndarray,
) -> np.ndarray:
    """Restore dark stone interiors lost by local adaptive thresholding.

    A polished black stone can be nearly uniform in its centre. Adaptive
    thresholding then retains its reflective rim but treats the centre like
    background. Only dark holes fully enclosed by the selected jewellery
    component are restored; ordinary openings retain the sampled background
    colour and remain empty.
    """
    component = (component_mask > 0).astype(np.uint8)
    component_area = int(cv2.countNonZero(component))
    if component_area <= 0:
        return component

    inverse = (component == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    if count <= 1:
        return component

    border_labels = set(
        np.unique(
            np.concatenate(
                (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
            )
        ).tolist()
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    outside_pixels = gray[component == 0]
    if outside_pixels.size == 0:
        return component
    background_luma = float(np.median(outside_pixels))
    dark_limit = background_luma - 18.0
    max_hole_area = max(16, int(round(component_area * 0.08)))
    restored = component.copy()

    for label in range(1, count):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4 or area > max_hole_area:
            continue
        pixels = labels == label
        values = gray[pixels]
        if (
            float(np.median(values)) <= dark_limit
            and float(np.mean(values <= dark_limit)) >= 0.75
        ):
            restored[pixels] = 1

    return restored


def separate_jewellery_items(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    min_area_px: int | None = None,
    min_area_ratio: float = 0.00045,
    padding_px: int = 18,
) -> list[dict[str, Any]]:
    """Return spatially separated jewels from the shared Otsu foreground mask.

    Each returned crop contains only its own connected component on white. This
    prevents neighbouring jewels from influencing SigLIP or a downstream
    dimension/stone detector. Objects must be placed with a visible gap; two
    touching objects are intentionally treated as one instead of inventing an
    unreliable split.
    """
    if image is None or image.size == 0:
        raise ValueError("A non-empty jewellery image is required")
    if mask is None or mask.shape[:2] != image.shape[:2]:
        raise ValueError("The jewellery mask must match the image dimensions")

    binary = (mask > 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    minimum = max(
        80,
        int(min_area_px or 0),
        int(round(binary.size * max(0.0, float(min_area_ratio)))),
    )
    items: list[dict[str, Any]] = []
    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        if component_width < 10 or component_height < 10:
            continue
        aspect = max(component_width, component_height) / max(1, min(component_width, component_height))
        extent = area / max(1, component_width * component_height)
        if aspect > 10.0 and area < minimum * 6:
            continue
        if extent < 0.01 and area < minimum * 10:
            continue

        x1 = max(0, x - max(0, padding_px))
        y1 = max(0, y - max(0, padding_px))
        x2 = min(width, x + component_width + max(0, padding_px))
        y2 = min(height, y + component_height + max(0, padding_px))
        local_component = (labels[y1:y2, x1:x2] == component_index).astype(np.uint8)
        source = image[y1:y2, x1:x2]
        local_mask = _restore_enclosed_dark_regions(source, local_component)
        crop_image = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
        crop_image[local_mask > 0] = source[local_mask > 0]
        items.append({
            "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "area_px": int(cv2.countNonZero(local_mask)),
            # Keep the untouched camera pixels for models trained on natural
            # backgrounds.  ``crop_bgr`` remains the Otsu-isolated version
            # used by classification and segmentation.
            "raw_crop_bgr": source.copy(),
            "crop_bgr": crop_image,
            "mask": local_mask,
        })

    # Stable reading order makes item numbers repeatable across runs.
    items.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    return items
