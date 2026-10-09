from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Any

import cv2
import numpy as np


BEAD_TINY_MAX_DIAMETER_MM = 3.0
BEAD_SMALL_MAX_DIAMETER_MM = 6.0
BEAD_COLOR_SAMPLE_SIDE = 24
BEAD_HUGE_COUNT = 100


def _positive_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    return None


def _metadata_image_size(metadata: dict[str, str]) -> tuple[int, int] | None:
    raw = metadata.get("imgsz")
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        value = raw
    if isinstance(value, (int, np.integer)):
        side = _positive_int(value)
        return (side, side) if side else None
    if isinstance(value, str):
        parts = value.lower().replace("x", " ").replace(",", " ").split()
        value = parts
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = (_positive_int(item) for item in value)
        if height and width:
            return height, width
    return None


def onnx_image_size(session: Any) -> tuple[int, int]:
    """Read detector height/width from Ultralytics metadata and validate its tensor shape."""
    model_input = session.get_inputs()[0]
    shape = list(model_input.shape)
    shape_size: tuple[int, int] | None = None
    if len(shape) == 4:
        height = _positive_int(shape[2])
        width = _positive_int(shape[3])
        if height and width:
            shape_size = (height, width)

    metadata = session.get_modelmeta().custom_metadata_map or {}
    metadata_size = _metadata_image_size(metadata)
    if metadata_size and shape_size and metadata_size != shape_size:
        raise ValueError(
            "ONNX detector imgsz metadata "
            f"{metadata_size} does not match its input tensor {shape_size}"
        )
    image_size = metadata_size or shape_size
    if image_size is None:
        raise ValueError(
            "ONNX detector must provide a fixed NCHW image size or Ultralytics imgsz metadata"
        )
    return image_size


def prepare_onnx_input(
    image_bgr: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, float, int, int]:
    """Letterbox an image to a detector's declared (height, width)."""
    input_height, input_width = image_size
    height, width = image_bgr.shape[:2]
    scale = min(input_width / float(width), input_height / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = (input_width - resized_width) // 2
    top = (input_height - resized_height) // 2
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = np.ascontiguousarray(
        canvas[:, :, ::-1].transpose(2, 0, 1),
        dtype=np.float32,
    ) / 255.0
    return tensor[None], scale, left, top


def _valid_measurement_scale(
    measurement_scale: dict[str, Any] | None,
) -> tuple[float, float] | None:
    try:
        scale_x = float((measurement_scale or {})["mm_per_pixel_x"])
        scale_y = float((measurement_scale or {})["mm_per_pixel_y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([scale_x, scale_y]).all() or scale_x <= 0.0 or scale_y <= 0.0:
        return None
    return scale_x, scale_y


def bead_size_category(diameter_mm: float) -> str:
    if diameter_mm < BEAD_TINY_MAX_DIAMETER_MM:
        return "tiny"
    if diameter_mm < BEAD_SMALL_MAX_DIAMETER_MM:
        return "small"
    return "large"


def _size_analysis(
    detections: list[dict[str, Any]],
    measurement_scale: dict[str, Any] | None,
) -> dict[str, Any]:
    scale = _valid_measurement_scale(measurement_scale)
    if scale is None:
        return {
            "available": False,
            "reason": "AprilTag pixel scale is unavailable",
            "counts": {"tiny": 0, "small": 0, "large": 0},
            "thresholds_mm": {
                "tiny_below": BEAD_TINY_MAX_DIAMETER_MM,
                "small_below": BEAD_SMALL_MAX_DIAMETER_MM,
                "large_from": BEAD_SMALL_MAX_DIAMETER_MM,
            },
        }

    scale_x, scale_y = scale
    diameters: list[float] = []
    counts = Counter({"tiny": 0, "small": 0, "large": 0})
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        width_mm = max(0, x2 - x1) * scale_x
        height_mm = max(0, y2 - y1) * scale_y
        # The geometric mean is stable for slightly non-square detector boxes
        # and preserves their calibrated physical area.
        diameter_mm = math.sqrt(width_mm * height_mm)
        category = bead_size_category(diameter_mm)
        detection.update(
            {
                "width_mm": round(width_mm, 2),
                "height_mm": round(height_mm, 2),
                "diameter_mm": round(diameter_mm, 2),
                "size_category": category,
            }
        )
        counts[category] += 1
        diameters.append(diameter_mm)

    return {
        "available": True,
        "calibration_source": "AprilTag",
        "measurement": "geometric mean of calibrated bounding-box width and height",
        "counts": {name: int(counts[name]) for name in ("tiny", "small", "large")},
        "minimum_diameter_mm": round(min(diameters), 2) if diameters else None,
        "median_diameter_mm": round(float(np.median(diameters)), 2) if diameters else None,
        "maximum_diameter_mm": round(max(diameters), 2) if diameters else None,
        "thresholds_mm": {
            "tiny_below": BEAD_TINY_MAX_DIAMETER_MM,
            "small_below": BEAD_SMALL_MAX_DIAMETER_MM,
            "large_from": BEAD_SMALL_MAX_DIAMETER_MM,
        },
    }


def _hsv_color_name(hue: float, saturation: float, value: float) -> str:
    if value < 55:
        return "black"
    # Neutral beads pick up a moderate yellow cast from nearby gold settings
    # and the warm testbed illumination. A wider neutral band keeps pearls and
    # silver/gray beads from being mislabeled as brown or gold.
    if saturation < 75:
        if value >= 185:
            return "white"
        return "gray"
    if 8 <= hue < 32:
        if hue < 15 and value < 100:
            return "brown"
        return "gold"
    if hue < 8 or hue >= 173:
        return "red"
    if hue < 20:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 100:
        return "cyan"
    if hue < 130:
        return "blue"
    if hue < 155:
        return "purple"
    return "pink"


def _sample_bead_color(
    image_bgr: np.ndarray,
    image_hsv: np.ndarray,
    bbox: list[int],
) -> tuple[str, str, int]:
    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_width, x2), min(image_height, y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown", "#000000", 0

    bgr_crop = image_bgr[y1:y2, x1:x2]
    hsv_crop = image_hsv[y1:y2, x1:x2]
    sample_width = min(BEAD_COLOR_SAMPLE_SIDE, bgr_crop.shape[1])
    sample_height = min(BEAD_COLOR_SAMPLE_SIDE, bgr_crop.shape[0])
    bgr_sample = cv2.resize(bgr_crop, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
    hsv_sample = cv2.resize(hsv_crop, (sample_width, sample_height), interpolation=cv2.INTER_NEAREST)

    yy, xx = np.ogrid[:sample_height, :sample_width]
    normalized_x = (xx + 0.5 - sample_width / 2.0) / max(1.0, sample_width / 2.0)
    normalized_y = (yy + 0.5 - sample_height / 2.0) / max(1.0, sample_height / 2.0)
    # Sampling only the inner ellipse avoids most background and adjacent
    # settings while retaining both the body color and reflective highlights.
    inner = normalized_x * normalized_x + normalized_y * normalized_y <= 0.58
    hsv_pixels = hsv_sample[inner]
    bgr_pixels = bgr_sample[inner]
    if not len(hsv_pixels):
        return "unknown", "#000000", 0

    color_votes = Counter(
        _hsv_color_name(float(h), float(s), float(v))
        for h, s, v in hsv_pixels
    )
    color = color_votes.most_common(1)[0][0]
    values = hsv_pixels[:, 2]
    low, high = np.percentile(values, [5, 95])
    trimmed = bgr_pixels[(values >= low) & (values <= high)]
    if not len(trimmed):
        trimmed = bgr_pixels
    blue, green, red = np.mean(trimmed, axis=0)
    color_hex = f"#{int(round(red)):02x}{int(round(green)):02x}{int(round(blue)):02x}"
    return color, color_hex, int(len(hsv_pixels))


def _color_analysis(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    if not detections:
        return {
            "counts": {},
            "dominant_color": None,
            "sampled_pixels": 0,
            "method": "bounded inner-ellipse HSV sampling",
        }
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    counts: Counter[str] = Counter()
    sampled_pixels = 0
    for detection in detections:
        color, color_hex, sample_count = _sample_bead_color(
            image_bgr,
            image_hsv,
            detection["bbox"],
        )
        detection["color"] = color
        detection["average_color_hex"] = color_hex
        counts[color] += 1
        sampled_pixels += sample_count
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    dominant_color = next(iter(ordered), None)
    return {
        "counts": ordered,
        "dominant_color": dominant_color,
        "dominant_share": round(counts[dominant_color] / len(detections), 3) if dominant_color else 0.0,
        "sampled_pixels": sampled_pixels,
        "maximum_samples_per_bead": BEAD_COLOR_SAMPLE_SIDE * BEAD_COLOR_SAMPLE_SIDE,
        "method": "single HSV conversion plus bounded inner-ellipse sampling",
    }


def _arrangement_analysis(detections: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(detections)
    if count < 2:
        return {
            "pattern": "isolated" if count else "none",
            "description": "One isolated bead is present." if count else "No beads were detected.",
            "repetitive": False,
        }

    boxes = np.asarray([item["bbox"] for item in detections], dtype=np.float32)
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
    diameters = np.sqrt(
        np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    )
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(count), nearest_indices]
    neighbor_diameters = diameters[nearest_indices]
    surface_gaps = np.maximum(0.0, nearest_distances - (diameters + neighbor_diameters) / 2.0)
    median_diameter = max(1.0, float(np.median(diameters)))
    normalized_gaps = surface_gaps / median_diameter
    size_cv = float(np.std(diameters) / max(1.0, float(np.mean(diameters))))
    distance_cv = float(
        np.std(nearest_distances) / max(1.0, float(np.mean(nearest_distances)))
    )
    close_neighbor_share = float(np.mean(normalized_gaps <= 0.75))
    repetitive = size_cv <= 0.30 and distance_cv <= 0.45
    continuous = count >= 5 and repetitive and close_neighbor_share >= 0.70
    well_spaced = not continuous and float(np.median(normalized_gaps)) >= 0.85
    if continuous:
        pattern = "continuous"
        description = "Continuous, repetitive beads are present."
    elif well_spaced:
        pattern = "well_spaced"
        description = "Well-spaced beads are present."
    else:
        pattern = "mixed_spacing"
        description = "Beads are present with mixed spacing or sizes."
    return {
        "pattern": pattern,
        "description": description,
        "repetitive": repetitive,
        "size_coefficient_of_variation": round(size_cv, 3),
        "nearest_distance_coefficient_of_variation": round(distance_cv, 3),
        "median_surface_gap_in_bead_diameters": round(float(np.median(normalized_gaps)), 3),
        "close_neighbor_share": round(close_neighbor_share, 3),
        "criteria": {
            "minimum_continuous_count": 5,
            "maximum_size_cv": 0.30,
            "maximum_nearest_distance_cv": 0.45,
            "minimum_close_neighbor_share": 0.70,
            "well_spaced_median_gap": 0.85,
        },
    }


def _size_description(size_analysis: dict[str, Any], total: int) -> str | None:
    if not size_analysis.get("available") or total == 0:
        return None
    counts = size_analysis["counts"]
    present = [name for name in ("large", "small", "tiny") if counts[name]]
    small_bead_count = counts["small"] + counts["tiny"]
    if small_bead_count >= BEAD_HUGE_COUNT:
        if counts["small"] and counts["tiny"]:
            detail = f"{counts['small']} small and {counts['tiny']} tiny"
            noun = "small and tiny beads"
        elif counts["small"]:
            detail = f"{counts['small']} small"
            noun = "small beads"
        else:
            detail = f"{counts['tiny']} tiny"
            noun = "tiny beads"
        large_suffix = f", plus {counts['large']} large" if counts["large"] else ""
        return f"A huge number of {noun} are present ({detail}{large_suffix})."
    if len(present) == 1:
        category = present[0]
        if total >= BEAD_HUGE_COUNT:
            return f"A huge number of {category} beads are present ({total})."
        return f"{category.capitalize()} beads are present ({total})."
    return (
        "Mixed bead sizes: "
        f"{counts['large']} large, {counts['small']} small, and {counts['tiny']} tiny."
    )


def _color_description(color_analysis: dict[str, Any], total: int) -> str | None:
    counts = color_analysis.get("counts") or {}
    if not counts or total == 0:
        return None
    ordered = list(counts.items())
    dominant, dominant_count = ordered[0]
    if dominant_count / total >= 0.80:
        return f"Predominantly {dominant} beads ({dominant_count} of {total})."
    shown = ", ".join(f"{name} {count}" for name, count in ordered[:4])
    remaining = sum(count for _, count in ordered[4:])
    suffix = f", other {remaining}" if remaining else ""
    return f"Bead colors: {shown}{suffix}."


def analyze_bead_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    measurement_scale: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrich detections with calibrated size, bounded HSV color, and spacing evidence."""
    size_analysis = _size_analysis(detections, measurement_scale)
    color_analysis = _color_analysis(image_bgr, detections)
    arrangement = _arrangement_analysis(detections)
    total = len(detections)
    insights = [f"{total} bead{'s' if total != 1 else ''} detected."]
    for statement in (
        _size_description(size_analysis, total),
        _color_description(color_analysis, total),
        arrangement["description"] if total else None,
    ):
        if statement:
            insights.append(statement)
    return {
        "size": size_analysis,
        "colors": color_analysis,
        "arrangement": arrangement,
        "insights": insights,
        "summary": " ".join(insights),
    }
