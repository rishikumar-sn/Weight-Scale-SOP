"""Print color/shape evidence for stored bead candidates during threshold tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def candidate_features(image: np.ndarray, bbox: list[int]) -> dict[str, float]:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    padding = int(round(max(width, height) * 0.12))
    crop = image[
        max(0, y1 - padding) : min(image.shape[0], y2 + padding),
        max(0, x1 - padding) : min(image.shape[1], x2 + padding),
    ]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    chromatic = saturation >= 55
    red = chromatic & ((hue <= 9) | (hue >= 165))
    dark_red = red & (value <= 185)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    return {
        "red_share": round(float(np.mean(red)), 3),
        "dark_red_share": round(float(np.mean(dark_red)), 3),
        "chromatic_share": round(float(np.mean(chromatic)), 3),
        "median_red_value": round(float(np.median(value[red])) if np.any(red) else 255.0, 1),
        "gray_std": round(float(np.std(gray)), 1),
        "edge_share": round(float(np.mean(edges > 0)), 3),
        "width": width,
        "height": height,
    }


def repeated_red_components(image: np.ndarray, jewel_mask: np.ndarray) -> dict:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (
        (jewel_mask > 0)
        & (saturation >= 70)
        & (value >= 35)
        & (value <= 210)
        & ((hue <= 10) | (hue >= 165))
    ).astype(np.uint8) * 255
    red = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    minimum_area = max(18, int(jewel_mask.size * 0.00004))
    maximum_area = max(minimum_area + 1, int(jewel_mask.size * 0.035))
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if not minimum_area <= area <= maximum_area or min(width, height) <= 2:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * cv2.contourArea(contour) / max(1.0, perimeter * perimeter)
        aspect = max(width, height) / max(1.0, min(width, height))
        if circularity >= 0.35 and aspect <= 2.2:
            components.append({"area": area, "circularity": round(float(circularity), 3)})
    areas = np.asarray([item["area"] for item in components], dtype=np.float32)
    return {
        "count": len(components),
        "median_area": round(float(np.median(areas)), 1) if areas.size else 0.0,
        "area_cv": round(float(np.std(areas) / max(1.0, np.mean(areas))), 3) if areas.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", type=Path)
    parser.add_argument("session_ids", nargs="+")
    args = parser.parse_args()
    for session_id in args.session_ids:
        manifest = json.loads((args.sessions / session_id / "manifest.json").read_text(encoding="utf-8"))
        item = manifest["result"]["items"][0]
        beads = item.get("beads") or {}
        classified = manifest["classification"]["items"][0]
        image = cv2.imread(classified["paths"]["raw"])
        jewel_mask = cv2.imread(classified["paths"]["mask"], cv2.IMREAD_GRAYSCALE)
        features = [
            {**candidate, **candidate_features(image, candidate["bbox"])}
            for candidate in beads.get("onnx_detections", [])
        ]
        print(json.dumps({
            "session": session_id,
            "candidate_count": len(features),
            "red_candidate_count": sum(item["red_share"] >= 0.25 for item in features),
            "dark_red_candidate_count": sum(
                item["dark_red_share"] >= 0.20 and item["median_red_value"] <= 165
                for item in features
            ),
            "repeated_red_components": repeated_red_components(image, jewel_mask),
            "candidate_features": features,
        }))


if __name__ == "__main__":
    main()
