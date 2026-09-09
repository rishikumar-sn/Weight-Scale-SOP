"""
Classical CV bangle OD/ID detector with CLI and PyQt6 GUI.

Examples:
    python bangle_detector.py --image path/to/bangle.jpg
    python bangle_detector.py --image path/to/bangle.jpg --scale 0.085
    python bangle_detector.py --image path/to/bangle.jpg --debug
    python bangle_detector.py --gui

Dependencies:
    pip install opencv-python numpy scipy matplotlib PyQt6

The detector is intentionally classical computer vision only: CLAHE,
bilateral filtering, adaptive/Otsu thresholding, morphology, color masking,
radial circle measurement, and HoughCircles fallback.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# ----------------------------- Data model -----------------------------


@dataclass
class CircleInfo:
    """Circle representation for OD/ID measurement."""

    center: tuple[float, float]
    radius: float
    method: str
    support_count: int = 0
    ellipse: Optional[tuple[tuple[float, float], tuple[float, float], float]] = None

    @property
    def diameter(self) -> float:
        return float(self.radius * 2.0)


# ----------------------------- Core pipeline -----------------------------


def preprocess(img: np.ndarray) -> np.ndarray:
    """Return CLAHE-enhanced, bilateral-filtered grayscale image."""
    if img is None or img.size == 0:
        raise ValueError("Empty image supplied to preprocess().")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # CLAHE reduces the effect of slow illumination gradients and shadows before
    # edge/threshold operations see the image.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Bilateral filtering suppresses sensor noise while preserving hard ring
    # boundaries better than Gaussian blur.
    return cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)


def _auto_canny(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Canny thresholds derived from image median for one sigma setting."""
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 40, 120
    return cv2.Canny(gray, lower, upper, L2gradient=True)


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Return multi-scale Canny edges using two sigma passes."""
    edges_tight = _auto_canny(gray, sigma=0.25)
    edges_loose = _auto_canny(gray, sigma=0.55)
    edges = cv2.bitwise_or(edges_tight, edges_loose)

    # A light close bridges tiny breaks caused by glare or low-contrast patches.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def segment_bangle(img: np.ndarray) -> np.ndarray:
    """Binary mask via adaptive threshold + Otsu OR, then morphology cleanup."""
    gray = preprocess(img)

    # Adaptive threshold catches local contrast under uneven lighting.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        3,
    )

    # Otsu catches globally separated foreground/background cases.
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick the Otsu polarity whose foreground area is plausible; this avoids
    # accidentally selecting the whole background on bright metal scenes.
    total = gray.shape[0] * gray.shape[1]
    inv_area = cv2.countNonZero(otsu_inv) / float(total)
    otsu_mask = otsu_inv if inv_area < 0.55 else otsu

    combined = cv2.bitwise_or(adaptive, otsu_mask)

    min_dim = min(gray.shape[:2])
    k_close = max(5, int(round(min_dim * 0.018)) | 1)
    k_open = max(3, int(round(min_dim * 0.006)) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return cleaned


def normalize_illumination(gray: np.ndarray, shadow_strength: int = 70) -> np.ndarray:
    """Dimension-calib-style shadow normalization before Otsu thresholding."""
    image_h, image_w = gray.shape[:2]
    min_size = min(image_w, image_h)
    bg_size = _odd_kernel(min_size * 0.22, 41)
    sh_size = _odd_kernel(min_size * 0.08, 15)
    strength = float(np.clip(shadow_strength, 0, 100)) / 100.0

    background = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
    divided = cv2.divide(gray, background, scale=255)

    shadow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (sh_size, sh_size)
    )
    dark_halo = cv2.morphologyEx(divided, cv2.MORPH_BLACKHAT, shadow_kernel)
    bright_halo = cv2.morphologyEx(divided, cv2.MORPH_TOPHAT, shadow_kernel)
    corrected = cv2.addWeighted(divided, 1.0, dark_halo, strength, 0)
    corrected = cv2.addWeighted(corrected, 1.0, bright_halo, -0.35 * strength, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(corrected)
    return cv2.GaussianBlur(normalized, (5, 5), 0)


def clean_otsu_mask(mask: np.ndarray) -> np.ndarray:
    """Connected-component and morphology cleanup from dimension_calib.py."""
    image_h, image_w = mask.shape[:2]
    min_size = min(image_w, image_h)
    cleaned = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(30, int(min_size * min_size * 0.0005))

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        touches_border = (
            x <= 1 or y <= 1 or x + w >= image_w - 1 or y + h >= image_h - 1
        )
        too_large = w > image_w * 0.75 or h > image_h * 0.75
        if area >= min_area and not touches_border and not too_large:
            cleaned[labels == label] = 255

    k = _odd_kernel(min_size * 0.018, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def _otsu_bangle_masks(
    img: np.ndarray,
    threshold_offsets: tuple[int, ...],
) -> list[np.ndarray]:
    """Build several illumination-normalized Otsu masks efficiently."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    corrected = normalize_illumination(gray)

    otsu_value, _ = cv2.threshold(
        corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    masks: list[np.ndarray] = []
    for threshold_offset in threshold_offsets:
        threshold_value = int(np.clip(otsu_value + threshold_offset, 0, 255))
        _, thresholded = cv2.threshold(
            corrected, threshold_value, 255, cv2.THRESH_BINARY
        )
        thresholded = cv2.bitwise_not(thresholded)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        masks.append(clean_otsu_mask(thresholded))
    return masks


def otsu_bangle_mask(img: np.ndarray, threshold_offset: int = 0) -> np.ndarray:
    """Build an Otsu mask using the process from dimension_calib.py."""
    return _otsu_bangle_masks(img, (threshold_offset,))[0]


def gold_bangle_mask(img: np.ndarray) -> np.ndarray:
    """Return a color-prior mask for gold/brown bangle pixels.

    The reference images have a pale background and a gold bangle. This mask is
    not used alone for measurement; it acts as a strong object prior so black
    ArUco markers, paper edges, and gray shadows are less likely to be chosen.
    
    Shadow removal is applied to prevent false inner diameter detection caused
    by shadow pixels being misclassified as bangle material.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)

    # Gold appears as yellow/orange/brown in HSV with enough saturation. The
    # value upper bound avoids swallowing white glare/background.
    hsv_gold = cv2.inRange(hsv, np.array([5, 22, 35]), np.array([45, 255, 245]))

    # Extra BGR evidence catches darker brown shadowed gold where hue can wobble.
    warm_excess = np.maximum(r, g).astype(np.int16) - b.astype(np.int16)
    bgr_gold = (
        (warm_excess > 18)
        & (r > 70)
        & (g > 55)
        & (s > 16)
        & (v < 250)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_or(hsv_gold, bgr_gold)
    
    # **Shadow detection and removal**
    # Shadows are detected as regions with:
    # - Valid gold hue (5-45 degrees)
    # - BUT significantly darker than surrounding gold (v < 80)
    # - AND lower saturation typical of shadows (s < 80)
    shadow_region = (
        (h >= 5) & (h <= 45)
        & (v < 80)
        & (s < 80)
    ).astype(np.uint8) * 255
    
    # Dilate shadow region slightly to ensure complete removal
    shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_region = cv2.dilate(shadow_region, shadow_kernel, iterations=1)
    
    # Remove detected shadows from the mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_region))
    
    marker_ignore = detect_marker_ignore_mask(img)
    mask[marker_ignore > 0] = 0

    min_dim = min(mask.shape[:2])
    open_k = max(3, int(round(min_dim * 0.004)) | 1)
    close_k = max(9, int(round(min_dim * 0.018)) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return mask


def detect_marker_ignore_mask(img: np.ndarray) -> np.ndarray:
    """Detect and mask square fiducial markers in the upper-left image area."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    dark = cv2.inRange(gray, 0, 95)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ignore = np.zeros_like(gray)

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if x > w * 0.30 or y > h * 0.30:
            continue
        if not (w * 0.04 <= bw <= w * 0.22 and h * 0.04 <= bh <= h * 0.22):
            continue
        aspect = bw / float(max(bh, 1))
        fill = area / float(max(bw * bh, 1))
        if 0.65 <= aspect <= 1.35 and fill > 0.12:
            pad = int(round(max(bw, bh) * 0.18))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            ignore[y0:y1, x0:x1] = 255
    return ignore


def _contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def find_ring_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find contour candidates filtered by area and circularity."""
    h, w = mask.shape[:2]
    img_area = float(h * w)
    min_area = max(80.0, img_area * 0.0004)
    max_area = img_area * 0.92

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    candidates: list[np.ndarray] = []

    for contour in contours:
        if len(contour) < 20:
            continue
        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < w * 0.03 or bh < h * 0.03:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        circularity = _contour_circularity(contour)
        if circularity < 0.12:
            continue
        candidates.append(contour)

    candidates.sort(key=cv2.contourArea, reverse=True)
    return candidates


def _odd_kernel(value: float, minimum: int) -> int:
    size = max(minimum, int(round(value)))
    return size + 1 if size % 2 == 0 else size


def _measure_contour_circle(contour: np.ndarray, method: str) -> Optional[CircleInfo]:
    """Measure a contour with an ellipse fit or true circle."""
    if contour is None or len(contour) == 0:
        return None
    
    if len(contour) >= 5:
        # Preferred: Ellipse fit (more robust for bangles at slight angles)
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (d1, d2), angle = ellipse
            avg_diameter = (d1 + d2) / 2.0
            return CircleInfo(
                center=(float(cx), float(cy)),
                radius=float(avg_diameter / 2.0),
                method=method,
                support_count=int(len(contour)),
                ellipse=ellipse,
            )
        except cv2.error:
            pass

    # Fallback: Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if radius <= 2.0 or not np.isfinite([cx, cy, radius]).all():
        return None
    return CircleInfo(
        center=(float(cx), float(cy)),
        radius=float(radius),
        method=method,
        support_count=int(len(contour)),
    )


def _circle_through_three_points(points: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Return the circle through three points, or None for a near-straight triplet."""
    p1, p2, p3 = points.astype(np.float64)
    twice_area = 2.0 * np.cross(p2 - p1, p3 - p1)
    if abs(float(twice_area)) < 1e-3:
        return None

    p1_sq = float(np.dot(p1, p1))
    p2_sq = float(np.dot(p2, p2))
    p3_sq = float(np.dot(p3, p3))
    cx = (
        p1_sq * (p2[1] - p3[1])
        + p2_sq * (p3[1] - p1[1])
        + p3_sq * (p1[1] - p2[1])
    ) / twice_area
    cy = (
        p1_sq * (p3[0] - p2[0])
        + p2_sq * (p1[0] - p3[0])
        + p3_sq * (p2[0] - p1[0])
    ) / twice_area
    radius = float(math.hypot(p1[0] - cx, p1[1] - cy))
    if not np.isfinite([cx, cy, radius]).all():
        return None
    return float(cx), float(cy), radius


def _least_squares_circle(points: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Fit a circle algebraically to a set of inlier points."""
    if len(points) < 3:
        return None
    points = points.astype(np.float64)
    design = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    target = np.sum(points * points, axis=1)
    try:
        cx, cy, constant = np.linalg.lstsq(design, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = float(constant + cx * cx + cy * cy)
    if radius_sq <= 0.0:
        return None
    radius = math.sqrt(radius_sq)
    if not np.isfinite([cx, cy, radius]).all():
        return None
    return float(cx), float(cy), float(radius)


def _robust_inner_circle(
    contour: np.ndarray,
    outer: CircleInfo,
) -> Optional[CircleInfo]:
    """Fit the ID while rejecting shadow/glare intrusions in the hole contour.

    A normal ellipse fit treats every contour point equally.  When a shadow is
    connected to the bangle wall, OpenCV traces that shadow as part of the hole
    and the fitted ID becomes too small.  RANSAC finds the boundary supported
    around most angular sectors, so a local damaged arc cannot move the result.
    """
    if contour is None or len(contour) < 12:
        return None
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) > 900:
        indices = np.linspace(0, len(points) - 1, 900, dtype=np.int32)
        points = points[indices]

    tolerance = max(2.0, outer.radius * 0.015)
    min_radius = outer.radius * 0.62
    max_radius = outer.radius * 0.96
    max_center_shift = max(5.0, outer.radius * 0.10)
    rng = np.random.default_rng(0)
    best: Optional[tuple[tuple[int, int, float], np.ndarray, tuple[float, float, float]]] = None
    iterations = min(1200, max(350, len(points) * 4))

    for _ in range(iterations):
        candidate = _circle_through_three_points(points[rng.choice(len(points), 3, replace=False)])
        if candidate is None:
            continue
        cx, cy, radius = candidate
        if not min_radius <= radius <= max_radius:
            continue
        if math.hypot(cx - outer.center[0], cy - outer.center[1]) > max_center_shift:
            continue

        radial_error = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)
        inliers = radial_error <= tolerance
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 10:
            continue
        angles = np.arctan2(points[inliers, 1] - cy, points[inliers, 0] - cx)
        angular_bins = np.unique(((angles + math.pi) * (36.0 / (2.0 * math.pi))).astype(np.int32) % 36)
        coverage = int(len(angular_bins))
        median_error = float(np.median(radial_error[inliers]))
        rank = (coverage, inlier_count, -median_error)
        if best is None or rank > best[0]:
            best = (rank, inliers, candidate)

    if best is None or best[0][0] < 18:
        return None

    inliers = best[1]
    fitted = best[2]
    for _ in range(3):
        refined = _least_squares_circle(points[inliers])
        if refined is None:
            break
        fitted = refined
        cx, cy, radius = fitted
        errors = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)
        new_inliers = errors <= tolerance
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers

    cx, cy, radius = fitted
    if not min_radius <= radius <= max_radius:
        return None
    if math.hypot(cx - outer.center[0], cy - outer.center[1]) > max_center_shift:
        return None
    return CircleInfo(
        center=(cx, cy),
        radius=radius,
        method="robust_inner_circle",
        support_count=int(np.count_nonzero(inliers)),
    )


def _circle_as_contour(circle: CircleInfo, samples: int = 180) -> np.ndarray:
    """Create a clean display contour for a fitted circle."""
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    points = np.column_stack(
        (
            circle.center[0] + np.cos(angles) * circle.radius,
            circle.center[1] + np.sin(angles) * circle.radius,
        )
    )
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def _refine_inner_circle_from_edges(
    inner: CircleInfo,
    outer: CircleInfo,
    edges: Optional[np.ndarray],
) -> CircleInfo:
    """Snap a mask-derived ID to the strongest nearby concentric image edge.

    Color/threshold masks can stop at the dark half of the metal wall.  The
    grayscale gradient still contains the actual metal-to-hole boundary and is
    substantially less sensitive to its brightness or hue.
    """
    if edges is None or edges.size == 0:
        return inner
    # Search the complete physically plausible ID range.  In a shadowed image
    # the hierarchy contour may follow the *shadow* inside the hole, making the
    # provisional ID tens of pixels too small.  Basing this window on that bad
    # radius prevents the real metal-to-hole edge from ever being considered.
    # The outer circle remains stable because its colour mask rejects neutral
    # gray shadows, so use it as the search anchor instead.
    low = outer.radius * 0.62
    high = outer.radius * 0.92
    if high <= low + 2.0:
        return inner

    radii = np.arange(math.ceil(low * 2.0) / 2.0, high + 0.25, 0.5)
    supports = np.asarray(
        [
            _circle_edge_support(edges, outer.center, float(radius), tolerance_px=1)
            for radius in radii
        ],
        dtype=np.float32,
    )
    if len(supports) >= 5:
        supports = np.convolve(
            supports,
            np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32),
            mode="same",
        )
    peak_indices = [
        index
        for index in range(1, len(radii) - 1)
        if supports[index] >= supports[index - 1]
        and supports[index] >= supports[index + 1]
        and supports[index] >= 0.18
    ]
    if not peak_indices:
        return inner

    # Moving outward from the empty hole, the first well-supported concentric
    # edge is the ID.  Stronger peaks after it are decorative grooves/highlights
    # within the metal wall and must not inflate the measurement.
    best_index = min(peak_indices, key=lambda index: float(radii[index]))
    radius = float(radii[best_index])
    return CircleInfo(
        center=outer.center,
        radius=radius,
        method="robust_inner_edge",
        support_count=int(round(float(supports[best_index]) * 1000.0)),
    )


def _find_inner_from_filled(mask: np.ndarray, outer_contour: np.ndarray) -> Optional[np.ndarray]:
    """Find the bangle hole when contour hierarchy did not expose a child."""
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [outer_contour], -1, 255, -1)
    hole_mask = cv2.subtract(filled, mask)
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    hole_contours, _ = cv2.findContours(
        hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_hole = cv2.contourArea(outer_contour) * 0.03
    valid = [c for c in hole_contours if cv2.contourArea(c) > min_hole]
    return max(valid, key=cv2.contourArea) if valid else None


def contour_circle_detection(
    mask: np.ndarray,
    edges: Optional[np.ndarray] = None,
) -> Optional[tuple[CircleInfo, CircleInfo, np.ndarray, np.ndarray]]:
    """Dimension-calib-style contour hierarchy detection, measured as circles/ellipses."""
    image_h, image_w = mask.shape[:2]
    min_size = min(image_w, image_h)
    k = _odd_kernel(min_size * 0.025, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    best_outer_idx = -1
    best_inner_idx = -1
    best_score = -1.0

    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        if w < min_size * 0.03 or h < min_size * 0.03:
            continue
        if w > image_w * 0.80 or h > image_h * 0.80:
            continue

        children = [
            ci for ci, item in enumerate(hierarchy)
            if item[3] == index and cv2.contourArea(contours[ci]) > 0
        ]
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.25:
            continue

        child_idx = -1
        child_area = 0.0
        if children:
            child_idx = int(max(children, key=lambda ci: cv2.contourArea(contours[ci])))
            child_area = cv2.contourArea(contours[child_idx])
            if child_area < area * 0.05:
                child_idx = -1
                child_area = 0.0

        score = area + child_area + circularity * 5000.0
        if score > best_score:
            best_outer_idx = index
            best_inner_idx = child_idx
            best_score = score

    if best_outer_idx == -1:
        # Fallback: Find largest top-level contour
        top_level = []
        for i, h_node in enumerate(hierarchy):
            if h_node[3] == -1:
                top_level.append((i, contours[i]))
        
        if not top_level:
            return None
        best_outer_idx, best_outer_contour = max(top_level, key=lambda x: cv2.contourArea(x[1]))
    else:
        best_outer_contour = contours[best_outer_idx]

    if best_inner_idx == -1:
        best_inner_contour = _find_inner_from_filled(connected, best_outer_contour)
    else:
        best_inner_contour = contours[best_inner_idx]

    outer = _measure_contour_circle(best_outer_contour, "contour_circle")
    # The inner boundary is especially vulnerable to a gray shadow being
    # attached to one side of the threshold contour.  Prefer a robust circular
    # fit and retain the ordinary ellipse only for genuinely elliptical views
    # where robust support is unavailable.
    inner = _robust_inner_circle(best_inner_contour, outer) if outer is not None else None
    if inner is None:
        inner = _measure_contour_circle(best_inner_contour, "contour_circle")
    if outer is None or inner is None:
        return None
    inner = _refine_inner_circle_from_edges(inner, outer, edges)
    if inner.radius >= outer.radius:
        return None

    center_dist = math.hypot(outer.center[0] - inner.center[0], outer.center[1] - inner.center[1])
    if center_dist > math.hypot(image_w, image_h) * 0.15:
        return None
    if inner.radius / outer.radius < 0.62:
        return None
    display_inner_contour = (
        _circle_as_contour(inner)
        if inner.method in {"robust_inner_circle", "robust_inner_edge"}
        else best_inner_contour
    )
    return outer, inner, best_outer_contour, display_inner_contour


def _select_circle_pair_consensus(
    candidates: list[tuple[CircleInfo, CircleInfo, np.ndarray, np.ndarray]],
) -> Optional[tuple[CircleInfo, CircleInfo, np.ndarray, np.ndarray]]:
    """Return the medoid of the largest geometrically consistent pair cluster."""
    if not candidates:
        return None

    clusters: list[list[int]] = []
    for outer, inner, _outer_contour, _inner_contour in candidates:
        cluster: list[int] = []
        for other_index, (other_outer, other_inner, _oc, _ic) in enumerate(candidates):
            center_distance = math.hypot(
                outer.center[0] - other_outer.center[0],
                outer.center[1] - other_outer.center[1],
            )
            if center_distance > max(6.0, outer.radius * 0.06):
                continue
            if abs(outer.radius - other_outer.radius) > max(5.0, outer.radius * 0.08):
                continue
            if abs(inner.radius - other_inner.radius) > max(5.0, inner.radius * 0.10):
                continue
            cluster.append(other_index)
        clusters.append(cluster)

    best_cluster = max(clusters, key=len)
    outer_median = float(np.median([candidates[i][0].radius for i in best_cluster]))
    inner_median = float(np.median([candidates[i][1].radius for i in best_cluster]))
    center_x = float(np.median([candidates[i][0].center[0] for i in best_cluster]))
    center_y = float(np.median([candidates[i][0].center[1] for i in best_cluster]))

    def deviation(index: int) -> float:
        outer, inner, _outer_contour, _inner_contour = candidates[index]
        return (
            abs(outer.radius - outer_median) / max(outer_median, 1.0)
            + abs(inner.radius - inner_median) / max(inner_median, 1.0)
            + math.hypot(outer.center[0] - center_x, outer.center[1] - center_y)
            / max(outer_median, 1.0)
        )

    return candidates[min(best_cluster, key=deviation)]


def otsu_consensus_circle_detection(
    img: np.ndarray,
    edges: np.ndarray,
    marker_ignore: Optional[np.ndarray] = None,
) -> tuple[
    Optional[tuple[CircleInfo, CircleInfo, np.ndarray, np.ndarray]],
    np.ndarray,
]:
    """Detect OD/ID from stable grayscale thresholds, independent of colour.

    A single Otsu cutoff can land on either side of a soft shadow. Nearby
    cutoffs are measured independently and the largest consistent circle
    cluster wins. This keeps lighting and gold hue out of the primary geometry
    path while retaining the original colour image for annotation.
    """
    offsets = (-25, -15, -8, 0, 8, 15, 25)
    masks = _otsu_bangle_masks(img, offsets)
    candidates: list[tuple[CircleInfo, CircleInfo, np.ndarray, np.ndarray]] = []
    zero_mask = masks[offsets.index(0)]
    for mask in masks:
        if marker_ignore is not None:
            mask[marker_ignore > 0] = 0
        pair = contour_circle_detection(mask, edges=edges)
        if pair is not None:
            candidates.append(pair)
    return _select_circle_pair_consensus(candidates), zero_mask


def hough_circle_detection(
    gray: np.ndarray, ignore_mask: Optional[np.ndarray] = None
) -> Optional[tuple[CircleInfo, CircleInfo]]:
    """Fallback OD/ID detection using HoughCircles."""
    if ignore_mask is not None:
        gray = gray.copy()
        # Paint ignored regions as background so square markers cannot become
        # fallback circles if radial measurement fails.
        bg_value = int(np.median(gray[ignore_mask == 0])) if np.any(ignore_mask == 0) else 220
        gray[ignore_mask > 0] = bg_value
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 8),
        param1=90,
        param2=24,
        minRadius=max(5, int(min_dim * 0.03)),
        maxRadius=int(min_dim * 0.48),
    )
    if circles is None:
        return None

    detected = np.squeeze(circles, axis=0)
    if detected.ndim != 2 or detected.shape[0] < 2:
        return None

    # Prefer two circles with nearby centers and substantially different radii.
    best_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
    best_score = float("inf")
    diag = math.hypot(w, h)
    for i in range(len(detected)):
        for j in range(i + 1, len(detected)):
            c1, c2 = detected[i], detected[j]
            r1, r2 = float(c1[2]), float(c2[2])
            if abs(r1 - r2) < min_dim * 0.035:
                continue
            center_dist = math.hypot(float(c1[0] - c2[0]), float(c1[1] - c2[1]))
            if center_dist > diag * 0.15:
                continue
            score = center_dist - abs(r1 - r2) * 0.02
            if score < best_score:
                best_score = score
                best_pair = (c1, c2)

    if best_pair is None:
        return None

    outer_c, inner_c = sorted(best_pair, key=lambda c: c[2], reverse=True)
    outer = CircleInfo(
        center=(float(outer_c[0]), float(outer_c[1])),
        radius=float(outer_c[2]),
        method="hough_circle",
    )
    inner = CircleInfo(
        center=(float(inner_c[0]), float(inner_c[1])),
        radius=float(inner_c[2]),
        method="hough_circle",
    )
    return outer, inner


def _largest_bangle_component(mask: np.ndarray) -> Optional[np.ndarray]:
    """Keep the largest non-border gold component as the bangle body."""
    h, w = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best_label = 0
    best_area = 0

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < max(250, int(h * w * 0.0005)):
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        if area > best_area:
            best_label = label
            best_area = int(area)

    if best_label == 0:
        return None
    component = np.zeros_like(mask)
    component[labels == best_label] = 255
    return component


def radial_circle_detection(mask: np.ndarray) -> Optional[tuple[CircleInfo, CircleInfo]]:
    """Measure OD/ID as circles from radial distances of bangle mask pixels."""
    component = _largest_bangle_component(mask)
    if component is None:
        return None

    points_y, points_x = np.where(component > 0)
    if len(points_x) < 100:
        return None

    x, y, bw, bh = cv2.boundingRect(component)
    center_x = x + bw / 2.0
    center_y = y + bh / 2.0
    distances = np.hypot(points_x.astype(np.float64) - center_x, points_y.astype(np.float64) - center_y)
    distances = distances[np.isfinite(distances)]
    if distances.size < 100:
        return None

    # Robust percentiles ignore isolated decorative bumps and tiny mask specks.
    inner_radius = float(np.percentile(distances, 4.0))
    outer_radius = float(np.percentile(distances, 98.5))
    if inner_radius <= 2.0 or outer_radius <= inner_radius:
        return None
    if inner_radius / outer_radius < 0.62:
        return None

    center = (float(center_x), float(center_y))
    outer = CircleInfo(center=center, radius=outer_radius, method="radial_mask", support_count=int(distances.size))
    inner = CircleInfo(center=center, radius=inner_radius, method="radial_mask", support_count=int(distances.size))
    return outer, inner


def _circle_edge_support(
    edges: np.ndarray,
    center: tuple[float, float],
    radius: float,
    tolerance_px: Optional[int] = None,
) -> float:
    """Measure how much of a proposed circle is supported by nearby edges."""
    if radius <= 1:
        return 0.0
    samples = int(np.clip(round(2.0 * math.pi * radius), 72, 720))
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    tolerance = (
        max(1, int(round(radius * 0.035)))
        if tolerance_px is None
        else max(0, int(tolerance_px))
    )
    supported = np.zeros(samples, dtype=bool)
    for offset in range(-tolerance, tolerance + 1):
        sample_radius = radius + offset
        xs = np.rint(center[0] + np.cos(angles) * sample_radius).astype(np.int32)
        ys = np.rint(center[1] + np.sin(angles) * sample_radius).astype(np.int32)
        valid = (
            (xs >= 0)
            & (xs < edges.shape[1])
            & (ys >= 0)
            & (ys < edges.shape[0])
        )
        supported[valid] |= edges[ys[valid], xs[valid]] > 0
    return float(supported.mean())


def finger_ring_circle_detection(
    img: np.ndarray,
    scale: Optional[float],
) -> tuple[CircleInfo, CircleInfo, float]:
    """Measure visible ring boundaries without forcing tilted bands into circles."""
    if __package__:
        from .finger_ring_detector import visible_ring_ellipses
    else:
        from finger_ring_detector import visible_ring_ellipses

    outer_fit, inner_fit = visible_ring_ellipses(img, scale)
    circles = [
        CircleInfo(center=fit[0], radius=sum(fit[1]) / 4.0,
                   method="finger_ring_edge_ellipse", ellipse=fit)
        for fit in (outer_fit, inner_fit)
    ]
    return circles[0], circles[1], 1.0


def detect_bangle(
    image_path: str | os.PathLike[str],
    scale: Optional[float] = None,
    debug: bool = False,
    jewel_type: Optional[str] = None,
) -> dict[str, Any]:
    """Full bangle detection pipeline. Returns a serializable result dict."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    normalized_jewel_type = str(jewel_type or "").strip().lower()
    is_finger_ring = "finger" in normalized_jewel_type and "ring" in normalized_jewel_type

    if is_finger_ring:
        outer, inner, zoom_factor = finger_ring_circle_detection(img, scale)
        od_px = outer.diameter
        id_px = inner.diameter
        wall_px = (od_px - id_px) / 2.0
        result: dict[str, Any] = {
            "image_path": str(image_path),
            "outer": asdict(outer),
            "inner": asdict(inner),
            "od_px": float(od_px),
            "id_px": float(id_px),
            "wall_thickness_px": float(wall_px),
            "scale_mm_per_px": float(scale) if scale is not None else None,
            "used_fallback": False,
            "detection_mode": "finger_ring_edge_ellipse",
            "zoom_factor": float(zoom_factor),
        }
        if scale is not None:
            result.update(
                {
                    "od_mm": float(od_px * scale),
                    "id_mm": float(id_px * scale),
                    "wall_thickness_mm": float(wall_px * scale),
                }
            )

        annotated = draw_results(img, result)
        output_path = _result_path(image_path)
        cv2.imwrite(str(output_path), annotated)
        result["annotated_path"] = str(output_path)
        return result

    gray = preprocess(img)
    mask = segment_bangle(img)
    edges = edge_map(gray)
    gold_mask = gold_bangle_mask(img)
    marker_ignore = detect_marker_ignore_mask(img)

    # Remove fiducial markers and paper borders from generic threshold/edge
    # sources, then add Otsu support near the gold prior for this use case.
    mask[marker_ignore > 0] = 0
    edges[marker_ignore > 0] = 0
    # Geometry is measured first on normalized grayscale/Otsu masks. The colour
    # image is used only as a fallback and as the annotation canvas.
    pair_data, otsu_mask = otsu_consensus_circle_detection(
        img, edges, marker_ignore
    )

    min_dim = min(mask.shape[:2])
    prior_k = _odd_kernel(min_dim * 0.035, 15)
    prior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prior_k, prior_k))
    gold_neighborhood = cv2.dilate(gold_mask, prior_kernel, iterations=1)
    otsu_near_gold = cv2.bitwise_and(otsu_mask, gold_neighborhood)
    mask = cv2.bitwise_or(mask, cv2.bitwise_or(gold_mask, otsu_near_gold))

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
    closed_edges[marker_ignore > 0] = 0
    contour_source = cv2.bitwise_or(mask, closed_edges)
    contour_source[marker_ignore > 0] = 0

    measurement_mask = cv2.bitwise_or(gold_mask, otsu_near_gold)
    used_fallback = False
    outer_contour = None
    inner_contour = None

    if pair_data is not None:
        outer, inner, outer_contour, inner_contour = pair_data
    else:
        color_pair = contour_circle_detection(measurement_mask, edges=edges)
        if color_pair is not None:
            outer, inner, outer_contour, inner_contour = color_pair
        else:
            pair = radial_circle_detection(measurement_mask)
            if pair is not None:
                outer, inner = pair
            else:
                pair = hough_circle_detection(gray, marker_ignore)
                used_fallback = pair is not None
                if pair is not None:
                    outer, inner = pair
                else:
                    raise RuntimeError("Could not detect valid OD/ID circles.")

    od_px = outer.diameter
    id_px = inner.diameter
    wall_px = (od_px - id_px) / 2.0

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "outer": asdict(outer),
        "inner": asdict(inner),
        "od_px": float(od_px),
        "id_px": float(id_px),
        "wall_thickness_px": float(wall_px),
        "scale_mm_per_px": float(scale) if scale is not None else None,
        "used_fallback": bool(used_fallback),
        "detection_mode": "bangle",
        "zoom_factor": 1.0,
    }

    if scale is not None:
        result.update(
            {
                "od_mm": float(od_px * scale),
                "id_mm": float(id_px * scale),
                "wall_thickness_mm": float(wall_px * scale),
            }
        )

    annotated = draw_results(img, result, outer_contour=outer_contour, inner_contour=inner_contour)
    output_path = _result_path(image_path)
    cv2.imwrite(str(output_path), annotated)
    result["annotated_path"] = str(output_path)

    if debug:
        _show_debug(img, gray, mask, edges, contour_source, annotated, otsu_mask, measurement_mask)

    return result


def _circle_from_result(data: dict[str, Any]) -> CircleInfo:
    ellipse_data = data.get("ellipse")
    if ellipse_data is not None:
        # data["ellipse"] might be a list of lists from JSON serialization
        ellipse = (
            (float(ellipse_data[0][0]), float(ellipse_data[0][1])),
            (float(ellipse_data[1][0]), float(ellipse_data[1][1])),
            float(ellipse_data[2]),
        )
    else:
        ellipse = None

    return CircleInfo(
        center=tuple(data["center"]),
        radius=float(data["radius"]),
        method=str(data["method"]),
        support_count=int(data.get("support_count", 0)),
        ellipse=ellipse,
    )


def draw_results(
    img: np.ndarray,
    result: dict[str, Any],
    outer_contour: Optional[np.ndarray] = None,
    inner_contour: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw OD/ID circles and measurement labels on a BGR image."""
    out = img.copy()
    outer = _circle_from_result(result["outer"])
    inner = _circle_from_result(result["inner"])

    # Colors: Green for Outer, Blue for Inner
    COLOR_OUTER = (0, 220, 0)
    COLOR_INNER = (255, 0, 0)

    # Draw raw contours if provided (thickness 2)
    if outer_contour is not None:
        cv2.drawContours(out, [outer_contour], -1, COLOR_OUTER, 2, cv2.LINE_AA)
    if inner_contour is not None:
        cv2.drawContours(out, [inner_contour], -1, COLOR_INNER, 2, cv2.LINE_AA)

    # Draw fits (Ellipse or Circle) - thickness 1
    for circle, color in [(outer, COLOR_OUTER), (inner, COLOR_INNER)]:
        if circle.ellipse is not None:
            cv2.ellipse(out, circle.ellipse, color, 1, cv2.LINE_AA)
        else:
            center = tuple(np.round(circle.center).astype(int))
            cv2.circle(out, center, int(round(circle.radius)), color, 1, cv2.LINE_AA)
        
        # Center point
        cv2.circle(out, tuple(np.round(circle.center).astype(int)), 3, color, -1, cv2.LINE_AA)

    # Annotate labels near center, like in dimension_calib_finale.py
    cx, cy = np.round(outer.center).astype(int)
    
    od_label = f"OD={result['od_mm']:.2f}mm" if result.get("od_mm") else f"OD={result['od_px']:.1f}px"
    id_label = f"ID={result['id_mm']:.2f}mm" if result.get("id_mm") else f"ID={result['id_px']:.1f}px"

    font_scale = 0.55 if outer.radius < 50 else 0.7
    text_thickness = 1 if outer.radius < 50 else 2
    outline_thickness = text_thickness + 2
    if outer.radius < 50 or outer.method.startswith("finger_ring"):
        label_x = cx + int(round(outer.radius)) + 8
        max_label_width = max(
            cv2.getTextSize(od_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0][0],
            cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0][0],
        )
        if label_x + max_label_width >= out.shape[1] - 5:
            label_x = max(5, cx - int(round(outer.radius)) - max_label_width - 8)
        od_y = max(18, cy - 3)
        id_y = min(out.shape[0] - 5, cy + 18)
    else:
        label_x = max(5, cx - 90)
        od_y = max(18, cy - 12)
        id_y = min(out.shape[0] - 5, cy + 18)

    # Background outline for readability
    cv2.putText(out, od_label, (label_x, od_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(out, od_label, (label_x, od_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_OUTER, text_thickness, cv2.LINE_AA)
    
    cv2.putText(out, id_label, (label_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(out, id_label, (label_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_INNER, text_thickness, cv2.LINE_AA)

    return out


def _result_path(image_path: str | os.PathLike[str]) -> Path:
    path = Path(image_path)
    return path.with_name(f"{path.stem}_result.jpg")


def _show_debug(
    img: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    contour_source: np.ndarray,
    annotated: np.ndarray,
    otsu_mask: Optional[np.ndarray] = None,
    measurement_mask: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Debug plotting unavailable: {exc}")
        return

    panels = [
        ("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "rgb"),
        ("CLAHE + Bilateral", gray, "gray"),
        ("Otsu Mask", otsu_mask if otsu_mask is not None else mask, "gray"),
        ("Measurement Mask", measurement_mask if measurement_mask is not None else mask, "gray"),
        ("Multi-scale Canny", edges, "gray"),
        ("Contour Source", contour_source, "gray"),
        ("Annotated", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "rgb"),
    ]
    plt.figure(figsize=(16, 8))
    for idx, (title, data, mode) in enumerate(panels, start=1):
        plt.subplot(2, 4, idx)
        plt.title(title)
        plt.axis("off")
        if mode == "gray":
            plt.imshow(data, cmap="gray")
        else:
            plt.imshow(data)
    plt.tight_layout()
    plt.show()


def print_result(result: dict[str, Any]) -> None:
    """Console output requested by the CLI requirements."""
    outer = _circle_from_result(result["outer"])
    inner = _circle_from_result(result["inner"])

    print(f"OD: {result['od_px']:.3f} px")
    print(f"ID: {result['id_px']:.3f} px")
    print(f"Wall thickness: {result['wall_thickness_px']:.3f} px")
    if result.get("scale_mm_per_px") is not None:
        print(f"OD: {result['od_mm']:.3f} mm")
        print(f"ID: {result['id_mm']:.3f} mm")
        print(f"Wall thickness: {result['wall_thickness_mm']:.3f} mm")

    print(
        "Outer circle: "
        f"center=({outer.center[0]:.2f}, {outer.center[1]:.2f}), "
        f"radius={outer.radius:.2f} px, "
        f"method={outer.method}"
    )
    print(
        "Inner circle: "
        f"center=({inner.center[0]:.2f}, {inner.center[1]:.2f}), "
        f"radius={inner.radius:.2f} px, "
        f"method={inner.method}"
    )
    print(f"Annotated image: {result['annotated_path']}")
    if result.get("used_fallback"):
        print("Fallback: HoughCircles was used.")


# ----------------------------- PyQt6 GUI -----------------------------


class _QtImportError(RuntimeError):
    pass


def _load_pyqt6():
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QDoubleSpinBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - import depends on environment.
        raise _QtImportError("PyQt6 is required for the GUI: pip install PyQt6") from exc
    return locals()


def run_gui() -> int:
    qt = _load_pyqt6()
    QApplication = qt["QApplication"]
    QMainWindow = qt["QMainWindow"]
    QWidget = qt["QWidget"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QTextEdit = qt["QTextEdit"]
    QVBoxLayout = qt["QVBoxLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    QDoubleSpinBox = qt["QDoubleSpinBox"]
    QCheckBox = qt["QCheckBox"]
    QImage = qt["QImage"]
    QPixmap = qt["QPixmap"]
    Qt = qt["Qt"]

    class ImageLabel(QLabel):
        def __init__(self) -> None:
            super().__init__("Load an image to detect OD / ID")
            self._bgr: Optional[np.ndarray] = None
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(760, 520)
            self.setStyleSheet("background:#181818;color:#aaa;border:1px solid #444;")

        def set_bgr(self, image: np.ndarray) -> None:
            self._bgr = image.copy()
            self._refresh()

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._refresh()

        def _refresh(self) -> None:
            if self._bgr is None:
                return
            rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    class BangleWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Bangle OD / ID Detector")
            self.resize(1100, 780)
            self.image_path: Optional[str] = None
            self.original: Optional[np.ndarray] = None
            self.annotated: Optional[np.ndarray] = None

            root = QWidget()
            layout = QVBoxLayout(root)

            controls = QHBoxLayout()
            self.btn_load = QPushButton("Load Image")
            self.btn_detect = QPushButton("Detect OD / ID")
            self.btn_save = QPushButton("Save Annotated")
            self.scale_check = QCheckBox("Use scale")
            self.scale_spin = QDoubleSpinBox()
            self.scale_spin.setDecimals(6)
            self.scale_spin.setRange(0.000001, 1000.0)
            self.scale_spin.setValue(0.085)
            self.scale_spin.setSuffix(" mm/px")
            self.btn_detect.setEnabled(False)
            self.btn_save.setEnabled(False)
            controls.addWidget(self.btn_load)
            controls.addWidget(self.btn_detect)
            controls.addWidget(self.scale_check)
            controls.addWidget(self.scale_spin)
            controls.addWidget(self.btn_save)
            controls.addStretch(1)

            self.image_label = ImageLabel()
            self.output = QTextEdit()
            self.output.setReadOnly(True)
            self.output.setMaximumHeight(150)
            self.output.setStyleSheet(
                "background:#101010;color:#8cffb2;font-family:Consolas,monospace;"
            )
            self.output.setPlainText("Load a bangle image, then click Detect OD / ID.")

            layout.addLayout(controls)
            layout.addWidget(self.image_label, 1)
            layout.addWidget(self.output)
            self.setCentralWidget(root)

            self.btn_load.clicked.connect(self.load_image)
            self.btn_detect.clicked.connect(self.detect)
            self.btn_save.clicked.connect(self.save_annotated)

        def load_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open bangle image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            )
            if not path:
                return
            image = cv2.imread(path)
            if image is None:
                QMessageBox.critical(self, "Load failed", "OpenCV could not load this image.")
                return
            self.image_path = path
            self.original = image
            self.annotated = None
            self.image_label.set_bgr(image)
            self.btn_detect.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.output.setPlainText(f"Loaded: {path}\nShape: {image.shape[1]} x {image.shape[0]} px")

        def detect(self) -> None:
            if not self.image_path:
                return
            scale = self.scale_spin.value() if self.scale_check.isChecked() else None
            try:
                result = detect_bangle(self.image_path, scale=scale, debug=False)
            except Exception as exc:
                QMessageBox.critical(self, "Detection failed", str(exc))
                return

            annotated = cv2.imread(result["annotated_path"])
            if annotated is not None:
                self.annotated = annotated
                self.image_label.set_bgr(annotated)
                self.btn_save.setEnabled(True)

            lines = [
                f"OD: {result['od_px']:.3f} px",
                f"ID: {result['id_px']:.3f} px",
                f"Wall thickness: {result['wall_thickness_px']:.3f} px",
            ]
            if scale is not None:
                lines.extend(
                    [
                        f"OD: {result['od_mm']:.3f} mm",
                        f"ID: {result['id_mm']:.3f} mm",
                        f"Wall thickness: {result['wall_thickness_mm']:.3f} mm",
                    ]
                )
            outer = _circle_from_result(result["outer"])
            inner = _circle_from_result(result["inner"])
            lines.extend(
                [
                    "",
                    f"Outer radius: {outer.radius:.2f} px | center=({outer.center[0]:.1f}, {outer.center[1]:.1f})",
                    f"Inner radius: {inner.radius:.2f} px | center=({inner.center[0]:.1f}, {inner.center[1]:.1f})",
                    f"Annotated saved: {result['annotated_path']}",
                ]
            )
            if result.get("used_fallback"):
                lines.append("Fallback: HoughCircles was used.")
            self.output.setPlainText("\n".join(lines))

        def save_annotated(self) -> None:
            if self.annotated is None:
                return
            default = _result_path(self.image_path or "bangle.jpg")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save annotated image", str(default), "JPEG (*.jpg);;PNG (*.png)"
            )
            if path:
                cv2.imwrite(path, self.annotated)
                self.output.append(f"\nSaved copy: {path}")

    app = QApplication(sys.argv)
    window = BangleWindow()
    window.show()
    return app.exec()


# ----------------------------- CLI -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect bangle OD/ID from an image.")
    parser.add_argument("--image", help="Path to bangle image.")
    parser.add_argument("--scale", type=float, default=None, help="Optional mm/pixel scale.")
    parser.add_argument("--debug", action="store_true", help="Show matplotlib debug panels.")
    parser.add_argument("--gui", action="store_true", help="Open PyQt6 GUI.")
    args = parser.parse_args()

    if args.gui or not args.image:
        return run_gui()

    result = detect_bangle(args.image, scale=args.scale, debug=args.debug)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
