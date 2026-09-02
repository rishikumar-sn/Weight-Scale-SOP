import cv2
import numpy as np

from Dimension.bangle_detector import (
    CircleInfo,
    _refine_inner_circle_from_edges,
    _robust_inner_circle,
    _select_circle_pair_consensus,
)


def _damaged_circle_contour(
    center=(220.0, 190.0),
    radius=82.0,
    damaged_radius=53.0,
):
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    radii = np.full_like(angles, radius)
    damaged = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) < 0.48
    radii[damaged] = damaged_radius + 5.0 * np.cos(angles[damaged] * 5.0)
    points = np.column_stack(
        (center[0] + np.cos(angles) * radii, center[1] + np.sin(angles) * radii)
    )
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def test_robust_inner_circle_ignores_attached_shadow_intrusion():
    contour = _damaged_circle_contour()
    outer = CircleInfo(center=(220.0, 190.0), radius=105.0, method="test")

    ordinary_ellipse = cv2.fitEllipse(contour)
    ordinary_radius = sum(ordinary_ellipse[1]) / 4.0
    robust = _robust_inner_circle(contour, outer)

    assert abs(ordinary_radius - 82.0) > 1.0
    assert robust is not None
    assert abs(robust.radius - 82.0) < 1.0
    assert np.linalg.norm(np.asarray(robust.center) - np.asarray(outer.center)) < 1.0


def test_robust_inner_circle_rejects_unrelated_non_circular_contour():
    contour = np.array([[[20, 20]], [[200, 20]], [[200, 40]], [[20, 40]]], dtype=np.int32)
    outer = CircleInfo(center=(110.0, 110.0), radius=100.0, method="test")

    assert _robust_inner_circle(contour, outer) is None


def test_edge_refinement_recovers_boundary_shifted_by_dark_metal():
    edges = np.zeros((440, 440), dtype=np.uint8)
    center = (220, 220)
    cv2.circle(edges, center, 86, 255, 1)
    outer = CircleInfo(center=center, radius=105.0, method="test")
    threshold_biased = CircleInfo(center=(216.0, 218.0), radius=77.0, method="test")

    refined = _refine_inner_circle_from_edges(threshold_biased, outer, edges)

    assert refined.method == "robust_inner_edge"
    assert abs(refined.radius - 86.0) <= 1.0
    assert refined.center == outer.center


def test_edge_refinement_recovers_boundary_when_shadow_severely_shrinks_contour():
    edges = np.zeros((440, 440), dtype=np.uint8)
    center = (220, 220)
    cv2.circle(edges, center, 82, 255, 1)
    outer = CircleInfo(center=center, radius=105.0, method="test")
    shadow_biased = CircleInfo(center=(214.0, 220.0), radius=65.0, method="test")

    refined = _refine_inner_circle_from_edges(shadow_biased, outer, edges)

    assert refined.method == "robust_inner_edge"
    assert abs(refined.radius - 82.0) <= 1.0
    assert refined.center == outer.center


def test_otsu_consensus_rejects_shadow_inflated_circle_pair():
    contour = np.empty((0, 1, 2), dtype=np.int32)

    def pair(outer_radius, inner_radius, center=(220.0, 220.0)):
        return (
            CircleInfo(center=center, radius=outer_radius, method="test"),
            CircleInfo(center=center, radius=inner_radius, method="test"),
            contour,
            contour,
        )

    candidates = [
        pair(100.0, 84.0),
        pair(101.0, 84.5),
        pair(99.5, 83.5),
        pair(116.0, 67.0, center=(229.0, 216.0)),
    ]

    selected = _select_circle_pair_consensus(candidates)

    assert selected is not None
    assert selected[0].radius < 102.0
    assert selected[1].radius > 83.0
