import cv2
import numpy as np

from backend.app.services.analysis_service import (
    BEAD_DETECTION_SCORE_THRESHOLD,
    BEAD_FALSE_POSITIVE_THRESHOLD,
    BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED,
    BEAD_MIN_VERIFICATION_RATIO,
    BEAD_NMS_LARGE_IOU,
    BEAD_NMS_MEDIUM_IOU,
    BEAD_NMS_SMALL_IOU,
    _adaptive_bead_nms,
    _bead_evidence_image,
    _bead_detector_only_decision,
    _bead_nms_iou_threshold,
    _bead_presence_decision,
    _box_center_in_region,
    _repeated_red_bead_evidence,
)


def test_bead_model_thresholds_match_production_settings():
    assert BEAD_DETECTION_SCORE_THRESHOLD == 0.50
    assert BEAD_FALSE_POSITIVE_THRESHOLD == 0.75
    assert BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED is False
    assert BEAD_MIN_VERIFICATION_RATIO == 0.20


def test_detector_only_mode_requires_two_candidates():
    assert _bead_detector_only_decision(0) == (False, "insufficient_detector_evidence")
    assert _bead_detector_only_decision(1) == (False, "insufficient_detector_evidence")
    assert _bead_detector_only_decision(2) == (True, "detector_consensus")


def test_nms_iou_threshold_tracks_bead_size_relative_to_image():
    image_shape = (1000, 1000, 3)

    assert _bead_nms_iou_threshold([0, 0, 20, 20], image_shape) == BEAD_NMS_SMALL_IOU
    assert _bead_nms_iou_threshold([0, 0, 40, 40], image_shape) == BEAD_NMS_MEDIUM_IOU
    assert _bead_nms_iou_threshold([0, 0, 80, 80], image_shape) == BEAD_NMS_LARGE_IOU


def test_adaptive_nms_suppresses_small_duplicates_but_keeps_large_neighbors():
    image_shape = (1000, 1000, 3)

    small_boxes = [[0, 0, 20, 20], [5, 0, 25, 20]]
    large_boxes = [[100, 100, 180, 180], [120, 100, 200, 180]]

    assert _adaptive_bead_nms(small_boxes, [0.9, 0.8], image_shape) == [0]
    assert _adaptive_bead_nms(large_boxes, [0.9, 0.8], image_shape) == [0, 1]


def test_adaptive_nms_returns_detections_in_score_order():
    boxes = [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]]

    assert _adaptive_bead_nms(boxes, [0.6, 0.9, 0.7], (500, 500, 3)) == [1, 2, 0]


def test_full_roi_detections_are_assigned_by_box_center():
    region = {"x": 100, "y": 200, "w": 300, "h": 250}

    assert _box_center_in_region([120, 220, 160, 260], region) is True
    assert _box_center_in_region([70, 220, 110, 260], region) is False
    assert _box_center_in_region([390, 220, 410, 260], region) is False
    assert _box_center_in_region([0, 0, 10, 10], None) is True


def test_colour_fallback_uses_only_the_assigned_item_region():
    image = np.zeros((500, 700, 3), dtype=np.uint8)
    region = {"x": 100, "y": 150, "w": 300, "h": 200}

    cropped, offset_x, offset_y = _bead_evidence_image(image, region)

    assert cropped.shape == (200, 300, 3)
    assert (offset_x, offset_y) == (100, 150)


def test_bead_presence_requires_verifier_consensus_not_one_weak_box():
    evidence = {"supported": False}

    assert _bead_presence_decision(40, 9, evidence)[:2] == (True, "verified_consensus")
    assert _bead_presence_decision(28, 8, evidence)[:2] == (True, "verified_consensus")
    assert _bead_presence_decision(17, 0, evidence)[0] is False
    assert _bead_presence_decision(1, 1, evidence)[0] is False


def test_repeated_similar_red_beads_enable_narrow_fallback():
    image = np.full((500, 700, 3), (185, 185, 185), dtype=np.uint8)
    for index in range(12):
        center = (60 + index * 50, 250 + (index % 2) * 8)
        cv2.circle(image, center, 14, (35, 45, 125), cv2.FILLED)
    evidence = _repeated_red_bead_evidence(image)

    assert evidence["supported"] is True
    detected, source, _ = _bead_presence_decision(3, 0, evidence)
    assert detected is True
    assert source == "repeated_red_bead_fallback"


def test_mixed_red_stone_sizes_do_not_enable_fallback():
    image = np.full((500, 700, 3), (185, 185, 185), dtype=np.uint8)
    radii = [5, 6, 7, 9, 12, 15, 19, 23, 27, 8, 14, 21]
    for index, radius in enumerate(radii):
        center = (45 + index * 53, 250)
        cv2.circle(image, center, radius, (35, 45, 150), cv2.FILLED)
    evidence = _repeated_red_bead_evidence(image)

    assert evidence["supported"] is False
