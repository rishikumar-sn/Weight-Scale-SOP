from backend.app.services.analysis_service import (
    BEAD_DETECTION_SCORE_THRESHOLD,
    BEAD_NMS_LARGE_IOU,
    BEAD_NMS_MEDIUM_IOU,
    BEAD_NMS_SMALL_IOU,
    _adaptive_bead_nms,
    _bead_nms_iou_threshold,
    _box_center_in_region,
)


def test_bead_model_thresholds_match_production_settings():
    assert BEAD_DETECTION_SCORE_THRESHOLD == 0.35


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
