from __future__ import annotations

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from backend.app.analysis.testbed_segmentation import TestbedSegmenter


def test_square_roi_masks_polygon_and_preserves_coordinate_mapping() -> None:
    image = np.zeros((10, 14, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(14, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(10, dtype=np.uint8)[:, None]
    mask = np.zeros((10, 14), dtype=np.uint8)
    polygon = np.array([[4, 3], [9, 3], [8, 6], [5, 6]], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)

    result = TestbedSegmenter._square_roi(image, mask, 0.91)

    assert result.image.shape == (6, 6, 3)
    assert result.mask.shape == (6, 6)
    assert result.source_roi == {"x": 4, "y": 3, "w": 6, "h": 4}
    assert result.processing_roi == {"x": 4, "y": 2, "w": 6, "h": 6}
    assert np.all(result.image[result.mask == 0] == 255)
    output_y, output_x = np.argwhere(result.mask > 0)[0]
    source_x = output_x + result.processing_roi["x"]
    source_y = output_y + result.processing_roi["y"]
    np.testing.assert_array_equal(result.image[output_y, output_x], image[source_y, source_x])


def test_clean_mask_keeps_only_largest_component_and_insets_boundary() -> None:
    segmenter = TestbedSegmenter.__new__(TestbedSegmenter)
    segmenter.inset_px = 1
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[3:5, 3:5] = 1
    mask[10:25, 8:23] = 1

    cleaned = segmenter._clean_mask(mask)

    assert cleaned[3, 3] == 0
    assert cleaned[10, 8] == 0
    assert cleaned[12, 10] == 1
    assert cv2.countNonZero(cleaned) < 15 * 15


def test_detection_below_required_confidence_fails_clearly() -> None:
    segmenter = TestbedSegmenter.__new__(TestbedSegmenter)
    segmenter.confidence_threshold = 0.50
    prediction = np.zeros((1, 37, 2), dtype=np.float32)
    prediction[0, 4, :] = (0.20, 0.49)

    with pytest.raises(RuntimeError, match="best confidence 0.490"):
        segmenter._best_detection(prediction)


def essae_segmenter(circle_score: float = .95) -> TestbedSegmenter:
    segmenter = TestbedSegmenter.__new__(TestbedSegmenter)
    segmenter.names = {0: "Circle", 1: "Test-Bed"}
    segmenter.testbed_class_id = 1
    segmenter.circle_class_id = 0
    segmenter.input_height = segmenter.input_width = 100
    segmenter.input_name = "images"
    segmenter.confidence_threshold = segmenter.mask_threshold = .5
    segmenter.inset_px = 2
    prototypes = np.full((1, 2, 100, 100), -10, np.float32)
    prototypes[0, 0, 10:90, 10:90] = 10
    prototypes[0, 0, 60:75, 60:75] = -10  # Missing platform pixels under an object.
    prototypes[0, 1, 30:50, 30:50] = 10
    prototypes[0, 1, 37:43, 37:43] = -10  # Missing pixels inside the hole prediction.
    rows = np.array([[50, 50, 80, 80, .01, .8, 1, 0],
                     [40, 40, 20, 20, circle_score, .01, 0, 1]], np.float32)
    segmenter.session = SimpleNamespace(run=lambda *_: [rows.T[None], prototypes])
    return segmenter


def test_essae_class_order_gap_filling_and_hole_margin() -> None:
    result = essae_segmenter().segment(np.zeros((100, 100, 3), np.uint8))
    assert result.confidence == pytest.approx(.8)
    assert result.source_mask[65, 65] == 1  # Restore interior platform gaps.
    assert result.source_mask[10, 50] == 0  # Trim the tray's outer edge.
    assert result.source_mask[40, 40] == 0  # Fill then exclude the entire hole.
    assert result.source_mask[40, 29] == 0  # Expand the hole to exclude its rim.
    assert result.source_mask[40, 25] == 1
    assert np.all(result.image[result.mask == 0] == 255)


def test_essae_missing_hole_blocks_capture() -> None:
    with pytest.raises(RuntimeError, match="circular hole was not detected"):
        essae_segmenter(.2).segment(np.zeros((100, 100, 3), np.uint8))
