from __future__ import annotations

import cv2
import numpy as np
import pytest

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
