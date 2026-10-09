from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.app.analysis.beads import (
    analyze_bead_detections,
    bead_size_category,
    onnx_image_size,
    prepare_onnx_input,
)


class FakeSession:
    def __init__(self, shape, metadata):
        self._input = SimpleNamespace(shape=shape)
        self._metadata = SimpleNamespace(custom_metadata_map=metadata)

    def get_inputs(self):
        return [self._input]

    def get_modelmeta(self):
        return self._metadata


def test_onnx_image_size_uses_and_validates_model_metadata():
    assert onnx_image_size(FakeSession([1, 3, 960, 960], {"imgsz": "[960, 960]"})) == (
        960,
        960,
    )
    assert onnx_image_size(FakeSession([1, 3, "height", "width"], {"imgsz": "[768, 1024]"})) == (
        768,
        1024,
    )
    with pytest.raises(ValueError, match="does not match"):
        onnx_image_size(FakeSession([1, 3, 640, 640], {"imgsz": "[960, 960]"}))


def test_letterbox_follows_rectangular_model_size():
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    tensor, scale, left, top = prepare_onnx_input(image, (768, 1024))

    assert tensor.shape == (1, 3, 768, 1024)
    assert scale == pytest.approx(1.28)
    assert left == 0
    assert top == 128


def test_calibrated_size_categories_use_physical_diameter():
    assert bead_size_category(2.99) == "tiny"
    assert bead_size_category(3.0) == "small"
    assert bead_size_category(5.99) == "small"
    assert bead_size_category(6.0) == "large"

    image = np.full((160, 360, 3), 220, dtype=np.uint8)
    detections = [
        {"bbox": [10, 50, 30, 70], "score": 0.9},
        {"bbox": [100, 45, 140, 85], "score": 0.9},
        {"bbox": [230, 30, 300, 100], "score": 0.9},
    ]
    result = analyze_bead_detections(
        image,
        detections,
        {"mm_per_pixel_x": 0.1, "mm_per_pixel_y": 0.1},
    )

    assert result["size"]["counts"] == {"tiny": 1, "small": 1, "large": 1}
    assert [item["size_category"] for item in detections] == ["tiny", "small", "large"]
    assert "1 large, 1 small, and 1 tiny" in result["summary"]


def test_hsv_color_sampling_counts_black_and_red_beads():
    image = np.full((120, 260, 3), 230, dtype=np.uint8)
    cv2.circle(image, (60, 60), 28, (15, 15, 15), cv2.FILLED)
    cv2.circle(image, (190, 60), 28, (20, 20, 210), cv2.FILLED)
    detections = [
        {"bbox": [30, 30, 90, 90], "score": 0.9},
        {"bbox": [160, 30, 220, 90], "score": 0.9},
    ]

    result = analyze_bead_detections(image, detections)

    assert result["colors"]["counts"] == {"black": 1, "red": 1}
    assert result["colors"]["sampled_pixels"] <= 2 * 24 * 24
    assert [item["color"] for item in detections] == ["black", "red"]


def test_regular_close_boxes_are_reported_as_continuous_and_repetitive():
    image = np.full((100, 280, 3), 220, dtype=np.uint8)
    detections = []
    for index in range(8):
        x1 = 10 + index * 31
        detections.append({"bbox": [x1, 35, x1 + 25, 60], "score": 0.9})

    result = analyze_bead_detections(image, detections)

    assert result["arrangement"]["pattern"] == "continuous"
    assert result["arrangement"]["repetitive"] is True
    assert "Continuous, repetitive beads" in result["summary"]


def test_regular_distant_boxes_are_reported_as_well_spaced():
    image = np.full((100, 500, 3), 220, dtype=np.uint8)
    detections = []
    for index in range(6):
        x1 = 10 + index * 80
        detections.append({"bbox": [x1, 35, x1 + 20, 55], "score": 0.9})

    result = analyze_bead_detections(image, detections)

    assert result["arrangement"]["pattern"] == "well_spaced"
    assert "Well-spaced beads" in result["summary"]


def test_hundreds_of_tiny_beads_get_huge_count_wording():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    detections = [{"bbox": [0, 0, 10, 10], "score": 0.9} for _ in range(100)]

    result = analyze_bead_detections(
        image,
        detections,
        {"mm_per_pixel_x": 0.1, "mm_per_pixel_y": 0.1},
    )

    assert "A huge number of tiny beads are present (100 tiny)." in result["summary"]
