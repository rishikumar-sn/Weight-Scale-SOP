from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.app.analysis.foreign_objects import ForeignObjectDetector
from backend.app.services.analysis_service import AnalysisService
from Classification.jewelry_classifier import PredictionResult


def detector_for_rows(rows, prototypes):
    detector = ForeignObjectDetector.__new__(ForeignObjectDetector)
    detector.input_width = detector.input_height = 100
    detector.input_name = "images"
    detector.model_path = Path("anomalies.onnx")
    detector.mask_threshold = .5
    detector.names = {0: "foreign_obj", 1: "jewellery", 2: "paper", 3: "plastic_packet"}
    detector.session = SimpleNamespace(run=lambda *_: [np.array(rows, np.float32).T[None], prototypes])
    return detector


@pytest.mark.parametrize("class_id,threshold", [(0, .45), (2, .25), (3, .35)])
def test_class_thresholds_and_jewellery_does_not_suppress_foreign(class_id, threshold):
    def row(score, category):
        result = [50, 50, 30, 30, 0, 0, 0, 0, 10]
        result[4 + category] = score
        return result

    image = np.full((100, 100, 3), 160, np.uint8)
    mask = np.ones((100, 100), np.uint8)
    for score, expected in [(threshold - .001, False), (threshold, True)]:
        detector = detector_for_rows([row(score, class_id), row(.99, 1)], np.ones((1, 1, 25, 25), np.float32))
        result = detector.detect(image, mask)
        assert bool(result["detections"]) is expected
        assert result["passed"] is not expected


def test_mask_polygon_filter_ignores_outside_but_keeps_crossing_object():
    # Both boxes have their centre outside the polygon. Only one actual
    # segmentation overlaps it; box intersection alone must not reject.
    prototypes = np.full((1, 2, 25, 25), -10, np.float32)
    prototypes[0, 0, 8:18, 2:5] = 10
    prototypes[0, 1, 8:18, 9:13] = 10
    rows = [[30, 50, 50, 60, .9, 0, 0, 0, 1, 0],
            [30, 50, 50, 60, 0, 0, .8, 0, 0, 1]]
    detector = detector_for_rows(rows, prototypes)
    mask = np.zeros((100, 100), np.uint8)
    mask[25:75, 40:80] = 1
    result = detector.detect(np.zeros((100, 100, 3), np.uint8), mask)
    assert [d["label"] for d in result["detections"]] == ["paper"]
    assert result["detections"][0]["bbox"]["x"] >= 40


def test_letterbox_masks_map_back_to_rectangular_raw_capture():
    detector = detector_for_rows([[50, 50, 20, 20, .9, 0, 0, 0, 10]],
                                 np.ones((1, 1, 25, 25), np.float32))
    mask = np.zeros((50, 100), np.uint8)
    mask[20:30, 45:55] = 255
    result = detector.detect(np.zeros((50, 100, 3), np.uint8), mask)
    assert not result["passed"]
    assert result["detections"][0]["testbed_pixels"] == 100
    with pytest.raises(ValueError, match="testbed mask"):
        detector.detect(np.zeros((50, 100, 3), np.uint8), np.ones((100, 100), np.uint8))


@pytest.fixture
def capture_service(monkeypatch, tmp_path):
    raw = np.full((80, 160, 3), (10, 80, 190), np.uint8)
    working = np.full((80, 80, 3), 200, np.uint8)
    paths = {}
    for name, pixels in {"original": raw, "working": working, "mask": np.full((80, 80), 255, np.uint8),
                         "testbed_source_mask": np.full((80, 160), 255, np.uint8)}.items():
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), pixels)
        paths[name] = str(path)

    class Repository:
        state = {"id": "capture", "paths": paths, "status": "captured"}

        def get(self, _id):
            return deepcopy(self.state)

        def save(self, state):
            self.state = deepcopy(state)

        def session_dir(self, _id):
            return tmp_path

    class Classifier:
        def check_non_jewelry(self, _image):
            return False, "passed"

        def classify_image(self, image, image_path):
            return PredictionResult(label="Bangle", confidence=.9, scores=[], image_path=image_path,
                                    original_image=image, cropped_image=image)

    separated = dict(bbox={"x": 0, "y": 0, "w": 80, "h": 80}, area_px=500,
                     crop_bgr=working, raw_crop_bgr=working, mask=np.ones((80, 80), np.uint8))
    monkeypatch.setattr("backend.app.services.analysis_service.separate_jewellery_items", lambda *_a, **_kw: [separated])
    service = AnalysisService(Repository())
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_a, **_kw: "review.jpg")
    yield service, raw
    service.shutdown()


@pytest.mark.parametrize("foreign", [False, True])
def test_second_gate_uses_full_raw_and_blocks_confirm_and_learning(capture_service, monkeypatch, foreign):
    service, raw = capture_service
    calls = []

    def detect(image, mask):
        np.testing.assert_array_equal(image, raw)
        assert mask.shape == raw.shape[:2]
        calls.append(True)
        return {"passed": not foreign, "detections": [{"label": "plastic_packet"}] if foreign else []}

    monkeypatch.setattr(service, "_get_foreign_objects", lambda: SimpleNamespace(detect=detect))
    state = service.classify("capture")
    assert calls == [True]
    assert state["status"] == ("rejected" if foreign else "awaiting_confirmation")
    if foreign:
        for save_only in (True, False):
            with pytest.raises(RuntimeError, match="other objects with jewellery"):
                service.confirm_and_start("capture", "Bangle", True, override_not_gold=True, save_only=save_only)
    else:
        monkeypatch.setattr(service._executor, "submit", lambda *_a, **_kw: None)
        assert service.confirm_and_start("capture", "Bangle", False)["status"] == "processing"


def test_missing_raw_mask_fails_closed(capture_service, monkeypatch):
    service, _raw = capture_service
    service.repository.state["paths"].pop("testbed_source_mask")
    monkeypatch.setattr(service, "_get_foreign_objects", lambda: pytest.fail("Do not infer without a mask"))
    with pytest.raises(RuntimeError, match="unavailable; recapture"):
        service.classify("capture")
    assert service.repository.state["status"] == "captured"


def test_model_failure_does_not_allow_confirmation(capture_service, monkeypatch):
    service, _raw = capture_service

    def unavailable():
        raise RuntimeError("Foreign-object model failed")

    monkeypatch.setattr(service, "_get_foreign_objects", unavailable)
    with pytest.raises(RuntimeError, match="model failed"):
        service.classify("capture")
    with pytest.raises(RuntimeError, match="Check the item type"):
        service.confirm_and_start("capture", "Bangle", False)
