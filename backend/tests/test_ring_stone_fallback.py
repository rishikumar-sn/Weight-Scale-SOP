from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.domain.workflow import route_for_label
from backend.app.services.analysis_service import AnalysisService


class Repository:
    def __init__(self, root, state):
        self.root, self.state = root, deepcopy(state)

    def session_dir(self, capture_id):
        return self.root / capture_id

    def get(self, capture_id):
        return deepcopy(self.state)

    def save(self, state):
        self.state = deepcopy(state)
        return deepcopy(state)


def test_hidden_ring_returns_capture_without_invented_measurements(tmp_path):
    path = tmp_path / "ring.png"
    image = np.full((400, 500, 3), 170, np.uint8)
    cv2.circle(image, (260, 210), 54, (35, 100, 140), -1)
    cv2.imwrite(str(path), image)
    state = {
        "id": "ring", "paths": {"working": str(path)},
        "classification": {"confirmed_label": "Finger Ring"},
        "calibration": {"available": True, "found": True,
                        "mm_per_pixel_x": .18, "mm_per_pixel_y": .18},
    }
    service = AnalysisService(Repository(tmp_path, state))
    try:
        result = service._run_dimension(state)
        assert result["applicable"] is False
        assert result["outer_diameter_mm"] is None
        assert result["inner_diameter_mm"] is None
        assert result["image"] != str(path)
        assert cv2.imread(result["image"]).shape[1] == 1536
        assert np.array_equal(cv2.imread(str(path)), image)
    finally:
        service.shutdown()


@pytest.mark.parametrize("applicable", [True, False])
def test_ring_fallback_runs_stones_only_when_dimensions_unavailable(tmp_path, monkeypatch, applicable):
    item = {"index": 1, "confirmed_label": "Finger Ring", "bbox": {},
            "route": route_for_label("Finger Ring")}
    state = {"id": "ring", "classification": {"count": 1, "items": [item]},
             "route": {"items": [item["route"]]}, "job": {}}
    repository = Repository(tmp_path, state)
    service = AnalysisService(repository)
    checked = []
    monkeypatch.setattr(service, "_run_dimension", lambda *_: {"applicable": applicable, "image": "ring.png"})
    monkeypatch.setattr(service, "_run_stones", lambda _, item: checked.append(item["index"]) or {"found": False})
    try:
        service._run("ring")
        saved = repository.state
        result = saved["result"]["items"][0]
        assert checked == ([] if applicable else [1])
        assert result["status"] == "complete"
        assert result["errors"] == []
        assert saved["route"]["stones"] is (not applicable)
        assert saved["classification"]["items"][0]["route"]["stones"] is (not applicable)
        assert ("stones" in saved["result"]) is (not applicable)
    finally:
        service.shutdown()


def test_unrelated_dimension_failure_is_not_hidden(tmp_path, monkeypatch):
    import backend.app.services.analysis_service as module
    service = AnalysisService(Repository(tmp_path, {}))
    monkeypatch.setattr(module, "detect_bangle", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unreadable capture")))
    state = {"id": "ring", "paths": {"working": "missing.png"},
             "classification": {"confirmed_label": "Finger Ring"},
             "calibration": {"available": True, "found": True,
                             "mm_per_pixel_x": .18, "mm_per_pixel_y": .18}}
    try:
        with pytest.raises(OSError, match="unreadable capture"):
            service._run_dimension(state)
    finally:
        service.shutdown()


def test_uncalibrated_ring_still_returns_image(tmp_path):
    state = {"id": "ring", "paths": {"working": "ring.png"},
             "classification": {"confirmed_label": "Finger Ring"}}
    service = AnalysisService(Repository(tmp_path, state))
    try:
        result = service._run_dimension(state)
        assert result["applicable"] is False
        assert result["image"] == "ring.png"
        assert "calibration" in result["reason"]
    finally:
        service.shutdown()


@pytest.mark.parametrize("name,face,expected", [
    ("plain", None, False),
    ("gold_shadow", (5, 18, 28), False),
    ("black_stone", (18, 18, 18), True),
    ("green_stone", (15, 80, 15), True),
    ("red_stone", (25, 25, 130), True),
    ("white_stone", (210, 210, 210), True),
    ("blue_stone", (180, 60, 30), True),
    ("purple_stone", (170, 40, 130), True),
])
def test_stone_pipeline_rejects_gold_and_shadows_but_retains_stones(tmp_path, name, face, expected):
    image = np.full((300, 300, 3), 180, np.uint8)
    mask = np.zeros((300, 300), np.uint8)
    cv2.circle(image, (150, 150), 75, (25, 125, 180), -1)
    cv2.circle(mask, (150, 150), 75, 255, -1)
    if face is not None:
        cv2.circle(image, (150, 150), 20, face, -1)
    image_path, mask_path = tmp_path / "input.png", tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    state = {"id": name, "paths": {"working": str(image_path),
             "prepared": str(image_path), "mask": str(mask_path)},
             "classification": {"count": 1, "confirmed_label": "Finger Ring"},
             "calibration": {"available": True, "found": True,
                             "mm_per_pixel_x": .18, "mm_per_pixel_y": .18}}
    service = AnalysisService(Repository(tmp_path, state))
    try:
        result = service._run_stones(state)
        assert result["found"] is expected
        assert (result["risk_level"] == "NONE") is (not expected)
        if not expected:
            assert json.loads(Path(result["report"]).read_text())["stone_instance_count"] == 0
    finally:
        service.shutdown()


@pytest.mark.parametrize("highlight", [(35, 170, 215), (230, 235, 240)])
def test_decorated_gold_recesses_and_glare_do_not_become_stones(tmp_path, highlight):
    image = np.full((300, 300, 3), 180, np.uint8)
    mask = np.zeros((300, 300), np.uint8)
    cv2.circle(image, (150, 150), 75, (25, 125, 180), -1)
    cv2.circle(mask, (150, 150), 75, 255, -1)
    for y in range(110, 191, 20):
        for x in range(110, 191, 20):
            cv2.circle(image, (x, y), 7, (5, 18, 28), -1)
            cv2.circle(image, (x-2, y-2), 2, highlight, -1)
    image_path, mask_path = tmp_path / "input.png", tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    state = {"id": "decorated", "paths": {"working": str(image_path),
             "prepared": str(image_path), "mask": str(mask_path)},
             "classification": {"count": 1},
             "calibration": {"available": True, "found": True,
                             "mm_per_pixel_x": .18, "mm_per_pixel_y": .18}}
    service = AnalysisService(Repository(tmp_path, state))
    try:
        result = service._run_stones(state)
        assert result["found"] is False
        assert result["risk_level"] == "NONE"
        assert json.loads(Path(result["report"]).read_text())["stone_instance_count"] == 0
    finally:
        service.shutdown()


@pytest.mark.parametrize("applicable", [True, False])
def test_testbench_ring_process_returns_fallback_stone_result(tmp_path, applicable):
    from process_testbench.backend.processes import finger_ring
    state = {"id": "ring"}
    repository = Repository(tmp_path, state)
    class Analysis:
        def _run_dimension(self, state):
            return {"applicable": applicable, "image": "ring.png"}

        def _run_stones(self, state):
            assert state["classification"]["confirmed_label"] == "Finger Ring"
            return {"found": False, "image": "stones.png"}

    saved = finger_ring.run(state, Analysis(), repository)
    assert saved["status"] == "complete"
    assert saved["result"]["image"] == "ring.png"
    assert ("stones" in saved["result"]) is (not applicable)
