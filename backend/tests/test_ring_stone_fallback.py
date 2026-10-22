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


@pytest.mark.parametrize("label", ["Finger Ring", "finger ring", "Ring"])
@pytest.mark.parametrize("calibrated", [True, False])
def test_ring_analysis_is_skipped_before_loading_images(tmp_path, label, calibrated):
    state = {"id": "ring", "classification": {"confirmed_label": label},
             "paths": {}, "calibration": {"available": calibrated, "found": calibrated}}
    service = AnalysisService(Repository(tmp_path, state))
    try:
        for operation in (service._run_dimension, service._run_stones):
            result = operation(state)
            assert result["status"] == "skipped"
            assert result["applicable"] is False
            assert "image" not in result
    finally:
        service.shutdown()


@pytest.mark.parametrize("stale_route", [False, True])
def test_ring_never_runs_analysis_even_with_old_queued_route(tmp_path, monkeypatch, stale_route):
    route = ({"key": "dimension", "dimension": True, "beads": True, "stones": True}
             if stale_route else route_for_label("Finger Ring"))
    item = {"index": 1, "confirmed_label": "Finger Ring", "bbox": {}, "route": route}
    state = {"id": "ring", "classification": {"count": 1, "items": [item]},
             "route": {"items": [route]}, "job": {}}
    repository = Repository(tmp_path, state)
    service = AnalysisService(repository)
    def unexpected(*args):
        pytest.fail("Ring analysis must not execute")
    for name in ("_run_dimension", "_run_stones", "_run_beads"):
        monkeypatch.setattr(service, name, unexpected)
    try:
        service._run("ring")
        saved = repository.state
        result = saved["result"]["items"][0]
        assert saved["status"] == "complete"
        assert result["errors"] == []
        for key in ("dimensions", "stones", "beads"):
            assert key not in result
        for key in ("dimension", "stones", "beads"):
            assert saved["route"][key] is False
        assert saved["classification"]["items"][0]["route"] == route_for_label("Finger Ring")
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
             "classification": {"count": 1, "confirmed_label": "Earrings / Nosepin"},
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


def test_testbench_ring_process_skips_all_analysis(tmp_path):
    from process_testbench.backend.processes import finger_ring
    state = {"id": "ring"}
    repository = Repository(tmp_path, state)
    saved = finger_ring.run(state, object(), repository)
    assert saved["status"] == "complete"
    assert saved["result"]["status"] == "skipped"
    assert "stones" not in saved["result"]
    assert "image" not in saved["result"]
