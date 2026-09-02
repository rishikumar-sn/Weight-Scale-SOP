from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from backend.app.analysis.vision import separate_jewellery_items
from backend.app.services.analysis_service import AnalysisService


def test_separate_jewellery_items_returns_stable_pair_and_isolated_crops() -> None:
    image = np.full((240, 420, 3), 210, dtype=np.uint8)
    mask = np.zeros((240, 420), dtype=np.uint8)
    cv2.circle(mask, (105, 120), 55, 1, 12)
    cv2.circle(mask, (315, 120), 48, 1, 10)
    image[mask > 0] = (20, 160, 220)
    mask[5:7, 5:7] = 1  # Sensor noise must not become a jewel.

    items = separate_jewellery_items(
        image,
        mask,
        min_area_px=100,
        min_area_ratio=0,
        padding_px=8,
    )

    assert len(items) == 2
    assert items[0]["bbox"]["x"] < items[1]["bbox"]["x"]
    for item in items:
        assert item["crop_bgr"].shape[:2] == item["mask"].shape
        assert item["raw_crop_bgr"].shape == item["crop_bgr"].shape
        assert np.all(item["crop_bgr"][item["mask"] == 0] == 255)
        assert np.all(item["raw_crop_bgr"][item["mask"] == 0] == 210)


def test_touching_objects_are_not_split_unreliably() -> None:
    image = np.full((180, 300, 3), 255, dtype=np.uint8)
    mask = np.zeros((180, 300), dtype=np.uint8)
    cv2.circle(mask, (105, 90), 45, 1, -1)
    cv2.circle(mask, (190, 90), 45, 1, -1)

    items = separate_jewellery_items(image, mask, min_area_px=100, min_area_ratio=0)

    assert len(items) == 1


class _Repository:
    def __init__(self, state: dict) -> None:
        self.state = deepcopy(state)

    def get(self, _capture_id: str):
        return deepcopy(self.state)

    def save(self, state: dict):
        self.state = deepcopy(state)
        return deepcopy(state)


def test_mixed_items_receive_independent_routes(monkeypatch) -> None:
    state = {
        "id": "capture-1",
        "classification": {
            "predicted_label": "Bangle, Necklace",
            "items": [
                {"index": 1, "predicted_label": "Bangle", "paths": {"working": "bangle.png"}},
                {"index": 2, "predicted_label": "Necklace", "paths": {"working": "necklace.png"}},
            ],
        },
        "paths": {},
    }
    repository = _Repository(state)
    service = AnalysisService(repository)
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_args, **_kwargs: "evidence.jpg")
    monkeypatch.setattr(service._executor, "submit", lambda *_args, **_kwargs: None)

    confirmed = service.confirm_and_start(
        "capture-1",
        None,
        False,
        [
            {"index": 1, "label": "Bangle", "learn": False},
            {"index": 2, "label": "Necklace", "learn": False},
        ],
    )
    service.shutdown()

    first, second = confirmed["classification"]["items"]
    assert first["route"] == {
        "key": "dimension", "dimension": True, "beads": False, "stones": False
    }
    assert second["route"]["beads"] is True
    assert second["route"]["stones"] is True
    assert confirmed["route"]["dimension"] is True
    assert confirmed["route"]["stones"] is True


def test_each_bangle_and_earring_is_analyzed_independently(monkeypatch) -> None:
    items = []
    for index, label in enumerate(("Bangle", "Bangle", "Earring", "Earring"), start=1):
        items.append({
            "index": index,
            "confirmed_label": label,
            "bbox": {"x": index * 10, "y": 10, "w": 20, "h": 20},
            "route": {
                "key": "dimension" if label == "Bangle" else "analysis",
                "dimension": label == "Bangle",
                "beads": False,
                "stones": label == "Earring",
            },
        })
    state = {
        "id": "capture-pairs",
        "captured_at": "2026-08-21T10:00:00+05:30",
        "updated_at": "2026-08-21T10:00:00+05:30",
        "status": "processing",
        "classification": {"items": items},
        "route": {"items": [item["route"] for item in items]},
        "job": {},
    }
    repository = _Repository(state)
    service = AnalysisService(repository)
    measured: list[int] = []
    stone_checked: list[int] = []
    monkeypatch.setattr(
        service,
        "_run_dimension",
        lambda _state, item: measured.append(item["index"]) or {"outer_diameter_mm": 60.0},
    )
    monkeypatch.setattr(
        service,
        "_run_stones",
        lambda _state, item: stone_checked.append(item["index"]) or {"found": False},
    )

    service._run("capture-pairs")
    service.shutdown()

    assert measured == [1, 2]
    assert stone_checked == [3, 4]
    assert repository.state["status"] == "complete"
    assert [item["status"] for item in repository.state["result"]["items"]] == [
        "complete", "complete", "complete", "complete"
    ]
