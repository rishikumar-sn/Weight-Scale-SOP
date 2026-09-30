from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
import pytest

from backend.app.analysis.vision import separate_jewellery_items
from backend.app.services.analysis_service import (
    AnalysisService,
    _ring_bangle_dimension_fallback,
)


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


def test_enclosed_black_stone_interior_is_preserved() -> None:
    image = np.full((180, 240, 3), 205, dtype=np.uint8)
    mask = np.zeros((180, 240), dtype=np.uint8)
    cv2.rectangle(mask, (30, 50), (120, 130), 1, -1)
    cv2.circle(mask, (130, 90), 14, 1, 6)
    image[mask > 0] = (35, 145, 205)
    cv2.circle(image, (130, 90), 11, (18, 18, 18), -1)

    items = separate_jewellery_items(
        image,
        mask,
        min_area_px=100,
        min_area_ratio=0,
        padding_px=8,
    )

    assert len(items) == 1
    item = items[0]
    local_x = 130 - item["bbox"]["x"]
    local_y = 90 - item["bbox"]["y"]
    assert item["mask"][local_y, local_x] == 1
    assert np.all(item["crop_bgr"][local_y, local_x] == (18, 18, 18))


def test_enclosed_background_opening_remains_removed() -> None:
    image = np.full((180, 240, 3), 205, dtype=np.uint8)
    mask = np.zeros((180, 240), dtype=np.uint8)
    cv2.rectangle(mask, (30, 50), (120, 130), 1, -1)
    cv2.circle(mask, (130, 90), 14, 1, 6)
    image[mask > 0] = (35, 145, 205)

    items = separate_jewellery_items(
        image,
        mask,
        min_area_px=100,
        min_area_ratio=0,
        padding_px=8,
    )

    assert len(items) == 1
    item = items[0]
    local_x = 130 - item["bbox"]["x"]
    local_y = 90 - item["bbox"]["y"]
    assert item["mask"][local_y, local_x] == 0
    assert np.all(item["crop_bgr"][local_y, local_x] == 255)


def test_dimension_fallback_resolves_only_ring_bangle_size_conflicts() -> None:
    mask = np.zeros((320, 320), dtype=np.uint8)
    cv2.circle(mask, (160, 160), 130, 1, 8)
    calibration = {
        "available": True,
        "found": True,
        "mm_per_pixel_x": 0.18,
        "mm_per_pixel_y": 0.18,
    }

    small_mask = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_NEAREST)
    label, small_evidence = _ring_bangle_dimension_fallback(
        "Bangle", small_mask, calibration
    )
    assert label == "Finger Ring"
    assert small_evidence["applied"] is True
    assert small_evidence["outer_span_mm"] <= 35.0

    label, large_evidence = _ring_bangle_dimension_fallback(
        "Finger Ring", mask, calibration
    )
    assert label == "Bangle"
    assert large_evidence["applied"] is True
    assert large_evidence["outer_span_mm"] >= 40.0

    label, unrelated_evidence = _ring_bangle_dimension_fallback(
        "Earrings / Nosepin", small_mask, calibration
    )
    assert label == "Earrings / Nosepin"
    assert unrelated_evidence["available"] is False


def test_dimension_fallback_keeps_prediction_without_calibration() -> None:
    mask = np.ones((50, 50), dtype=np.uint8)

    label, evidence = _ring_bangle_dimension_fallback("Bangle", mask, None)

    assert label == "Bangle"
    assert evidence["applied"] is False


def test_raw_semantic_rejection_runs_before_masked_category_prediction(
    monkeypatch,
    tmp_path,
) -> None:
    working_path = tmp_path / "working.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(working_path), np.full((80, 80, 3), 255, dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.full((80, 80), 255, dtype=np.uint8))
    state = {
        "id": "foreign-object",
        "paths": {"working": str(working_path), "mask": str(mask_path)},
        "settings_snapshot": {"analysis": {"item_separation": {}}},
    }

    class Repository:
        def get(self, _capture_id):
            return deepcopy(state)

        def session_dir(self, capture_id):
            return tmp_path / capture_id

    class Classifier:
        def check_non_jewelry(self, _image):
            return True, "foreign object"

        def classify_image(self, *_args, **_kwargs):
            raise AssertionError("Masked category prediction must not run")

    separated_item = {
        "bbox": {"x": 0, "y": 0, "w": 50, "h": 50},
        "area_px": 1000,
        "raw_crop_bgr": np.full((50, 50, 3), (0, 0, 255), dtype=np.uint8),
        "crop_bgr": np.full((50, 50, 3), 255, dtype=np.uint8),
        "mask": np.ones((50, 50), dtype=np.uint8),
    }
    monkeypatch.setattr(
        "backend.app.services.analysis_service.separate_jewellery_items",
        lambda *_args, **_kwargs: [separated_item],
    )
    service = AnalysisService(Repository())
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())

    with pytest.raises(RuntimeError, match="non-jewellery object"):
        service.classify("foreign-object")
    service.shutdown()


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
