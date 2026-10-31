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


@pytest.mark.parametrize(
    "labels",
    [
        ("Not Gold Jewelry",),
        ("Bangle", "Not Gold Jewelry"),
    ],
)
def test_not_gold_prediction_blocks_single_and_multi_item_analysis(monkeypatch, labels) -> None:
    items = [
        {"index": index, "predicted_label": item_label, "paths": {"working": f"{index}.png"}}
        for index, item_label in enumerate(labels, start=1)
    ]
    repository = _Repository({"id": "capture-rejected", "classification": {"items": items}, "paths": {}})
    service = AnalysisService(repository)
    submitted: list[str] = []
    monkeypatch.setattr(service._executor, "submit", lambda *_args, **_kwargs: submitted.append("started"))

    with pytest.raises(RuntimeError, match="classified as Not Gold Jewelry"):
        service.confirm_and_start(
            "capture-rejected",
            None,
            False,
            [
                {"index": item["index"], "label": "Bangle", "learn": False}
                for item in items
            ],
        )

    assert submitted == []
    assert repository.state["classification"]["items"] == items
    service.shutdown()


def test_not_gold_prediction_cannot_be_overridden_or_relearned_as_gold(monkeypatch) -> None:
    item = {"index": 1, "predicted_label": "Not Gold Jewelry", "paths": {"working": "item.png"}}
    repository = _Repository({"id": "capture-override", "status": "awaiting_confirmation", "classification": {"items": [item]}, "paths": {}})
    service = AnalysisService(repository)
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_args, **_kwargs: "evidence.jpg")
    monkeypatch.setattr(service._executor, "submit", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="confirmed as Not Gold Jewelry"):
        service.confirm_and_start(
            "capture-override",
            None,
            False,
            [{"index": 1, "label": "Not Gold Jewelry", "learn": False}],
            override_not_gold=True,
        )

    for save_only in (False, True):
        with pytest.raises(RuntimeError, match="cannot be overwritten"):
            service.confirm_and_start(
                "capture-override", None, True,
                [{"index": 1, "label": "Bangle", "learn": True}],
                override_not_gold=True, save_only=save_only,
            )
    assert repository.state["classification"]["items"] == [item]
    service.shutdown()


@pytest.mark.parametrize("foreign_first", [True, False])
@pytest.mark.parametrize("gate", ["semantic", "hsv"])
def test_mixed_foreign_and_gold_capture_keeps_preview_and_confirmation(monkeypatch, tmp_path, foreign_first, gate):
    from Classification.jewelry_classifier import PredictionResult

    working, mask_path = tmp_path / "working.png", tmp_path / "mask.png"
    cv2.imwrite(str(working), np.full((80, 160, 3), 200, np.uint8))
    cv2.imwrite(str(mask_path), np.full((80, 160), 255, np.uint8))
    repository = _Repository({"id": "mixed", "paths": {"working": str(working), "mask": str(mask_path)}})
    repository.session_dir = lambda _id: tmp_path
    separated = []
    for index, foreign in enumerate((foreign_first, not foreign_first)):
        crop = np.full((50, 50, 3), (0, 0, 255) if foreign else (20, 160, 220), np.uint8)
        separated.append(dict(bbox={"x": index * 70, "y": 0, "w": 50, "h": 50},
                              area_px=1000, crop_bgr=crop, raw_crop_bgr=crop,
                              mask=np.ones((50, 50), np.uint8)))

    class Classifier:
        def check_non_jewelry(self, image):
            return gate == "semantic" and bool(np.asarray(image)[0, 0, 0] == 255), "semantic check"

        def classify_image(self, image, image_path):
            foreign = np.asarray(image)[0, 0, 0] == 255
            assert not foreign or gate == "hsv"  # The raw eliminator stays authoritative.
            return PredictionResult(label="Not Gold Jewelry" if foreign else "Bangle",
                                    confidence=.9, scores=[], image_path=image_path,
                                    original_image=image, cropped_image=image,
                                    decision_source="obvious_non_gold_color" if foreign else "siglip")

    service = AnalysisService(repository)
    monkeypatch.setattr("backend.app.services.analysis_service.separate_jewellery_items", lambda *_a, **_kw: separated)
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_a, **_kw: "mixed-preview.jpg")
    monkeypatch.setattr(service, "_get_foreign_objects", lambda: pytest.fail("Existing rejection must skip the second model"))
    monkeypatch.setattr(service, "_save_rejected_review", lambda *_a: pytest.fail("No rejection preview"))
    try:
        state = service.classify("mixed")
        assert state["status"] == "awaiting_confirmation"
        assert state["paths"]["evidence"] == "mixed-preview.jpg"
        assert state["classification"]["labels"] == (["Not Gold Jewelry", "Bangle"] if foreign_first else ["Bangle", "Not Gold Jewelry"])
    finally:
        service.shutdown()


def test_empty_capture_rejects_without_classifier_or_review(monkeypatch, tmp_path):
    working, mask_path = tmp_path / "working.png", tmp_path / "mask.png"
    cv2.imwrite(str(working), np.full((50, 50, 3), 200, np.uint8))
    cv2.imwrite(str(mask_path), np.zeros((50, 50), np.uint8))
    repository = _Repository({"id": "empty", "status": "captured",
                              "paths": {"working": str(working), "mask": str(mask_path)}})
    service = AnalysisService(repository)
    monkeypatch.setattr("backend.app.services.analysis_service.separate_jewellery_items", lambda *_a, **_kw: [])
    monkeypatch.setattr(service, "_get_classifier", lambda: pytest.fail("No object to classify"))
    monkeypatch.setattr(service, "_save_rejected_review", lambda *_a: pytest.fail("No rejection preview"))
    try:
        with pytest.raises(RuntimeError, match="No separate jewellery item"):
            service.classify("empty")
        assert repository.state["status"] == "captured"
    finally:
        service.shutdown()


@pytest.mark.parametrize("decision_source", ["empty_background", "obvious_non_gold_color", "strong_non_jewelry", "gallery"])
def test_only_not_gold_predictions_reject_without_review(monkeypatch, tmp_path, decision_source):
    from Classification.jewelry_classifier import PredictionResult

    working, mask_path = tmp_path / "working.png", tmp_path / "mask.png"
    crop = np.full((50, 50, 3), 200, np.uint8)
    cv2.imwrite(str(working), crop)
    cv2.imwrite(str(mask_path), np.full((50, 50), 255, np.uint8))
    repository = _Repository({"id": "non-gold", "status": "captured",
                              "paths": {"working": str(working), "mask": str(mask_path)}})
    repository.session_dir = lambda _id: tmp_path
    separated = dict(bbox={"x": 0, "y": 0, "w": 50, "h": 50}, area_px=1000,
                     crop_bgr=crop, raw_crop_bgr=crop, mask=np.ones((50, 50), np.uint8))

    class Classifier:
        def check_non_jewelry(self, _image):
            return False, "semantic check"

        def classify_image(self, image, image_path):
            return PredictionResult(label="Not Gold Jewelry", confidence=.9, scores=[], image_path=image_path,
                                    original_image=image, cropped_image=image, decision_source=decision_source)

    service = AnalysisService(repository)
    monkeypatch.setattr("backend.app.services.analysis_service.separate_jewellery_items", lambda *_a, **_kw: [separated])
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())
    from types import SimpleNamespace
    monkeypatch.setattr(service, "_check_foreign_objects", lambda *_: {
        "passed": True, "detections": [], "jewellery_instances": [
            {"component_overlaps": [{"component_index": 1, "component_share": .95}]}],
    })
    monkeypatch.setattr(service, "_get_jewel_thread", lambda: SimpleNamespace(classify=lambda _: {"passed": True}))
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_a, **_kw: pytest.fail("No classified preview"))
    monkeypatch.setattr(service, "_save_rejected_review", lambda *_a: None)
    try:
        result = service.classify("non-gold")
        assert result["status"] == "rejected"
        assert result["rejection_kind"] == "non_jewellery"
        assert result["classification"]["items"] == []
    finally:
        service.shutdown()


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


@pytest.mark.parametrize("selected_label", ["Not Gold Jewelry", "Necklace"])
def test_recapture_saves_corrections_without_starting_analysis(monkeypatch, selected_label) -> None:
    item = {"index": 1, "predicted_label": "Bangle", "paths": {"working": "old-item.png"}}
    repository = _Repository({
        "id": "recapture", "status": "awaiting_confirmation",
        "classification": {"items": [item]}, "paths": {},
    })
    service = AnalysisService(repository)
    learned = []
    classifier = type("Classifier", (), {
        "learn_correction": lambda _self, path, label: learned.append((path, label)),
    })()
    monkeypatch.setattr(service, "_get_classifier", lambda: classifier)
    monkeypatch.setattr(service._executor, "submit", lambda *_args, **_kwargs: pytest.fail("Analysis must not start"))
    for _ in range(2):
        saved = service.confirm_and_start(
            "recapture", None, False,
            [{"index": 1, "label": selected_label, "learn": True}], save_only=True,
        )
    assert learned == [("old-item.png", selected_label)]
    assert saved["status"] == "awaiting_confirmation"
    assert saved["classification"]["items"][0]["correction_label"] == selected_label
    assert not saved["classification"].get("confirmed")
    service.shutdown()


def test_recapture_learns_unchanged_not_gold_once_without_starting_analysis(monkeypatch):
    repository = _Repository({
        "id": "mixed-recapture", "status": "awaiting_confirmation", "paths": {},
        "classification": {"items": [
            {"index": 1, "predicted_label": "Bangle", "paths": {"working": "gold.png"}},
            {"index": 2, "predicted_label": "Not Gold Jewelry", "paths": {"working": "foreign.png"}},
        ]},
    })
    service = AnalysisService(repository)
    learned = []
    classifier = type("Classifier", (), {
        "learn_correction": lambda _self, path, label: learned.append((path, label)),
    })()
    monkeypatch.setattr(service, "_get_classifier", lambda: classifier)
    monkeypatch.setattr(service._executor, "submit", lambda *_a, **_kw: pytest.fail("Analysis must not start"))
    try:
        for _ in range(2):
            state = service.confirm_and_start("mixed-recapture", None, False, [
                {"index": 1, "label": "Bangle", "learn": False},
                {"index": 2, "label": "Not Gold Jewelry", "learn": False},
            ], save_only=True)
        assert learned == [("foreign.png", "Not Gold Jewelry")]
        assert state["status"] == "awaiting_confirmation"
        assert state["classification"]["items"][1]["gallery_saved_label"] == "Not Gold Jewelry"
    finally:
        service.shutdown()


def test_confirm_saves_manual_correction_for_gallery(monkeypatch) -> None:
    repository = _Repository({
        "id": "confirm-gallery", "status": "awaiting_confirmation", "paths": {},
        "classification": {"items": [
            {"index": 1, "predicted_label": "Bangle", "paths": {"working": "item.png"}},
        ]},
    })
    service = AnalysisService(repository)
    learned = []
    classifier = type("Classifier", (), {
        "learn_correction": lambda _self, path, label: learned.append((path, label)),
    })()
    monkeypatch.setattr(service, "_get_classifier", lambda: classifier)
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_args, **_kwargs: "evidence.jpg")
    monkeypatch.setattr(service._executor, "submit", lambda *_args, **_kwargs: None)
    confirmed = service.confirm_and_start(
        "confirm-gallery", None, False,
        [{"index": 1, "label": "Necklace", "learn": True}],
    )
    assert learned == [("item.png", "Necklace")]
    assert confirmed["status"] == "processing"
    service.shutdown()


@pytest.mark.parametrize("indices", [(1, 1), (1, 3)])
def test_recapture_validates_all_indices_before_learning(monkeypatch, indices) -> None:
    repository = _Repository({
        "id": "invalid", "status": "awaiting_confirmation", "paths": {},
        "classification": {"items": [
            {"index": index, "predicted_label": "Bangle", "paths": {"working": f"{index}.png"}}
            for index in (1, 2)
        ]},
    })
    service = AnalysisService(repository)
    monkeypatch.setattr(service, "_get_classifier", lambda: pytest.fail("Invalid input must not teach gallery"))
    with pytest.raises(RuntimeError, match="every detected jewel"):
        service.confirm_and_start(
            "invalid", None, False,
            [{"index": index, "label": "Not Gold Jewelry", "learn": True} for index in indices],
            save_only=True,
        )
    service.shutdown()
