from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.app.analysis.component_presence import (
    blocks_jewellery_labels, component_presence_check,
)
from backend.app.analysis import annotations
from backend.app.services.analysis_service import AnalysisService
from Classification.jewelry_classifier import PredictionResult


def evidence(*shares):
    return {"passed": True, "detections": [], "jewellery_instances": [
        {"component_overlaps": [{"component_index": index, "component_share": share,
                                  "pixels": int(share * 1000)}
                                 for index, share in enumerate(shares, 1)]}]}


@pytest.mark.parametrize("check,count,expected", [
    (evidence(.9, 0), 2, False),
    (evidence(.9, .19), 2, False),
    (evidence(.9, .2), 2, True),
    ({"passed": True, "detections": []}, 1, False),
    (evidence(), 0, False),
])
def test_every_candidate_needs_positive_mask_evidence(check, count, expected):
    assert component_presence_check(check, count)["passed"] is expected


@pytest.fixture
def service_case(monkeypatch, tmp_path):
    image = np.full((150, 250, 3), 220, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (70, 75), 35, 255, 10)
    cv2.rectangle(mask, (190, 20), (203, 130), 255, -1)
    paths = {}
    for name, value in {"original": image, "working": image, "mask": mask,
                         "testbed_source_mask": np.ones(mask.shape, np.uint8) * 255,
                         "testbed_mask": np.ones(mask.shape, np.uint8) * 255}.items():
        path = tmp_path/f"{name}.png"
        cv2.imwrite(str(path), value); paths[name] = str(path)

    class Repository:
        state = {"id": "candidate", "status": "captured", "paths": paths,
                 "rois": {"processing": {"x": 0, "y": 0, "w": 250, "h": 150}},
                 "classification": {"count": 2, "items": [{"index": 1, "predicted_label": "Bangle"}]}}

        def get(self, _id): return deepcopy(self.state)
        def save(self, state): self.state = deepcopy(state)
        def session_dir(self, _id): return tmp_path

    service = AnalysisService(Repository())
    monkeypatch.setattr(service, "_save_classified_evidence", lambda *_a, **_kw: "evidence.jpg")
    monkeypatch.setattr(service, "_get_classifier", lambda: pytest.fail("Must not classify unverified fragments"))
    monkeypatch.setattr(service, "_get_jewel_thread", lambda: pytest.fail("Presence must run before thread classification"))
    yield service
    service.shutdown()


@pytest.mark.parametrize("label", ["paper", "plastic_packet", "foreign_obj"])
def test_any_anomaly_blocks_all_categories_even_with_jewellery_evidence(service_case, monkeypatch, label):
    service = service_case
    check = evidence(.95, .95)
    check.update(passed=False, detections=[{"label": label, "bbox": {"x": 15,"y":15,"w":80,"h":80}}])
    monkeypatch.setattr(service, "_check_foreign_objects", lambda *_: check)
    result = service.classify("candidate")
    assert result["status"] == "rejected"
    assert result["classification"]["items"] == []
    assert result["classification"]["count"] == 0
    assert result["route"] == result["result"] == {}
    with pytest.raises(RuntimeError):
        service.confirm_and_start("candidate", "Finger Ring", True, save_only=True)


def test_unsupported_shadow_with_real_jewel_rejects_without_inventing_count(service_case, monkeypatch):
    service = service_case
    monkeypatch.setattr(service, "_check_foreign_objects", lambda *_: evidence(.95, 0))
    result = service.classify("candidate")
    assert result["rejection_kind"] == "unverified_foreground"
    assert not result["component_presence_check"]["passed"]
    assert result["classification"]["items"] == []
    assert (service.repository.session_dir("candidate")/"capture/classified_evidence.jpg").is_file()


def test_gate_failure_clears_predictions_from_previous_attempt(service_case, monkeypatch):
    service = service_case
    def fail(*_): raise RuntimeError("simulated ONNX failure")
    monkeypatch.setattr(service, "_check_foreign_objects", fail)
    result = service.classify("candidate")
    assert result["status"] == "rejected"
    assert result["rejection_kind"] == "verification_unavailable"
    assert result["classification"]["count"] == 0
    assert result["foreign_object_check"]["error"] == "simulated ONNX failure"


def test_missing_instance_evidence_cannot_fall_back_to_siglip(service_case, monkeypatch):
    service = service_case
    monkeypatch.setattr(service, "_check_foreign_objects", lambda *_: {"passed": True,"detections": []})
    result = service.classify("candidate")
    assert result["status"] == "rejected"
    assert result["classification"]["count"] == 0
    assert not result["component_presence_check"]["instance_evidence_available"]


def test_supported_candidates_keep_independent_labels(service_case, monkeypatch):
    service = service_case
    monkeypatch.setattr(service, "_check_foreign_objects", lambda *_: evidence(.95, .9))
    monkeypatch.setattr(service, "_get_jewel_thread", lambda: SimpleNamespace(classify=lambda _: {"passed": True}))
    calls = []
    class Classifier:
        def check_non_jewelry(self, _): return False, "clear"
        def classify_image(self, image, image_path):
            calls.append(image_path)
            return PredictionResult(label="Finger Ring", confidence=.9, scores=[], image_path=image_path,
                                    original_image=image, cropped_image=image)
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())
    result = service.classify("candidate")
    assert result["status"] == "awaiting_confirmation"
    assert result["classification"]["count"] == len(calls) == 2


def test_legacy_anomaly_review_hides_j_labels_and_preserves_a_label(monkeypatch):
    boxes = []
    monkeypatch.setattr(annotations, "label_box", lambda _image, _box, label, **_kw: boxes.append(label))
    state = {"foreign_object_check": {"passed": False, "detections": [
        {"label": "plastic_packet", "bbox": {"x": 20, "y": 20, "w": 60, "h": 60}}]}}
    items = [{"index": 2, "predicted_label": "Earrings / Nosepin", "bbox": {"x": 30,"y":30,"w":40,"h":40}}]
    before = deepcopy(items)
    annotations.render_capture_review(np.full((120,120,3), 200, np.uint8), state, items)
    assert boxes == ["A1: Plastic packet"]
    assert items == before


@pytest.mark.parametrize("gate", [
    {"foreign_object_check": {"passed": False}},
    {"foreign_object_check": {"passed": True, "detections": [{"label": "paper"}]}},
    {"component_presence_check": {"passed": False}},
])
def test_public_and_confirmation_paths_cannot_restore_blocked_labels(service_case, gate):
    service = service_case
    service.repository.state.update(gate, status="awaiting_confirmation")
    source = deepcopy(service.repository.state)
    public = service.public_state(source)
    assert blocks_jewellery_labels(source)
    assert public["classification"]["items"] == []
    assert source == service.repository.state
    with pytest.raises(RuntimeError):
        service.confirm_and_start("candidate", "Bangle", True, save_only=True)
