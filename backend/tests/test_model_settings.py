from copy import deepcopy

import pytest
from pydantic import ValidationError

from backend.app.core.config import SettingsStore
from backend.app.domain.schemas import ModelSettings
from backend.app.services.analysis_service import AnalysisService
from Classification.jewelry_classifier import PredictionResult
from test_component_presence import service_case, evidence


def test_settings_defaults_persist_and_apply_without_restarting(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    assert all(store.get()["models"].values())
    captured = deepcopy(store.get())
    store.update({"models": {"real_print": False, "thread_only": False}})
    assert store.get()["models"]["real_print"] is False
    assert SettingsStore(store.path).get()["models"]["thread_only"] is False
    assert captured["models"]["real_print"] is True
    assert store.get()["models"]["foreign_objects"] is True


def test_settings_reject_unknown_model():
    with pytest.raises(ValidationError):
        ModelSettings(unknown_model=False)


@pytest.mark.parametrize("disabled", [
    {"foreign_objects": False},
    {"component_presence": False},
    {"thread_only": False},
    {"real_print": False},
    {"semantic_validation": False},
    {"bangle_placement": False},
    {"rim_filter": False},
    ModelSettings().model_dump() | {key: False for key in ModelSettings.model_fields},
])
def test_disabled_checks_never_invoke_model_and_allow_capture(service_case, monkeypatch, disabled):
    service = service_case
    service.repository.state["settings_snapshot"] = {"models": disabled}
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled check was invoked")
    monkeypatch.setattr(service, "_check_foreign_objects",
                        forbidden if disabled.get("foreign_objects") is False else lambda *_: evidence(.95, .95))
    if disabled.get("component_presence") is False:
        monkeypatch.setattr("backend.app.services.analysis_service.component_presence_check", forbidden)
    if disabled.get("thread_only") is False:
        monkeypatch.setattr(service, "_get_jewel_thread", forbidden)
    if disabled.get("real_print") is False:
        monkeypatch.setattr(service, "_check_real_print", forbidden)
    if disabled.get("rim_filter") is False:
        monkeypatch.setattr(service, "_exclude_verified_rim_strips", forbidden)
    if disabled.get("bangle_placement") is False:
        monkeypatch.setattr(service, "_check_bangle_placement", forbidden)
    class Classifier:
        check_non_jewelry = forbidden if disabled.get("semantic_validation") is False else lambda *_: (False, "clear")
        def classify_image(self, image, image_path):
            return PredictionResult(label="Finger Ring", confidence=.9, scores=[], image_path=image_path,
                                    original_image=image, cropped_image=image)
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())
    state = service.classify("candidate")
    assert state["status"] == "awaiting_confirmation"
    assert state["classification"]["count"] == 2
    for key, field in (("foreign_objects", "foreign_object_check"), ("thread_only", "thread_only_check"),
                       ("real_print", "real_print_check")):
        if disabled.get(key) is False:
            assert state[field]["skipped"] is True


def test_disabled_models_are_not_preloaded(monkeypatch):
    service = AnalysisService(None)
    try:
        monkeypatch.setattr(service, "_get_classifier", lambda: None)
        monkeypatch.setattr(service, "_get_beads", lambda: None)
        def forbidden():
            pytest.fail("Disabled model preloaded")
        for name in ("_get_foreign_objects", "_get_jewel_thread", "_get_real_print"):
            monkeypatch.setattr(service, name, forbidden)
        service.preload_models({"models": {"foreign_objects": False, "thread_only": False, "real_print": False}})
    finally:
        service.shutdown()


def test_disabled_thread_segmentation_continues_stone_analysis(service_case, monkeypatch):
    from backend.app.services import analysis_service as module
    from backend.app.domain.workflow import BEAD_LABELS
    state = service_case.repository.get("candidate")
    state["settings_snapshot"] = {"models": {"thread_segmentation": False}}
    state["classification"]["confirmed_label"] = next(iter(BEAD_LABELS))
    state["paths"]["prepared"] = state["paths"]["working"]
    def forbidden():
        pytest.fail("Disabled thread segmentation invoked")
    monkeypatch.setattr(service_case, "_get_thread", forbidden)
    monkeypatch.setattr(module.stone_detection, "build_candidates_from_component_mask", lambda *_a, **_kw: [])
    class ReachedStoneAnalysis(Exception):
        pass
    def analyze(*args, **kwargs):
        assert kwargs["stone_exclusion_mask"] is None
        raise ReachedStoneAnalysis()
    monkeypatch.setattr(module.stone_detection, "analyze_image_bgr", analyze)
    with pytest.raises(ReachedStoneAnalysis):
        service_case._run_stones(state)
