from pathlib import Path
from threading import RLock
import weakref

import pytest

from backend.app.core.config import SettingsStore, TESTBED_MODEL_PATHS
from backend.app.services.capture_service import CaptureService


def service_for_settings(path):
    service = CaptureService.__new__(CaptureService)
    service.settings_store = SettingsStore(path)
    service._testbed_lock = RLock()
    service._testbed_segmenter = None
    return service


def test_apply_switches_models_releases_previous_and_restores_on_startup(tmp_path, monkeypatch):
    previous = None
    loaded = []

    class Segmenter:
        def __init__(self, path, *, inset_px, outer_inset_ratio):
            assert previous is None or previous() is None
            self.model_path = Path(path)
            loaded.append((self.model_path, inset_px, outer_inset_ratio))

    monkeypatch.setattr("backend.app.services.capture_service.TestbedSegmenter", Segmenter)
    service = service_for_settings(tmp_path / "settings.json")
    for model in ("embsys_wm", "essae_jx500", "embsys_wm"):
        saved = service.apply_scale(model)
        assert saved["scale"]["model"] == model
        assert service._get_testbed_segmenter() is service._testbed_segmenter
        previous = weakref.ref(service._testbed_segmenter)
    assert loaded == [(TESTBED_MODEL_PATHS["embsys_wm"], 2, 0.0),
                      (TESTBED_MODEL_PATHS["essae_jx500"], 12, 0.035),
                      (TESTBED_MODEL_PATHS["embsys_wm"], 2, 0.0)]
    service._testbed_segmenter = None
    restarted = service_for_settings(tmp_path / "settings.json")
    assert restarted._get_testbed_segmenter().model_path == TESTBED_MODEL_PATHS["embsys_wm"]


def test_load_failure_does_not_save_new_scale_selection(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("missing model")

    monkeypatch.setattr("backend.app.services.capture_service.TestbedSegmenter", fail)
    service = service_for_settings(tmp_path / "settings.json")
    with pytest.raises(FileNotFoundError, match="missing model"):
        service.apply_scale("essae_jx500")
    assert service.settings_store.get()["scale"]["model"] == "embsys_wm"
    assert service._testbed_segmenter is None
