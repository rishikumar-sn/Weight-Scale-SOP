from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest

from backend.app.hardware.adaptive import AdaptiveCameraPipeline
from backend.app.hardware.anti_flicker import AntiFlickerPipeline
from backend.app.hardware.camera import CameraService


def hardware():
    return Mock(supported=True, error="", _original_powerline=None)


@pytest.mark.parametrize("mode", ["normal", "driver_50", "driver_60", "average_30", "average_90", "row", "hybrid"])
def test_fixed_methods_do_not_measure_lighting(mode):
    pipeline = AntiFlickerPipeline(hardware(), lambda state: None)
    pipeline.configure(mode, 0, force=True)
    with patch("backend.app.hardware.anti_flicker.prepare_flicker_sample", side_effect=AssertionError("Unexpected lighting check")):
        for index in range(95):
            pipeline.process_frame(np.full((90, 120, 3), 80 + index, np.uint8), index / 30)


def test_adaptive_filter_keeps_its_initial_selection_when_scene_changes():
    updates = []
    pipeline = AntiFlickerPipeline(hardware(), updates.append)
    pipeline.configure("adaptive", 0, force=True)
    frame = np.full((90, 120, 3), 100, np.uint8)
    with patch("backend.app.hardware.anti_flicker.temporal_flicker_score", return_value=.2):
        for index in range(120):
            pipeline.process_frame(frame, index / 30)
    assert pipeline.selection_locked and pipeline.adaptive_averaging
    assert updates[-1]["ready"]
    with patch("backend.app.hardware.anti_flicker.prepare_flicker_sample", side_effect=AssertionError("Unexpected lighting check")):
        for index in range(60):
            pipeline.process_frame(255 - frame, 4 + index / 30)
            assert updates[-1]["ready"]
    pipeline.configure("adaptive", 10, force=True)
    assert not pipeline.selection_locked
    assert not updates[-1]["ready"]


def test_adaptive_shutter_stops_metering_after_initial_setup():
    writes = Mock(side_effect=lambda value: value)
    pipeline = AdaptiveCameraPipeline({"rotation": 0}, None, 100, writes)
    frame = np.full((96, 160, 3), 190, np.uint8)
    for index in range(40):
        pipeline.apply(frame, index / 10)
    assert pipeline.configuration_locked
    writes.reset_mock()
    with patch.object(pipeline, "_band_score", side_effect=AssertionError("Unexpected lighting check")):
        for index in range(40):
            changed = np.full_like(frame, 20 if index % 2 else 240)
            changed[20:60, 30:70] = (20, 80, 180)
            assert pipeline.apply(changed, 4 + index / 10) is not None
            assert pipeline.status()["ready"]
    writes.assert_not_called()


def test_capture_readiness_stays_ready_until_reconnect_but_rejects_stale_frames():
    camera = CameraService(lambda: {"camera": {"rotation": 0}})
    camera._method_pipeline = object()
    camera._method_status = {"ready": True}
    camera._frame = np.zeros((90, 120, 3), np.uint8)
    camera._frame_time = datetime.now().astimezone()
    assert camera.status()["capture_ready"]
    camera._method_status = {"ready": False}
    assert camera.status()["capture_ready"]
    camera.snapshot_frame()
    camera._frame_time -= timedelta(seconds=3)
    with pytest.raises(RuntimeError, match="current camera image"):
        camera.snapshot_frame()
    assert not camera.status()["capture_ready"]
    assert not camera.request_reconnect()["capture_ready"]
    assert not camera._capture_filter_ready
