import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from backend.app.core.config import SettingsStore
from backend.app.hardware.camera import CameraService


def test_legacy_windows_settings_and_shared_calibration_survive_migration(tmp_path):
    path = tmp_path / "app.json"
    path.write_text(json.dumps({
        "camera": {"temporal_average_frames": 90, "rotation": 180},
        "scale": {"port": "COM12"},
        "apriltag": {"width_mm": 12.5},
    }))
    # A Windows installation first opened on Linux must retain its Windows setup.
    linux = SettingsStore(path, platform="linux").get()
    windows = SettingsStore(path, platform="windows").get()
    assert windows["camera"]["temporal_average_frames"] == 90
    assert windows["camera"]["anti_flicker_mode"] == "temporal_average"
    assert windows["camera"]["rotation"] == 180
    assert windows["scale"]["port"] == "COM12"
    assert linux["camera"]["driver_managed_exposure"] is False
    assert linux["camera"]["linux_image_controls"]["brightness"] == 96
    assert linux["scale"]["port"] == ""
    assert linux["apriltag"]["width_mm"] == windows["apriltag"]["width_mm"] == 12.5


def test_saved_methods_survive_restarts_and_switching_platforms(tmp_path):
    path = tmp_path / "app.json"
    windows = SettingsStore(path, platform="windows")
    windows.update({"camera": {"anti_flicker_mode": "average_90"}})
    linux = SettingsStore(path, platform="linux")
    linux.update({"camera": {"anti_flicker_mode": "shutter_50"},
                  "scale": {"port": "/dev/serial/by-id/my-scale"}})
    windows = SettingsStore(path, platform="windows")
    assert windows.get()["camera"]["anti_flicker_mode"] == "average_90"
    assert windows.get()["scale"]["port"] == "COM12"
    windows.update({"rois": {"processing": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.5}}})
    linux = SettingsStore(path, platform="linux")
    assert linux.get()["camera"]["anti_flicker_mode"] == "shutter_50"
    assert linux.get()["scale"]["port"] == "/dev/serial/by-id/my-scale"
    assert linux.get()["rois"]["processing"]["x"] == 0.1


def test_original_windows_ninety_frame_filter_restarts_and_warms_up(tmp_path):
    path = tmp_path / "app.json"
    path.write_text(json.dumps({"camera": {"temporal_average_frames": 90},
                                "scale": {"port": "COM12"}}))
    SettingsStore(path, platform="windows")
    camera = CameraService(SettingsStore(path, platform="windows").get)
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0
    with patch("backend.app.hardware.camera.os", SimpleNamespace(name="nt")), \
         patch("backend.app.hardware.camera.cv2.VideoCapture", return_value=cap), \
         patch("backend.app.hardware.camera.ExposureController.apply", return_value={}) as exposure:
        camera._open()
    assert exposure.call_args.args[4] is True
    assert camera._temporal_average.frame_count == 90
    frame = np.full((90, 120, 3), 100, dtype=np.uint8)
    for _ in range(89):
        camera._frame = camera._temporal_average.apply(frame)
    camera._frame_time = datetime.now().astimezone()
    assert camera._anti_flicker_status()["averaged_frames"] == 89
    with pytest.raises(RuntimeError, match="stabilizing"):
        camera.snapshot_frame()
    camera._frame = camera._temporal_average.apply(frame)
    assert camera.snapshot_frame()[1]["anti_flicker"]["ready"] is True


@pytest.mark.parametrize("platform, backend", [("windows", cv2.CAP_DSHOW), ("linux", cv2.CAP_V4L2)])
def test_restarted_camera_uses_active_profile_and_backend(tmp_path, platform, backend):
    path = tmp_path / "app.json"
    store = SettingsStore(path, platform=platform)
    if platform == "windows":
        store.update({"camera": {"anti_flicker_mode": "average_90"}})
    # Reconstruct both services as a new backend process would.
    store = SettingsStore(path, platform=platform)
    camera = CameraService(store.get)
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0
    hardware = Mock(supported=False, error="Driver control unavailable", _original_powerline=None)
    with patch("backend.app.hardware.camera.os", SimpleNamespace(name="nt" if platform == "windows" else "posix")), \
         patch("backend.app.hardware.camera.cv2.VideoCapture", return_value=cap) as open_capture, \
         patch("backend.app.hardware.camera.ExposureController.apply", return_value={}) as exposure, \
         patch("backend.app.hardware.camera.WindowsPowerLineController", return_value=hardware):
        camera._open()
    open_capture.assert_called_once_with(0, backend)
    if platform == "windows":
        assert camera._method_pipeline.mode == "average_90"
        assert camera._method_pipeline.average.target_frames == 90
        assert camera._anti_flicker_status()["ready"] is False
        camera._frame = np.full((90, 120, 3), 100, dtype=np.uint8)
        camera._frame_time = datetime.now().astimezone()
        with pytest.raises(RuntimeError, match="stabilizing"):
            camera.snapshot_frame()
        for index in range(90):
            camera._method_pipeline.process_frame(camera._frame, index / 30)
        assert camera.snapshot_frame()[1]["anti_flicker"]["ready"] is True
        exposure.assert_not_called()
    else:
        assert camera._temporal_average.frame_count == 1
        assert exposure.call_args.args[4] is False
        assert exposure.call_args.args[5]["brightness"] == 96
