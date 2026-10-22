from datetime import datetime
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.core.config import SettingsStore
from backend.app.domain.schemas import CameraRotationSettings
from backend.app.hardware.camera import CameraService, TemporalFrameAverage


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_rotation_and_areas_survive_restart_without_changing_other_settings(tmp_path, platform):
    path = tmp_path / "settings.json"
    store = SettingsStore(path, platform=platform)
    roi = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
    original = store.update({"rois": {"apriltag": roi}})
    saved = store.update_camera_rotation(90)
    restarted = SettingsStore(path, platform=platform).get()
    assert restarted == saved
    assert saved["camera"]["rotation"] == 90
    assert saved["camera"]["width"] == 1920
    assert saved["camera"]["height"] == 1080
    assert saved["rois"]["apriltag"] == pytest.approx(
        {"x": 0.6, "y": 0.4, "width": 0.3, "height": 0.4}
    )
    assert saved["rois"]["processing"] is None
    assert saved["camera"] | {"rotation": 270} == original["camera"]
    # Applying the same option twice must not turn either the image or its areas.
    assert store.update_camera_rotation(90) == saved
    restored = store.update_camera_rotation(-90)
    assert SettingsStore(path, platform=platform).get() == restored
    assert restored["camera"]["rotation"] == 270
    assert restored["rois"]["apriltag"] == pytest.approx(roi)


@pytest.mark.parametrize("rotation", [0, 180, 270, 45, "right", None])
def test_rotation_request_rejects_unsupported_values(rotation):
    with pytest.raises(ValidationError):
        CameraRotationSettings(rotation=rotation)


def test_apply_turns_cached_frame_immediately_and_discards_old_preview():
    settings = {"camera": {"rotation": 270}}
    camera = CameraService(lambda: settings)
    camera._active_config = dict(settings["camera"])
    sensor = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    camera._frame = cv2.rotate(sensor, cv2.ROTATE_90_COUNTERCLOCKWISE)
    camera._frame_time = datetime.now().astimezone()
    camera._capture_filter_ready = True
    camera._sequence = 5
    camera._jpeg_sequence = 4
    camera._jpeg = b"old preview"
    settings["camera"]["rotation"] = 90
    camera.set_rotation(90)
    frame, metadata = camera.snapshot_frame()
    np.testing.assert_array_equal(frame, cv2.rotate(sensor, cv2.ROTATE_90_CLOCKWISE))
    assert metadata["rotation"] == 90
    assert metadata["resolution"] == [2, 3]
    assert camera._jpeg is None
    assert camera._jpeg_sequence > 5
    assert camera._preview_pending[2] > camera._jpeg_sequence
    assert not camera._reconnect.is_set()
    assert camera._capture_filter_ready


def test_rotation_during_live_average_does_not_mix_orientations_or_reopen():
    settings = {"camera": {"rotation": 270}}
    camera = CameraService(lambda: settings)
    camera._active_config = dict(settings["camera"])
    camera._temporal_average = TemporalFrameAverage(3)
    sensor = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    outputs = []

    def read():
        if camera._frame is not None:
            outputs.append(camera._frame.copy())
        if len(outputs) == 1:
            settings["camera"]["rotation"] = 90
            camera.set_rotation(90)
        elif len(outputs) == 3:
            camera._stop.set()
        return True, sensor.copy()

    cap = Mock()
    cap.read.side_effect = read
    with patch.object(camera, "_open", return_value=cap) as open_camera:
        camera._run()
    open_camera.assert_called_once()
    expected = cv2.rotate(sensor, cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(outputs[0], cv2.rotate(sensor, cv2.ROTATE_90_COUNTERCLOCKWISE))
    for frame in outputs[1:]:
        np.testing.assert_array_equal(frame, expected)
    assert len(camera._temporal_average._frames) == 3


def test_encoder_cannot_publish_old_orientation_after_apply():
    camera = CameraService(lambda: {"camera": {"rotation": 90}})
    camera._active_config = {"rotation": 270, "preview_quality": 100}
    sensor = np.zeros((24, 40, 3), dtype=np.uint8)
    sensor[:12] = 220
    camera._frame = cv2.rotate(sensor, cv2.ROTATE_90_COUNTERCLOCKWISE)
    camera._frame_time = datetime.now().astimezone()
    camera._sequence = 1
    camera._preview_pending = (camera._frame, camera._frame_time, 1)
    encode = cv2.imencode
    calls = []

    def apply_while_encoding(extension, frame, options):
        calls.append(frame.copy())
        if len(calls) == 1:
            camera.set_rotation(90)
        else:
            assert camera._jpeg is None
            camera._stop.set()
        return encode(extension, frame, options)

    with patch("backend.app.hardware.camera.cv2.imencode", side_effect=apply_while_encoding):
        camera._encode_preview()
    assert len(calls) == 2
    decoded = cv2.imdecode(np.frombuffer(camera._jpeg, np.uint8), cv2.IMREAD_COLOR)
    expected = cv2.rotate(sensor, cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_allclose(decoded, expected, atol=2)
