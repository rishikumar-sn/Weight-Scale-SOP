import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import cv2
import numpy as np

from backend.app.hardware.camera import CameraService, TemporalFrameAverage


def test_ninety_frame_average_cancels_alternating_vertical_bands():
    average = TemporalFrameAverage(90)
    band = np.empty((64, 64, 3), dtype=np.uint8)
    for index in range(90):
        band[:, :32] = 40 if index % 2 == 0 else 160
        band[:, 32:] = 160 if index % 2 == 0 else 40
        filtered = average.apply(band)
    assert int(filtered[10, 10, 0]) == 100
    assert int(filtered[10, 50, 0]) == 100


def test_exact_average_keeps_full_window_during_scene_change():
    average = TemporalFrameAverage(90)
    previous = np.full((64, 64, 3), 80, dtype=np.uint8)
    for _ in range(90):
        average.apply(previous)

    packet = previous.copy()
    packet[8:56, 8:56] = 220
    filtered = average.apply(packet)
    assert int(filtered[32, 32, 0]) == 82
    for _ in range(89):
        filtered = average.apply(packet)
    assert int(filtered[32, 32, 0]) == 220


def test_preview_resize_keeps_capture_frame_full_resolution():
    captured = np.zeros((1920, 1080, 3), dtype=np.uint8)
    preview = CameraService._preview_frame(captured, 960)
    assert preview.shape == (960, 540, 3)
    assert captured.shape == (1920, 1080, 3)


def test_preview_encoder_publishes_latest_frame_without_changing_capture():
    camera = CameraService(lambda: {"camera": {"rotation": 0}})
    camera._active_config = {"preview_max_dimension": 960, "preview_quality": 78}
    captured = np.full((1920, 1080, 3), 75, dtype=np.uint8)
    encoder = threading.Thread(target=camera._encode_preview)
    encoder.start()
    try:
        with camera._preview_ready:
            camera._frame = captured
            camera._preview_pending = (captured, datetime.now().astimezone(), 1)
            camera._preview_ready.notify()
        with camera._preview_ready:
            published = camera._preview_ready.wait_for(
                lambda: camera._jpeg_sequence == 1, timeout=3
            )
        assert published
        preview = cv2.imdecode(np.frombuffer(camera.jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert preview.shape == (960, 540, 3)
        assert camera._frame.shape == (1920, 1080, 3)
    finally:
        camera._stop.set()
        with camera._preview_ready:
            camera._preview_ready.notify_all()
        encoder.join(timeout=3)


def test_mjpeg_stream_waits_for_camera_reconnection():
    camera = CameraService(lambda: {"camera": {"rotation": 0}})
    stream = camera.mjpeg()
    result: list[bytes] = []
    # Simulate more than five seconds passing immediately. The old stream
    # implementation stopped at that point instead of waiting for reconnection.
    with patch("backend.app.hardware.camera.time.monotonic", side_effect=[0.0, 6.0]):
        reader = threading.Thread(target=lambda: result.append(next(stream)))
        reader.start()
        try:
            time.sleep(0.05)
            assert reader.is_alive()

            with camera._preview_ready:
                camera._jpeg = b"reconnected-frame"
                camera._jpeg_time = datetime.now().astimezone()
                camera._jpeg_sequence = 1
                camera._preview_ready.notify_all()

            reader.join(timeout=1)
            assert result == [
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\nreconnected-frame\r\n"
            ]
        finally:
            camera._stop.set()
            with camera._preview_ready:
                camera._preview_ready.notify_all()
            reader.join(timeout=1)


def test_reconnect_request_invalidates_old_camera_frame():
    camera = CameraService(lambda: {"camera": {"rotation": 0}})
    with camera._lock:
        camera._frame = np.zeros((10, 10, 3), dtype=np.uint8)
        camera._frame_time = datetime.now().astimezone()
        camera._jpeg = b"old-frame"
        camera._jpeg_time = datetime.now().astimezone()

    status = camera.request_reconnect()

    assert camera._reconnect.is_set()
    assert status["connected"] is False
    assert status["status"] == "Camera reconnecting"
    assert camera.jpeg() is None


def test_stale_frame_is_reported_as_reconnecting():
    camera = CameraService(lambda: {"camera": {"rotation": 0}})
    with camera._lock:
        camera._status = "Camera connected"
        camera._frame = np.zeros((10, 10, 3), dtype=np.uint8)
        camera._frame_time = datetime.now().astimezone() - timedelta(seconds=3)

    status = camera.status()

    assert status["connected"] is False
    assert status["status"] == "Camera reconnecting"
