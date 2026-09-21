from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Iterator

import cv2
import numpy as np

from .exposure import ExposureController


logger = logging.getLogger(__name__)


class CameraReconnectRequested(RuntimeError):
    """Ask the camera owner thread to release and reopen the device."""


class FlickerStabilizer:
    """Remove residual rolling bands and frame-to-frame AC light variation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("software_anti_flicker", False))
        self.row_normalize = bool(config.get("anti_flicker_row_normalize", False))
        self.alpha = max(0.01, min(0.5, float(config.get("anti_flicker_temporal_alpha", 0.08))))
        self.gain_limit = max(1.01, min(2.0, float(config.get("anti_flicker_gain_limit", 1.35))))
        self._target_luma: float | None = None
        self.last_gain = 1.0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame.ndim != 3 or frame.shape[2] != 3:
            return frame
        corrected = self._normalize_horizontal_bands(frame) if self.row_normalize else frame
        return self._normalize_frame_luma(corrected)

    def _normalize_frame_luma(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > 360:
            sample_height = max(1, int(round(gray.shape[0] * 360 / gray.shape[1])))
            gray = cv2.resize(gray, (360, sample_height), interpolation=cv2.INTER_AREA)
        current = float(np.percentile(gray, 60))
        if current <= 1.0:
            return frame
        if self._target_luma is None:
            self._target_luma = current
            return frame
        self._target_luma = (1.0 - self.alpha) * self._target_luma + self.alpha * current
        gain = float(np.clip(self._target_luma / current, 1.0 / self.gain_limit, self.gain_limit))
        self.last_gain = gain
        if abs(gain - 1.0) < 0.01:
            return frame
        return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    def _normalize_horizontal_bands(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height < 80 or width < 80:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample_width = min(480, width)
        if sample_width < width:
            sample_height = max(40, int(round(height * sample_width / width)))
            gray = cv2.resize(gray, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
        row_luma = np.median(gray.astype(np.float32), axis=1)
        kernel = max(9, ((gray.shape[0] // 12) * 2) + 1)
        smooth = cv2.GaussianBlur(row_luma.reshape(-1, 1), (1, kernel), 0).reshape(-1)
        target = float(np.median(smooth))
        if target <= 1.0:
            return frame
        gains = np.clip(target / np.maximum(smooth, 1.0), 1.0 / self.gain_limit, self.gain_limit)
        if gains.shape[0] != height:
            gains = cv2.resize(gains.reshape(-1, 1), (1, height), interpolation=cv2.INTER_LINEAR).reshape(height)
        if float(np.max(np.abs(gains - 1.0))) < 0.015:
            return frame
        return np.clip(frame.astype(np.float32) * gains.reshape(height, 1, 1), 0, 255).astype(np.uint8)


class TemporalFrameAverage:
    """Average recent frames to cancel moving LED/rolling-shutter bands."""

    def __init__(self, frame_count: int) -> None:
        self.frame_count = max(1, min(90, int(frame_count)))
        self._frames: deque[np.ndarray] = deque()
        self._sum: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.frame_count <= 1:
            return frame
        if self._sum is None or self._sum.shape != frame.shape:
            self._frames.clear()
            self._sum = np.zeros(frame.shape, dtype=np.float32)
        if len(self._frames) == self.frame_count:
            oldest = self._frames.popleft()
            np.subtract(self._sum, oldest, out=self._sum, casting="unsafe")
        stored = frame.copy()
        self._frames.append(stored)
        np.add(self._sum, stored, out=self._sum, casting="unsafe")
        return cv2.convertScaleAbs(self._sum, alpha=1.0 / len(self._frames))


class CameraService:
    def __init__(self, settings_provider) -> None:
        self._settings_provider = settings_provider
        self._lock = threading.RLock()
        self._preview_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._thread: threading.Thread | None = None
        self._encoder_thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._jpeg_time: datetime | None = None
        self._jpeg_sequence = 0
        self._preview_pending: tuple[np.ndarray, datetime, int] | None = None
        self._frame_time: datetime | None = None
        self._sequence = 0
        self._fps_window_started = time.monotonic()
        self._fps_window_frames = 0
        self._processed_fps = 0.0
        self._stream_fps = 0.0
        self._stream_window_started = time.monotonic()
        self._stream_window_frames = 0
        self._status = "Camera is starting"
        self._error = ""
        self._actual_resolution = (0, 0)
        self._preview_resolution = (0, 0)
        self._exposure: dict[str, Any] = {}
        self._flicker: FlickerStabilizer | None = None
        self._temporal_average: TemporalFrameAverage | None = None
        self._active_config: dict[str, Any] = {}

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._reconnect.clear()
            self._preview_pending = None
            self._jpeg = None
            self._jpeg_time = None
            self._stream_fps = 0.0
            self._stream_window_started = time.monotonic()
            self._stream_window_frames = 0
            self._encoder_thread = threading.Thread(target=self._encode_preview, name="camera-preview-encoder", daemon=True)
            self._thread = threading.Thread(target=self._run, name="camera-service", daemon=True)
            self._encoder_thread.start()
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._preview_ready:
            thread = self._thread
            encoder_thread = self._encoder_thread
            self._preview_ready.notify_all()

        # VideoCapture is owned by the camera thread. Releasing it here while
        # that thread is inside cap.read() can crash in the native camera
        # backend, especially with DirectShow on Windows.
        if thread and thread.is_alive():
            thread.join(timeout=5)
        if encoder_thread and encoder_thread.is_alive():
            encoder_thread.join(timeout=5)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
            if self._encoder_thread is encoder_thread and (encoder_thread is None or not encoder_thread.is_alive()):
                self._encoder_thread = None

    def request_reconnect(self) -> dict[str, Any]:
        """Invalidate old frames and ask the camera thread to reopen hardware."""
        self._reconnect.set()
        with self._preview_ready:
            self._status = "Camera reconnecting"
            self._error = ""
            self._frame = None
            self._frame_time = None
            self._jpeg = None
            self._jpeg_time = None
            self._preview_pending = None
            self._processed_fps = 0.0
            self._stream_fps = 0.0
            self._preview_ready.notify_all()
        return self.status()

    @staticmethod
    def _rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
        rotation %= 360
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def _preview_frame(frame: np.ndarray, max_dimension: int) -> np.ndarray:
        longest = max(frame.shape[:2])
        if max_dimension <= 0 or longest <= max_dimension:
            return frame
        scale = max_dimension / longest
        width = max(1, round(frame.shape[1] * scale))
        height = max(1, round(frame.shape[0] * scale))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _encode_preview(self) -> None:
        while not self._stop.is_set():
            with self._preview_ready:
                while self._preview_pending is None and not self._stop.is_set():
                    self._preview_ready.wait(timeout=0.2)
                if self._stop.is_set():
                    return
                frame, captured_at, sequence = self._preview_pending
                self._preview_pending = None
            try:
                preview = self._preview_frame(
                    frame, int(self._active_config.get("preview_max_dimension", 960))
                )
                ok, encoded = cv2.imencode(
                    ".jpg", preview,
                    [cv2.IMWRITE_JPEG_QUALITY, int(self._active_config["preview_quality"])],
                )
                if not ok:
                    continue
                with self._lock:
                    if sequence <= self._jpeg_sequence:
                        continue
                    self._jpeg = encoded.tobytes()
                    self._jpeg_time = captured_at
                    self._jpeg_sequence = sequence
                    self._preview_resolution = (preview.shape[1], preview.shape[0])
                    self._preview_ready.notify_all()
                    self._stream_window_frames += 1
                    elapsed = time.monotonic() - self._stream_window_started
                    if elapsed >= 1.0:
                        self._stream_fps = round(self._stream_window_frames / elapsed, 1)
                        self._stream_window_started = time.monotonic()
                        self._stream_window_frames = 0
            except Exception:
                logger.exception("Could not encode camera preview frame")

    def _open(self) -> cv2.VideoCapture:
        config = self._settings_provider()["camera"]
        self._active_config = dict(config)
        index = int(config["index"])
        # CAP_ANY commonly selects DirectShow again on Windows, so it is not a
        # real fallback when a long-running DirectShow session has failed.
        # Try the two independent Windows capture stacks explicitly.
        backends = (
            (("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF))
            if os.name == "nt"
            else (("V4L2", cv2.CAP_V4L2), ("automatic", cv2.CAP_ANY))
        )
        cap: cv2.VideoCapture | None = None
        failures: list[str] = []
        for backend_name, backend in backends:
            candidate = cv2.VideoCapture(index, backend)
            if candidate.isOpened():
                cap = candidate
                self._exposure = {"backend": backend_name}
                break
            candidate.release()
            failures.append(backend_name)
        if cap is None:
            raise RuntimeError(
                f"Could not open camera index {index} using {' or '.join(failures)}"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))
        cap.set(cv2.CAP_PROP_FPS, int(config["fps"]))
        # DirectShow renegotiates the format when resolution changes. Select
        # MJPEG last or the BRIO falls back to a roughly 5 FPS Full-HD mode.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        exposure = ExposureController().apply(
            cap,
            index,
            int(config["power_line_hz"]),
            int(config["shutter_denominator"]),
            bool(config.get("driver_managed_exposure", os.name == "nt")),
        )
        self._exposure.update(exposure)
        self._flicker = FlickerStabilizer(config)
        self._temporal_average = TemporalFrameAverage(int(config.get("temporal_average_frames", 9)))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        self._exposure["stream"] = {
            "codec": codec,
            "reported_fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 2),
        }
        for _ in range(8):
            cap.grab()
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            cap: cv2.VideoCapture | None = None
            try:
                cap = self._open()
                with self._preview_ready:
                    # A successful open also satisfies a reconnect request made
                    # while the device was unavailable or still opening.
                    self._reconnect.clear()
                    self._capture = cap
                    self._status = "Camera connected"
                    self._error = ""
                    self._preview_pending = None
                    self._jpeg = None
                    self._jpeg_time = None
                    self._stream_fps = 0.0
                    self._stream_window_started = time.monotonic()
                    self._stream_window_frames = 0
                    self._fps_window_started = time.monotonic()
                    self._fps_window_frames = 0
                    self._processed_fps = 0.0
                while not self._stop.is_set():
                    if self._reconnect.is_set():
                        raise CameraReconnectRequested("Camera reconnect requested")
                    ok, frame = cap.read()
                    if self._reconnect.is_set():
                        raise CameraReconnectRequested("Camera reconnect requested")
                    if not ok or frame is None:
                        raise RuntimeError("Camera stopped returning frames")
                    if self._flicker is not None:
                        frame = self._flicker.apply(frame)
                    rotation = int(self._active_config["rotation"])
                    frame = self._rotate(frame, rotation)
                    if self._temporal_average is not None:
                        frame = self._temporal_average.apply(frame)
                    now = datetime.now().astimezone()
                    with self._preview_ready:
                        self._frame = frame
                        self._frame_time = now
                        self._sequence += 1
                        self._actual_resolution = (frame.shape[1], frame.shape[0])
                        self._preview_pending = (frame, now, self._sequence)
                        self._preview_ready.notify()
                        self._fps_window_frames += 1
                        elapsed = time.monotonic() - self._fps_window_started
                        if elapsed >= 1.0:
                            self._processed_fps = round(self._fps_window_frames / elapsed, 1)
                            self._fps_window_started = time.monotonic()
                            self._fps_window_frames = 0
            except CameraReconnectRequested:
                self._reconnect.clear()
                logger.info("Reopening camera after reconnect request")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Camera stream failed; retrying in 1 second: %s", exc)
                with self._lock:
                    if not self._stop.is_set():
                        self._status = "Camera reconnecting"
                        self._error = str(exc)
                self._stop.wait(1)
            finally:
                if cap is not None:
                    cap.release()
                with self._lock:
                    if self._capture is cap:
                        self._capture = None
        with self._lock:
            self._status = "Camera stopped"

    def snapshot_frame(self) -> tuple[np.ndarray, dict[str, Any]]:
        with self._lock:
            if self._frame is None or self._frame_time is None:
                raise RuntimeError("Camera frame is not ready")
            frame_age_ms = max(
                0,
                int(
                    (datetime.now().astimezone() - self._frame_time).total_seconds()
                    * 1000
                ),
            )
            if frame_age_ms >= 2000:
                raise RuntimeError("A current camera image is not available. Please try again.")
            return self._frame.copy(), {
                "sequence": self._sequence,
                "processed_fps": self._processed_fps,
                "captured_at": self._frame_time.isoformat(timespec="milliseconds"),
                "resolution": list(self._actual_resolution),
                "preview_resolution": list(self._preview_resolution),
                "rotation": int(self._settings_provider()["camera"]["rotation"]),
                "exposure": dict(self._exposure),
                "anti_flicker": {
                    "software_enabled": bool(self._flicker and self._flicker.enabled),
                    "row_normalize": bool(self._flicker and self._flicker.row_normalize),
                    "temporal_gain": round(self._flicker.last_gain, 4) if self._flicker else 1.0,
                    "averaged_frames": self._temporal_average.frame_count if self._temporal_average else 1,
                    "temporal_method": "exact_rolling_average",
                },
            }

    def jpeg(self) -> bytes | None:
        with self._lock:
            if self._jpeg is None or self._jpeg_time is None:
                return None
            age_ms = (datetime.now().astimezone() - self._jpeg_time).total_seconds() * 1000
            return self._jpeg if age_ms < 2000 else None

    def mjpeg(self) -> Iterator[bytes]:
        last_sequence = -1
        while not self._stop.is_set():
            with self._preview_ready:
                self._preview_ready.wait_for(
                    lambda: self._stop.is_set()
                    or (
                        self._jpeg_sequence != last_sequence
                        and self._jpeg is not None
                        and self._jpeg_time is not None
                    ),
                    timeout=0.5,
                )
                if self._stop.is_set():
                    return
                sequence = self._jpeg_sequence
                payload = self._jpeg
                jpeg_time = self._jpeg_time
            if sequence == last_sequence or payload is None or jpeg_time is None:
                continue
            if (datetime.now().astimezone() - jpeg_time).total_seconds() < 2:
                last_sequence = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            else:
                # Remember stale frames so the wait condition sleeps until a
                # newly captured frame is published after reconnection.
                last_sequence = sequence

    def status(self) -> dict[str, Any]:
        with self._lock:
            age_ms = None
            if self._frame_time:
                age_ms = max(0, int((datetime.now().astimezone() - self._frame_time).total_seconds() * 1000))
            connected = self._frame is not None and age_ms is not None and age_ms < 2000
            status = self._status
            if not connected and status == "Camera connected":
                status = "Camera reconnecting"
            return {
                "connected": connected,
                "status": status,
                "error": self._error,
                "resolution": list(self._actual_resolution),
                "preview_resolution": list(self._preview_resolution),
                "rotation": int(self._settings_provider()["camera"]["rotation"]),
                "frame_age_ms": age_ms,
                "sequence": self._sequence,
                "processed_fps": self._processed_fps if connected else 0.0,
                "stream_fps": self._stream_fps if connected else 0.0,
                "exposure": dict(self._exposure),
                "anti_flicker": {
                    "software_enabled": bool(self._flicker and self._flicker.enabled),
                    "row_normalize": bool(self._flicker and self._flicker.row_normalize),
                    "temporal_gain": round(self._flicker.last_gain, 4) if self._flicker else 1.0,
                    "averaged_frames": self._temporal_average.frame_count if self._temporal_average else 1,
                    "temporal_method": "exact_rolling_average",
                },
            }
