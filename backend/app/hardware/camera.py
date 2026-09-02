from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Iterator

import cv2
import numpy as np

from .exposure import ExposureController


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
        # The installed AC lighting was tested with windows up to 90 frames;
        # that window produced the cleanest image under the actual fixture.
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._frame_time: datetime | None = None
        self._sequence = 0
        self._status = "Camera is starting"
        self._error = ""
        self._actual_resolution = (0, 0)
        self._exposure: dict[str, Any] = {}
        self._flicker: FlickerStabilizer | None = None
        self._temporal_average: TemporalFrameAverage | None = None
        self._active_config: dict[str, Any] = {}

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="camera-service", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread

        # VideoCapture is owned by the camera thread. Releasing it here while
        # that thread is inside cap.read() can crash in the native camera
        # backend, especially with DirectShow on Windows.
        if thread and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

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

    def _open(self) -> cv2.VideoCapture:
        config = self._settings_provider()["camera"]
        self._active_config = dict(config)
        index = int(config["index"])
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))
        cap.set(cv2.CAP_PROP_FPS, int(config["fps"]))
        # DirectShow renegotiates the format when resolution changes. Select
        # MJPEG last or the BRIO falls back to a roughly 5 FPS Full-HD mode.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._exposure = ExposureController().apply(
            cap,
            index,
            int(config["power_line_hz"]),
            int(config["shutter_denominator"]),
            bool(config.get("driver_managed_exposure", os.name == "nt")),
        )
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
                with self._lock:
                    self._capture = cap
                    self._status = "Camera connected"
                    self._error = ""
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("Camera stopped returning frames")
                    if self._flicker is not None:
                        frame = self._flicker.apply(frame)
                    rotation = int(self._active_config["rotation"])
                    frame = self._rotate(frame, rotation)
                    if self._temporal_average is not None:
                        frame = self._temporal_average.apply(frame)
                    quality = int(self._active_config["preview_quality"])
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality],
                    )
                    now = datetime.now().astimezone()
                    with self._lock:
                        self._frame = frame
                        if encoded_ok:
                            self._jpeg = encoded.tobytes()
                        self._frame_time = now
                        self._sequence += 1
                        self._actual_resolution = (frame.shape[1], frame.shape[0])
            except Exception as exc:  # noqa: BLE001
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
                "captured_at": self._frame_time.isoformat(timespec="milliseconds"),
                "resolution": list(self._actual_resolution),
                "rotation": int(self._settings_provider()["camera"]["rotation"]),
                "exposure": dict(self._exposure),
                "anti_flicker": {
                    "software_enabled": bool(self._flicker and self._flicker.enabled),
                    "row_normalize": bool(self._flicker and self._flicker.row_normalize),
                    "temporal_gain": round(self._flicker.last_gain, 4) if self._flicker else 1.0,
                    "averaged_frames": self._temporal_average.frame_count if self._temporal_average else 1,
                },
            }

    def jpeg(self) -> bytes | None:
        with self._lock:
            if self._jpeg is None or self._frame_time is None:
                return None
            age_ms = (datetime.now().astimezone() - self._frame_time).total_seconds() * 1000
            return self._jpeg if age_ms < 2000 else None

    def mjpeg(self) -> Iterator[bytes]:
        last_sequence = -1
        while not self._stop.is_set():
            with self._lock:
                sequence = self._sequence
            if sequence == last_sequence:
                time.sleep(0.015)
                continue
            payload = self.jpeg()
            if payload:
                last_sequence = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"

    def status(self) -> dict[str, Any]:
        with self._lock:
            age_ms = None
            if self._frame_time:
                age_ms = max(0, int((datetime.now().astimezone() - self._frame_time).total_seconds() * 1000))
            return {
                "connected": self._frame is not None and age_ms is not None and age_ms < 2000,
                "status": self._status,
                "error": self._error,
                "resolution": list(self._actual_resolution),
                "rotation": int(self._settings_provider()["camera"]["rotation"]),
                "frame_age_ms": age_ms,
                "sequence": self._sequence,
                "exposure": dict(self._exposure),
                "anti_flicker": {
                    "software_enabled": bool(self._flicker and self._flicker.enabled),
                    "row_normalize": bool(self._flicker and self._flicker.row_normalize),
                    "temporal_gain": round(self._flicker.last_gain, 4) if self._flicker else 1.0,
                    "averaged_frames": self._temporal_average.frame_count if self._temporal_average else 1,
                },
            }
