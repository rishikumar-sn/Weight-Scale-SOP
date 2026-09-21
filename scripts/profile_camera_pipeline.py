"""Measure the live camera stages without starting the backend or scale."""

from __future__ import annotations

import statistics
import time

import cv2

from backend.app.core.config import SettingsStore
from backend.app.hardware.camera import CameraService


def main() -> None:
    service = CameraService(SettingsStore().get)
    capture = service._open()
    times: dict[str, list[float]] = {key: [] for key in ("read", "rotate", "average", "jpeg")}
    try:
        for _ in range(45):
            t = time.perf_counter()
            ok, frame = capture.read()
            times["read"].append(time.perf_counter() - t)
            if not ok or frame is None:
                raise RuntimeError("Camera did not return a frame")
            t = time.perf_counter()
            frame = service._rotate(frame, int(service._active_config["rotation"]))
            times["rotate"].append(time.perf_counter() - t)
            t = time.perf_counter()
            frame = service._temporal_average.apply(frame)
            times["average"].append(time.perf_counter() - t)
            t = time.perf_counter()
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(service._active_config["preview_quality"])])
            times["jpeg"].append(time.perf_counter() - t)
    finally:
        capture.release()
    for stage, values in times.items():
        print(f"{stage}: median {statistics.median(values) * 1000:.1f} ms, mean {statistics.mean(values) * 1000:.1f} ms")
    total = sum(sum(values) for values in times.values())
    print(f"total: {len(times['read']) / total:.1f} fps, resolution {frame.shape[1]}x{frame.shape[0]}")
    for longest_side in (max(frame.shape[:2]), 1280, 960):
        started = time.perf_counter()
        for _ in range(15):
            scale = min(1.0, longest_side / max(frame.shape[:2]))
            preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, int(service._active_config["preview_quality"])])
        print(f"preview {longest_side}px: {(time.perf_counter() - started) * 1000 / 15:.1f} ms/frame")


if __name__ == "__main__":
    main()
