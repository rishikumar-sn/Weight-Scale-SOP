"""Capture a short camera burst and measure AC-light rolling bands.

Run from the project root:
    python scripts/diagnose_camera_flicker.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def band_score(frame: np.ndarray) -> float:
    """Return robust horizontal-band amplitude as a percentage of image luma."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    # Ignore borders and use row medians so jewellery/details have little influence.
    gray = gray[height // 10 : height - height // 10, width // 10 : width - width // 10]
    row_luma = np.median(gray, axis=1)
    kernel = max(31, (len(row_luma) // 5) | 1)
    smooth = cv2.GaussianBlur(row_luma.reshape(-1, 1), (1, kernel), 0).reshape(-1)
    residual = row_luma - smooth
    amplitude = float(np.percentile(residual, 95) - np.percentile(residual, 5))
    return 100.0 * amplitude / max(1.0, float(np.median(row_luma)))


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/camera-flicker-diagnostic"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.index, args.width, args.height, args.fps)
    try:
        for _ in range(args.warmup):
            if not cap.grab():
                raise RuntimeError("Camera stopped during warm-up")
        frames: list[np.ndarray] = []
        started = time.perf_counter()
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped while capturing the burst")
            frames.append(frame)
        elapsed = time.perf_counter() - started
    finally:
        cap.release()

    counts = [count for count in (1, 2, 3, 5, 7, 9, 12, 15, 20, 30, 45, 60, 90) if count <= len(frames)]
    stack = np.zeros_like(frames[0], dtype=np.float32)
    results = []
    count_set = set(counts)
    for index, frame in enumerate(frames, start=1):
        stack += frame
        if index in count_set:
            averaged = cv2.convertScaleAbs(stack, alpha=1.0 / index)
            score = band_score(averaged)
            filename = f"average_{index:02d}.jpg"
            cv2.imwrite(str(args.output / filename), averaged, [cv2.IMWRITE_JPEG_QUALITY, 92])
            results.append({"frames": index, "band_score_percent": round(score, 3), "image": filename})

    best = min(item["band_score_percent"] for item in results)
    # Prefer the shortest average within 10% of the cleanest result. A longer
    # window adds motion blur and capture latency for little visible benefit.
    target = best * 1.10
    recommended = next(item["frames"] for item in results if item["band_score_percent"] <= target)
    report = {
        "captured_frames": len(frames),
        "measured_fps": round(len(frames) / elapsed, 2),
        "resolution": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        "results": results,
        "best_band_score_percent": best,
        "recommendation_frames": recommended,
        "recommendation_rule": "shortest window within 10% of the lowest measured band score",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
