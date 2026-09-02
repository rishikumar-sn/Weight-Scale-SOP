from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analysis.vision import (
    build_jewellery_mask,
    crop,
    detect_apriltag,
    marker_ignore_mask,
    roi_to_pixels,
    translate_roi_to_crop,
)


class CaptureService:
    def __init__(self, camera, scale, settings_store, repository) -> None:
        self.camera = camera
        self.scale = scale
        self.settings_store = settings_store
        self.repository = repository

    @staticmethod
    def _save(path: Path, image: np.ndarray) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save image: {path}")
        return str(path)

    @staticmethod
    def _evidence_image(
        image: np.ndarray,
        weight: float | None,
        captured_at: datetime,
        caption: str | None = None,
    ) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        weight_text = f"{weight:.2f} g" if weight is not None else "Unavailable"
        timestamp_text = captured_at.strftime("%d-%m-%Y  %I:%M:%S %p")
        scale = max(0.8, min(width, height) / 900.0)
        margin = max(18, int(26 * scale))
        panel_height = max(112, int(150 * scale))
        panel_left = margin
        panel_top = height - margin - panel_height
        panel_right = width - margin
        panel_bottom = height - margin

        # A single high-contrast panel keeps the evidence visible while making
        # its two most important facts immediately readable.
        overlay = result.copy()
        cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (10, 16, 25), -1)
        cv2.addWeighted(overlay, 0.93, result, 0.07, 0, result)
        accent_width = max(5, int(7 * scale))
        cv2.rectangle(result, (panel_left, panel_top), (panel_left + accent_width, panel_bottom), (45, 154, 220), -1)

        content_left = panel_left + max(18, int(26 * scale))
        label_y = panel_top + max(25, int(33 * scale))
        value_y = panel_bottom - max(18, int(23 * scale))
        divider_x = panel_left + int((panel_right - panel_left) * 0.47)
        label_scale = max(0.46, 0.52 * scale)
        weight_scale = max(1.05, 1.65 * scale)
        timestamp_scale = max(0.58, 0.76 * scale)
        label_thickness = max(1, int(2 * scale))
        value_thickness = max(2, int(3 * scale))
        gold = (83, 199, 244)

        cv2.putText(
            result, (caption or "Captured weight").upper(), (content_left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, label_scale, gold, label_thickness, cv2.LINE_AA,
        )
        cv2.putText(
            result, weight_text, (content_left, value_y), cv2.FONT_HERSHEY_SIMPLEX,
            weight_scale, (255, 255, 255), value_thickness, cv2.LINE_AA,
        )
        cv2.line(
            result, (divider_x, panel_top + int(22 * scale)),
            (divider_x, panel_bottom - int(22 * scale)), (100, 108, 118),
            max(1, int(2 * scale)), cv2.LINE_AA,
        )
        timestamp_x = divider_x + max(18, int(25 * scale))
        cv2.putText(
            result, "CAPTURED DATE & TIME", (timestamp_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, label_scale, gold, label_thickness, cv2.LINE_AA,
        )
        cv2.putText(
            result, timestamp_text, (timestamp_x, value_y), cv2.FONT_HERSHEY_SIMPLEX,
            timestamp_scale, (255, 255, 255), value_thickness, cv2.LINE_AA,
        )
        return result

    def capture(self) -> dict[str, Any]:
        frame, camera_snapshot = self.camera.snapshot_frame()
        scale_snapshot = self.scale.snapshot()
        weight = scale_snapshot.get("weight_g")
        reading_age_ms = scale_snapshot.get("age_ms")
        if (
            weight is None
            or reading_age_ms is None
            or float(reading_age_ms) > 3000
            or not scale_snapshot.get("connected")
        ):
            raise RuntimeError(
                "A current scale reading is not available. Check the USB scale and try again."
            )
        now = datetime.now().astimezone()
        capture_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_dir = self.repository.session_dir(capture_id)
        source_dir = session_dir / "capture"
        source_dir.mkdir(parents=True, exist_ok=True)
        settings = self.settings_store.get()
        height, width = frame.shape[:2]
        processing_roi = roi_to_pixels(settings["rois"].get("processing"), width, height)
        apriltag_roi = roi_to_pixels(settings["rois"].get("apriltag"), width, height)
        processing = crop(frame, processing_roi)
        working_tag_roi = translate_roi_to_crop(
            apriltag_roi, processing_roi, processing.shape[1], processing.shape[0]
        )
        calibration: dict[str, Any]
        try:
            tag = settings["apriltag"]
            detected = detect_apriltag(
                frame,
                apriltag_roi,
                int(tag["id"]),
                float(tag["width_mm"]),
                float(tag["height_mm"]),
            )
            calibration = {"available": True, **detected}
        except Exception as exc:  # noqa: BLE001
            calibration = {"available": False, "message": str(exc)}
        ignore_mask = marker_ignore_mask(processing.shape, working_tag_roi)
        clean_processing = processing.copy()
        clean_processing[ignore_mask > 0] = 255
        prepared, jewellery_mask = build_jewellery_mask(clean_processing, ignore_mask)
        evidence = self._evidence_image(frame, weight, now)
        paths = {
            "original": self._save(source_dir / "original.png", frame),
            "evidence": self._save(source_dir / "evidence.jpg", evidence),
            "working": self._save(source_dir / "working.png", clean_processing),
            "prepared": self._save(source_dir / "prepared.png", prepared),
            "mask": self._save(source_dir / "mask.png", jewellery_mask * 255),
        }
        state: dict[str, Any] = {
            "id": capture_id,
            "status": "captured",
            "captured_at": now.isoformat(timespec="milliseconds"),
            "updated_at": now.isoformat(timespec="milliseconds"),
            "weight_g": weight,
            "weight": scale_snapshot,
            "camera": camera_snapshot,
            "settings_snapshot": settings,
            "rois": {
                "processing": processing_roi,
                "apriltag": apriltag_roi,
                "working_apriltag": working_tag_roi,
            },
            "calibration": calibration,
            "paths": paths,
            "classification": {},
            "route": {},
            "result": {},
            "job": {
                "status": "idle",
                "stage": "",
                "message": "",
                "percent": 0,
                "seconds_remaining": 0,
                "error": None,
            },
        }
        self.repository.save(state)
        return state

    def capture_tare(self, mode: str) -> dict[str, Any]:
        """Save a packet tare image and its live scale reading without analysis."""
        if mode not in {"pledge", "release"}:
            raise ValueError("Tare mode must be pledge or release")

        frame, camera_snapshot = self.camera.snapshot_frame()
        scale_snapshot = self.scale.snapshot()
        weight = scale_snapshot.get("weight_g")
        reading_age_ms = scale_snapshot.get("age_ms")
        if (
            weight is None
            or reading_age_ms is None
            or float(reading_age_ms) > 3000
            or not scale_snapshot.get("connected")
        ):
            raise RuntimeError(
                "A current scale reading is not available. Check the USB scale and try again."
            )

        now = datetime.now().astimezone()
        capture_id = f"tare_{mode}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_dir = self.repository.session_dir(capture_id) / "capture"
        evidence = self._evidence_image(
            frame,
            float(weight),
            now,
            caption=f"{mode.title()} Tare Weight",
        )
        paths = {
            "original": self._save(source_dir / "original.png", frame),
            "evidence": self._save(source_dir / "evidence.jpg", evidence),
        }
        state: dict[str, Any] = {
            "id": capture_id,
            "capture_type": "tare",
            "tare_mode": mode,
            "status": "complete",
            "captured_at": now.isoformat(timespec="milliseconds"),
            "updated_at": now.isoformat(timespec="milliseconds"),
            "weight_g": float(weight),
            "weight": scale_snapshot,
            "camera": camera_snapshot,
            "paths": paths,
            "classification": {},
            "route": {},
            "result": {"tare_mode": mode},
            "job": {
                "status": "complete",
                "stage": "Complete",
                "message": f"{mode.title()} tare weight captured",
                "percent": 100,
                "seconds_remaining": 0,
                "error": None,
            },
        }
        self.repository.save(state)
        return state
