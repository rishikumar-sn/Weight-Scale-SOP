from __future__ import annotations

import json
import uuid
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np

from ..analysis.vision import (
    apriltag_roi_from_detection,
    build_jewellery_mask,
    crop,
    detect_apriltag,
    marker_ignore_mask,
    roi_to_normalized,
    roi_to_pixels,
    translate_roi_to_crop,
)
from ..analysis.testbed_segmentation import TestbedSegmenter
from ..core.config import TESTBED_MODEL_PATHS
from .packet_scan_service import PacketScanService
from .report_service import ReportService


class CaptureService:
    def __init__(self, camera, scale, settings_store, repository, artifact_finalizer=None) -> None:
        self.camera = camera
        self.scale = scale
        self.settings_store = settings_store
        self.repository = repository
        self.artifact_finalizer = artifact_finalizer
        self.packet_scanner = PacketScanService()
        self.reports = ReportService()
        self._testbed_segmenter: TestbedSegmenter | None = None
        self._testbed_lock = RLock()
        self._testbed_model = None
        self._event_lock = RLock()
        self._scale_events = OrderedDict()
        self.scale.on_reading = self.record_scale_reading

    def record_scale_reading(self, reading: dict[str, Any]) -> None:
        # Freeze pixels in the serial reader, before UI polling or model inference.
        try:
            if reading.get("unit") != "g":
                raise RuntimeError("Change the weight scale mode to grams (g), then press the scale button again.")
            frame, camera_status = self.camera.snapshot_frame()
            event = (time.monotonic(), frame, camera_status, reading, None)
        except Exception as exc:
            event = (time.monotonic(), None, None, reading, str(exc))
        with self._event_lock:
            self._scale_events[reading["reading_id"]] = event
            while len(self._scale_events) > 4:
                self._scale_events.popitem(last=False)

    def _capture_inputs(self, reading_id: int | None):
        settings = self.settings_store.get()
        model = settings.get("scale", {}).get("model") if isinstance(settings, dict) else None
        if model != "accet_cg1003l":
            if reading_id is not None:
                raise RuntimeError("Scale selection changed. Capture again with the selected scale.")
            frame, camera_status = self.camera.snapshot_frame()
            return frame, camera_status, self.scale.snapshot()
        if reading_id is None:
            raise RuntimeError("Press the ACCET scale button to capture the image and weight.")
        with self._event_lock:
            event = self._scale_events.pop(reading_id, None)
        if event is None or time.monotonic() - event[0] > 3:
            raise RuntimeError("This scale event expired or was already captured. Press the scale button again.")
        _, frame, camera_status, reading, error = event
        if error:
            raise RuntimeError(error)
        if reading.get("unit") != "g" or reading.get("weight_g") is None:
            raise RuntimeError("Change the weight scale mode to grams (g), then press the scale button again.")
        return frame, camera_status, reading

    def _get_testbed_segmenter(self, model: str | None = None) -> TestbedSegmenter:
        with self._testbed_lock:
            model = model or self.settings_store.get()["scale"].get("model", "embsys_wm")
            path = TESTBED_MODEL_PATHS[model]
            if self._testbed_segmenter is None or self._testbed_segmenter.model_path != path or self._testbed_model != model:
                # Drop the previous ONNX session before loading the other scale.
                self._testbed_segmenter = None
                self._testbed_segmenter = TestbedSegmenter(
                    path,
                    inset_px=12 if model == "essae_jx500" else 2,
                    outer_inset_ratio=0.035 if model in {"essae_jx500", "accet_cg1003l"} else 0.0,
                    exclude_circle=model != "accet_cg1003l",
                )
            self._testbed_model = model
            return self._testbed_segmenter

    def apply_scale(self, model: str) -> dict[str, Any]:
        with self._testbed_lock:
            if model not in TESTBED_MODEL_PATHS:
                raise ValueError("Unsupported scale model")
            self._get_testbed_segmenter(model)
            with self._event_lock:
                self._scale_events.clear()
            return self.settings_store.select_scale(model)

    @staticmethod
    def _save(path: Path, image: np.ndarray) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save image: {path}")
        return str(path)

    def refresh_apriltag_roi(self) -> dict[str, Any]:
        """Detect the configured tag in the current frame and persist its padded ROI."""
        frame, _ = self.camera.snapshot_frame()
        settings = self.settings_store.get()
        tag = settings["apriltag"]
        detected = detect_apriltag(
            frame,
            None,
            int(tag["id"]),
            float(tag["width_mm"]),
            float(tag["height_mm"]),
        )
        pixel_roi = apriltag_roi_from_detection(frame.shape, detected)
        height, width = frame.shape[:2]
        normalized_roi = roi_to_normalized(pixel_roi, width, height)
        saved = self.settings_store.update({"rois": {"apriltag": normalized_roi}})
        return {"settings": saved, "roi": normalized_roi, "detection": detected}

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

    def capture(self, reading_id: int | None = None) -> dict[str, Any]:
        frame, camera_snapshot, scale_snapshot = self._capture_inputs(reading_id)
        settings = self.settings_store.get()
        height, width = frame.shape[:2]
        apriltag_roi = roi_to_pixels(settings["rois"].get("apriltag"), width, height)
        tag = settings["apriltag"]
        try:
            detected = detect_apriltag(
                frame, apriltag_roi, int(tag["id"]),
                float(tag["width_mm"]), float(tag["height_mm"]),
            )
        except RuntimeError as exc:
            raise RuntimeError("Remove the object on the marker and recapture.") from exc
        calibration = {"available": True, **detected}
        effective_apriltag_roi = apriltag_roi_from_detection(frame.shape, detected)
        if scale_snapshot.get("model") != "accet_cg1003l" and scale_snapshot.get("unit") not in (None, "g"):
            raise RuntimeError("ct capture is not allowed. Switch the scale to grams (g) and try again.")
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
        with self._testbed_lock:
            settings = self.settings_store.get()
            segmenter = self._get_testbed_segmenter()
            testbed = segmenter.segment(frame, apriltag=detected)
            testbed_model_path = segmenter.model_path
            del segmenter
        processing = testbed.image
        processing_roi = testbed.processing_roi
        working_tag_roi = translate_roi_to_crop(
            effective_apriltag_roi, processing_roi, processing.shape[1], processing.shape[0]
        )
        ignore_mask = marker_ignore_mask(processing.shape, working_tag_roi)
        clean_processing = processing.copy()
        clean_processing[ignore_mask > 0] = 255
        # Treat everything outside the segmented platform as an erase region.
        # build_jewellery_mask expands erase regions slightly, which prevents
        # the polygon-to-white transition from becoming a false jewellery item.
        analysis_ignore_mask = cv2.bitwise_or(
            ignore_mask, np.where(testbed.mask > 0, 0, 255).astype(np.uint8)
        )
        prepared, jewellery_mask = build_jewellery_mask(
            clean_processing, analysis_ignore_mask
        )
        jewellery_mask = cv2.bitwise_and(
            jewellery_mask.astype(np.uint8), testbed.mask.astype(np.uint8)
        )
        prepared[testbed.mask == 0] = 255
        prepared[jewellery_mask == 0] = 255
        evidence = self._evidence_image(frame, weight, now)
        source_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "original": self._save(source_dir / "original.png", frame),
            "evidence": self._save(source_dir / "evidence.jpg", evidence),
            "working": self._save(source_dir / "working.png", clean_processing),
            "prepared": self._save(source_dir / "prepared.png", prepared),
            "mask": self._save(source_dir / "mask.png", jewellery_mask * 255),
            "testbed_mask": self._save(source_dir / "testbed_mask.png", testbed.mask * 255),
            "testbed_source_mask": self._save(
                source_dir / "testbed_source_mask.png", testbed.source_mask * 255
            ),
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
                "testbed_source": testbed.source_roi,
                "apriltag": effective_apriltag_roi,
                "configured_apriltag": apriltag_roi,
                "working_apriltag": working_tag_roi,
            },
            "testbed": {
                "model": str(testbed_model_path),
                "confidence": testbed.confidence,
                "hole_validation": testbed.hole_validation,
                "masking": "polygon_with_white_square_padding",
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

    def capture_tare(self, mode: str, reading_id: int | None = None) -> dict[str, Any]:
        """Save a packet tare image with its scale reading and verified packet ID."""
        if mode not in {"pledge", "release"}:
            raise ValueError("Tare mode must be pledge or release")

        frame, camera_snapshot, scale_snapshot = self._capture_inputs(reading_id)
        if scale_snapshot.get("model") == "accet_cg1003l":
            tag = self.settings_store.get()["apriltag"]
            detected = detect_apriltag(frame, None, int(tag["id"]), float(tag["width_mm"]), float(tag["height_mm"]))
            apriltag_roi_from_detection(frame.shape, detected)
        if scale_snapshot.get("model") != "accet_cg1003l" and scale_snapshot.get("unit") not in (None, "g"):
            raise RuntimeError("ct capture is not allowed. Switch the scale to grams (g) and try again.")
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
        packet = self.packet_scanner.scan(frame)
        capture_id = f"tare_{mode}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_dir = self.repository.session_dir(capture_id) / "capture"
        paths = {
            "original": self._save(source_dir / "original.png", frame),
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
            "result": {"tare_mode": mode, "packet": packet},
            "job": {
                "status": "complete",
                "stage": "Complete",
                "message": f"{mode.title()} tare captured. {packet['message']}",
                "percent": 100,
                "seconds_remaining": 0,
                "error": None,
            },
        }
        evidence_path = source_dir / "evidence.png"
        self.reports.generate_tare_result_image(state).save(evidence_path)
        state["paths"]["evidence"] = str(evidence_path)
        self.repository.save(state)
        return state
