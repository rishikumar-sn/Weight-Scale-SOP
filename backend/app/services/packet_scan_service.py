from __future__ import annotations

import os
from threading import Lock

from ..packet_scanner.config import BLUR_THRESHOLD, DETECTION_CONFIDENCE, MODEL_PATH, NMS_IOU_THRESHOLD


class PacketScanService:
    """Share the packet detector and OCR models across tare captures."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._scanner = None

    def _load(self) -> None:
        if self._scanner is not None:
            return
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Packet scanner model is missing: {MODEL_PATH}")
        try:
            # PaddleX otherwise uses ten CPU threads, which can starve the
            # camera producer on this six-core capture PC during a tare scan.
            os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", "4")
            import paddleocr  # noqa: F401
            import zxingcpp  # noqa: F401
            from ..packet_scanner.detector import Detector
            from ..packet_scanner.scanner import Scanner
            scanner = Scanner(Detector(MODEL_PATH))
            scanner.preload_ocr()
            if scanner.ocr_error:
                raise RuntimeError(f"Packet OCR is unavailable: {scanner.ocr_error}")
            self._scanner = scanner
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"Packet scanner dependencies are unavailable: {exc}") from exc

    def preload(self) -> None:
        with self._lock:
            self._load()

    def scan(self, frame) -> dict:
        with self._lock:
            self._load()
            result = self._scanner.scan(
                frame, DETECTION_CONFIDENCE, NMS_IOU_THRESHOLD, BLUR_THRESHOLD
            )
            if self._scanner.ocr_error:
                error = self._scanner.ocr_error
                self._scanner = None
                raise RuntimeError(f"Packet OCR is unavailable: {error}")
            return {
                "status": result.status,
                "packet_number": result.ocr or None,
                "barcode": result.barcode or None,
                "barcode_format": result.barcode_format or None,
                "barcode_matched": result.status == "MATCH",
                "message": result.message,
                "ocr_confidence": result.ocr_confidence,
                "detection_confidence": result.detection_confidence,
                "blur_score": result.blur_score,
                "provider": result.provider,
                "detections": [
                    {"confidence": detection.confidence, "points": detection.points.tolist()}
                    for detection in result.detections
                ],
            }
