import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .barcode_reader import read_barcode, region
from .config import BARCODE_REGION_RATIO
from .image_quality import blur_score
from .perspective import warp_perspective
from .validator import validate


@dataclass
class ScanResult:
    status: str = "NOT_FOUND"
    message: str = "No target detected."
    barcode: str = ""
    ocr: str = ""
    barcode_format: str = ""
    detection_confidence: float = 0.0
    ocr_confidence: float = 0.0
    blur_score: float = 0.0
    provider: str = ""
    timings: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    detections: list = field(default_factory=list)


class Scanner:
    def __init__(self, detector):
        self.detector = detector
        self.ocr_reader = None
        self.ocr_error = ""

    def preload_ocr(self):
        if self.ocr_reader is None and not self.ocr_error:
            try:
                from .ocr_reader import OCRReader
                self.ocr_reader = OCRReader()
            except Exception as exc:
                self.ocr_error = str(exc)
                logging.exception("PaddleOCR initialization failed")

    def scan(self, image, confidence, iou, blur_threshold):
        start = time.perf_counter()
        result = ScanResult(provider=self.detector.provider)
        result.views["original"] = image.copy()
        detections, letterboxed, ms = self.detector.detect(image, confidence, iou)
        result.timings["yolo_ms"] = ms
        result.views["letterbox"] = letterboxed
        result.detections = detections
        detected = image.copy()
        for idx, det in enumerate(detections):
            color = (0, 255, 0) if idx == 0 else (0, 200, 255)
            cv2.polylines(detected, [np.round(det.points).astype(np.int32)], True, color, 3)
            anchor = np.round(det.points[0]).astype(int)
            cv2.putText(detected, f"{det.confidence:.2f}", tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        result.views["detected"] = detected
        if not detections:
            result.timings["total_ms"] = (time.perf_counter() - start) * 1000
            return result
        result.detection_confidence = detections[0].confidence
        tick = time.perf_counter()
        rectified = warp_perspective(image, detections[0].points)
        result.timings["rectification_ms"] = (time.perf_counter() - tick) * 1000
        result.views["rectified"] = rectified
        result.blur_score = blur_score(rectified)
        if result.blur_score < blur_threshold:
            result.message = "Target detected but image is blurred. Please hold packet steady."
            result.timings["total_ms"] = (time.perf_counter() - start) * 1000
            return result
        # A confident match on the original OCR crop gives the same result as the
        # full variant search. Try it in both directions before slower filters.
        orientations = (("0", rectified), ("180", cv2.rotate(rectified, cv2.ROTATE_180)))
        barcode_reads = []
        for orientation, oriented in orientations:
            tick = time.perf_counter()
            barcode, fmt, processed, variant = read_barcode(oriented)
            result.timings["barcode_ms"] = result.timings.get("barcode_ms", 0) + (time.perf_counter() - tick) * 1000
            barcode_reads.append((orientation, oriented, barcode, fmt, processed, variant))
        self.preload_ocr()
        candidates = []
        barcode_values = {entry[2] for entry in barcode_reads if entry[2]}
        if self.ocr_reader is not None and len(barcode_values) <= 1:
            for orientation, oriented, barcode, fmt, processed, variant in barcode_reads:
                tick = time.perf_counter()
                quick = self.ocr_reader.quick_match(oriented, barcode)
                result.timings["ocr_ms"] = result.timings.get("ocr_ms", 0) + (time.perf_counter() - tick) * 1000
                if quick is not None:
                    ocr, ocr_conf, digit_roi, ocr_processed, ocr_variant = quick
                    candidates.append((4, ocr_conf, "MATCH", barcode, fmt, ocr, oriented,
                                       digit_roi, ocr_processed, processed, orientation, variant, ocr_variant))
                    break
        if not candidates:
            for orientation, oriented, barcode, fmt, processed, variant in barcode_reads:
                tick = time.perf_counter()
                if self.ocr_reader is None:
                    h = oriented.shape[0]
                    digit_roi = oriented[int(h * 0.35):]
                    ocr, ocr_conf, ocr_processed, ocr_variant = "", 0.0, digit_roi, "unavailable"
                else:
                    ocr, ocr_conf, digit_roi, ocr_processed, ocr_variant = self.ocr_reader.read(oriented)
                result.timings["ocr_ms"] = result.timings.get("ocr_ms", 0) + (time.perf_counter() - tick) * 1000
                status = validate(barcode, ocr)
                rank = {"MATCH": 4, "MISMATCH": 3, "BARCODE_ONLY": 2, "OCR_ONLY": 1, "NOT_FOUND": 0}[status]
                candidates.append((rank, ocr_conf, status, barcode, fmt, ocr, oriented, digit_roi, ocr_processed, processed, orientation, variant, ocr_variant))
                if status == "MATCH":
                    break
        _, result.ocr_confidence, result.status, result.barcode, result.barcode_format, result.ocr, oriented, digit_roi, ocr_processed, processed, orientation, variant, ocr_variant = max(candidates, key=lambda item: (item[0], item[1]))
        result.views.update(rectified=oriented, barcode_roi=region(oriented, BARCODE_REGION_RATIO), digit_roi=digit_roi,
                            ocr_processed=ocr_processed, barcode_processed=processed)
        result.message = {
            "MATCH": "Target detected. Barcode and OCR digits match.",
            "MISMATCH": "Barcode and printed digits differ. Check the packet.",
            "BARCODE_ONLY": "Barcode read; printed digits could not be verified.",
            "OCR_ONLY": "Printed digits read; barcode could not be verified.",
            "NOT_FOUND": "Target detected, but neither value could be read. Try again.",
        }[result.status]
        if not result.barcode and oriented.shape[1] < 260:
            result.message += " Barcode is too small in this frame; move the packet closer or use higher camera resolution."
        if self.ocr_error:
            result.message += f" PaddleOCR unavailable: {self.ocr_error}"
        result.timings["total_ms"] = (time.perf_counter() - start) * 1000
        logging.info("Scan: %s barcode=%s ocr=%s orientation=%s barcode_variant=%s ocr_variant=%s times=%s",
                     result.status, result.barcode, result.ocr, orientation, variant, ocr_variant, result.timings)
        return result
