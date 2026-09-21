from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend.app.services.capture_service import CaptureService
from backend.app.services.packet_scan_service import PacketScanService
from backend.app.storage.repository import CaptureRepository
from backend.app.packet_scanner.scanner import Scanner
from backend.app.packet_scanner.ocr_reader import OCRReader


class TarePacketCaptureTests(unittest.TestCase):
    def test_quick_ocr_rejects_a_lower_confidence_match(self):
        reader = OCRReader.__new__(OCRReader)
        reader._items = Mock(return_value=iter([
            ("41233285", 0.91), ("41233284", 0.98),
        ]))
        self.assertIsNone(reader.quick_match(np.zeros((100, 100, 3), dtype=np.uint8), "41233285"))

    def test_unmatched_quick_ocr_runs_full_search(self):
        detector = Mock(provider="CPUExecutionProvider")
        detector.detect.return_value = (
            [Mock(confidence=0.95, points=np.zeros((4, 2), dtype=np.float32))],
            np.zeros((100, 100, 3), dtype=np.uint8), 1.0,
        )
        scanner = Scanner(detector)
        scanner.ocr_reader = Mock()
        scanner.ocr_reader.quick_match.return_value = None
        scanner.ocr_reader.read.return_value = (
            "41233284", 0.98, np.zeros((50, 100, 3), dtype=np.uint8),
            np.zeros((50, 100, 3), dtype=np.uint8), "original",
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("backend.app.packet_scanner.scanner.warp_perspective", return_value=image), \
             patch("backend.app.packet_scanner.scanner.blur_score", return_value=80.0), \
             patch("backend.app.packet_scanner.scanner.read_barcode", return_value=("41233285", "Code128", image, "original")):
            result = scanner.scan(image, 0.4, 0.45, 35)
        self.assertEqual(result.status, "MISMATCH")
        self.assertEqual(scanner.ocr_reader.read.call_count, 2)

    def test_mismatched_values_are_never_reported_as_matched(self):
        service = PacketScanService()
        service._scanner = Mock()
        service._scanner.ocr_error = ""
        service._scanner.scan.return_value = Mock(
            status="MISMATCH", ocr="41233285", barcode="41233284",
            barcode_format="Code128", message="Barcode and printed digits differ.",
            ocr_confidence=0.98, detection_confidence=0.95, blur_score=80.0,
            provider="CPUExecutionProvider", detections=[],
        )
        packet = service.scan(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(packet["packet_number"], "41233285")
        self.assertEqual(packet["barcode"], "41233284")
        self.assertFalse(packet["barcode_matched"])

    def test_both_tare_modes_save_packet_result_beside_photo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = CaptureRepository(root / "app.db", root / "sessions")
            frame = np.full((1080, 720, 3), (40, 80, 120), dtype=np.uint8)
            camera = Mock()
            camera.snapshot_frame.return_value = (frame, {"connected": True})
            scale = Mock()
            scale.snapshot.return_value = {"connected": True, "weight_g": 12.34, "age_ms": 100}
            service = CaptureService(camera, scale, Mock(), repository)
            service.packet_scanner.scan = Mock(return_value={
                "status": "MATCH", "packet_number": "41233285", "barcode": "41233285",
                "barcode_matched": True, "message": "Barcode and OCR digits match.",
                "detections": [],
            })

            for mode in ("pledge", "release"):
                state = service.capture_tare(mode)
                self.assertEqual(state["result"]["packet"]["packet_number"], "41233285")
                self.assertTrue(state["result"]["packet"]["barcode_matched"])
                self.assertEqual(state["tare_mode"], mode)
                self.assertEqual(repository.get(state["id"])["result"]["packet"]["status"], "MATCH")
                image = cv2.imread(state["paths"]["evidence"])
                self.assertIsNotNone(image)
                self.assertEqual(image.shape[:2], (1080, 1200))
                np.testing.assert_array_equal(image[500, 500], frame[500, 500])


if __name__ == "__main__":
    unittest.main()
