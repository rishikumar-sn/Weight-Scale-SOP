import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from backend.app.analysis.beads import prepare_onnx_input
from backend.app.services.analysis_service import AnalysisService, OnnxBeadDetector


class BeadFullFrameTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Distinct pixels outside the testbed prove inference sees the original
        # frame, including its background, rather than a masked or cropped view.
        self.image = np.full((200, 400, 3), (30, 70, 120), dtype=np.uint8)
        self.image[40:140, 100:280] = (180, 190, 200)
        self.mask = np.zeros((200, 400), dtype=np.uint8)
        self.mask[40:140, 100:280] = 255
        self.mask[70:80, 140:150] = 0
        original = self.root / "original.png"
        mask_path = self.root / "testbed_source_mask.png"
        cv2.imwrite(str(original), self.image)
        cv2.imwrite(str(mask_path), self.mask)
        self.state = {
            "id": "capture",
            "paths": {"original": str(original), "testbed_source_mask": str(mask_path)},
            # Negative virtual origin from square testbed padding.
            "rois": {"processing": {"x": 100, "y": -20, "w": 200, "h": 200}},
            "calibration": {
                "available": True, "found": True,
                "mm_per_pixel_x": 0.1, "mm_per_pixel_y": 0.1,
            },
        }
        self.detector = OnnxBeadDetector.__new__(OnnxBeadDetector)
        self.detector.input_name = "images"
        self.detector.input_size = (200, 400)
        self.detector.model_name = "bead_finder.onnx"
        self.detector.model_metadata = {}
        # Two jewels, a polygon hole, outside the platform, an item boundary,
        # and outside the processing ROI. All scores pass the confidence gate.
        rows = np.array([
            [125, 75, 50, 20, 0.9],
            [205, 75, 10, 10, 0.9],
            [145, 75, 8, 8, 0.9],
            [20, 75, 10, 10, 0.9],
            [170, 75, 8, 8, 0.9],
            [310, 75, 10, 10, 0.9],
        ], dtype=np.float32)
        self.detector.session = Mock()
        self.detector.session.run.return_value = [rows.T[None]]
        repository = SimpleNamespace(session_dir=lambda capture_id: self.root / capture_id)
        self.service = AnalysisService(repository)
        self.service._beads = self.detector

    def test_multiple_jewels_use_full_frame_and_filter_their_own_regions(self):
        first = {"index": 1, "bbox": {"x": 10, "y": 70, "w": 60, "h": 70}}
        second = {"index": 2, "bbox": {"x": 90, "y": 70, "w": 60, "h": 70}}
        results = [self.service._run_beads(self.state, item) for item in (first, second)]
        expected_tensor = prepare_onnx_input(self.image, self.detector.input_size)[0]
        for call in self.detector.session.run.call_args_list:
            np.testing.assert_array_equal(call.args[1]["images"], expected_tensor)
        self.assertEqual([result["count"] for result in results], [1, 1])
        self.assertEqual(results[0]["detections"][0]["bbox"], [100, 65, 150, 85])
        self.assertEqual(results[1]["detections"][0]["bbox"], [200, 70, 210, 80])
        self.assertEqual(results[0]["detection_region"], {"x": 110, "y": 50, "w": 60, "h": 70})
        # Boxes crossing the region edge retain their full calibrated size.
        self.assertEqual(results[0]["detections"][0]["width_mm"], 5.0)
        for result in results:
            self.assertEqual(result["input_source"], "full_frame")
            self.assertEqual(result["coordinate_space"], "full_frame")
            self.assertEqual(cv2.imread(result["image"]).shape, self.image.shape)
            self.assertTrue(Path(result["onnx_image"]).is_file())

    def test_without_item_only_accepts_predictions_inside_testbed_polygon(self):
        result = self.service._run_beads(self.state)
        self.assertEqual(result["count"], 3)
        centers = [(box[0] + box[2]) / 2 for box in (d["bbox"] for d in result["detections"])]
        self.assertEqual(centers, [125, 205, 170])

    def test_missing_testbed_mask_requires_recapture(self):
        del self.state["paths"]["testbed_source_mask"]
        with self.assertRaisesRegex(RuntimeError, "mask is unavailable"):
            self.service._run_beads(self.state)
        self.detector.session.run.assert_not_called()

    def test_mask_dimensions_must_match_original_frame(self):
        with self.assertRaisesRegex(ValueError, "full-frame image dimensions"):
            self.detector.run(self.image, detection_mask=np.ones((100, 100), dtype=np.uint8))
        self.detector.session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
