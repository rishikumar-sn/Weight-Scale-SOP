from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from .testbed_segmentation import TestbedSegmenter


FOREIGN_OBJECT_THRESHOLDS = {"foreign_obj": 0.45, "plastic_packet": 0.35, "paper": 0.25}


class ForeignObjectDetector(TestbedSegmenter):
    """Run the segmentation model on raw pixels, then filter by the testbed polygon.

    Reuse the testbed decoder's letterboxing and prototype-mask mapping so both
    models retain the original camera coordinates, including non-square images.
    """

    def __init__(self, model_path: str | Path) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Foreign-object segmentation model not found: {model_path}")
        super().__init__(model_path)
        metadata = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(metadata.get("names", "{}"))
        self.names = ({int(key): value for key, value in names.items()}
                      if isinstance(names, dict) else dict(enumerate(names)))
        if set(self.names.values()) != {*FOREIGN_OBJECT_THRESHOLDS, "jewellery"}:
            raise RuntimeError(f"Unexpected foreign-object model classes: {self.names}")
        if set(self.names) != set(range(len(self.names))):
            raise RuntimeError("Foreign-object model class indices must be contiguous")

    def detect(self, image_bgr: np.ndarray, testbed_mask: np.ndarray) -> dict:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Foreign-object detection requires the raw BGR capture")
        if testbed_mask is None or testbed_mask.shape != image_bgr.shape[:2] or not np.any(testbed_mask):
            raise ValueError("Foreign-object detection requires the full-frame testbed mask")
        tensor, transform = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 2:
            raise RuntimeError("Foreign-object model must return detections and mask prototypes")
        predictions, prototypes = map(np.asarray, outputs)
        class_count = len(self.names)
        if (prototypes.ndim != 4 or prototypes.shape[0] != 1
                or predictions.ndim != 3 or predictions.shape[0] != 1
                or predictions.shape[1] != 4 + class_count + prototypes.shape[1]):
            raise RuntimeError(f"Unsupported foreign-object model outputs: {predictions.shape}, {prototypes.shape}")
        rows = predictions[0].T
        scores = rows[:, 4:4 + class_count]
        classes = np.argmax(scores, axis=1)
        detections = []
        for class_id, label in self.names.items():
            if label not in FOREIGN_OBJECT_THRESHOLDS:
                continue
            threshold = FOREIGN_OBJECT_THRESHOLDS[label]
            indices = np.flatnonzero((classes == class_id) & np.isfinite(rows).all(axis=1)
                                     & (scores[:, class_id] >= threshold)
                                     & (rows[:, 2] > 0) & (rows[:, 3] > 0))
            if not len(indices):
                continue
            boxes = rows[indices, :4].copy()
            boxes[:, :2] -= boxes[:, 2:] / 2
            confidences = scores[indices, class_id]
            # Suppress duplicates separately for each class. Jewellery must never
            # suppress a paper/packet detection covering the same jewel.
            kept = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.0, 0.45)
            for local_index in np.asarray(kept, dtype=int).reshape(-1):
                row = rows[indices[local_index]]
                x, y, width, height = boxes[local_index]
                box = np.array([x, y, x + width, y + height], dtype=np.float32)
                mask = self._decode_mask(row[4 + class_count:], prototypes, box,
                                         transform, image_bgr.shape[:2])
                overlap = (mask > 0) & (testbed_mask > 0)
                if not np.any(overlap):
                    continue
                bx, by, bw, bh = cv2.boundingRect(overlap.astype(np.uint8))
                detections.append({"label": label, "confidence": float(confidences[local_index]),
                                   "threshold": threshold, "testbed_pixels": int(np.count_nonzero(overlap)),
                                   "bbox": {"x": bx, "y": by, "w": bw, "h": bh}})
        return {"passed": not detections, "model": str(self.model_path),
                "thresholds": dict(FOREIGN_OBJECT_THRESHOLDS), "detections": detections,
                "region_filter": "segmentation_mask_intersects_testbed_polygon"}
