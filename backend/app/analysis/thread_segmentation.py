from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from .testbed_segmentation import TestbedSegmenter


class ThreadSegmenter(TestbedSegmenter):
    """YOLO11s-seg on the raw full capture; retain all thread instances."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path, confidence_threshold=0.25, exclude_circle=False)
        if self.names != {0: "Thread"} or (self.input_height, self.input_width) != (960, 960):
            raise RuntimeError(f"Unexpected thread model: {self.names}, {self.input_height}x{self.input_width}")

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Thread detection requires the original full-frame BGR capture")
        tensor, transform = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 2:
            raise RuntimeError("Thread model must return predictions and mask prototypes")
        prediction, prototypes = outputs
        if (prototypes.ndim != 4 or prediction.ndim != 3 or prediction.shape[0] != 1
                or prediction.shape[1] != 5 + prototypes.shape[1]):
            raise RuntimeError(f"Unsupported thread outputs: {prediction.shape}, {prototypes.shape}")
        rows = prediction[0].T
        indices = np.flatnonzero(np.isfinite(rows).all(axis=1)
                                 & (rows[:, 4] >= self.confidence_threshold)
                                 & (rows[:, 2] > 0) & (rows[:, 3] > 0))
        union = np.zeros(image.shape[:2], np.uint8)
        confidences = []
        if len(indices):
            boxes = rows[indices, :4].copy()
            boxes[:, :2] -= boxes[:, 2:] / 2
            kept = cv2.dnn.NMSBoxes(boxes.tolist(), rows[indices, 4].tolist(), 0.0, 0.45)
            for index in np.asarray(kept, dtype=int).reshape(-1):
                row = rows[indices[index]]
                x, y, w, h = boxes[index]
                mask = self._decode_mask(row[5:], prototypes, np.array([x, y, x+w, y+h]),
                                         transform, image.shape[:2])
                if np.any(mask):
                    union |= mask
                    confidences.append(float(row[4]))
        return union, {"model": self.model_path.name, "input_source": "raw_full_frame",
                       "input_shape": list(image.shape), "imgsz": 960,
                       "confidence_threshold": self.confidence_threshold,
                       "mask_threshold": self.mask_threshold, "confidences": confidences}


def project_thread_mask(source: np.ndarray, shape: tuple[int, int], x: int, y: int) -> np.ndarray:
    """Map camera pixels into an unscaled jewel crop, including virtual padding."""
    local = np.zeros(shape, np.uint8)
    height, width = shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(source.shape[1], x + width), min(source.shape[0], y + height)
    if x2 > x1 and y2 > y1:
        local[y1-y:y2-y, x1-x:x2-x] = source[y1:y2, x1:x2]
    return local
