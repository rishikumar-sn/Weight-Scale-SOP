from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class TestbedRoi:
    """A square, white-backed view whose non-white pixels are inside the testbed."""

    image: np.ndarray
    mask: np.ndarray
    source_mask: np.ndarray
    source_roi: dict[str, int]
    processing_roi: dict[str, int]
    confidence: float


class TestbedSegmenter:
    """Decode a one-class Ultralytics YOLOv8 segmentation ONNX export."""

    __test__ = False

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.50,
        mask_threshold: float = 0.50,
        inset_px: int = 2,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Testbed segmentation model not found: {self.model_path}")
        self.confidence_threshold = float(confidence_threshold)
        self.mask_threshold = float(mask_threshold)
        self.inset_px = max(0, int(inset_px))
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise RuntimeError(f"Unsupported testbed model input shape: {shape}")
        self.input_name = model_input.name
        self.input_height = int(shape[2])
        self.input_width = int(shape[3])

    def segment(self, image_bgr: np.ndarray) -> TestbedRoi:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Testbed segmentation requires a BGR colour image")

        tensor, transform = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) < 2:
            raise RuntimeError("Testbed model did not return detection and mask outputs")
        prediction, prototypes = outputs[:2]
        row, box, confidence = self._best_detection(prediction)
        source_mask = self._decode_mask(row[5:], prototypes, box, transform, image_bgr.shape[:2])
        source_mask = self._clean_mask(source_mask)
        if not np.any(source_mask):
            raise RuntimeError("The detected testbed mask is empty")
        return self._square_roi(image_bgr, source_mask, confidence)

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        height, width = image_bgr.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        left = (self.input_width - resized_width) // 2
        top = (self.input_height - resized_height) // 2
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = cv2.resize(
            image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return tensor, {
            "scale": scale,
            "left": left,
            "top": top,
            "resized_width": resized_width,
            "resized_height": resized_height,
        }

    def _best_detection(
        self, prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        rows = np.asarray(prediction, dtype=np.float32)
        if rows.ndim != 3 or rows.shape[0] != 1:
            raise RuntimeError(f"Unsupported testbed detection output shape: {rows.shape}")
        rows = rows[0].T
        if rows.shape[1] < 6:
            raise RuntimeError(f"Unsupported testbed prediction width: {rows.shape[1]}")
        scores = rows[:, 4]
        valid = np.flatnonzero(np.isfinite(scores) & (scores >= self.confidence_threshold))
        if not len(valid):
            best = float(np.nanmax(scores)) if scores.size else 0.0
            raise RuntimeError(
                f"Testbed was not detected (best confidence {best:.3f}, "
                f"required {self.confidence_threshold:.3f})"
            )
        index = int(valid[np.argmax(scores[valid])])
        row = rows[index]
        center_x, center_y, width, height = row[:4]
        box = np.array(
            [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2],
            dtype=np.float32,
        )
        return row, box, float(scores[index])

    def _decode_mask(
        self,
        coefficients: np.ndarray,
        prototypes: np.ndarray,
        box: np.ndarray,
        transform: dict[str, Any],
        source_shape: tuple[int, int],
    ) -> np.ndarray:
        proto = np.asarray(prototypes, dtype=np.float32)
        if proto.ndim != 4 or proto.shape[0] != 1:
            raise RuntimeError(f"Unsupported testbed mask output shape: {proto.shape}")
        proto = proto[0]
        channels, mask_height, mask_width = proto.shape
        if coefficients.shape[0] != channels:
            raise RuntimeError(
                f"Mask coefficient count {coefficients.shape[0]} does not match {channels} prototypes"
            )
        logits = coefficients @ proto.reshape(channels, -1)
        mask = (1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))).reshape(
            mask_height, mask_width
        )

        scaled_box = box.copy()
        scaled_box[[0, 2]] *= mask_width / self.input_width
        scaled_box[[1, 3]] *= mask_height / self.input_height
        x1, y1, x2, y2 = scaled_box
        columns = np.arange(mask_width, dtype=np.float32)[None, :]
        rows = np.arange(mask_height, dtype=np.float32)[:, None]
        mask *= (columns >= x1) & (columns < x2) & (rows >= y1) & (rows < y2)
        mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        left = int(transform["left"])
        top = int(transform["top"])
        resized_width = int(transform["resized_width"])
        resized_height = int(transform["resized_height"])
        mask = mask[top : top + resized_height, left : left + resized_width]
        source_height, source_width = source_shape
        mask = cv2.resize(mask, (source_width, source_height), interpolation=cv2.INTER_LINEAR)
        return (mask >= self.mask_threshold).astype(np.uint8)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return np.zeros_like(binary)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        if self.inset_px:
            size = self.inset_px * 2 + 1
            binary = cv2.erode(
                binary,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                iterations=1,
            )
        return binary

    @staticmethod
    def _square_roi(image_bgr: np.ndarray, source_mask: np.ndarray, confidence: float) -> TestbedRoi:
        x, y, width, height = cv2.boundingRect(source_mask)
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not calculate testbed bounds")
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        cropped_image = image_bgr[y : y + height, x : x + width]
        cropped_mask = source_mask[y : y + height, x : x + width]
        square_image = np.full((side, side, 3), 255, dtype=np.uint8)
        square_mask = np.zeros((side, side), dtype=np.uint8)
        target = square_image[pad_top : pad_top + height, pad_left : pad_left + width]
        target_mask = square_mask[pad_top : pad_top + height, pad_left : pad_left + width]
        target[cropped_mask > 0] = cropped_image[cropped_mask > 0]
        target_mask[cropped_mask > 0] = 1
        return TestbedRoi(
            image=square_image,
            mask=square_mask,
            source_mask=source_mask,
            source_roi={"x": x, "y": y, "w": width, "h": height},
            # This virtual square origin maps coordinates in the padded output
            # back to the original camera frame without losing the white padding.
            processing_roi={
                "x": x - pad_left,
                "y": y - pad_top,
                "w": side,
                "h": side,
            },
            confidence=confidence,
        )
