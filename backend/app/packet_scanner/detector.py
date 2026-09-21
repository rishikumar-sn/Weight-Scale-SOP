import ast
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from .obb_utils import letterbox, rotated_nms, scale_points_to_original, xywhr_to_four_points


@dataclass
class Detection:
    confidence: float
    points: np.ndarray
    box: np.ndarray


class Detector:
    def __init__(self, model_path):
        available = ort.get_available_providers()
        providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()
        self.provider = self.session.get_providers()[0]
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.class_names = ast.literal_eval(metadata.get("names", "{0: 'target'}"))
        logging.info("Model provider: %s", self.provider)
        logging.info("Model inputs: %s", [(i.name, i.shape, i.type) for i in self.session.get_inputs()])
        logging.info("Model outputs: %s", [(o.name, o.shape, o.type) for o in self.outputs])
        logging.info("Model metadata: %s", metadata)

    def detect(self, image, confidence=0.4, iou=0.45):
        if len(self.input.shape) != 4 or self.input.shape[1] != 3:
            raise ValueError(f"Unsupported model input: {self.input.shape}")
        size = int(self.input.shape[-1])
        canvas, scale, padding = letterbox(image, size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        start = time.perf_counter()
        raw = self.session.run([self.outputs[0].name], {self.input.name: tensor})[0]
        elapsed = (time.perf_counter() - start) * 1000
        predictions = np.squeeze(raw, axis=0)
        # Ultralytics raw OBB: [cx, cy, w, h, class scores..., angle] per anchor.
        # This model is [1, 6, 8400], i.e. one class plus angle, with no exported NMS.
        if predictions.ndim != 2:
            raise ValueError(f"Unsupported OBB output: {raw.shape}")
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        classes = len(self.class_names)
        if predictions.shape[1] != 5 + classes:
            raise ValueError(f"Expected {5 + classes} OBB values, got {predictions.shape[1]}; output {raw.shape}")
        scores = predictions[:, 4:4 + classes].max(axis=1)
        mask = np.isfinite(predictions).all(axis=1) & (scores >= confidence) & (predictions[:, 2] > 0) & (predictions[:, 3] > 0)
        selected = predictions[mask]
        scores = scores[mask]
        kept = rotated_nms(selected[:, [0, 1, 2, 3, -1]].tolist(), scores.tolist(), iou)
        detections = []
        for index in kept:
            row = selected[index]
            box = row[[0, 1, 2, 3, -1]]
            points = scale_points_to_original(xywhr_to_four_points(box), scale, padding, image.shape)
            detections.append(Detection(float(scores[index]), points, box.copy()))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        logging.info("Detections: %d; inference %.1f ms", len(detections), elapsed)
        return detections, canvas, elapsed
