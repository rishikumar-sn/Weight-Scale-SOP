"""Three-class jewel/thread gate, exported from jewel-thread-small.pt."""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


THREAD_ONLY_MESSAGE = (
    "Thread or tassel without a jewel detected. Place jewellery on the testbed "
    "and recapture. Analysis is blocked."
)


class JewelThreadClassifier:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.size = int(model_input.shape[2])
        if model_input.shape != [1, 3, self.size, self.size]:
            raise RuntimeError(f"Unexpected jewel/thread input: {model_input.shape}")
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
        if self.names != {0: "jewel_with_thread", 1: "jewel_without_thread", 2: "thread_only"}:
            raise RuntimeError(f"Unexpected jewel/thread classes: {self.names}")

    def prepare(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("Jewel/thread classification requires a raw BGR item crop")
        # Match Ultralytics classification: PIL bilinear shortest-edge resize,
        # centre crop, RGB float32 / 255 (mean=0, std=1).
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = rgb.size
        if width <= height:
            target = (self.size, int(self.size * height / width))
        else:
            target = (int(self.size * width / height), self.size)
        rgb = rgb.resize(target, Image.Resampling.BILINEAR)
        left = round((target[0] - self.size) / 2)
        top = round((target[1] - self.size) / 2)
        rgb = rgb.crop((left, top, left + self.size, top + self.size))
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0)

    def classify(self, image: np.ndarray) -> dict:
        scores = np.asarray(self.session.run(None, {self.input_name: self.prepare(image)})[0]).reshape(-1)
        if (scores.shape != (3,) or not np.isfinite(scores).all()
                or np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-4)):
            raise RuntimeError("Invalid jewel/thread classification output; recapture")
        index = int(np.argmax(scores))
        return {"model": self.model_path.name, "input_source": "raw_item_crop",
                "label": self.names[index], "confidence": float(scores[index]),
                "passed": self.names[index] != "thread_only",
                "scores": {self.names[i]: float(score) for i, score in enumerate(scores)}}
