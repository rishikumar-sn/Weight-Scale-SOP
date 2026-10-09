"""Evaluate the exported bead ONNX model against a YOLO-format split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.analysis.beads import onnx_image_size, prepare_onnx_input  # noqa: E402
from backend.app.services.analysis_service import _adaptive_bead_nms  # noqa: E402


def infer(session: ort.InferenceSession, image: np.ndarray, threshold: float) -> list[dict]:
    height, width = image.shape[:2]
    tensor, scale, left, top = prepare_onnx_input(image, onnx_image_size(session))
    output = np.squeeze(session.run(None, {session.get_inputs()[0].name: tensor})[0])
    if output.ndim == 2 and output.shape[0] == 5:
        output = output.T
    boxes: list[list[int]] = []
    scores: list[float] = []
    for cx, cy, box_width, box_height, score, *_ in output:
        if float(score) < threshold:
            continue
        box = [
            int(round((float(cx) - float(box_width) / 2 - left) / scale)),
            int(round((float(cy) - float(box_height) / 2 - top) / scale)),
            int(round((float(cx) + float(box_width) / 2 - left) / scale)),
            int(round((float(cy) + float(box_height) / 2 - top) / scale)),
        ]
        box[0], box[1] = max(0, box[0]), max(0, box[1])
        box[2], box[3] = min(width - 1, box[2]), min(height - 1, box[3])
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            scores.append(float(score))
    return [
        {"bbox": boxes[index], "score": scores[index]}
        for index in _adaptive_bead_nms(boxes, scores, image.shape)
    ]


def ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        _, center_x, center_y, box_width, box_height = map(float, values[:5])
        boxes.append([
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        ])
    return boxes


def iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def evaluate(model: Path, split_dir: Path, thresholds: list[float]) -> dict:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    images = sorted(path for path in (split_dir / "images").iterdir() if path.is_file())
    results = []
    for threshold in thresholds:
        true_positive = false_positive = false_negative = detected_images = 0
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            truths = ground_truth(
                split_dir / "labels" / f"{image_path.stem}.txt",
                image.shape[1],
                image.shape[0],
            )
            predictions = infer(session, image, threshold)
            detected_images += bool(predictions)
            unmatched = set(range(len(truths)))
            for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
                matches = [(iou(prediction["bbox"], truths[index]), index) for index in unmatched]
                best_iou, best_index = max(matches, default=(0.0, -1))
                if best_iou >= 0.5:
                    true_positive += 1
                    unmatched.remove(best_index)
                else:
                    false_positive += 1
            false_negative += len(unmatched)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        results.append({
            "threshold": threshold,
            "images": len(images),
            "images_with_detections": detected_images,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision_at_iou_0_5": round(precision, 4),
            "recall_at_iou_0_5": round(recall, 4),
        })
    return {"model": str(model), "split": str(split_dir), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/detection/bead_finder.onnx")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.6, 0.75])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.dataset / args.split, args.thresholds), indent=2))


if __name__ == "__main__":
    main()
