from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.analysis.beads import onnx_image_size, prepare_onnx_input  # noqa: E402
from backend.app.services.analysis_service import _adaptive_bead_nms  # noqa: E402


def detector_scores(
    session: ort.InferenceSession,
    image: np.ndarray,
    minimum_score: float,
) -> list[float]:
    height, width = image.shape[:2]
    tensor, scale, left, top = prepare_onnx_input(image, onnx_image_size(session))
    output = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])
    predictions = np.squeeze(output)
    if predictions.ndim == 2 and predictions.shape[0] == 5:
        predictions = predictions.T

    boxes: list[list[int]] = []
    scores: list[float] = []
    if predictions.ndim != 2 or predictions.shape[1] < 5:
        return scores
    for cx, cy, box_width, box_height, score, *_ in predictions:
        if (
            not np.isfinite([cx, cy, box_width, box_height, score]).all()
            or float(score) < minimum_score
        ):
            continue
        x1 = max(0, int(round((float(cx) - float(box_width) / 2 - left) / scale)))
        y1 = max(0, int(round((float(cy) - float(box_height) / 2 - top) / scale)))
        x2 = min(width - 1, int(round((float(cx) + float(box_width) / 2 - left) / scale)))
        y2 = min(height - 1, int(round((float(cy) + float(box_height) / 2 - top) / scale)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
    return [scores[index] for index in _adaptive_bead_nms(boxes, scores, image.shape)]


def historical_item_labels(session_dir: Path) -> dict[int, bool]:
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    labels: dict[int, bool] = {}
    for item in (manifest.get("result") or {}).get("items") or []:
        beads = item.get("beads")
        if isinstance(beads, dict) and "beads_detected" in beads:
            labels[int(item["index"])] = bool(beads["beads_detected"])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the bead ONNX model on saved raw item crops.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/detection/bead_finder.onnx")
    parser.add_argument("--sessions", type=Path, default=PROJECT_ROOT / "data/sessions")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.50, 0.60, 0.75])
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    totals = {threshold: Counter() for threshold in args.thresholds}
    disagreements: list[tuple[str, bool, int]] = []
    evaluated = 0
    for session_dir in sorted(args.sessions.iterdir()):
        labels = historical_item_labels(session_dir)
        for raw_path in sorted((session_dir / "items").glob("item_*_raw.png")):
            item_index = int(raw_path.stem.split("_")[1])
            if item_index not in labels:
                continue
            image = cv2.imread(str(raw_path))
            if image is None:
                continue
            scores = detector_scores(session, image, min(args.thresholds))
            reference = labels[item_index]
            evaluated += 1
            for threshold in args.thresholds:
                count = sum(score >= threshold for score in scores)
                predicted = count >= 2
                totals[threshold][("positive" if reference else "negative", "hit" if predicted else "miss")] += 1
                totals[threshold]["detections"] += count
                if threshold == args.thresholds[0] and predicted != reference:
                    disagreements.append((str(raw_path.relative_to(PROJECT_ROOT)), reference, count))

    print(f"Evaluated {evaluated} historical raw item crops")
    print("Reference is the result stored by the prior production pipeline; review disagreements visually.")
    for threshold in args.thresholds:
        counts = totals[threshold]
        tp = counts[("positive", "hit")]
        fn = counts[("positive", "miss")]
        fp = counts[("negative", "hit")]
        tn = counts[("negative", "miss")]
        print(
            f"threshold={threshold:.2f} TP={tp} FN={fn} FP={fp} TN={tn} "
            f"candidate_detections={counts['detections']}"
        )
    print(f"Disagreements at threshold={args.thresholds[0]:.2f} (stored_reference, candidates):")
    for path, reference, count in disagreements:
        print(f"  {path} reference={reference} candidates={count}")


if __name__ == "__main__":
    main()
