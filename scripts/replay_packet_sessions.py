"""Run the production packet scanner on saved sessions and save review images.

Source images and manifests are read only. --expect verifies both independent
readers against a supplied identifier, without feeding that identifier to OCR.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", "4")

from backend.app.packet_scanner.config import (
    BLUR_THRESHOLD, DETECTION_CONFIDENCE, MODEL_PATH, NMS_IOU_THRESHOLD,
)
from backend.app.packet_scanner.detector import Detector
from backend.app.packet_scanner.scanner import Scanner


def review_image(result, session_id):
    canvas = np.full((1060, 1440, 3), (28, 24, 20), dtype=np.uint8)

    def text(value, x, y, scale=.7, color=(235, 235, 235)):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def picture(image, x, y, width, height):
        if image is None or not image.size:
            return
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        factor = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        h, w = resized.shape[:2]
        canvas[y:y+h, x:x+w] = resized

    text(session_id, 24, 38, .75)
    text(f"{result.status} | Independent barcode / OCR comparison", 24, 77, .8,
         (105, 230, 120) if result.status == "MATCH" else (80, 190, 255))
    picture(result.views["detected"], 24, 100, 520, 935)
    x = 575
    text(f"Barcode: {result.barcode or '(unread)'}", x, 132, .8)
    text(f"OCR:     {result.ocr or '(unread)'}", x, 175, .8)
    text(f"OCR confidence: {result.ocr_confidence:.4%}", x, 218)
    text(f"Detector confidence: {result.detection_confidence:.2%}", x, 254)
    text(f"Blur score: {result.blur_score:.1f} (minimum {BLUR_THRESHOLD:.0f})", x, 290)
    text("Rectified label from detected box", x, 343)
    picture(result.views.get("rectified"), x, 360, 835, 210)
    text("Barcode decoder input", x, 595)
    picture(result.views.get("barcode_processed"), x, 615, 835, 160)
    text("Independent OCR input", x, 810)
    picture(result.views.get("digit_roi"), x, 830, 835, 150)
    text("Full ID preserved; letters and leading zeros included.", x, 1020, .62)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/audits/packet-replay")
    parser.add_argument("--expect", help="Assert MATCH and this exact ID for every input")
    args = parser.parse_args()
    scanner = Scanner(Detector(MODEL_PATH))
    scanner.preload_ocr()
    if scanner.ocr_error:
        raise RuntimeError(scanner.ocr_error)
    rows = []
    failed = False
    for session in args.sessions:
        source = session / "capture/original.png"
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Cannot read {source}")
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        result = scanner.scan(frame, DETECTION_CONFIDENCE, NMS_IOU_THRESHOLD, BLUR_THRESHOLD)
        destination = args.output / session.name
        destination.mkdir(parents=True, exist_ok=True)
        for name, view in result.views.items():
            if view is not None and view.size:
                if not cv2.imwrite(str(destination / f"{name}.png"), view):
                    raise RuntimeError(f"Could not save view {name}")
        if not cv2.imwrite(str(destination / "result.png"), review_image(result, session.name)):
            raise RuntimeError(f"Could not save result for {session.name}")
        row = {
            "session": session.name, "source": str(source.resolve()),
            "before": manifest.get("result", {}).get("packet", {}),
            "after": {
                "status": result.status, "barcode": result.barcode, "ocr": result.ocr,
                "barcode_format": result.barcode_format, "ocr_confidence": result.ocr_confidence,
                "detection_confidence": result.detection_confidence, "blur_score": result.blur_score,
                "message": result.message, "timings": result.timings,
                "detections": [{"confidence": det.confidence, "points": det.points.tolist()}
                               for det in result.detections],
            },
            "result_image": str((destination / "result.png").resolve()),
        }
        (destination / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(json.dumps(row["after"]), flush=True)
        if args.expect and (result.status != "MATCH" or result.barcode != args.expect or result.ocr != args.expect):
            failed = True
    (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit("At least one session failed --expect; inspect saved results.")


if __name__ == "__main__":
    main()
