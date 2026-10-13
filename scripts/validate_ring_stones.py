"""Replay dimension-rejected ring captures and save stone evidence for review.

Uses the same segmentation and stone service as production. Source captures
and saved sessions are never overwritten. Optional labels map source filenames
to true (stones present) or false (stone-free) for a scored regression audit.
"""
import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.vision import build_jewellery_mask
from backend.app.services.analysis_service import AnalysisService


class Repository:
    def __init__(self, root):
        self.root = root

    def session_dir(self, capture_id):
        return self.root / capture_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=Path, default=ROOT / "data/ring-validation/report/results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ring-validation/stones")
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    source_rows = json.loads(args.dimensions.read_text())["images"]
    sources = [Path(row["source"]) for row in source_rows
               if row["group"] == "dataset" and "skipped" in row["after"]
               and "_roi" in Path(row["source"]).stem]
    args.output.mkdir(parents=True, exist_ok=True)
    service = AnalysisService(Repository(args.output))
    cv2.setNumThreads(1)
    rows, cards = [], []
    try:
        for index, path in enumerate(sources):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Cannot read {path}")
            # The dataset ROI includes its marker; production erases the
            # configured marker area before segmenting jewellery as well.
            erase = np.zeros(image.shape[:2], np.uint8)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            corners, _, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
            for corner in corners:
                x, y, w, h = cv2.boundingRect(corner[0])
                # Erase the printed border/label as well as the tag's coded
                # square, matching the larger configured marker ROI.
                padding = int(max(w, h) * .40)
                cv2.rectangle(erase, (max(0, x-padding), max(0, y-padding)),
                              (x+w+padding, y+h+padding), 255, -1)
            prepared, mask = build_jewellery_mask(image, erase_mask=erase)
            run_dir = args.output / f"{index:03}"
            run_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(run_dir / "input.png"), image)
            cv2.imwrite(str(run_dir / "mask.png"), mask * 255)
            cv2.imwrite(str(run_dir / "prepared.png"), prepared)
            state = {"id": f"{index:03}", "classification": {"confirmed_label": "Finger Ring"},
                     "paths": {"working": str(run_dir / "input.png"),
                               "mask": str(run_dir / "mask.png"),
                               "prepared": str(run_dir / "prepared.png")}}
            result = service._run_stones(state)
            report = json.loads(Path(result["report"]).read_text())
            row = {"source": str(path), "result": result,
                   "instance_count": report.get("stone_instance_count"),
                   "expected_stones": labels.get(path.name)}
            rows.append(row)
            gallery = Path(result["image"]).relative_to(args.output).as_posix()
            cards.append(f'<article><h2>{html.escape(path.name)}</h2><p>{html.escape(result["risk_status"])}; '
                         f'{row["instance_count"]} regions</p><div><img src="{index:03}/input.png">'
                         f'<a href="{gallery}"><img src="{gallery}"></a></div></article>')
            print(f"{index + 1}/{len(sources)} {path.name}: {result['risk_status']}, {row['instance_count']} regions", flush=True)
    finally:
        service.shutdown()
    negatives = [r for r in rows if r["expected_stones"] is False]
    positives = [r for r in rows if r["expected_stones"] is True]
    summary = {"captures": len(rows), "detections": sum(r["result"]["found"] for r in rows),
               "labelled_stone_free": len(negatives),
               "false_positives": sum(r["result"]["found"] for r in negatives) if negatives else None,
               "labelled_stone_present": len(positives),
               "false_negatives": sum(not r["result"]["found"] for r in positives) if positives else None}
    (args.output / "results.json").write_text(json.dumps({"summary": summary, "images": rows}, indent=2))
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Ring stone audit</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:auto}article{border-top:1px solid #aaa}'
        'div{display:flex;align-items:flex-start}div>img,div>a{width:49%}a>img{width:100%}h2{font-size:16px}</style>'
        '<h1>Dimension-unavailable ring stone audit</h1><p>Unlabelled captures are visual review only. '
        'Supply --labels to score false detections. No AprilTag scale is available for these dataset images.</p>'
        f'<pre>{html.escape(json.dumps(summary, indent=2))}</pre>' + ''.join(cards), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
