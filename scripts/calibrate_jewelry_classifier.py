from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Classification.jewelry_classifier import (
    JewelryZeroShotClassifier,
    canonicalize_jewelry_label,
)


DATASET_CLASSES = {
    "Bangle": ("Bangle", "Bangle"),
    "Earings": ("Earings", "Earrings / Nosepin"),
    "Mattal": ("Mattal", "Mattal"),
    "Necklace": ("Necklace", "Chain / Necklace"),
    "Rings": ("Rings", "Finger Ring"),
}
SUPPORTED_DATASET_LABELS = {definition[1] for definition in DATASET_CLASSES.values()}


@dataclass(frozen=True)
class DatasetRecord:
    capture_id: str
    expected_label: str
    roi_path: str
    full_path: str | None
    split: str


def _capture_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_roi", "_full"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _split(capture_id: str, calibration_percent: int, seed: str) -> str:
    value = int(hashlib.sha256(f"{seed}:{capture_id}".encode()).hexdigest()[:8], 16) % 100
    return "calibration" if value < calibration_percent else "validation"


def discover_dataset(
    root: Path,
    calibration_percent: int,
    seed: str,
) -> tuple[list[DatasetRecord], list[dict[str, str]]]:
    records: list[DatasetRecord] = []
    rejected: list[dict[str, str]] = []
    for folder, (expected_prefix, label) in DATASET_CLASSES.items():
        folder_path = root / folder
        if not folder_path.is_dir():
            rejected.append({"path": str(folder_path), "reason": "missing class folder"})
            continue
        by_capture: dict[str, dict[str, Path]] = {}
        for path in sorted(folder_path.glob("*.png")):
            capture_id = _capture_id(path)
            prefix = capture_id.split("_", 1)[0]
            if prefix != expected_prefix:
                rejected.append({
                    "path": str(path),
                    "reason": f"folder/prefix mismatch: {folder} versus {prefix}",
                })
                continue
            kind = "roi" if path.stem.endswith("_roi") else "full" if path.stem.endswith("_full") else "unknown"
            if kind == "unknown":
                rejected.append({"path": str(path), "reason": "filename lacks _roi or _full suffix"})
                continue
            by_capture.setdefault(capture_id, {})[kind] = path
        for capture_id, paths in sorted(by_capture.items()):
            if "roi" not in paths:
                rejected.append({"path": str(paths.get("full", folder_path)), "reason": "capture lacks ROI image"})
                continue
            records.append(DatasetRecord(
                capture_id=capture_id,
                expected_label=label,
                roi_path=str(paths["roi"]),
                full_path=str(paths["full"]) if "full" in paths else None,
                split=_split(capture_id, calibration_percent, seed),
            ))
    return records, rejected


def historical_records(session_root: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for manifest_path in sorted(session_root.glob("*/manifest.json")):
        try:
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in (state.get("classification") or {}).get("items") or []:
            label = canonicalize_jewelry_label(str(item.get("confirmed_label") or ""))
            path = Path(str((item.get("paths") or {}).get("working") or ""))
            if label not in SUPPORTED_DATASET_LABELS or not path.is_file():
                continue
            records.append(DatasetRecord(
                capture_id=f"{manifest_path.parent.name}:item_{item.get('index', 0)}",
                expected_label=label,
                roi_path=str(path),
                full_path=None,
                split="historical_replay",
            ))
    return records


def add_calibration_gallery(
    classifier: JewelryZeroShotClassifier,
    records: Iterable[DatasetRecord],
) -> int:
    added = 0
    for index, record in enumerate(records, start=1):
        # ROI matches production; the paired full frame adds robustness to the
        # compact tray and AprilTag without becoming a second validation sample.
        paths = [record.roi_path]
        if record.full_path:
            paths.append(record.full_path)
        for path in paths:
            before = len(classifier.gallery.labels)
            classifier.gallery.add(
                classifier.embedding_for_path(path),
                record.expected_label,
                save=False,
            )
            added += len(classifier.gallery.labels) - before
        if index % 25 == 0:
            print(f"Embedded {index} calibration captures", flush=True)
    classifier.gallery.save()
    return added


def evaluate(
    classifier: JewelryZeroShotClassifier,
    records: Iterable[DatasetRecord],
    image_kind: str = "roi",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    source: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        image_path = record.full_path if image_kind == "full" and record.full_path else record.roi_path
        prediction = classifier.classify_path(image_path)
        predicted = canonicalize_jewelry_label(prediction.label)
        confusion[(record.expected_label, predicted)] += 1
        source["gallery" if prediction.gallery_match else "siglip"] += 1
        rows.append({
            **asdict(record),
            "evaluated_path": image_path,
            "predicted_label": predicted,
            "correct": predicted == record.expected_label,
            "confidence": prediction.confidence,
            "gallery_match": prediction.gallery_match,
            "gallery_similarity": prediction.gallery_similarity,
            "gallery_support": prediction.gallery_support,
            "gallery_margin": prediction.gallery_margin,
            "top_scores": [asdict(score) for score in prediction.scores[:3]],
        })
        if index % 25 == 0:
            print(f"Evaluated {index} captures", flush=True)
    labels = sorted({label for pair in confusion for label in pair}, key=str.casefold)
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        total = sum(count for (truth, _), count in confusion.items() if truth == label)
        correct = confusion[(label, label)]
        predicted_total = sum(count for (_, predicted), count in confusion.items() if predicted == label)
        per_class[label] = {
            "samples": total,
            "correct": correct,
            "recall": correct / total if total else 0.0,
            "precision": correct / predicted_total if predicted_total else 0.0,
        }
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "decision_source": dict(source),
        "per_class": per_class,
        "confusion": [
            {"expected": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in sorted(confusion.items())
        ],
        "predictions": rows,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Calibrate and evaluate the jewellery classifier.")
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--calibration-percent", type=int, default=70)
    value.add_argument("--seed", default="weight-integration-v1")
    value.add_argument("--build-gallery", action="store_true")
    value.add_argument("--disable-gallery", action="store_true")
    value.add_argument("--historical-root", type=Path)
    value.add_argument("--image-kind", choices=("roi", "full"), default="roi")
    return value


def main() -> None:
    args = parser().parse_args()
    records, rejected = discover_dataset(args.dataset_root, args.calibration_percent, args.seed)
    calibration = [record for record in records if record.split == "calibration"]
    validation = [record for record in records if record.split == "validation"]
    classifier = JewelryZeroShotClassifier()
    gallery_added = add_calibration_gallery(classifier, calibration) if args.build_gallery else 0
    if args.disable_gallery:
        classifier.gallery.embeddings = None
        classifier.gallery.labels = []
    report: dict[str, object] = {
        "dataset_root": str(args.dataset_root),
        "split_seed": args.seed,
        "calibration_percent": args.calibration_percent,
        "clean_unique_captures": len(records),
        "calibration_captures": len(calibration),
        "validation_captures": len(validation),
        "rejected_files": rejected,
        "gallery_entries_added": gallery_added,
        "evaluated_image_kind": args.image_kind,
        "validation": evaluate(classifier, validation, args.image_kind),
    }
    if args.historical_root:
        replay = historical_records(args.historical_root)
        report["historical_replay"] = evaluate(classifier, replay, "roi")
        report["historical_replay_note"] = (
            "Diagnostic replay only; confirmed historical images can overlap the correction gallery."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["validation"]
    print(json.dumps({key: summary[key] for key in ("samples", "correct", "accuracy", "decision_source", "per_class")}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
