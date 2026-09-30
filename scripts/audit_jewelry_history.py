from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Classification.jewelry_classifier import (  # noqa: E402
    JewelryZeroShotClassifier,
    canonicalize_jewelry_label,
)
from backend.app.services.analysis_service import (  # noqa: E402
    _ring_bangle_dimension_fallback,
)


def confirmed_items(session_root: Path, image_kind: str):
    for manifest_path in sorted(session_root.glob("*/manifest.json")):
        try:
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in (state.get("classification") or {}).get("items") or []:
            paths = item.get("paths") or {}
            image_path = Path(str(paths.get(image_kind) or paths.get("working") or ""))
            confirmed = canonicalize_jewelry_label(item.get("confirmed_label") or "")
            if item.get("confirmed") and confirmed and image_path.is_file():
                yield (
                    state.get("id", manifest_path.parent.name),
                    item,
                    image_path,
                    confirmed,
                    state.get("calibration") or {},
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay confirmed jewellery items with SigLIP and no learned gallery."
    )
    parser.add_argument("--sessions", type=Path, default=PROJECT_ROOT / "data" / "sessions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-kind", choices=("working", "raw"), default="working")
    parser.add_argument(
        "--enable-gallery",
        action="store_true",
        help="Include the production correction gallery in the final prediction.",
    )
    parser.add_argument(
        "--dimension-fallback",
        action="store_true",
        help="Apply the calibrated Finger Ring/Bangle size fallback.",
    )
    args = parser.parse_args()

    records = list(confirmed_items(args.sessions, args.image_kind))
    if args.limit:
        records = records[-args.limit :]

    classifier = JewelryZeroShotClassifier()
    if not args.enable_gallery:
        classifier.gallery.embeddings = None
        classifier.gallery.labels = []

    rows = []
    confusion: Counter[tuple[str, str]] = Counter()
    for index, (session_id, old_item, image_path, expected, calibration) in enumerate(
        records, start=1
    ):
        prediction = classifier.classify_path(image_path)
        predicted = canonicalize_jewelry_label(prediction.label)
        dimension_evidence = None
        if args.dimension_fallback:
            mask_path = Path(str((old_item.get("paths") or {}).get("mask") or ""))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                predicted, dimension_evidence = _ring_bangle_dimension_fallback(
                    predicted,
                    mask,
                    calibration,
                )
        confusion[(expected, predicted)] += 1
        rows.append(
            {
                "session": session_id,
                "item_index": int(old_item.get("index", 0)),
                "image": str(image_path),
                "confirmed": expected,
                "previous_prediction": canonicalize_jewelry_label(
                    old_item.get("predicted_label") or ""
                ),
                "prediction_without_gallery": predicted,
                "siglip_label": prediction.model_label,
                "siglip_confidence": prediction.model_confidence,
                "decision_source": prediction.decision_source,
                "dimension_fallback": dimension_evidence,
                "gold_verification_reason": prediction.gold_verification_reason,
                "correct": predicted == expected,
            }
        )
        if index % 25 == 0:
            print(f"Replayed {index}/{len(records)} items", flush=True)

    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    siglip_correct = sum(
        row["siglip_label"] == row["confirmed"] for row in rows
    )
    false_not_gold = sum(
        row["prediction_without_gallery"] == "Not Gold Jewelry"
        and row["confirmed"] != "Not Gold Jewelry"
        for row in rows
    )
    report = {
        "gallery_disabled": not args.enable_gallery,
        "dimension_fallback_enabled": args.dimension_fallback,
        "image_kind": args.image_kind,
        "items": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "siglip_correct": siglip_correct,
        "siglip_accuracy": siglip_correct / total if total else 0.0,
        "false_not_gold": false_not_gold,
        "confusion": [
            {"confirmed": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in sorted(confusion.items())
        ],
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "items",
                    "correct",
                    "accuracy",
                    "siglip_correct",
                    "siglip_accuracy",
                    "false_not_gold",
                )
            },
            indent=2,
        )
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
