"""Replay narrow foreign checks without changing sessions or correction galleries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Classification.jewelry_classifier import JewelryZeroShotClassifier


def main() -> None:
    output = ROOT / "data/audits/foreign_objects_20261012"
    cache_dir = output / "image_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    classifier = JewelryZeroShotClassifier()
    # Limit replay CPU contention without changing production inference settings.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    classifier.session = ort.InferenceSession(
        str(classifier.onnx_model_path), sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    embed = classifier._embed_image
    encoder = classifier.onnx_model_path
    identity = f"{encoder.stat().st_size}:{encoder.stat().st_mtime_ns}".encode()

    def cached_embed(image):
        key = hashlib.sha256(image.tobytes() + str(image.size).encode() + identity).hexdigest()
        path = cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path, allow_pickle=False)
        vector = embed(image)
        np.save(path, vector)
        return vector

    classifier._embed_image = cached_embed
    rows = []
    records = []
    for session_root in (ROOT / "data/sessions", ROOT / "process_testbench/data/sessions"):
        for manifest in sorted(session_root.glob("*/manifest.json")):
            state = json.loads(manifest.read_text(encoding="utf-8"))
            for item in (state.get("classification") or {}).get("items") or []:
                paths = item.get("paths") or {}
                if Path(paths.get("working", "")).is_file():
                    records.append((state, item))

    vectors = classifier.foreign_object_embeddings
    for index, (state, item) in enumerate(records, 1):
        previous_raw = current_raw = (None, "raw crop unavailable in saved session")
        raw_path = Path(item["paths"].get("raw", ""))
        if raw_path.is_file():
            with Image.open(raw_path) as image:
                image = image.convert("RGB")
                classifier.foreign_object_embeddings = None
                previous_raw = classifier.check_non_jewelry(image)
                classifier.foreign_object_embeddings = vectors
                current_raw = classifier.check_non_jewelry(image)
        with Image.open(item["paths"]["working"]) as image:
            image = image.convert("RGB")
            classifier.foreign_object_embeddings = None
            previous = classifier.classify_image(image)
            classifier.foreign_object_embeddings = vectors
            current = classifier.classify_image(image)
        rows.append(dict(session=state["id"], index=item["index"],
                         confirmed_label=item.get("confirmed_label"),
                         saved_label=item.get("predicted_label"),
                         previous_label=previous.label, current_label=current.label,
                         previous_raw_rejected=previous_raw[0], raw_rejected=current_raw[0],
                         raw_reason=current_raw[1], reason=current.gold_verification_reason,
                         label_changed=previous.label != current.label,
                         raw_changed=previous_raw[0] != current_raw[0],
                         scores_unchanged=previous.scores == current.scores,
                         model_unchanged=(previous.model_label, previous.model_confidence) ==
                                         (current.model_label, current.model_confidence)))
        if index % 25 == 0:
            print(f"Replayed {index}/{len(records)}", flush=True)
            (output / "progress.json").write_text(json.dumps({
                "processed": index, "total": len(records),
                "changes": [r for r in rows if r["label_changed"] or r["raw_changed"]],
            }, indent=2), encoding="utf-8")

    history = [r for r in rows if "20261012" not in r["session"]]
    changed = [r for r in rows if r["label_changed"] or r["raw_changed"]]
    history_changes = [r for r in history if r["label_changed"] or r["raw_changed"]]
    def outcome(row, previous=False):
        raw = row["previous_raw_rejected" if previous else "raw_rejected"]
        label = row["previous_label" if previous else "current_label"]
        return "rejected" if raw or label == "Not Gold Jewelry" else label

    preserved = all(
        outcome(r, previous=True) == outcome(r)
        or r["saved_label"] == r["current_label"] == "Not Gold Jewelry"
        for r in history_changes
    )
    report = dict(items=len(rows), historical_items=len(history),
                  raw_crops_available=sum(r["raw_rejected"] is not None for r in rows),
                  historical_changes=history_changes,
                  historical_jewellery_decisions_preserved=preserved,
                  model_scores_unchanged=all(r["scores_unchanged"] and r["model_unchanged"] for r in rows),
                  changed_items=changed, rows=rows)
    (output / "replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"rows", "changed_items", "historical_changes"}}, indent=2), flush=True)
    assert report["model_scores_unchanged"], "Jewellery model scores changed"
    assert preserved, "A historical jewellery decision changed"


if __name__ == "__main__":
    main()
