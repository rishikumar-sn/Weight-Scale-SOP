"""Audit button-cell confusion without loading or changing learned corrections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Classification.jewelry_classifier import CorrectionGallery, JewelryZeroShotClassifier

def main() -> None:
    output = ROOT / "data/audits/battery_cell_20261013"
    cache_dir = ROOT / "data/audits/foreign_objects_20261012/image_embeddings"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Loading a gallery can migrate/deduplicate it; avoid touching user data.
    with patch.object(CorrectionGallery, "load", lambda self: None):
        classifier = JewelryZeroShotClassifier()
    definitions = classifier.foreign_object_definitions
    vectors = classifier.foreign_object_embeddings
    battery = vectors[list(definitions).index("button_battery")]
    baseline_indices = [n for n, name in enumerate(definitions) if name != "button_battery"]
    baseline_definitions = {name: entry for name, entry in definitions.items()
                            if name != "button_battery"}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    classifier.session = ort.InferenceSession(str(classifier.onnx_model_path),
        sess_options=options, providers=["CPUExecutionProvider"])
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
    torch.set_num_threads(4)
    model = classifier._load_text_model()
    model.eval()
    finger = classifier.group_prompt_config["finger"]
    ring = classifier.prompt_config["classes"]["Finger Ring"]["prompts"]
    with torch.no_grad():
        individual = np.stack([classifier._encode_prompt_set(model, [p]) for p in finger + ring])
    del model
    rows = []
    for session_root in (ROOT / "data/sessions", ROOT / "process_testbench/data/sessions"):
        for manifest in sorted(session_root.glob("*/manifest.json")):
            state = json.loads(manifest.read_text(encoding="utf-8"))
            for item in (state.get("classification") or {}).get("items") or []:
                for kind in ("raw", "working"):
                    path = Path((item.get("paths") or {}).get(kind) or "")
                    if not path.is_file():
                        continue
                    with Image.open(path) as im:
                        vector = classifier.embedding_for_image(im)
                    similarities = vector @ classifier.text_embeddings.T
                    probabilities = classifier._compute_class_probabilities(similarities, vector)
                    object_similarity = float(vector @ battery)
                    classifier.foreign_object_definitions = baseline_definitions
                    classifier.foreign_object_embeddings = vectors[baseline_indices]
                    previously_rejected = classifier._has_strong_non_jewelry_evidence(vector)[0]
                    classifier.foreign_object_definitions = definitions
                    classifier.foreign_object_embeddings = vectors
                    rejected, reason = classifier._has_strong_non_jewelry_evidence(vector)
                    row = dict(session=state["id"], index=item["index"], kind=kind,
                        confirmed=item.get("confirmed_label"), saved=item.get("predicted_label"),
                        model=classifier.labels[int(np.argmax(probabilities))],
                        battery=object_similarity, jewellery=float(np.max(similarities)),
                        margin=object_similarity-float(np.max(similarities)),
                        gold=classifier._group_similarity(vector, "gold_verification"),
                        previously_rejected=previously_rejected, rejected=rejected, reason=reason,
                        new_rejection=rejected and not previously_rejected)
                    if state["id"].startswith("20261013_172"):
                        row["classes"] = dict(zip(classifier.labels, map(float, similarities)))
                        row["groups"] = dict(zip(classifier.group_labels, map(float, vector @ classifier.group_embeddings.T)))
                        ranked = np.argsort(vector @ individual.T)[::-1][:5]
                        row["top_ring_prompts"] = [{"prompt": (finger+ring)[n],
                            "similarity": float(vector @ individual[n])} for n in ranked]
                    rows.append(row)
                    if len(rows) % 200 == 0:
                        print(f"Audited {len(rows)} crops", flush=True)
    (output / "probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        if row["session"].startswith("20261013"):
            print(json.dumps({k:v for k,v in row.items() if k not in {"classes", "groups", "top_ring_prompts"}}), flush=True)
    print(f"Audited {len(rows)} crops; gallery disabled", flush=True)
    # Eight visually reviewed views of the same test battery, not independent
    # battery samples. Other unconfirmed experiments are not accuracy labels.
    battery_sessions = {"20261013_172014_b9e0b064", "20261013_172030_6aea93eb",
        "20261013_172103_2b619648", "20261013_172126_e331131c",
        "20261013_172141_f4465ba7", "20261013_172146_ba8c2975",
        "20261013_172153_0da68e31", "20261013_172215_56b71f78"}
    battery_raw = [r for r in rows if r["session"] in battery_sessions and r["kind"] == "raw"]
    regressions = [r for r in rows if r["new_rejection"] and r["confirmed"]
                   and r["confirmed"] != "Not Gold Jewelry"]
    report = dict(gallery_disabled=True, crops=len(rows),
        confirmed_jewellery_crops=sum(bool(r["confirmed"]) and r["confirmed"] != "Not Gold Jewelry" for r in rows),
        battery_raw_crops=len(battery_raw), battery_raw_rejected=sum(r["rejected"] for r in battery_raw),
        confirmed_jewellery_regressions=regressions,
        changed_crops=[r for r in rows if r["new_rejection"]])
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert len(battery_raw) == 8 and all(r["rejected"] for r in battery_raw)
    assert not regressions, "Battery gate rejected confirmed jewellery"


if __name__ == "__main__":
    main()
