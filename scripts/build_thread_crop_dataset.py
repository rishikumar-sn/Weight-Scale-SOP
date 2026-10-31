"""Build a resumable jewel/thread classification dataset from saved item crops.

Run with the application's Python environment. Source captures are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.thread_segmentation import ThreadSegmenter, project_thread_mask
from backend.app.domain.workflow import label_key, uses_thread_classifier
from backend.app.analysis.vision import separate_jewellery_items
from StoneDetection.gemstone_colors import original_color_crop

LABELS = {
    "jewel_with_thread": "Jewel_with_Thread",
    "jewel_without_thread": "Jewel_without_Thread",
    "thread_only": "Thread-Only",
    "empty_test_bed": "Empty-Test-Bed",
    "empty_testbed": "Empty-Test-Bed",
}
NEGATIVE_LABELS = {"not gold jewelry", "not gold jewellery", "non jewellery", "unknown", ""}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def resolve_saved(session, value, subfolder, fallback):
    # Prefer the local session, including Linux/Windows manifests moved to a new PC.
    name = Path(str(value or "").replace("\\", "/")).name
    for path in [session / subfolder / name, session / subfolder / fallback, Path(str(value or ""))]:
        if path.is_file():
            return path
    return None


def item_label(item):
    return str(item.get("confirmed_label") or item.get("correction_label") or item.get("predicted_label") or "").strip()


def thread_assignment(overlap, foreground, min_pixels=24, min_share=.005):
    if foreground <= 0:
        return "review", "empty_foreground_mask"
    share = overlap / foreground
    if share >= .8:
        return "review", "mostly_thread_mask_needs_human_jewel_check"
    if overlap >= min_pixels and share >= min_share:
        return "Jewel_with_Thread", "thread_segmentation_intersects_this_jewel"
    if overlap:
        return "review", "weak_thread_overlap"
    return "Jewel_without_Thread", "no_thread_mask_overlap_auto_suggestion"


class DatasetBuilder:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.segmenter = None
        self.records = []
        self.keys = set()
        self.hashes = {}
        self.log_path = self.output / "manifest.jsonl"
        self.state_path = self.output / "state.json"
        config = dict(sessions=str(args.sessions.resolve()), model_sha256=sha256(args.thread_model),
                      seed_dirs=[str(p.resolve()) for p in args.seed_dir], min_thread_pixels=args.min_thread_pixels,
                      min_thread_share=args.min_thread_share, thread_confidence=args.thread_confidence,
                      min_prediction_confidence=args.min_prediction_confidence, version=2)
        self.state = dict(config=config, completed={}, errors={})
        if self.state_path.exists():
            if not args.resume:
                raise RuntimeError("Output already exists. Use --resume or choose a new output folder.")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["config"] != config:
                raise RuntimeError("Dataset options/model changed; choose a new output folder to avoid mixing labels.")
        elif self.log_path.exists():
            raise RuntimeError("Manifest exists without resume state; choose a new output folder.")
        self.save_state()
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("output") and not (self.output / row["output"]).is_file():
                    raise RuntimeError(f"Resume output missing: {row['output']}")
                self.records.append(row)
                self.keys.add(row["key"])
                if row.get("pixel_sha256") and row.get("output"):
                    self.hashes[row["pixel_sha256"]] = row["output"]

    def save_state(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def record(self, row, image=None, folder=None):
        if row["key"] in self.keys:
            return
        if image is not None:
            digest = pixel_hash(image)
            row["pixel_sha256"] = digest
            if digest in self.hashes:
                row["duplicate_of"] = self.hashes[digest]
            else:
                name = row["key"].replace(":", "_").replace("/", "_") + ".png"
                relative = Path(folder) / name
                target = self.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), image):
                    raise RuntimeError(f"Could not save {target}")
                row["output"] = relative.as_posix()
                self.hashes[digest] = row["output"]
        self.records.append(row)
        self.keys.add(row["key"])
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()

    def add_seeds(self):
        for directory in self.args.seed_dir:
            if not directory.is_dir():
                raise FileNotFoundError(f"Reviewed seed directory missing: {directory}")
            for path in sorted(directory.rglob("*")):
                normalized = path.parent.name.lower().replace("-", "_")
                label = LABELS.get(normalized)
                if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not label:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"Could not read reviewed seed: {path}")
                digest = pixel_hash(image)
                key = "seed:" + digest
                self.record(dict(key=key, session=path.stem.split("_item_")[0], source=str(path.resolve()),
                                 assigned_label=label, label_basis="manually_reviewed_seed", needs_label_review=False,
                                 input_variant="raw_item_crop"), image, "images/" + label)

    def thread_mask(self, session, state, original):
        cache = self.output / "_cache/thread_masks" / (session.name + ".png")
        if cache.is_file():
            mask = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == original.shape[:2]:
                return mask
        if self.segmenter is None:
            self.segmenter = ThreadSegmenter(self.args.thread_model)
            self.segmenter.confidence_threshold = self.args.thread_confidence
        mask, _ = self.segmenter.detect(original)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(cache), mask * 255):
            raise RuntimeError("Could not save thread-mask cache")
        return mask

    def process_session(self, path):
        session = path.parent
        fingerprint = sha256(path)
        if session.name in self.state["completed"]:
            if self.state["completed"][session.name] != fingerprint:
                raise RuntimeError("A completed source manifest changed; use a new output folder.")
            return "resumed"
        state = json.loads(path.read_text(encoding="utf-8"))
        classification = state.get("classification") or {}
        items = classification.get("items") or []
        if not items and (classification.get("confirmed_label") or classification.get("predicted_label")):
            working_path = resolve_saved(session, (state.get("paths") or {}).get("working"), "capture", "working.png")
            mask_path = resolve_saved(session, (state.get("paths") or {}).get("mask"), "capture", "mask.png")
            working = cv2.imread(str(working_path)) if working_path else None
            old_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
            if working is not None and old_mask is not None and old_mask.shape == working.shape[:2]:
                components = separate_jewellery_items(working, old_mask)
                for index, component in enumerate(components, 1):
                    item = dict(classification, index=index, bbox=component["bbox"],
                                paths={"working": str(working_path)},
                                _mask_override=component["mask"], _working_override=component["crop_bgr"])
                    if len(components) != 1 or int(classification.get("count") or 1) != 1:
                        item["_forced_review"] = "legacy_multiple_objects_without_per_item_labels"
                    items.append(item)
        original = None
        original_path = resolve_saved(session, (state.get("paths") or {}).get("original"), "capture", "original.png")
        if original_path:
            original = cv2.imread(str(original_path))
        origin = (state.get("rois") or {}).get("processing")
        source_thread = None
        for position, item in enumerate(items, 1):
            index = int(item.get("index", position))
            key = f"{session.name}:item_{index:02d}"
            if key in self.keys:
                continue
            label = item_label(item)
            row = dict(key=key, session=session.name, item_index=index, jewellery_label=label,
                       confirmed=bool(item.get("confirmed_label") or item.get("confirmed")),
                       input_variant="raw_item_crop", needs_label_review=True)
            if label_key(label) in NEGATIVE_LABELS:
                self.record(dict(row, excluded_reason="not_a_jewellery_prediction"))
                continue
            paths = item.get("paths") or {}
            raw_path = resolve_saved(session, paths.get("raw"), "items", f"item_{index:02d}_raw.png")
            working_path = resolve_saved(session, paths.get("working"), "items", f"item_{index:02d}.png")
            mask_path = resolve_saved(session, paths.get("mask"), "items", f"item_{index:02d}_mask.png")
            mask = item.get("_mask_override")
            if mask is None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
            bbox = item.get("bbox") or {}
            ox = int((origin or {}).get("x", 0)) + int(bbox.get("x", 0))
            oy = int((origin or {}).get("y", 0)) + int(bbox.get("y", 0))
            crop = cv2.imread(str(raw_path)) if raw_path else None
            row["source"] = str(raw_path) if raw_path else str(original_path)
            if crop is None and original is not None and origin and mask is not None and bbox:
                crop = original_color_crop(original, ox, oy, mask.shape)
                row["reconstructed_from_original"] = True
            if crop is None:
                # Preserve inaccessible legacy variants for review; never silently
                # mix white-isolated crops with live raw-crop training inputs.
                crop = item.get("_working_override")
                if crop is None:
                    crop = cv2.imread(str(working_path)) if working_path else None
                if crop is None:
                    self.record(dict(row, excluded_reason="no_readable_item_crop"))
                    continue
                row.update(assigned_label="review", label_basis="raw_crop_unavailable", input_variant="otsu_working_crop")
                self.record(row, crop, "review/raw_crop_unavailable")
                continue
            if item.get("_forced_review"):
                row.update(assigned_label="review", label_basis=item["_forced_review"])
                self.record(row, crop, "review/" + item["_forced_review"])
                continue
            blocked = ((state.get("foreign_object_check") or {}).get("passed") is False
                       or (state.get("component_presence_check") or {}).get("passed") is False
                       or state.get("status") == "rejected")
            low_confidence = not row["confirmed"] and float(item.get("confidence") or 0) < self.args.min_prediction_confidence
            if blocked or low_confidence:
                reason = "rejected_or_unverified_capture" if blocked else "low_siglip_confidence"
                row.update(assigned_label="review", label_basis=reason)
                self.record(row, crop, "review/" + reason)
                continue
            if not uses_thread_classifier(label):
                row.update(assigned_label="Jewel_without_Thread", label_basis="non_chain_jewellery_prediction")
            elif original is None or not origin or not bbox or mask is None or mask.shape != crop.shape[:2]:
                row.update(assigned_label="review", label_basis="thread_coordinate_mapping_unavailable")
            else:
                if source_thread is None:
                    source_thread = self.thread_mask(session, state, original)
                local = project_thread_mask(source_thread, mask.shape, ox, oy)
                overlap = int(np.count_nonzero((local > 0) & (mask > 0)))
                foreground = int(np.count_nonzero(mask))
                assigned, basis = thread_assignment(overlap, foreground, self.args.min_thread_pixels, self.args.min_thread_share)
                row.update(assigned_label=assigned, label_basis=basis, thread_overlap_pixels=overlap,
                           foreground_pixels=foreground, thread_overlap_share=round(overlap / max(1, foreground), 6))
            assigned = row["assigned_label"]
            folder = "review/" + row["label_basis"] if assigned == "review" else "images/" + assigned
            self.record(row, crop, folder)
        # Failed old gates may have cleared jewellery labels. Keep their crops
        # for human review; the known-inaccurate classifier is not a training label.
        if not items:
            for item in (state.get("thread_only_check") or {}).get("items") or []:
                index = int(item.get("item_index", 0))
                raw = resolve_saved(session, (item.get("paths") or {}).get("raw"), "items", f"item_{index:02d}_raw.png")
                crop = cv2.imread(str(raw)) if raw else None
                if crop is not None:
                    self.record(dict(key=f"{session.name}:unlabelled_{index:02d}", session=session.name,
                                     item_index=index, source=str(raw), assigned_label="review", needs_label_review=True,
                                     label_basis="gate_prediction_not_ground_truth", old_gate_label=item.get("label")),
                                crop, "review/unlabelled_gate_crops")
        self.state["completed"][session.name] = fingerprint
        self.state["errors"].pop(session.name, None)
        self.save_state()
        return "processed"

    def summarize(self, status):
        outputs = [r for r in self.records if r.get("output")]
        result = dict(status=status, output=str(self.output), completed_sessions=len(self.state["completed"]),
                      labelled_images=dict(Counter(r["assigned_label"] for r in outputs if r.get("assigned_label") != "review")),
                      review_images=sum(r.get("assigned_label") == "review" for r in outputs),
                      exact_pixel_duplicates=sum("duplicate_of" in r for r in self.records),
                      excluded=Counter(r["excluded_reason"] for r in self.records if "excluded_reason" in r),
                      errors=self.state["errors"], source_item_records=len(self.records))
        (self.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        import csv
        fields = ["output", "assigned_label", "jewellery_label", "session", "item_index", "confirmed", "needs_label_review", "label_basis", "thread_overlap_share", "source", "duplicate_of"]
        with (self.output / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        from html import escape
        cards = []
        for row in outputs:
            path = escape(row["output"], quote=True)
            caption = escape(str(row.get("jewellery_label") or row.get("assigned_label")))
            basis = escape(str(row.get("label_basis", "")))
            cards.append(f'<article><a href="{path}"><img loading="lazy" src="{path}"></a><b>{caption}</b><small>{escape(row["assigned_label"])}<br>{basis}<br>{escape(row["session"])}</small></article>')
        html = ('<!doctype html><meta charset="utf-8"><title>Thread dataset review</title>'
                '<style>body{font:16px system-ui;background:#eef2f6;margin:24px}main{display:flex;flex-wrap:wrap;gap:16px}article{width:220px;padding:12px;background:white}img{width:220px;height:200px;object-fit:contain}b,small{display:block;margin-top:8px;overflow-wrap:anywhere}</style>'
                '<h1>Jewel/thread cropped dataset</h1><p>Automatic labels are suggestions. Review before training; folder names are in labels.csv.</p><main>'
                + ''.join(cards) + '</main>')
        (self.output / "review.html").write_text(html, encoding="utf-8")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=ROOT / "data/sessions")
    parser.add_argument("--output", type=Path, default=ROOT / "data/datasets/jewel-thread-crops-20261031")
    parser.add_argument("--thread-model", type=Path, default=ROOT / "models/segmentation/thread.onnx")
    parser.add_argument("--seed-dir", type=Path, action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-id", action="append", help="Optional session names for a smoke check")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 processes every session")
    parser.add_argument("--thread-confidence", type=float, default=.5)
    parser.add_argument("--min-thread-pixels", type=int, default=24)
    parser.add_argument("--min-thread-share", type=float, default=.005)
    parser.add_argument("--min-prediction-confidence", type=float, default=.5)
    args = parser.parse_args()
    args.seed_dir = args.seed_dir or [ROOT / "data/datasets/jewel-thread-reviewed-seeds-20261031"]
    if not args.sessions.is_dir():
        parser.error("Session directory is missing")
    if not all(0 <= value <= 1 for value in (args.thread_confidence, args.min_thread_share, args.min_prediction_confidence)):
        parser.error("Confidence/share must be between zero and one")
    if args.min_thread_pixels < 1 or args.max_sessions < 0:
        parser.error("Thread pixel count must be positive; max sessions cannot be negative")
    if args.output.resolve().is_relative_to(args.sessions.resolve()):
        parser.error("Output must be outside source sessions")
    builder = DatasetBuilder(args)
    paths = sorted(args.sessions.glob("*/manifest.json"))
    if args.session_id:
        paths = [path for path in paths if path.parent.name in set(args.session_id)]
    if args.max_sessions:
        paths = paths[:args.max_sessions]
    start = time.monotonic()
    try:
        builder.add_seeds()
        for number, path in enumerate(paths, 1):
            try:
                action = builder.process_session(path)
            except Exception as exc:
                action = "ERROR: " + str(exc)
                builder.state["errors"][path.parent.name] = str(exc)
                builder.save_state()
            print(f"[{number}/{len(paths)}] {path.parent.name}: {action}; elapsed {time.monotonic()-start:.0f}s", flush=True)
            if number % 10 == 0:
                builder.summarize("running")
    except KeyboardInterrupt:
        builder.summarize("interrupted_resume_available")
        print("Interrupted. Run the same command with --resume to continue.", flush=True)
        raise SystemExit(130)
    status = "complete_with_errors" if builder.state["errors"] else "complete"
    result = builder.summarize(status)
    (builder.output / "DONE.txt").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DATASET READY: " + str(builder.output), flush=True)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(1 if builder.state["errors"] else 0)


if __name__ == "__main__":
    main()
