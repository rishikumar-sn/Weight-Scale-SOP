"""Compare bounded stone recovery with the previous pipeline on saved captures.

Does not change captures, manifests or saved production predictions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2

import audit_stone_sessions as audit


def compare(path: Path, output: Path) -> dict:
    detection = audit.stone_detection
    v2 = detection._stone_v2
    original_generate = v2.generate_stone_candidates
    original_serialize = detection.serialize_regions
    original_analyze = detection.analyze_image_bgr
    runs = {}
    key = path.parents[3].name + "_" + path.parent.name
    try:
        for enabled in (False, True):
            cv2.setNumThreads(1)
            cv2.setRNGSeed(42)
            fingerprints = []
            region_details = []

            def generate(*args, **kwargs):
                return original_generate(*args, **kwargs, recover_overflow=enabled)

            def serialize(regions):
                fingerprints.extend(
                    (r["color"], hashlib.sha256(r["mask"].tobytes()).hexdigest())
                    for r in regions
                )
                region_details.extend({k: r.get(k) for k in ("color", "area_px", "bbox", "overflow_recovery")}
                                      for r in regions)
                return original_serialize(regions)

            def analyze(*args, **kwargs):
                result = original_analyze(*args, **kwargs)
                runs[enabled] = {"report": result["report"], "fingerprints": fingerprints, "regions": region_details}
                if enabled:
                    cv2.imwrite(str(output / (key + ".jpg")), result["result_gallery_bgr"])
                return result

            v2.generate_stone_candidates = generate
            detection.serialize_regions = serialize
            detection.analyze_image_bgr = analyze
            audit.analyze_saved_item(path.parents[3], path.parent)
        previous, current = runs[False], runs[True]
        missing = [fp for fp in previous["fingerprints"] if fp not in current["fingerprints"]]
        return {
            "session": path.parents[3].name,
            "item": path.parent.name,
            "previous_count": previous["report"]["stone_instance_count"],
            "current_count": current["report"]["stone_instance_count"],
            "previous_area_px": sum(j["stone_area_px"] for j in previous["report"]["jewels"]),
            "current_area_px": sum(j["stone_area_px"] for j in current["report"]["jewels"]),
            "previous_masks_changed_or_removed": len(missing),
            "added_regions": [detail for fp, detail in zip(current["fingerprints"], current["regions"])
                              if fp not in previous["fingerprints"]],
            "overflow_candidates": sum(j["stone_analysis_v2"].get("overflow_candidate_count", 0)
                                       for j in current["report"]["jewels"]),
        }
    finally:
        v2.generate_stone_candidates = original_generate
        detection.serialize_regions = original_serialize
        detection.analyze_image_bgr = original_analyze


def _worker(args):
    path, output = args
    try:
        return compare(path, output)
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", help="Session names; omit to compare the full archive")
    parser.add_argument("--output", type=Path, default=audit.PROJECT_ROOT / "data/audits/stone-recovery")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((audit.PROJECT_ROOT / "data/sessions").glob("*/results/stones/item_*/result.json"))
    if args.sessions:
        paths = [p for p in paths if p.parents[3].name in args.sessions]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(_worker, [(p, args.output) for p in paths]):
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(r.get("error") or r.get("previous_masks_changed_or_removed", 0) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
