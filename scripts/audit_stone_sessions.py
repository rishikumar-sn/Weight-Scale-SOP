"""Re-run saved stone captures without overwriting production artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_path in (PROJECT_ROOT, PROJECT_ROOT / "StoneDetection"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import jewel_gem_hsv_report as stone_detection  # noqa: E402


def _color_counts(report: dict[str, Any]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(region.get("color") or "Unknown")
                for jewel in report.get("jewels") or []
                for region in jewel.get("regions") or []
            ).items()
        )
    )


def _black_region_details(report: dict[str, Any]) -> list[dict[str, Any]]:
    details = []
    for jewel in report.get("jewels") or []:
        for region in jewel.get("regions") or []:
            if region.get("color") != "Black":
                continue
            refinement = region.get("refinement_diagnostics") or {}
            details.append(
                {
                    "bbox": region.get("bbox"),
                    "area_px": region.get("area_px"),
                    "source_methods": region.get("source_methods"),
                    "broad_gold_overlap": refinement.get("gold_overlap"),
                    "strict_gold_overlap": refinement.get("strict_gold_overlap"),
                    "metal_surround_ratio": refinement.get("metal_surround_ratio"),
                    "compactness": refinement.get("compactness"),
                }
            )
    return details


def _resolve_path(session_dir: Path, value: str | None, fallback: Path) -> Path:
    if value:
        path = Path(value)
        if path.exists():
            return path
    return fallback


def analyze_saved_item(session_dir: Path, item_dir: Path) -> dict[str, Any]:
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    stored_path = item_dir / "result.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    item_name = item_dir.name
    item_index = int(item_name.rsplit("_", 1)[-1])
    items = (manifest.get("classification") or {}).get("items") or []
    item = next((entry for entry in items if int(entry.get("index", -1)) == item_index), None)
    if item is None:
        raise RuntimeError(f"{item_name} is absent from the manifest classification")

    paths = item.get("paths") or {}
    image_path = _resolve_path(
        session_dir,
        paths.get("working"),
        session_dir / "items" / f"{item_name}.png",
    )
    mask_path = _resolve_path(
        session_dir,
        paths.get("mask"),
        session_dir / "items" / f"{item_name}_mask.png",
    )
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise RuntimeError(f"could not read {image_path} or {mask_path}")
    mask = (mask > 0).astype(np.uint8)
    candidates = stone_detection.build_candidates_from_component_mask(
        image,
        mask,
        min_area=max(80, int(mask.size * 0.0005)),
        max_candidates=12,
        reject_border_touching=False,
        min_area_ratio_to_largest=0.01,
    )
    candidates = [
        candidate
        for candidate in candidates
        if stone_detection.has_enough_gold_for_stone_detection(
            candidate["crop_bgr"], candidate["crop_mask"]
        )
    ]
    settings = (
        manifest.get("analysis_settings")
        or (manifest.get("settings_snapshot") or {}).get("analysis")
        or {}
    )
    calibration = manifest.get("calibration") or {}
    scale = None
    if calibration.get("available") and calibration.get("found"):
        scale_x = float(calibration["mm_per_pixel_x"])
        scale_y = float(calibration["mm_per_pixel_y"])
        if np.isfinite([scale_x, scale_y]).all() and scale_x > 0 and scale_y > 0:
            scale = {"mm_per_pixel_x": scale_x, "mm_per_pixel_y": scale_y}
    current = stone_detection.analyze_image_bgr(
        image,
        source_name=str(image_path),
        zoom_scale=1,
        preset_candidates=candidates,
        external_mask=mask,
        color_correction=settings.get("color_correction"),
        background_calibration=settings.get("background_calibration"),
        analysis_normalization=settings.get("normalization"),
        measurement_scale=scale,
        learned_stone_profiles=settings.get("learned_stone_profiles"),
        fastsam_model=None,
        fastsam_lock=None,
    )["report"]
    stored_colors = _color_counts(stored)
    current_colors = _color_counts(current)
    return {
        "session": session_dir.name,
        "item": item_name,
        "stored_instance_count": int(stored.get("stone_instance_count") or 0),
        "current_instance_count": int(current.get("stone_instance_count") or 0),
        "stored_colors": stored_colors,
        "current_colors": current_colors,
        "current_black_regions": _black_region_details(current),
        "black_instance_delta": current_colors.get("Black", 0) - stored_colors.get("Black", 0),
        "non_black_changed": {
            key: [stored_colors.get(key, 0), current_colors.get(key, 0)]
            for key in sorted((set(stored_colors) | set(current_colors)) - {"Black"})
            if stored_colors.get(key, 0) != current_colors.get(key, 0)
        },
    }


def _audit_path(result_path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    # Archive replay is parallelized by process; prevent every worker from
    # creating its own full-size OpenCV thread pool.
    cv2.setNumThreads(1)
    session_dir = result_path.parents[3]
    try:
        return analyze_saved_item(session_dir, result_path.parent), None
    except Exception as exc:  # noqa: BLE001 - audit must continue across captures
        return None, {
            "session": session_dir.name,
            "item": result_path.parent.name,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="*", help="Session names; omit to audit all")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "sessions")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    selected = set(args.sessions)
    result_paths = sorted(args.root.glob("*/results/stones/item_*/result.json"))
    if selected:
        result_paths = [path for path in result_paths if path.parents[3].name in selected]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    worker_count = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        outcomes = executor.map(_audit_path, result_paths)
        for index, (result, error) in enumerate(outcomes, start=1):
            result_path = result_paths[index - 1]
            session_dir = result_path.parents[3]
            if result is not None:
                results.append(result)
            if error is not None:
                errors.append(error)
            print(f"[{index}/{len(result_paths)}] {session_dir.name}/{result_path.parent.name}", flush=True)

    payload = {
        "audited_item_count": len(results),
        "error_count": len(errors),
        "items_with_added_black": sum(item["black_instance_delta"] > 0 for item in results),
        "total_added_black_instances": sum(max(0, item["black_instance_delta"]) for item in results),
        "items_with_non_black_changes": sum(bool(item["non_black_changed"]) for item in results),
        "results": results,
        "errors": errors,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
