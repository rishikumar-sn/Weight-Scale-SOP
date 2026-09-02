"""Write side-by-side metrics/artifacts for legacy and non-gold HSV residual stone analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_path in (PROJECT_ROOT, PROJECT_ROOT / "StoneDetection"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import jewel_gem_hsv_report as analysis  # noqa: E402
import stone_analysis_v2 as v2  # noqa: E402


def summarize(result: dict) -> dict:
    report = result["report"]
    return {
        "stone_instance_count": report.get("stone_instance_count"),
        "stone_area_px": report.get("stone_area_px_total"),
        "stone_coverage_percent": report.get("stone_percentage"),
        "stone_area_mm2": (report.get("stone_measurements") or {}).get("total_area_mm2"),
        "risk": report.get("stone_surface_risk"),
        "segmentation_methods": report.get("segmentation_method_counts"),
    }


def run(image, mask, source, scale, residual_enabled: bool) -> dict:
    original = v2._non_gold_color_components
    if not residual_enabled:
        v2._non_gold_color_components = lambda *_args, **_kwargs: []
    try:
        return analysis.analyze_image_bgr(
            image,
            source_name=source,
            zoom_scale=1,
            preset_candidates=analysis.build_candidates_from_component_mask(
                image,
                mask,
                min_area=max(80, int(mask.size * 0.0005)),
                max_candidates=12,
                reject_border_touching=False,
                min_area_ratio_to_largest=0.01,
            ),
            external_mask=mask,
            measurement_scale=scale,
            fastsam_model=None,
            fastsam_lock=None,
        )
    finally:
        v2._non_gold_color_components = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mm-per-pixel-x", type=float)
    parser.add_argument("--mm-per-pixel-y", type=float)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise SystemExit("Could not read image or mask")
    mask = (mask > 0).astype("uint8")
    scale = None
    if args.mm_per_pixel_x and args.mm_per_pixel_y:
        scale = {
            "mm_per_pixel_x": args.mm_per_pixel_x,
            "mm_per_pixel_y": args.mm_per_pixel_y,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = run(image, mask, str(args.image), scale, residual_enabled=False)
    proposed = run(image, mask, str(args.image), scale, residual_enabled=True)
    cv2.imwrite(str(args.output / "baseline.png"), baseline["result_gallery_bgr"])
    cv2.imwrite(str(args.output / "non_gold_hsv_residual.png"), proposed["result_gallery_bgr"])
    payload = {"baseline": summarize(baseline), "non_gold_hsv_residual": summarize(proposed)}
    (args.output / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
