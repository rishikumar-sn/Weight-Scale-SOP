"""Render saved reference captures without inference or changes to session files.

Historical stone overlays and dimension outlines are reused as rendered pixels:
their raw masks/fitted geometry are not persisted by the existing application.
"""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.annotations import add_side_panel, render_capture_review, render_bead_evidence
from backend.app.services.report_service import ReportService
from StoneDetection.jewel_gem_hsv_report import build_result_gallery


def main():
    output = ROOT / "WeightScaleSOP_UI_Redesign_Kit" / "previews_v2"
    output.mkdir(parents=True, exist_ok=True)
    necklace_path = ROOT / "data/sessions/20261021_113208_6860b238"
    bangle_path = ROOT / "data/sessions/20261021_113848_5815741c"
    necklace = json.loads((necklace_path / "manifest.json").read_text())
    bangle = json.loads((bangle_path / "manifest.json").read_text())
    original = cv2.imread(str(necklace_path / "capture/original.png"))
    items = necklace["classification"]["items"]
    results = necklace["result"]["items"][0]
    beads = results["beads"]
    capture = render_capture_review(original, necklace, items)
    bead = render_bead_evidence(original, beads["detections"], beads, beads["detection_region"])

    # Exact image region from the previously exported, known legacy layout.
    # This retains every saved mask pixel without reconstructing fake masks.
    old_stones = cv2.imread(str(necklace_path / "results/stones/item_01/stones.png"))
    stone_photo = old_stones[:, :675].copy()
    foreground = np.any(stone_photo < 245, axis=2).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(foreground))
    stone_photo = stone_photo[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12].copy()
    stone_data = json.loads((necklace_path / "results/stones/item_01/result.json").read_text())
    jewel = stone_data["jewels"][0]
    stone = build_result_gallery([dict(jewel_id=1, overlay_bgr=stone_photo,
        detected_colors=jewel["detected_colors"], stone_status=results["stones"]["risk_status"])], [
        ("RESULT", results["stones"]["risk_status"]),
        ("STONE SURFACE AREA", f"{results['stones']['stone_area_mm2']:.2f} mm²"),
    ])
    old_dimension = cv2.imread(str(bangle_path / "items/item_01_result.jpg"))
    dimension_photo = old_dimension[:, :520].copy()
    dimensions = bangle["result"]["items"][0]["dimensions"]
    dimension = add_side_panel(dimension_photo, "DIMENSION ANALYSIS", [
        ("OUTER DIAMETER (OD)", f"{dimensions['outer_diameter_mm']:.2f} mm"),
        ("INNER DIAMETER (ID)", f"{dimensions['inner_diameter_mm']:.2f} mm"),
    ], {"OUTER DIAMETER (OD)": (0, 145, 150), "INNER DIAMETER (ID)": (235, 120, 30)},
       note="Dimensions are measured on the detected outer and inner boundaries.")
    images = [("01_capture_review", capture), ("02_bead_analysis", bead),
              ("03_stone_analysis", stone), ("04_dimension_analysis", dimension)]
    for name, image in images:
        cv2.imwrite(str(output / f"{name}.png"), image)
    rejected = render_capture_review(original, necklace, [], rejected=True)
    cv2.imwrite(str(output / "presence_badge_example.png"), rejected)
    (output / "capture_summary_restored.png").write_bytes(ReportService().generate_result_image(necklace).getvalue())
    sheet = Image.new("RGB", (1536, 1024), "white")
    for i, (_, image) in enumerate(images):
        thumb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        thumb.thumbnail((768, 512), Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((i % 2) * 768, (i // 2) * 512))
    sheet.save(output / "00_all_four_designs.png")
    (output / "README.txt").write_text(
        "Real saved reference captures; no inference run and no session file overwritten.\n"
        "Capture and beads rendered from original photograph and saved metadata.\n"
        "Stone preview reuses historical overlay pixels, including their old opacity.\n"
        "New captures use the active 52% fill and 1px contour renderer.\n"
        "Dimension preview reuses historical fitted outlines; no geometry invented.\n"
        "presence_badge_example.png demonstrates the rejected display with the same photo;\n"
        "it is NOT a new classification of that jewellery capture.\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
