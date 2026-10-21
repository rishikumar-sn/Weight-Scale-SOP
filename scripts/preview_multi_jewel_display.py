"""Recompose real saved multi-jewel captures; no inference or session writes."""
from pathlib import Path
import json
import sys

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.annotations import (
    add_side_panel, identify_result_image, jewel_reference,
    render_bead_evidence, render_capture_review,
)
from StoneDetection.jewel_gem_hsv_report import build_result_gallery


def load(name):
    return json.loads((ROOT / "data/sessions" / name / "manifest.json").read_text())


def main():
    out = ROOT / "WeightScaleSOP_UI_Redesign_Kit/previews_multi"
    out.mkdir(parents=True, exist_ok=True)
    previews = []
    for count, name in [(2, "20261017_164543_d392f801"), (4, "20260928_115814_cd477382"),
                        (5, "20261012_113634_022820ee"), (13, "20260929_154230_7a0c439e")]:
        state = load(name)
        photo = cv2.imread(state["paths"]["original"])
        review = render_capture_review(photo, state, state["classification"]["items"])
        cv2.imwrite(str(out / f"review_{count}_jewels.png"), review)
        if count == 4:
            previews.append(review)
    state = load("20261017_164543_d392f801")
    photo = cv2.imread(state["paths"]["original"])
    for result in state["result"]["items"]:
        index = result["index"]
        item = next(item for item in state["classification"]["items"] if item["index"] == index)
        reference = jewel_reference(item)
        if result.get("beads"):
            beads = result["beads"]
            rendered = render_bead_evidence(photo, beads["detections"], beads, beads["detection_region"])
            rendered = identify_result_image(rendered, reference)
            cv2.imwrite(str(out / f"J{index}_beads.png"), rendered)
            previews.append(rendered)
        if result.get("stones"):
            stones = result["stones"]
            saved = cv2.imread(stones["image"])
            # This historical layout has headers at y=0..251 and an image
            # tile at x=20..659, y=252..672, verified visually before reuse.
            source = saved[252:673, 20:660].copy()
            report = json.loads(Path(stones["report"]).read_text())
            jewel = report["jewels"][0]
            area = stones.get("stone_area_mm2")
            rendered = build_result_gallery([dict(jewel_id=1, overlay_bgr=source,
                detected_colors=jewel["detected_colors"], stone_status=stones["risk_status"])], [
                    ("RESULT", stones["risk_status"]),
                    ("STONE SURFACE AREA", f"{area:.2f} mm²" if area is not None else "Not reported"),
                ], item_reference=reference)
            cv2.imwrite(str(out / f"J{index}_stones.png"), rendered)
            previews.append(rendered)
        if result.get("dimensions", {}).get("applicable") is False:
            dimensions = result["dimensions"]
            rendered = add_side_panel(cv2.imread(item["paths"]["working"]), "DIMENSION ANALYSIS", [
                ("OUTER DIAMETER (OD)", "Unavailable"), ("INNER DIAMETER (ID)", "Unavailable")],
                note=dimensions.get("reason", "Dimensions unavailable"))
            rendered = identify_result_image(rendered, reference)
            cv2.imwrite(str(out / f"J{index}_dimensions_unavailable.png"), rendered)
            previews.append(rendered)
    # A second actual multi-item session demonstrates separate bangle results.
    state = load("20261015_133714_6d06b806")
    for result in state["result"]["items"]:
        item = next(item for item in state["classification"]["items"] if item["index"] == result["index"])
        dimensions = result["dimensions"]
        rendered = add_side_panel(cv2.imread(dimensions["image"]), "DIMENSION ANALYSIS", [
            ("OUTER DIAMETER (OD)", f"{dimensions['outer_diameter_mm']:.2f} mm"),
            ("INNER DIAMETER (ID)", f"{dimensions['inner_diameter_mm']:.2f} mm"),
        ], note="Saved measured boundaries; historical annotation colors retained.")
        rendered = identify_result_image(rendered, jewel_reference(item))
        cv2.imwrite(str(out / f"bangles_J{item['index']}_dimensions.png"), rendered)
        if item["index"] == 2:
            previews.append(rendered)
    sheet = Image.new("RGB", (1536, 512 * ((len(previews) + 1) // 2)), "white")
    for i, preview in enumerate(previews):
        thumb = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        thumb.thumbnail((768, 512), Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((i % 2) * 768, (i // 2) * 512))
    sheet.save(out / "multi_jewel_examples.png")
    (out / "README.txt").write_text(
        "Display previews from real saved 2-, 4-, 5- and 13-jewel captures.\n"
        "All original files, classifications and measurements are unchanged.\n"
        "Stone masks and dimension outlines are reused from historical exports;\n"
        "their original annotation colors/opacity remain visible in these previews.\n"
        "The overall Capture Summary renderer is unchanged.\n"
        "The contact sheet combines different saved sessions to demonstrate layouts.\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
