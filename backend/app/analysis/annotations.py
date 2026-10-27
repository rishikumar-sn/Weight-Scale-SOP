"""Consistent, readable typography for generated evidence images (BGR in/out)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAVY = (20, 30, 42)
GOLD = (198, 151, 66)
WHITE = (248, 250, 252)
MUTED = (182, 194, 208)
GREEN_BGR = (174, 184, 18)
BEAD_SIZE_COLORS_BGR = {
    "tiny": (210, 140, 45),
    "small": (150, 145, 0),
    "large": (30, 120, 235),
}
UNKNOWN_BEAD_BGR = (145, 130, 115)


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False):
    for path in (
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def label_box(image: np.ndarray, bbox, text: str, color=GREEN_BGR,
              occupied: list | None = None) -> None:
    """Keep text independent of object width; clamp and avoid earlier labels."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    thickness = max(1, round(min(h, w) / 750))
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    size = max(12, round(min(h, w) / 45))
    pad = max(4, size // 3)
    while size > 10 and draw.textlength(text, font=font(size)) + pad * 2 > w:
        size -= 1
    lines = textwrap.wrap(text, width=max(1, int((w - pad * 2) / max(1, size * 0.6)))) or [text]
    tw = min(w, int(max(draw.textlength(line, font=font(size)) for line in lines)) + pad * 2 + 1)
    th = len(lines) * (size + 4) + pad * 2
    x = max(0, min(x1, w - tw))
    y = y1 - th - 4 if y1 >= th + 4 else min(h - th, y2 + 4)
    y = max(0, y)
    if occupied is not None:
        def overlaps(left, top):
            return any(left < r[2] and left + tw > r[0] and top < r[3] and top + th > r[1] for r in occupied)
        candidates = [(x, y), (max(0, min(x2 - tw, w - tw)), y)]
        candidates += [(x, max(0, min(h - th, y2 + 4))),
                       (max(0, min(w - tw, x2 + 4)), max(0, min(h - th, y1))),
                       (max(0, x1 - tw - 4), max(0, min(h - th, y1)))]
        nearby = [(left, top) for top in range(0, max(1, h - th + 1), th + 4)
                  for left in range(0, max(1, w - tw + 1), tw + 4)]
        candidates += sorted(nearby, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)
        for left, top in candidates:
            if not overlaps(left, top):
                x, y = left, top
                break
        occupied.append((x, y, x + tw, y + th))
    draw.rounded_rectangle((x, y, x + tw - 1, y + th - 1), radius=3, fill=NAVY,
                           outline=tuple(reversed(color)), width=1)
    for i, line in enumerate(lines):
        draw.text((x + pad, y + pad + i * (size + 4)), line, font=font(size), fill=WHITE, anchor="lt")
    image[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _wrap(draw, text, face, width):
    """Pixel-measured wrapping, including unbroken identifiers."""
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and draw.textlength(candidate, font=face) > width:
                lines.append(line)
                line = ""
            for char in ((" " if line else "") + word):
                if line and draw.textlength(line + char, font=face) > width:
                    lines.append(line)
                    line = ""
                line += char
        lines.append(line)
    return lines or ["Unavailable"]


def display_crop(image: np.ndarray, region: dict | None = None, *, padding: float = .025) -> np.ndarray:
    """Crop a rendering copy only; callers must never feed this to inference.

    Region coordinates remain in the original image space. Clamp padded/negative
    tray bounds; invalid metadata falls back to the complete photograph.
    """
    if not region:
        return image.copy()
    h, w = image.shape[:2]
    x, y, rw, rh = (float(region.get(k, 0)) for k in ("x", "y", "w", "h"))
    if not np.isfinite([x, y, rw, rh]).all() or rw <= 0 or rh <= 0:
        return image.copy()
    margin = max(4, round(max(rw, rh) * padding))
    x1, y1 = max(0, int(np.floor(x)) - margin), max(0, int(np.floor(y)) - margin)
    x2, y2 = min(w, int(np.ceil(x + rw)) + margin), min(h, int(np.ceil(y + rh)) + margin)
    return image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else image.copy()


@lru_cache(maxsize=64)
def review_font(size: int, bold: bool = False):
    """Times New Roman on Windows; bundled metric-compatible serif elsewhere."""
    name = "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf"
    for path in (
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path(__file__).resolve().parents[3] / "assets" / "fonts" / name,
        Path("/usr/share/fonts/truetype/liberation2") / name,
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return font(size, bold)


def add_side_panel(image: np.ndarray, title: str, fields: list[tuple[str, str]],
                   field_colors: dict[str, tuple[int, int, int]] | None = None,
                   *, presence_ok: bool | None = None, note: str | None = None,
                   jewellery_items: list[tuple[int, str]] | None = None,
                   weights: dict | None = None) -> np.ndarray:
    """Typeset evidence on a fixed 1536 x 1024 export, preserving photo zoom.

    All layout fitting happens in the information panel. Source crops, detection
    annotations and the contain-fit/enlargement limits are unchanged.
    """
    width, height, header, split = 1536, 1024, 144, 864
    left, right, top = 916, 1490, 184
    available = right - left
    paper, ink, muted, rule, accent = "#FFFDFA", "#171713", "#626568", "#DDD8D0", "#94632E"
    upper = title.upper()
    stage = 2 if "BEAD" in upper else 3 if "STONE" in upper else 4 if "DIMENSION" in upper else 1
    heading = {2: "Bead detection", 3: "Stone analysis", 4: "Dimension analysis"}.get(stage, title.capitalize())
    rows = [(k, "Unavailable" if v is None else str(v)) for k, v in fields if k.upper() not in {"NOTE", "DATE & TIME"}]
    date = next((str(v) for k, v in fields if k.upper() == "DATE & TIME"), "")
    notes = [str(v) for k, v in fields if k.upper() == "NOTE"]
    if note:
        notes.append(note)
    footer = " ".join(notes) if stage != 4 else ""
    canvas = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(canvas)
    draw.text((44, 42), heading, font=review_font(52, True), fill=ink, anchor="lt")
    if jewellery_items:
        draw.text((44, 108), f"{len(jewellery_items)} pieces on the tray", font=review_font(24), fill=muted, anchor="lt")
    badge = "IMAGE OK" if presence_ok is True else "IMAGE IS NOT PROPER" if presence_ok is False else ""
    if badge:
        badge_ink = accent if presence_ok else "#A23931"
        badge_width = draw.textlength(badge, font=review_font(21, True)) + 28
        draw.rectangle((right - badge_width, 28, right, 66), outline=badge_ink, width=2)
        draw.text((right - badge_width / 2, 47), badge, font=review_font(21, True), fill=badge_ink, anchor="mm")
    if date:
        draw.text((right, 100), date, font=review_font(25), fill=muted, anchor="rt")
    draw.line((0, header, width, header), fill=rule)
    # Keep exactly the existing display bounds, crop scaling and positioning.
    photo = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    photo_height = height - header - 24
    inset = .80 if stage in {2, 3, 4} else 1.
    factor = min(2., (split - 24) * inset / photo.width, photo_height * inset / photo.height)
    if factor != 1:
        photo = photo.resize((round(photo.width * factor), round(photo.height * factor)), Image.Resampling.LANCZOS)
    canvas.paste(photo, (12 + (split - 24 - photo.width) // 2, header + 12 + (photo_height - photo.height) // 2))
    draw.line((split, header, split, height), fill=rule)
    footer_lines = _wrap(draw, footer, review_font(22), available) if footer else []
    bottom = height - 32 - len(footer_lines) * 28 - (24 if footer_lines else 0)
    weight_note_lines = []
    weight_table_top = None
    if weights is not None and stage == 3:
        weight_note_lines = _wrap(draw, weights.get("note") or "", review_font(20), available)
        weight_height = 224 + len(weight_note_lines) * 24
        weight_table_top = bottom - weight_height
        bottom = weight_table_top - 28
    total = None
    if stage == 2:
        try:
            total = max(0, int(next(v for k, v in rows if k in {"DETECTED", "BEADS DETECTED"})))
        except (ValueError, StopIteration):
            pass

    # Measure before painting. Dense trays use tighter table rows, retaining all
    # J references and wrapped labels rather than growing/shrinking the photo.
    for scale in (1., .94, .88, .82, .76, .70, .64, .58, .52):
        small = max(18, round(23 * scale))
        normal = max(22, round(36 * scale))
        leading, padding = round(10 * scale), round(26 * scale)
        prepared = []
        for caption, value in rows:
            category = caption.upper().replace(" BEADS", "")
            table = stage == 2 and category in {"TINY", "SMALL", "LARGE", "UNCLASSIFIED SIZE"}
            items = jewellery_items if caption.startswith("JEWELLERY TYPES") else None
            label = "BEADS DETECTED" if caption == "DETECTED" else caption
            captions = _wrap(draw, label, review_font(small), available)
            if items:
                item_size = max(21, round(29 * scale))
                content = [(i, _wrap(draw, text, review_font(item_size), available - 70)) for i, text in items]
                content_height = sum(len(lines) * (item_size + 5) + round(10 * scale) for _, lines in content)
                value_size = item_size
            else:
                metric = any(word in caption for word in ("WEIGHT", "AREA", "DIAMETER")) or caption in {"DETECTED", "BEADS DETECTED"}
                secondary = caption in {"STONE MASK", "LEARNED REFERENCES", "SURFACE REFLECTION", "OTHER OBJECTS / LOCATION"}
                value_size = max(22, round((68 if metric else 29 if secondary else 40) * scale))
                content = _wrap(draw, value, review_font(value_size), available)
                content_height = len(content) * (value_size + leading)
            row_height = (max(small, normal) + padding * 2 if table else len(captions) * (small + 5) + leading + content_height + padding * 2)
            prepared.append((caption, value, captions, content, value_size, row_height, items, table, category))
        if sum(row[5] for row in prepared) <= bottom - top:
            break
    y = top
    for caption, value, captions, content, value_size, row_height, items, table, category in prepared:
        ty = y
        if table:
            label = f"{category.capitalize()} beads" if category != "UNCLASSIFIED SIZE" else "Unclassified size"
            draw.text((left, ty), label, font=review_font(small + 2), fill=ink, anchor="lt")
            draw.text((right - 105, ty), value, font=review_font(normal), fill=ink, anchor="rt")
            if total is not None:
                try:
                    percent = max(0, int(value)) / total * 100 if total else 0
                    draw.text((right, ty + 5), f"{percent:.1f}%", font=review_font(small), fill=muted, anchor="rt")
                except ValueError:
                    pass
        else:
            for line in captions:
                draw.text((left, ty), line, font=review_font(small), fill=muted, anchor="lt")
                ty += small + 5
            ty += leading
            if items:
                for index, lines in content:
                    draw.text((left, ty), f"J{index}", font=review_font(value_size), fill=accent, anchor="lt")
                    for line in lines:
                        draw.text((left + 70, ty), line, font=review_font(value_size), fill=ink, anchor="lt")
                        ty += value_size + 5
                    ty += round(10 * scale)
                    draw.line((left, ty - 5, right, ty - 5), fill=rule)
            else:
                value_ink = "#A23931" if caption in {"RESULT", "ASSESSMENT"} and "high" in value.lower() else ink
                for line in content:
                    number, separator, unit = line.rpartition(" ")
                    if separator and unit in {"g", "mm", "mm²", "mm2"}:
                        draw.text((left, ty), number, font=review_font(value_size), fill=value_ink, anchor="lt")
                        unit_x = left + draw.textlength(number, font=review_font(value_size)) + 14
                        unit_size = max(18, round(value_size * .52))
                        # Align the smaller unit with the number's baseline.
                        baseline = ty + draw.textbbox((0, 0), number, font=review_font(value_size), anchor="lt")[3]
                        draw.text((unit_x, baseline), unit, font=review_font(unit_size), fill=muted, anchor="lb")
                    else:
                        draw.text((left, ty), line, font=review_font(value_size), fill=value_ink, anchor="lt")
                    ty += value_size + leading
        y += row_height
        if y < bottom - 8:
            draw.line((left, y - padding, right, y - padding), fill=rule)
    if weight_table_top is not None:
        from ..domain.weights import weight_rows
        table_y = weight_table_top
        draw.line((left, table_y, right, table_y), fill=rule)
        draw.text((left, table_y + 18), "JEWELLERY WEIGHTS (GRAMS)", font=review_font(22), fill=muted, anchor="lt")
        table_y += 54
        columns = (left, left + 220, left + 370)
        for row_index, row in enumerate(weight_rows(weights)):
            for column, text in zip(columns, row):
                draw.text((column, table_y), text, font=review_font(21, row_index == 0), fill=muted if row_index == 0 else ink, anchor="lt")
            table_y += 34
            draw.line((left, table_y - 7, right, table_y - 7), fill=rule)
        table_y += 12
        for line in weight_note_lines:
            draw.text((left, table_y), line, font=review_font(20), fill=muted, anchor="lt")
            table_y += 24
    for index, line in enumerate(footer_lines):
        draw.text((left, height - 32 - len(footer_lines) * 28 + index * 28), line, font=review_font(22), fill=muted, anchor="lt")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def render_capture_review(image: np.ndarray, state: dict, items: list[dict], *,
                          confirmed: bool = False, rejected: bool = False) -> np.ndarray:
    """Display-only tray crop, labels and presence badge from existing decisions."""
    from datetime import datetime

    photo = image.copy()
    rois = state.get("rois") or {}
    processing = rois.get("processing") or {}
    bounds = []
    tray = rois.get("testbed_source") or processing
    if tray and tray.get("w", 0) > 0 and tray.get("h", 0) > 0:
        bounds.append((tray["x"], tray["y"], tray["x"] + tray["w"], tray["y"] + tray["h"]))
    labels, occupied = [], []
    for position, item in enumerate(items, 1):
        label = str((item.get("confirmed_label") if confirmed else item.get("predicted_label")) or "Unavailable")
        index = item.get("index", position)
        labels.append((index, label))
        bbox = item.get("bbox") or {}
        if bbox.get("w", 0) > 0 and bbox.get("h", 0) > 0:
            x, y = bbox.get("x", 0) + processing.get("x", 0), bbox.get("y", 0) + processing.get("y", 0)
            box = (x, y, x + bbox["w"], y + bbox["h"])
            label_box(photo, box, f"J{index}", occupied=occupied)
            bounds.append(box)
    anomaly_locations = []
    for number, detection in enumerate((state.get("foreign_object_check") or {}).get("detections") or [], 1):
        if str(detection.get("label", "")).casefold() in {"jewellery", "jewelry", "jewel"}:
            continue
        bbox = detection.get("bbox") or {}
        if bbox.get("w", 0) <= 0 or bbox.get("h", 0) <= 0:
            continue
        # Anomaly boxes already use raw camera coordinates, unlike item crops.
        x, y = bbox["x"], bbox["y"]
        box = (x, y, x + bbox["w"], y + bbox["h"])
        name = str(detection["label"]).replace("_", " ").capitalize()
        label_box(photo, box, f"A{number}: {name}", color=(75, 61, 198), occupied=occupied)
        bounds.append(box)
        tray_x, tray_y = tray.get("x", 0), tray.get("y", 0)
        relative_x = (x + bbox["w"] / 2 - tray_x) / max(1, tray.get("w", image.shape[1]))
        relative_y = (y + bbox["h"] / 2 - tray_y) / max(1, tray.get("h", image.shape[0]))
        horizontal = "left" if relative_x < 1 / 3 else "right" if relative_x > 2 / 3 else "centre"
        vertical = "top" if relative_y < 1 / 3 else "bottom" if relative_y > 2 / 3 else "middle"
        anomaly_locations.append(f"A{number}: {name} — {vertical} {horizontal}")
    bounds.extend(occupied)
    corners = (state.get("calibration") or {}).get("corners") or []
    if corners:
        points = np.asarray(corners).reshape(-1, 2)
        bounds.append((*points.min(axis=0), *points.max(axis=0)))
    if bounds:
        x, y = min(b[0] for b in bounds), min(b[1] for b in bounds)
        region = dict(x=x, y=y, w=max(b[2] for b in bounds) - x, h=max(b[3] for b in bounds) - y)
        photo = display_crop(photo, region)
    known = bool(labels) and all(label != "Unavailable" for _, label in labels)
    non_jewel = any(label.strip().casefold() in {"not gold jewelry", "non-jewellery", "non-jewelry"} for _, label in labels)
    presence_ok = False if rejected or non_jewel else True if known else None
    weight = state.get("weight_g")
    try:
        date = datetime.fromisoformat(str(state.get("captured_at"))).strftime("%d-%m-%Y %I:%M:%S %p")
    except ValueError:
        date = "Unavailable"
    types = labels[0][1] if len(labels) == 1 else "\n".join(f"J{index}  {label}" for index, label in labels) or "Unavailable"
    content = "Jewellery present" if presence_ok is True else "Image is not proper" if presence_ok is False else "Presence not checked"
    fields = [
        ("JEWELLERY TYPE" if len(labels) <= 1 else f"JEWELLERY TYPES / {len(labels)} ITEMS", types),
        ("TOTAL CAPTURED WEIGHT" if len(labels) > 1 else "CAPTURED WEIGHT", f"{float(weight):.2f} g" if weight is not None else "Unavailable"),
        ("DATE & TIME", date), ("IMAGE CONTENT", content),
    ]
    if anomaly_locations:
        fields.append(("OTHER OBJECTS / LOCATION", "\n".join(anomaly_locations)))
    if rejected and state.get("rejection_reason"):
        fields.append(("RECAPTURE", str(state["rejection_reason"])))
    return add_side_panel(photo, "CAPTURE CONFIRMATION" if confirmed else "CAPTURE REVIEW", fields,
                          presence_ok=presence_ok, jewellery_items=labels if len(labels) > 1 else None)


def render_bead_evidence(image: np.ndarray, detections: list[dict], analysis: dict,
                         region: dict | None = None, *, title="BEAD ANALYSIS") -> np.ndarray:
    """Draw completed detections on a copy; do not run or prepare any model."""
    photo = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        color = BEAD_SIZE_COLORS_BGR.get(detection.get("size_category"), UNKNOWN_BEAD_BGR)
        cv2.rectangle(photo, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    # Include entire accepted boxes even when their center lies at an ROI edge.
    if region and detections:
        all_boxes = [d["bbox"] for d in detections]
        x = min(region["x"], *(b[0] for b in all_boxes))
        y = min(region["y"], *(b[1] for b in all_boxes))
        right = max(region["x"] + region["w"], *(b[2] for b in all_boxes))
        bottom = max(region["y"] + region["h"], *(b[3] for b in all_boxes))
        region = dict(x=x, y=y, w=right - x, h=bottom - y)
    photo = display_crop(photo, region)
    size = analysis.get("size") or {}
    fields = [("DETECTED", str(len(detections)))]
    colors = {name.upper(): tuple(reversed(color)) for name, color in BEAD_SIZE_COLORS_BGR.items()}
    if size.get("available") or not detections:
        counts = size.get("counts") or {}
        fields.extend((name.upper(), str(counts.get(name, 0))) for name in ("tiny", "small", "large"))
    else:
        fields.append(("UNCLASSIFIED SIZE", str(len(detections))))
        colors["UNCLASSIFIED SIZE"] = tuple(reversed(UNKNOWN_BEAD_BGR))
        description = (analysis.get("arrangement") or {}).get("description")
        if description:
            fields.append(("ARRANGEMENT", description))
    return add_side_panel(photo, title, fields, field_colors=colors)


def identify_result_image(image: np.ndarray, reference: str | None) -> np.ndarray:
    """Put the capture's J-number/type on a finished export, never on model input."""
    if not reference:
        return image
    output = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(output)
    size = 26
    text = str(reference)
    while size > 18 and draw.textlength(text, font=review_font(size)) > output.width - 100:
        size -= 1
    while text and draw.textlength(text, font=review_font(size)) > output.width - 100:
        text = text[:-2].rstrip() + "…"
    draw.text((output.width - 46, 108), text, font=review_font(size), fill="#626568", anchor="rt")
    return cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)


def jewel_reference(item: dict | None) -> str | None:
    if not item:
        return None
    label = item.get("confirmed_label") or item.get("predicted_label") or item.get("label") or "Jewellery"
    return f"J{item['index']} / {label}"
