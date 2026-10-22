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


def _icon(draw, kind, x, y, color="#243E58", size=80):
    """Small vector-style icons, rasterized with the same renderer as the cards."""
    def box(points):
        return tuple(v * size / 80 + (x if i % 2 == 0 else y) for i, v in enumerate(points))
    def line(points, width=5):
        draw.line([tuple(box(p)) for p in points], fill=color, width=width, joint="curve")
    def oval(bounds, fill=None, width=5):
        draw.ellipse(box(bounds), fill=fill, outline=color, width=width)
    if kind == "jewel":
        draw.arc(box((5, -35, 75, 65)), 0, 180, fill=color, width=5)
        oval((28, 52, 52, 76)); oval((16, 49, 29, 62), width=3); oval((51, 49, 64, 62), width=3)
    elif kind == "weight":
        line([(14, 13), (66, 13)]); line([(40, 13), (40, 29)])
        line([(24, 29), (56, 29), (69, 70), (11, 70), (24, 29)])
        draw.arc(box((26, 37, 54, 62)), 180, 360, fill=color, width=4)
        line([(40, 52), (46, 41)], 4)
    elif kind == "calendar":
        draw.rounded_rectangle(box((12, 16, 68, 72)), radius=5, outline=color, width=5)
        line([(12, 32), (68, 32)]); line([(26, 8), (26, 24)]); line([(54, 8), (54, 24)])
        for dx in (24, 39, 54):
            for dy in (44, 58):
                draw.rectangle(box((dx - 3, dy - 3, dx + 3, dy + 3)), fill=color)
    elif kind == "camera":
        draw.rounded_rectangle(box((6, 23, 74, 69)), radius=7, outline=color, width=5)
        line([(19, 23), (26, 13), (51, 13), (58, 23)])
        oval((26, 32, 54, 60)); oval((61, 30, 64, 33), fill=color, width=1)
    elif kind == "beads":
        for dx, dy in ((40, 14), (16, 39), (64, 39), (40, 65)):
            oval((dx - 10, dy - 10, dx + 10, dy + 10), width=5)
    elif kind == "gem":
        line([(5, 28), (23, 7), (57, 7), (75, 28), (40, 73), (5, 28)])
        line([(5, 28), (75, 28)], 4)
        line([(23, 7), (30, 28), (40, 73), (50, 28), (57, 7)], 3)
    elif kind == "area":
        draw.rounded_rectangle(box((10, 10, 70, 70)), radius=12, outline=color, width=5)
        for a, b in [((21, 19), (33, 19)), ((47, 19), (59, 19)), ((21, 61), (33, 61)), ((47, 61), (59, 61)), ((19, 24), (19, 36)), ((61, 44), (61, 56))]:
            line([a, b], 3)
    elif kind in {"outer", "inner"}:
        line([(7, 40), (73, 40)], 5)
        line([(18, 29), (7, 40), (18, 51)], 5)
        line([(62, 29), (73, 40), (62, 51)], 5)
        if kind == "outer":
            line([(40, 5), (40, 75)])
            line([(29, 16), (40, 5), (51, 16)])
            line([(29, 64), (40, 75), (51, 64)])
        else:
            draw.arc(box((8, 4, 72, 76)), 215, 325, fill=color, width=5)
            draw.arc(box((8, 4, 72, 76)), 35, 145, fill=color, width=5)
    elif kind == "dot":
        oval((21, 21, 59, 59), fill=color, width=1)
    else:
        oval((15, 15, 65, 65), fill=color, width=1)
        draw.text((x + size / 2, y + size / 2), "i", font=font(30, True), fill="white", anchor="mm")


def _card_kind(caption):
    caption = caption.upper()
    for keyword, kind in [("JEWELLERY", "jewel"), ("WEIGHT", "weight"), ("DATE", "calendar"),
                          ("TIME", "calendar"), ("CONTENT", "camera"), ("QUALITY", "camera"),
                          ("DETECTED", "beads"), ("RESULT", "gem"), ("ASSESSMENT", "gem"),
                          ("AREA", "area"), ("OUTER", "outer"), ("INNER", "inner")]:
        if keyword in caption:
            return kind
    return "info"


def add_side_panel(image: np.ndarray, title: str, fields: list[tuple[str, str]],
                   field_colors: dict[str, tuple[int, int, int]] | None = None,
                   *, presence_ok: bool | None = None, note: str | None = None,
                   jewellery_items: list[tuple[int, str]] | None = None) -> np.ndarray:
    """Reference-style 1536 x 1024 *display export*, independent of source size.

    Accepts an already annotated rendering copy. Contain-fit retains all pixels
    of the selected display crop. The model input and its coordinates never pass
    through this layout. Additional rows extend only the exported canvas.
    """
    width, height, header, split = 1536, 1024, 144, 864
    panel_x, right, top, gap = 890, 1508, 176, 18
    text_x, text_right = panel_x + 126, right - 28
    available = text_right - text_x
    title_upper = title.upper()
    stage = 2 if "BEAD" in title_upper else 3 if "STONE" in title_upper else 4 if "DIMENSION" in title_upper else 1
    heading = {1: title_upper, 2: title_upper, 3: "STONE ANALYSIS", 4: "DIMENSION ANALYSIS"}[stage]
    rows = list(fields)
    # Notes are quiet footer text, never a result card. Dimensions need none.
    notes = [str(value) for caption, value in rows if caption.upper() == "NOTE"]
    rows = [(caption, value) for caption, value in rows if caption.upper() != "NOTE"]
    if note:
        notes.append(note)
    footer = " ".join(notes) if stage != 4 else ""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    footer_lines = _wrap(draw, f"Note: {footer}", font(20), right - panel_x - 24) if footer else []
    footer_height = len(footer_lines) * 27 + 24 if footer_lines else 0
    total = None
    if stage == 2:
        total_value = next((v for k, v in fields if k in {"DETECTED", "BEADS DETECTED"}), None)
        try:
            total = max(0, int(total_value))
        except (TypeError, ValueError):
            pass
    item_rows = [(index, _wrap(draw, label, font(34, True), right - panel_x - 132))
                 for index, label in jewellery_items or []]
    prepared = []
    for caption, value in rows:
        value = "Unavailable" if value is None else str(value)
        category = caption.upper().replace(" BEADS", "")
        progress = stage == 2 and category in {"TINY", "SMALL", "LARGE", "UNCLASSIFIED SIZE"}
        kind = "dot" if progress or caption == "STONE MASK" else _card_kind(caption)
        caption_text = "BEADS DETECTED" if caption == "DETECTED" else f"{category} BEADS" if category in {"TINY", "SMALL", "LARGE"} else caption
        caption_lines = _wrap(draw, caption_text, font(30), available)
        secondary = caption in {"DATE & TIME", "NOTE", "STONE MASK", "LEARNED REFERENCES"}
        value_size = 32 if secondary or item_rows else 48
        # Keep the risk phrase and familiar short types on a single line where possible.
        if len(value) <= 26 and not secondary:
            while value_size > 36 and draw.textlength(value, font=font(value_size, True)) > available:
                value_size -= 1
        value_lines = _wrap(draw, value, font(value_size, not secondary), available)
        min_height = 48 + len(caption_lines) * 36 + 10 + len(value_lines) * (value_size + 10) + (32 if progress else 0)
        if item_rows and caption.startswith("JEWELLERY TYPES"):
            min_height = 28 + len(caption_lines) * 36 + 18 + sum(len(lines) * 42 + 12 for _, lines in item_rows) + 20
        prepared.append((caption, caption_lines, value_lines, value_size, secondary, progress, kind, min_height, value))
    natural_height = (height - top - 26 - footer_height - gap * max(0, len(prepared) - 1)) / max(1, len(prepared))
    # Three large, spaced cards or four compact cards, matching the supplied screens.
    heights = [max(row[7], 132 if item_rows else min(244, int(natural_height))) for row in prepared]
    height = max(height, top + sum(heights) + gap * max(0, len(prepared) - 1) + 26 + footer_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for yy in range(header):
        amount = yy / header
        color = tuple(round(a * (1 - amount) + b * amount) for a, b in zip((29, 49, 68), (19, 35, 52)))
        draw.line((0, yy, width, yy), fill=color)
    draw.ellipse((50, 27, 144, 121), fill="#2485F6", outline="#65AEFF", width=2)
    draw.text((97, 71), f"{stage}.", font=font(57), fill="white", anchor="mm")
    badge = "Image OK" if presence_ok is True else "Image is not proper" if presence_ok is False else f"Total Beads: {total}" if total is not None else None
    badge_width = int(draw.textlength(badge, font=font(28, True))) + 70 if badge else 0
    title_size = 54
    while title_size > 30 and draw.textlength(heading, font=font(title_size, True)) > width - 220 - badge_width:
        title_size -= 1
    draw.text((184, 70), heading, font=font(title_size, True), fill="white", anchor="lm")
    if badge:
        bx = width - badge_width - 32
        badge_color = "#0BA87A" if presence_ok is True else "#C63D4B" if presence_ok is False else "#29435E"
        draw.rounded_rectangle((bx, 32, width - 32, 112), radius=24, fill=badge_color)
        draw.text((bx + badge_width / 2, 70), badge, font=font(28, True), fill="white", anchor="mm")
    draw.rounded_rectangle((1, header, width - 2, height - 2), radius=22, fill="#FFFFFF", outline="#DCE5ED", width=2)
    draw.rectangle((split, header + 22, width - 4, height - 24), fill="#FBFCFE")
    photo = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Display resampling only: both enlargement and reduction preserve aspect ratio.
    photo_height = min(height, 1024) - header - 24 if item_rows else height - header - 24
    # Leave breathing room and limit enlargement of small camera crops.
    inset = .80 if stage in {2, 3, 4} else 1.
    factor = min(2., (split - 24) * inset / photo.width, photo_height * inset / photo.height)
    if factor != 1:
        photo = photo.resize((round(photo.width * factor), round(photo.height * factor)), Image.Resampling.LANCZOS)
    canvas.paste(photo, (12 + (split - 24 - photo.width) // 2, header + 12 + (photo_height - photo.height) // 2))
    draw.line((split, header + 8, split, height - 8), fill="#E5EBF1", width=2)
    y = top
    for row, row_height in zip(prepared, heights):
        caption, captions, values, value_size, secondary, progress, kind, _, raw_value = row
        color = (field_colors or {}).get(caption, "#233F5B")
        fill, ink = "#F0F4F8", "#14283E"
        if caption in {"RESULT", "ASSESSMENT"} and "high" in raw_value.lower():
            fill, ink, color = "#FFF0F3", "#D81D42", "#D81D42"
        elif caption == "IMAGE CONTENT":
            color = "#08A57F" if presence_ok is True else "#C63D4B" if presence_ok is False else "#617083"
        draw.rounded_rectangle((panel_x, y, right, y + row_height), radius=24, fill=fill)
        _icon(draw, kind, panel_x + 25, y + 28 if item_rows else y + 44, color, size=44 if item_rows and caption.startswith("JEWELLERY TYPES") else 60 if item_rows else 80)
        ty = y + 28
        for line in captions:
            draw.text((text_x, ty), line, font=font(30), fill=ink if caption == "RESULT" else "#30485F", anchor="lt")
            ty += 36
        ty += 12
        if item_rows and caption.startswith("JEWELLERY TYPES"):
            for index, lines in item_rows:
                draw.rounded_rectangle((panel_x + 26, ty, panel_x + 100, ty + 40), radius=10, fill="#233F5B")
                draw.text((panel_x + 63, ty + 20), f"J{index}", font=font(24, True), fill="white", anchor="mm")
                for line in lines:
                    draw.text((panel_x + 112, ty), line, font=font(34, True), fill=ink, anchor="lt")
                    ty += 42
                ty += 12
            y += row_height + gap
            continue
        for line in values:
            draw.text((text_x, ty), line, font=font(value_size, not secondary), fill=ink, anchor="lt")
            ty += value_size + 10
        if progress:
            try:
                count = max(0, int(raw_value))
                share = count / total if total and total > 0 else None
            except (TypeError, ValueError):
                share = None
            if share is not None:
                percent = f"{share * 100:.1f}%"
                draw.text((text_right, ty - value_size - 10), percent, font=font(28), fill="#30485F", anchor="rt")
            bar_y = y + row_height - 34
            draw.rounded_rectangle((text_x, bar_y, text_right, bar_y + 17), radius=8, fill="#DDE6EE")
            if share is not None and share > 0:
                end = text_x + max(17, round((text_right - text_x) * min(1., share)))
                draw.rounded_rectangle((text_x, bar_y, end, bar_y + 17), radius=8, fill=color)
        y += row_height + gap
    for index, line in enumerate(footer_lines):
        draw.text((panel_x + 12, height - footer_height + index * 27), line,
                  font=font(20), fill="#617083", anchor="lt")
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
    return add_side_panel(photo, "CAPTURE CONFIRMATION" if confirmed else "CAPTURE REVIEW", [
        ("JEWELLERY TYPE" if len(labels) <= 1 else f"JEWELLERY TYPES / {len(labels)} ITEMS", types),
        ("TOTAL CAPTURED WEIGHT" if len(labels) > 1 else "CAPTURED WEIGHT", f"{float(weight):.2f} g" if weight is not None else "Unavailable"),
        ("DATE & TIME", date), ("IMAGE CONTENT", content),
    ], presence_ok=presence_ok, jewellery_items=labels if len(labels) > 1 else None)


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
    while size > 18 and draw.textlength(text, font=font(size)) > output.width - 220:
        size -= 1
    while text and draw.textlength(text, font=font(size)) > output.width - 220:
        text = text[:-2].rstrip() + "…"
    draw.text((184, 108), text, font=font(size), fill="#D1E3F3", anchor="lt")
    return cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)


def jewel_reference(item: dict | None) -> str | None:
    if not item:
        return None
    label = item.get("confirmed_label") or item.get("predicted_label") or item.get("label") or "Jewellery"
    return f"J{item['index']} / {label}"
