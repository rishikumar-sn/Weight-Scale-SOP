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
GREEN_BGR = (118, 155, 36)


@lru_cache(maxsize=64)
def font(size: int):
    for path in (
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
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
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    size = max(14, round(h / 28))
    pad = max(6, size // 3)
    while size > 10 and draw.textlength(text, font=font(size)) + pad * 2 > w:
        size -= 1
    lines = textwrap.wrap(text, width=max(1, int((w - pad * 2) / max(1, size * 0.6)))) or [text]
    tw = min(w, int(max(draw.textlength(line, font=font(size)) for line in lines)) + pad * 2 + 1)
    th = len(lines) * (size + 4) + pad * 2
    x = max(0, min(x1, w - tw))
    y = y1 - th - 4 if y1 >= th + 4 else min(h - th, y2 + 4)
    y = max(0, y)
    if occupied is not None:
        while y + th <= h and any(x < r[2] and x + tw > r[0] and y < r[3] and y + th > r[1] for r in occupied):
            y += th + 4
        y = max(0, min(y, h - th))
        occupied.append((x, y, x + tw, y + th))
    draw.rounded_rectangle((x, y, x + tw - 1, y + th - 1), radius=3, fill=NAVY,
                           outline=tuple(reversed(color)), width=1)
    for i, line in enumerate(lines):
        draw.text((x + pad, y + pad + i * (size + 4)), line, font=font(size), fill=WHITE, anchor="lt")
    image[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def add_side_panel(image: np.ndarray, title: str, fields: list[tuple[str, str]],
                   field_colors: dict[str, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Use the space beside a portrait photo; never increase its height.

    Type scales with photo height so it remains readable in a fitted UI card.
    Colors are RGB, matching Pillow and the legend swatches.
    """
    height, width = image.shape[:2]
    panel_width = max(340, round(height * .95))
    panel = Image.new("RGB", (panel_width, height), NAVY)
    draw = ImageDraw.Draw(panel)
    pad = max(8, round(height / 28))
    gap = max(3, round(height / 160))

    def wrap(text, text_font, available):
        lines, line = [], ""
        for word in str(text).split():
            candidate = f"{line} {word}".strip()
            if line and draw.textlength(candidate, font=text_font) > available:
                lines.append(line)
                line = ""
            for character in ((" " if line else "") + word):
                if line and draw.textlength(line + character, font=text_font) > available:
                    lines.append(line)
                    line = ""
                line += character
        return lines + [line]

    size = max(28, round(height / 23))
    while True:
        caption_size = max(10, round(size * .9))
        title_size = round(size * 1.15)
        title_lines = wrap(title, font(title_size), panel_width - pad * 2)
        top = pad + len(title_lines) * (title_size + 8) + pad
        rows = []
        for caption, value in fields:
            color = (field_colors or {}).get(caption)
            caption_lines = wrap(caption, font(caption_size), panel_width - pad * 2 - (caption_size + 12 if color else 0))
            lines = wrap(value, font(size), panel_width - pad * 2)
            rows.append((top, caption_lines, lines, color))
            top += len(caption_lines) * (caption_size + 6) + gap + len(lines) * (size + gap) + pad
        if top <= height - pad or size <= 8:
            break
        size -= 1

    draw.line((0, 0, 0, height), fill=GOLD, width=max(2, round(height / 500)))
    y = pad
    for line in title_lines:
        draw.text((pad, y), line, font=font(title_size), fill=GOLD, anchor="lt")
        y += title_size + 8
    for y, captions, lines, color in rows:
        x = pad
        if color:
            swatch = max(8, round(caption_size * .65))
            draw.rectangle((x, y + 3, x + swatch, y + 3 + swatch), fill=color)
            x += caption_size + 12
        for caption in captions:
            draw.text((x, y), caption, font=font(caption_size), fill=MUTED, anchor="lt")
            y += caption_size + 6
        y += gap
        for line in lines:
            draw.text((pad, y), line, font=font(size), fill=WHITE, anchor="lt")
            y += size + gap
    return np.concatenate((image, cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)), axis=1)
