"""Validate complete packet identifiers without guessing OCR characters."""

import re

from .config import ALLOWED_DIGIT_LENGTHS


def packet_identifier(text):
    # OCR can insert spaces around a separator or use a typographic dash.
    # Keep letters, leading zeros, case and separators: O/0 and I/1 are distinct.
    value = re.sub(r"\s+", "", text.strip()).translate(str.maketrans("‐‑‒–—−", "------"))
    if not value.isascii():
        return ""
    if value.isdigit():
        return value if len(value) in ALLOWED_DIGIT_LENGTHS else ""
    # Support new label layouts without tying identifiers to a branch prefix.
    # Pure words and arbitrary OCR prose must not become packet numbers.
    if (7 <= len(value) <= 32
            and re.fullmatch(r"[A-Za-z0-9]+(?:[-/._][A-Za-z0-9]+)*", value)
            and any(char.isalpha() for char in value)
            and any(char.isdigit() for char in value)):
        return value
    return ""
