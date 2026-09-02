from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DIMENSION_LABELS = {"bangle", "finger ring"}
BEAD_LABELS = {
    "chain necklace",
    # Keep historical records routable even though these are no longer exposed
    # as prediction or confirmation choices.
    "chain",
    "necklace",
    "dollar chain",
    "haram",
    "kasu mala",
    "kasu malai",
    "mangalsutra",
}


def label_key(value: str | None) -> str:
    return " ".join(str(value or "").replace("/", " ").split()).casefold()


def route_for_label(label: str) -> dict[str, Any]:
    key = label_key(label)
    if key == "not gold jewelry":
        return {
            "key": "no_analysis",
            "dimension": False,
            "beads": False,
            "stones": False,
        }
    if key in DIMENSION_LABELS:
        return {
            "key": "dimension",
            "dimension": True,
            "beads": False,
            "stones": False,
        }
    return {
        "key": "analysis",
        "dimension": False,
        "beads": key in BEAD_LABELS,
        "stones": True,
    }


def class_labels(prompt_path: Path) -> list[str]:
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    labels = list((payload.get("classes") or {}).keys())
    additions = ["Other Gold Jewellery", "Not Gold Jewelry"]
    return sorted(set(labels + additions), key=str.casefold)
