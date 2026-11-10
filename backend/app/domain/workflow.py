from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DIMENSION_LABELS = {"bangle"}
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


THREAD_CLASSIFIER_LABELS = BEAD_LABELS | {
    "mattal", "ear chain", "ear chains", "chain bracelet", "waist chain", "hip chain",
}


def uses_thread_classifier(label: str | None) -> bool:
    """Only explicit chain-type predictions use the provisional thread model."""
    return label_key(label) in THREAD_CLASSIFIER_LABELS


def route_for_label(label: str, *, bangle_orientation: str | None = None) -> dict[str, Any]:
    key = label_key(label)
    if key in {"coin", "bar"}:
        return {"key": "object_review", "dimension": False, "beads": False, "stones": False}
    if key in {"finger ring", "ring"}:
        return {
            "key": "dimension",
            "dimension": True,
            "beads": False,
            "stones": False,
        }
    if key == "not gold jewelry":
        return {
            "key": "no_analysis",
            "dimension": False,
            "beads": False,
            "stones": False,
        }
    if key in DIMENSION_LABELS:
        if bangle_orientation is not None and bangle_orientation != "flat":
            return {
                "key": "analysis",
                "dimension": False,
                "beads": False,
                "stones": True,
            }
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
    additions = ["Other Gold Jewellery", "Not Gold Jewelry", "Coin", "Bar"]
    return sorted(set(labels + additions), key=str.casefold)
