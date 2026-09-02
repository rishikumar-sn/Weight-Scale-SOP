from __future__ import annotations

import json
from pathlib import Path

from backend.app.domain.workflow import class_labels, route_for_label
from Classification.jewelry_classifier import (
    CorrectionGallery,
    canonicalize_jewelry_label,
)
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "Classification" / "jewelry_prompts.json"
COMBINED_LABEL = "Chain / Necklace"
LEGACY_LABELS = {
    "chain",
    "Necklace",
    "Dollar chain",
    "Haram",
    "Kasu Mala",
    "Mangalsutra",
}


def test_prompts_expose_one_combined_neck_class() -> None:
    payload = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    classes = payload["classes"]

    assert COMBINED_LABEL in classes
    assert LEGACY_LABELS.isdisjoint(classes)
    prompts = " ".join(classes[COMBINED_LABEL]["prompts"]).casefold()
    for subtype in ("necklace", "haram", "kasu mala", "dollar chain", "mangalsutra"):
        assert subtype in prompts


def test_combined_label_is_selectable_and_uses_bead_analysis() -> None:
    labels = class_labels(PROMPT_PATH)

    assert COMBINED_LABEL in labels
    assert LEGACY_LABELS.isdisjoint(labels)
    assert route_for_label(COMBINED_LABEL) == {
        "key": "analysis",
        "dimension": False,
        "beads": True,
        "stones": True,
    }


def test_confirmation_choices_use_reduced_production_taxonomy() -> None:
    assert set(class_labels(PROMPT_PATH)) == {
        "Bangle",
        "Bracelet",
        "Chain / Necklace",
        "Earrings / Nosepin",
        "Finger Ring",
        "Mattal",
        "Not Gold Jewelry",
        "Other Gold Jewellery",
    }


def test_historical_labels_are_canonicalized() -> None:
    assert canonicalize_jewelry_label("Haram") == "Chain / Necklace"
    assert canonicalize_jewelry_label("Earing / Jumkha") == "Earrings / Nosepin"
    assert canonicalize_jewelry_label("bracelet") == "Bracelet"
    assert canonicalize_jewelry_label("Finger ring") == "Finger Ring"


def test_gallery_requires_consensus_when_it_disagrees(tmp_path: Path) -> None:
    gallery = CorrectionGallery(tmp_path / "gallery.npz")
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gallery.add(np.array([0.94, 0.341, 0.0, 0.0], dtype=np.float32), "Mattal")
    first = gallery.search(query, expected_label="Chain / Necklace")
    assert first.label is None

    gallery.add(np.array([0.95, 0.0, 0.312, 0.0], dtype=np.float32), "Mattal")
    gallery.add(np.array([0.96, 0.0, 0.0, 0.28], dtype=np.float32), "Mattal")
    result = gallery.search(query, expected_label="Chain / Necklace")
    assert result.label == "Mattal"
    assert result.support >= 3
