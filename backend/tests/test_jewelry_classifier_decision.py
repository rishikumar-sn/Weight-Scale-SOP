from __future__ import annotations

import numpy as np
from PIL import Image

from Classification.jewelry_classifier import GallerySearchResult, JewelryZeroShotClassifier


def test_thin_jewellery_crop_is_not_treated_as_empty() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
    pixels[40:60, 40:60] = (180, 130, 20)

    assert classifier._check_background_empty(Image.fromarray(pixels)) is False
    assert classifier._check_background_empty(Image.new("RGB", (100, 100), "white")) is True


def test_green_rubber_band_is_rejected_by_foreground_colour() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
    pixels[30:70, 25:75] = (70, 145, 45)

    rejected, reason = classifier._check_obvious_non_gold_color(
        Image.fromarray(pixels)
    )

    assert rejected is True
    assert "green-dominant" in reason


def test_small_green_stones_do_not_reject_gold_jewellery() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
    pixels[30:70, 25:75] = (185, 130, 35)
    pixels[40:50, 40:60] = (40, 130, 45)

    rejected, _ = classifier._check_obvious_non_gold_color(Image.fromarray(pixels))

    assert rejected is False


def test_only_strong_non_jewelry_evidence_triggers_rejection() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    classifier._gold_verification_scores = lambda _embedding: (0.017, 0.044, 0.054)
    rejected, _ = classifier._has_strong_non_jewelry_evidence(
        np.array([1.0], dtype=np.float32)
    )
    assert rejected is True

    classifier._gold_verification_scores = lambda _embedding: (0.089, 0.091, 0.052)
    rejected, _ = classifier._has_strong_non_jewelry_evidence(
        np.array([1.0], dtype=np.float32)
    )
    assert rejected is False


def test_non_gold_prompt_result_is_advisory_for_detected_item() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    classifier.labels = ["Chain / Necklace", "Bangle"]
    classifier.text_embeddings = np.eye(2, dtype=np.float32)
    classifier._crop_foreground = lambda image: image
    classifier._embed_image = lambda _image: np.array([1.0, 0.0], dtype=np.float32)
    classifier._compute_class_probabilities = lambda _similarities, _embedding: np.array(
        [0.96, 0.04], dtype=np.float32
    )
    classifier._verify_gold_embedding = lambda _image, _embedding: (
        False,
        0.2,
        "not gold",
    )
    classifier._has_strong_non_jewelry_evidence = lambda _embedding: (
        False,
        "not strong",
    )
    classifier._check_obvious_non_gold_color = lambda _image: (False, "not green")
    classifier.gallery = type(
        "Gallery",
        (),
        {"search": lambda _self, _embedding, expected_label: GallerySearchResult(None, 0.0)},
    )()

    result = classifier.classify_image(Image.new("RGB", (20, 20), "black"))

    assert result.label == "Chain / Necklace"
    assert result.model_label == "Chain / Necklace"
    assert result.decision_source == "siglip"
    assert result.gold_verification_reason.startswith("advisory only")


def test_gallery_is_applied_after_and_preserves_siglip_result_metadata() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    classifier.labels = ["Chain / Necklace", "Bangle"]
    classifier.text_embeddings = np.eye(2, dtype=np.float32)
    classifier._crop_foreground = lambda image: image
    classifier._embed_image = lambda _image: np.array([1.0, 0.0], dtype=np.float32)
    classifier._compute_class_probabilities = lambda _similarities, _embedding: np.array(
        [0.90, 0.10], dtype=np.float32
    )
    classifier._verify_gold_embedding = lambda _image, _embedding: (
        True,
        0.2,
        "gold",
    )
    classifier._has_strong_non_jewelry_evidence = lambda _embedding: (
        False,
        "not strong",
    )
    classifier._check_obvious_non_gold_color = lambda _image: (False, "not green")
    classifier.gallery = type(
        "Gallery",
        (),
        {"search": lambda _self, _embedding, expected_label: GallerySearchResult("Bangle", 0.99, 1, 0.2)},
    )()

    result = classifier.classify_image(Image.new("RGB", (20, 20), "black"))

    assert result.model_label == "Chain / Necklace"
    assert result.label == "Bangle"
    assert result.decision_source == "gallery_correction"


def test_group_probability_does_not_penalize_groups_with_more_classes() -> None:
    classifier = JewelryZeroShotClassifier.__new__(JewelryZeroShotClassifier)
    classifier.group_labels = ["wrist", "finger"]
    classifier.group_index_by_label = {"wrist": 0, "finger": 1}
    classifier.group_to_class_indices = {"wrist": [0, 1], "finger": [2]}
    classifier.group_embeddings = np.eye(2, dtype=np.float32)

    probabilities = classifier._compute_class_probabilities(
        np.array([0.126, 0.099, 0.103], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
    )

    assert int(np.argmax(probabilities)) == 0
