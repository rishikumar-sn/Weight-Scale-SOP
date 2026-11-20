"""Conservative physical-size prior for compact earrings mistaken for chains."""
from __future__ import annotations

import cv2
import numpy as np

MAX_SPAN_MM = 25.0
MAX_DIAGONAL_MM = 30.0
MAX_SIMILARITY_GAP = 0.025


def small_footprint_fallback(label, scores, item_mask, calibration, *, gallery_match=False):
    """Return a label and auditable evidence; never infer size from resized crops.

    A small pendant or coiled chain can share this footprint. Require semantic
    support for earrings, and leave explicit learned examples authoritative.
    """
    evidence = dict(applied=False, available=False, input_label=label, output_label=label)

    def retain(reason):
        evidence['reason'] = reason
        return label, evidence

    if label != 'Chain / Necklace' or gallery_match:
        return retain('not an unlearned chain/necklace prediction')
    calibration = calibration or {}
    if not calibration.get('available') or not calibration.get('found'):
        return retain('AprilTag calibration unavailable')
    try:
        scales = np.array([float(calibration['mm_per_pixel_x']),
                           float(calibration['mm_per_pixel_y'])])
    except (KeyError, TypeError, ValueError):
        return retain('AprilTag calibration scales unavailable')
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        return retain('AprilTag calibration scales invalid')
    mask = np.asarray(item_mask)
    if mask.ndim != 2:
        return retain('item mask invalid')
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return retain('item mask has no foreground')
    physical_points = (points * scales).astype(np.float32)
    _, sides, _ = cv2.minAreaRect(physical_points)
    # Include the pixel extent, rather than measuring only pixel centres.
    width, height = np.asarray(sides) + max(scales)
    span, diagonal = float(max(width, height)), float(np.hypot(width, height))
    evidence.update(available=True, span_mm=span, diagonal_mm=diagonal,
                    max_span_mm=MAX_SPAN_MM, max_diagonal_mm=MAX_DIAGONAL_MM,
                    max_similarity_gap=MAX_SIMILARITY_GAP)
    if span > MAX_SPAN_MM or diagonal > MAX_DIAGONAL_MM:
        return retain('footprint exceeds small-jewellery limits')
    by_label = {score.label: score for score in scores}
    ear = by_label.get('Earrings / Nosepin')
    chain = by_label.get('Chain / Necklace')
    ring = by_label.get('Finger Ring')
    if ear is None or chain is None or ring is None:
        return retain('small-jewellery scores unavailable')
    values = [ear.similarity, chain.similarity, ear.confidence, ring.confidence]
    if not np.isfinite(values).all():
        return retain('small-jewellery scores invalid')
    gap = float(chain.similarity - ear.similarity)
    evidence['similarity_gap'] = gap
    if ear.confidence <= ring.confidence or gap > MAX_SIMILARITY_GAP:
        return retain('insufficient semantic support for earrings/nosepin')
    evidence.update(applied=True, output_label=ear.label,
                    reason='small calibrated footprint with earrings/nosepin support',
                    candidate_confidence=float(ear.confidence))
    return ear.label, evidence
