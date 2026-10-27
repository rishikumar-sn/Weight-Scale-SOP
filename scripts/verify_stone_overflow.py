"""Verify crowded colored stones survive budgeting while metal/glare do not."""
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'StoneDetection'))
import stone_analysis_v2 as v


def main():
    image = np.full((360, 640, 3), 255, np.uint8)
    jewel = np.zeros(image.shape[:2], np.uint8)
    gold = np.zeros_like(jewel)
    seeds = []
    expected = []
    # Sixty independently segmented pink faces, including more than the former
    # 24 overflow slots. Low confidence simulates crowded full-ROI proposals.
    for i in range(60):
        x, y = 20 + (i % 12) * 50, 30 + (i // 12) * 55
        mask = np.zeros_like(jewel)
        cv2.ellipse(mask, (x, y), (13, 9), 0, 0, 360, 255, -1)
        image[mask > 0] = (110, 25, 180)
        jewel |= mask
        expected.append(mask)
        seeds.append(dict(mask=mask, seed_mask=mask, confidence=.65, color='Pink'))
    # High-priority warm metal fragments fill the normal processing budget.
    for i in range(48):
        mask = np.zeros_like(jewel)
        cv2.circle(mask, (5 + i * 13, 320), 2, 255, -1)
        image[mask > 0] = (15, 100, 150)
        jewel |= mask
        gold |= mask
        seeds.insert(0, dict(mask=mask, circularity=1.0, color='Yellow/Gold'))
    negatives = []
    for x, color in [(30, (35, 70, 100)), (90, (225, 225, 225)), (150, (25, 25, 25))]:
        mask = np.zeros_like(jewel)
        cv2.ellipse(mask, (x, 285), (13, 9), 0, 0, 360, 255, -1)
        image[mask > 0] = color
        jewel |= mask
        negatives.append(mask)
        seeds.append(dict(mask=mask, confidence=.62, color='White/Colorless'))
    # Isolate the independent color-seed recovery path from complementary seeds.
    with patch.object(v, '_non_gold_color_components', return_value=[]), \
         patch.object(v, '_non_gold_neutral_components', return_value=[]), \
         patch.object(v, '_structural_components', return_value=[]):
        limited = v.generate_stone_candidates(image, jewel, seeds, gold, recover_overflow=False)
        candidates = v.generate_stone_candidates(image, jewel, seeds, gold)
    assert len(limited) == 48
    recovered = [c for c in candidates if c.get('overflow_recovery')]
    assert len(recovered) == 60, len(recovered)
    union = np.zeros_like(jewel)
    for c in recovered:
        union |= c['seed_mask']
    assert all(np.all(union[m > 0] > 0) for m in expected)
    assert all(not np.any(union[m > 0]) for m in negatives)
    instances, diagnostics = v.build_final_stone_instances(image, jewel, candidates, gold, gold)
    final = np.zeros_like(jewel)
    for instance in instances:
        final |= instance['mask']
    assert all(np.mean(final[m > 0] > 0) >= .95 for m in expected)
    assert diagnostics['fastsam_candidate_calls'] == 0
    # More than eight touching beads must split before the per-face size cap.
    strand = np.zeros((80, 640), np.uint8)
    for i in range(12):
        cv2.ellipse(strand, (30 + i * 50, 40), (17, 12), 0, 0, 360, 255, -1)
    cv2.line(strand, (30, 40), (580, 40), 255, 2)
    strand_image = np.full((*strand.shape, 3), 255, np.uint8)
    strand_image[strand > 0] = (110, 25, 180)
    faces = v._non_gold_color_components(strand_image, strand, np.zeros_like(strand))
    assert len(faces) == 12, len(faces)
    assert all(cv2.countNonZero(face['mask']) < cv2.countNonZero(strand) * .16 for face in faces)
    print('Passed: 60 colored beads recovered; 12 touching beads split; warm metal, white glare and dark shadows excluded.')


if __name__ == '__main__':
    main()
