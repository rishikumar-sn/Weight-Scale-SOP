"""Replay saved stone captures and expose candidate-budget losses offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import audit_stone_sessions as audit


def replay(session: str, output: Path) -> dict:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    detection = audit.stone_detection
    v2 = detection._stone_v2
    generate = v2.generate_stone_candidates
    serialize = detection.serialize_regions
    analyze = detection.analyze_image_bgr
    rows = []
    target = output / session
    target.mkdir(parents=True, exist_ok=True)
    context = {}

    def candidates(image, jewel, seeds, gold, **kwargs):
        pool = generate(image, jewel, seeds, gold, **kwargs,
                        max_candidates=10000, recover_overflow=False)
        selected = generate(image, jewel, seeds, gold, **kwargs)
        context.update(image=image, jewel=jewel, gold=gold)
        canvas = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for index, candidate in enumerate(pool):
            color = v2.classify_stone_instance_color(image, candidate['seed_mask'], gold_mask=gold)
            rows.append(dict(rank=index + 1, bbox=candidate['bbox'],
                             area=candidate['seed_area_px'], confidence=candidate['confidence'],
                             sources=candidate['source_methods'], color=color['color'],
                             color_confidence=color['color_confidence'],
                             non_gold_chromatic_share=float(np.mean(((hsv[:, :, 1] >= 48) &
                                 (hsv[:, :, 2] >= 24) & (gold == 0))[candidate['seed_mask'] > 0])),
                             shape=v2._shape_metrics(candidate['seed_mask']),
                             metal_surround=v2._metal_surround_ratio(candidate['seed_mask'], gold)))
            x, y, w, h = candidate['bbox']
            if candidate['seed_area_px'] >= 70:
                cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0) if index < 48 else (0, 0, 255), 1)
                cv2.putText(canvas, str(index+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 0, 255), 1)
        cv2.imwrite(str(target / 'candidates.jpg'), canvas)
        cv2.imwrite(str(target / 'prepared.png'), image)
        cv2.imwrite(str(target / 'gold.png'), gold)
        return selected

    def regions(items):
        canvas = context['image'].copy()
        union = np.zeros(canvas.shape[:2], np.uint8)
        for region in items:
            union |= region['mask']
        canvas[union > 0] = (canvas[union > 0] * .5 + np.array([0, 255, 0]) * .5).astype(np.uint8)
        cv2.imwrite(str(target / 'overlay.jpg'), canvas)
        cv2.imwrite(str(target / 'mask.png'), union)
        return serialize(items)

    def result(*args, **kwargs):
        value = analyze(*args, **kwargs)
        (target / 'report.json').write_text(json.dumps(value['report'], indent=2), encoding='utf-8')
        return value

    try:
        v2.generate_stone_candidates = candidates
        detection.serialize_regions = regions
        detection.analyze_image_bgr = result
        path = audit.PROJECT_ROOT / 'data/sessions' / session
        summary = audit.analyze_saved_item(path, path / 'results/stones/item_01')
        (target / 'candidates.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        return summary
    finally:
        v2.generate_stone_candidates = generate
        detection.serialize_regions = serialize
        detection.analyze_image_bgr = analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='+')
    parser.add_argument('--output', type=Path, default=audit.PROJECT_ROOT / 'data/audits/dark-stones/baseline')
    args = parser.parse_args()
    for session in args.sessions:
        print(json.dumps(replay(session, args.output)), flush=True)
