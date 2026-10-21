"""Compare current dark-face recovery with a Git revision on saved captures.

Writes only offline audit artifacts, never saved session predictions/settings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import subprocess
import types

import cv2

import audit_stone_sessions as audit


def compare(task):
    path, output, revision = task
    cv2.setNumThreads(1)
    detection = audit.stone_detection
    v2 = detection._stone_v2
    generate, serialize, analyze = v2.generate_stone_candidates, detection.serialize_regions, detection.analyze_image_bgr
    baseline = types.ModuleType('baseline_stone_analysis')
    source = subprocess.check_output(
        ['git', 'show', f'{revision}:StoneDetection/stone_analysis_v2.py'],
        cwd=audit.PROJECT_ROOT, text=True,
    )
    exec(compile(source, '<baseline_stone_analysis>', 'exec'), baseline.__dict__)
    key = path.parents[3].name + '_' + path.parent.name
    target = output / key
    target.mkdir(parents=True, exist_ok=True)
    runs = {}
    try:
        for name, generator in [('before', baseline.generate_stone_candidates), ('after', generate)]:
            cv2.setRNGSeed(42)
            regions = []

            def save_regions(items):
                regions.extend(dict(color=r['color'], area_px=r['area_px'], bbox=r['bbox'],
                                    fingerprint=hashlib.sha256(r['mask'].tobytes()).hexdigest()) for r in items)
                return serialize(items)

            def save_result(*args, **kwargs):
                result = analyze(*args, **kwargs)
                cv2.imwrite(str(target / f'{name}.jpg'), result['result_gallery_bgr'])
                runs[name] = dict(regions=regions, report=result['report'])
                return result

            v2.generate_stone_candidates = generator
            detection.serialize_regions = save_regions
            detection.analyze_image_bgr = save_result
            audit.analyze_saved_item(path.parents[3], path.parent)
        before, after = runs['before']['regions'], runs['after']['regions']
        old = {(r['color'], r['fingerprint']) for r in before}
        new = {(r['color'], r['fingerprint']) for r in after}
        result = dict(session=path.parents[3].name, item=path.parent.name,
                      before_count=len(before), after_count=len(after),
                      before_area_px=sum(r['area_px'] for r in before),
                      after_area_px=sum(r['area_px'] for r in after),
                      changed_or_removed_before_masks=len(old-new),
                      added_regions=[r for r in after if (r['color'], r['fingerprint']) not in old])
        (target / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        v2.generate_stone_candidates, detection.serialize_regions, detection.analyze_image_bgr = generate, serialize, analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='*')
    parser.add_argument('--review-only', action='store_true', help='Render existing comparison.json without replaying')
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, default=audit.PROJECT_ROOT / 'data/audits/dark-stones/comparison')
    args = parser.parse_args()
    paths = [p for p in (audit.PROJECT_ROOT / 'data/sessions').glob('*/results/stones/item_*/result.json')
             if p.parents[3].name in args.sessions]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.review_only:
        results = json.loads((args.output / 'comparison.json').read_text(encoding='utf-8'))
    else:
        if not args.sessions:
            parser.error('Specify session names or --review-only')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(compare, [(p, args.output, args.revision) for p in sorted(paths)]))
    (args.output / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    cards = []
    for row in results:
        key = row['session'] + '_' + row['item']
        cards.append(f'<article><h2>{html.escape(key)}</h2>'
                     f'<p>Accepted regions: {row["before_count"]} → {row["after_count"]}; '
                     f'area: {row["before_area_px"]} → {row["after_area_px"]} pixels; '
                     f'prior masks changed/removed: {row["changed_or_removed_before_masks"]}.</p>'
                     f'<div><figure><figcaption>Before</figcaption><img src="{key}/before.jpg"></figure>'
                     f'<figure><figcaption>After</figcaption><img src="{key}/after.jpg"></figure></div></article>')
    (args.output / 'review.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Dark stone recovery review</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#eef2f5}article{background:white;padding:18px;margin:20px 0}'
        'h2{font-size:18px}div{display:flex}figure{margin:8px;width:50%}img{width:100%}</style>'
        '<h1>Dark stone recovery: saved capture replay</h1>'
        '<p>This comparison measures prediction changes. Region counts are not ground-truth stone counts. '
        'White recovery is unchanged; raw warm-metal and strict-gold rejection still apply.</p>' + ''.join(cards),
        encoding='utf-8',
    )
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != 'added_regions'}))
    if any(r['changed_or_removed_before_masks'] for r in results):
        raise SystemExit(1)
