"""Save a review gallery without changing dataset images or saved sessions.

Example:
    python scripts/validate_ring_dimensions.py --images C:/Datasets/Rings \
        --sessions data/sessions --output data/ring-validation/report

Dataset measurements use pixels; session measurements use the saved AprilTag
scale. The gallery is a visual review, not a ground-truth accuracy benchmark.
"""

import argparse
from dataclasses import asdict
import html
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dimension import bangle_detector as detector


def measure(module, image, scale):
    outer, inner, _ = module.finger_ring_circle_detection(image, scale)
    result = {'outer': asdict(outer), 'inner': asdict(inner),
              'od_px': outer.diameter, 'id_px': inner.diameter}
    if scale is not None:
        result.update(od_mm=outer.diameter * scale, id_mm=inner.diameter * scale)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--sessions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Optional previous detector module for comparison')
    parser.add_argument('--reuse-baseline', action='store_true', help='Reuse before measurements from an existing report')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cached = {}
    if args.reuse_baseline:
        previous = json.loads((args.output / 'results.json').read_text())
        cached = {row['source']: row.get('before') for row in previous['images']}
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location('ring_baseline', args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
    entries = [(p, None, 'dataset') for p in sorted(args.images.iterdir())
               if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
               and not p.stem.endswith('_result')]
    if args.sessions:
        for manifest in sorted(args.sessions.glob('*/manifest.json')):
            session = json.loads(manifest.read_text(encoding='utf-8'))
            label = str(session.get('classification', {}).get('confirmed_label', '')).lower()
            if 'finger' not in label or 'ring' not in label:
                continue
            calibration = session.get('calibration') or {}
            x, y = calibration.get('mm_per_pixel_x'), calibration.get('mm_per_pixel_y')
            scale = math.sqrt(x * y) if x and y and x > 0 and y > 0 else None
            entries.append((Path(session['paths']['working']), scale, 'session'))
    rows, cards = [], []
    for index, (path, scale, group) in enumerate(entries):
        image = cv2.imread(str(path))
        row = {'source': str(path), 'group': group, 'scale_mm_per_pixel': scale}
        views = [image.copy(), image.copy()]
        for key, module, view_index in [('before', baseline, 0), ('after', detector, 1)]:
            if module is None:
                continue
            try:
                result = cached.get(str(path)) if key == 'before' else None
                if result is None:
                    result = measure(module, image, scale)
                if 'skipped' in result:
                    raise RuntimeError(result['skipped'])
                row[key] = result
                views[view_index] = module.draw_results(image, result)
            except RuntimeError as exc:
                row[key] = {'skipped': str(exc)}
        accepted = 'outer' in row['after']
        status = 'Measured' if accepted else 'Skipped: insufficient visible boundaries'
        name = path.name if group == 'dataset' else path.parent.parent.name
        links = []
        for title, view in zip(('before', 'after'), views):
            filename = f'{index:03}_{title}.jpg'
            cv2.imwrite(str(args.output / filename), view)
            # Crops make pixel alignment inspectable; full images remain linked.
            if accepted:
                cx, cy = row['after']['outer']['center']
                radius = max(row['after']['outer']['ellipse'][1]) / 2
                half = int(radius + 55)
                crop = view[max(0, int(cy) - half):int(cy) + half,
                            max(0, int(cx) - half):int(cx) + half]
            else:
                crop = view
            thumb = f'{index:03}_{title}_preview.jpg'
            height, width = crop.shape[:2]
            cv2.imwrite(str(args.output / thumb), cv2.resize(crop, (480, round(height * 480 / width))))
            links.append(f'<a href="{filename}"><img src="{thumb}" loading="lazy"><span>{title}</span></a>')
        cards.append(f'<article><h2>{html.escape(name)}</h2><p>{status}</p><div>{"".join(links)}</div></article>')
        rows.append(row)
        print(f'{index + 1}/{len(entries)} {group}: {name}: {status}', flush=True)
    summary = {group: {'total': sum(r['group'] == group for r in rows),
                       'measured': sum(r['group'] == group and 'outer' in r['after'] for r in rows),
                       'skipped': sum(r['group'] == group and 'skipped' in r['after'] for r in rows)}
               for group in ('dataset', 'session')}
    (args.output / 'results.json').write_text(json.dumps({'summary': summary, 'images': rows}, indent=2))
    (args.output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Finger ring dimension review</title>'
        '<style>body{font:16px system-ui;background:#181b20;color:#eee;max-width:1100px;margin:30px auto}'
        'article{border-top:1px solid #555;padding:16px}h2{font-size:16px}div{display:flex;gap:12px}'
        'a{color:#ddd;width:48%}img{width:100%}span{display:block}</style>'
        '<h1>Finger ring dimensions: before / after</h1>'
        '<p>Green: outer band; blue: opening. Click an image for full-frame evidence. '
        'Skipped images have no fitted overlay. Dataset results are in pixels; '
        'session results use saved calibration. Tilted fits describe projected ellipses; '
        'use flat captures for physical sizing. No ground-truth diameter labels were supplied.</p>'
        f'<pre>{html.escape(json.dumps(summary, indent=2))}</pre>' + ''.join(cards), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
