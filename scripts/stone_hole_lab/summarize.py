"""Apply the lab verifier to cached diagnostics and create a review report."""
from collections import Counter
import html
import json
from pathlib import Path

import cv2
import numpy as np

from verifier import evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/audits/stone-hole-lab'


def main():
    rows, changed = [], []
    colors_before, colors_after = Counter(), Counter()
    labels = json.loads(Path(__file__).with_name('review_labels.json').read_text())['items']
    label_checks = []
    for p in sorted((OUT / 'runs').glob('*/diagnostics.json')):
        data = json.loads(p.read_text())
        # Earlier cached runs predate explicit jewel IDs in diagnostics.
        # Recover them from the unmodified replay report before grouping.
        report = json.loads((p.parent / 'baseline_report.json').read_text())
        flat = [(j['jewel_id'], region) for j in report['jewels'] for region in j['regions']]
        if len(flat) != len(data['regions']):
            raise RuntimeError(f"Diagnostic/report region mismatch: {data['key']}")
        for row, (jewel_id, region) in zip(data['regions'], flat):
            if row['bbox'] != region['bbox'] or row['color'] != region['color']:
                raise RuntimeError(f"Diagnostic/report order mismatch: {data['key']}")
            row['jewel_id'] = jewel_id
            row['learned_matches'] = region.get('learned_matches')
        decisions = evaluate(data['regions'], data['registration'])
        rejected = [r for r, d in zip(data['regions'], decisions) if d['decision'] == 'reject_background_hole']
        kept = [r for r, d in zip(data['regions'], decisions) if d['decision'] != 'reject_background_hole']
        colors_before.update(r['color'] for r in data['regions'])
        colors_after.update(r['color'] for r in kept)
        item = {'key': data['key'], 'reference_geometry_compatible': data['registration']['valid'],
                'baseline_count': len(data['regions']), 'retained_count': len(kept),
                'rejected_background_holes': len(rejected),
                'baseline_area_px': sum(r['area_px'] for r in data['regions']),
                'retained_area_px_including_uncertain': sum(r['area_px'] for r in kept),
                'uncertain_white_count': sum(d['decision'] == 'uncertain' for d in decisions),
                'decisions': [dict(jewel_id=r['jewel_id'], region_id=r['region_id'], color=r['color'],
                                   mask_sha256=r['mask_sha256'], **d)
                              for r, d in zip(data['regions'], decisions)]}
        rows.append(item)
        for rid in labels.get(data['key'], {}).get('background_hole_region_ids', []):
            ds = [d for r, d in zip(data['regions'], decisions) if r['region_id'] == rid]
            label_checks.append({'key': data['key'], 'region_id': rid, 'expected': 'reject_background_hole',
                                 'passed': len(ds) == 1 and ds[0]['decision'] == 'reject_background_hole'})
        for rid in labels.get(data['key'], {}).get('true_white_region_ids', []):
            ds = [d for r, d in zip(data['regions'], decisions) if r['region_id'] == rid]
            label_checks.append({'key': data['key'], 'region_id': rid, 'expected': 'preserve',
                                 'passed': len(ds) == 1 and ds[0]['decision'] != 'reject_background_hole'})
        if rejected:
            changed.append(item)
        raw_path = p.parent / 'raw_item.png'
        full_coordinates = raw_path.exists()
        if not full_coordinates:
            raw_path = p.parent / 'raw.png'
        if raw_path.exists() and (rejected or data['key'].startswith('live_')):
            raw = cv2.imread(str(raw_path))
            overlay = raw.copy()
            union = np.zeros(raw.shape[:2], 'uint8')
            for region, decision in zip(data['regions'], decisions):
                mask_name = (f"jewel_{region['jewel_id']:02d}_region_{region['region_id']:03d}.png"
                             if full_coordinates else f"region_{region['region_id']:03d}.png")
                mask_path = p.parent / mask_name
                mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else None
                if mask is None or mask.shape != union.shape:
                    continue
                reject = decision['decision'] == 'reject_background_hole'
                if not reject:
                    union = cv2.bitwise_or(union, mask)
                color = (0, 70, 255) if reject else ((0, 170, 255) if decision['decision'] == 'uncertain' else (0, 190, 0))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 1)
                points = cv2.findNonZero(mask)
                x, y, _, _ = cv2.boundingRect(points)
                cv2.putText(overlay, str(region['region_id']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
            cv2.imwrite(str(p.parent / 'proposal_overlay.jpg'), overlay)
            cv2.imwrite(str(p.parent / 'retained_union.png'), union)
            item['retained_union_area_px'] = int(cv2.countNonZero(union))

    archive = [r for r in rows if not r['key'].startswith('live_')]
    compatible = [r for r in archive if r['reference_geometry_compatible']]
    summary = {'archive_replayed_items': len(archive), 'live_controls': len(rows)-len(archive),
               'reference_geometry_compatible_archive_items': len(compatible),
               'reference_geometry_incompatible_archive_items': len(archive)-len(compatible),
               'archive_items_changed': sum(r['rejected_background_holes'] > 0 for r in archive),
               'rejected_archive_holes': sum(r['rejected_background_holes'] for r in archive),
               'colored_region_counts_preserved': all(colors_before[c] == colors_after[c] for c in colors_before if c != 'White/Colorless'),
               'baseline_colors': dict(colors_before), 'retained_colors': dict(colors_after),
               'review_label_checks': label_checks,
               'integration_ready': False,
               'limitations': ['Pilot thresholds calibrated on the two target images, not an independent accuracy benchmark.',
                              'Incompatible reference geometry disables rejection; unchanged predictions on these items are not a reference safety test.',
                              'Uncertain white candidates retain their original masks and remain included in retained area.',
                              'Latest pendant white ornaments remain missed; a multicolor clasp opening and isolated gaps remain unresolved.',
                              'The broad raw-cleanup ablation removes a reviewed true white center stone and is rejected.']}
    payload = {'summary': summary, 'items': rows}
    (OUT / 'comparison.json').write_text(json.dumps(payload, indent=2))
    cards = []
    for r in rows:
        if r['rejected_background_holes'] or r['key'].startswith('live_'):
            key = html.escape(r['key'])
            has_images = (OUT / 'runs' / r['key'] / 'proposal_overlay.jpg').exists()
            cards.append(f'<article><h2>{key}</h2><p>{r["baseline_count"]} baseline → {r["retained_count"]} retained; '
                         f'{r["rejected_background_holes"]} background holes rejected; '
                         f'{r["uncertain_white_count"]} uncertain white regions retained.</p>'
                         + (f'<div class="pair"><figure><img src="runs/{key}/baseline_overlay.jpg"><figcaption>Current baseline</figcaption></figure>'
                         f'<figure><img src="runs/{key}/proposal_overlay.jpg"><figcaption>Red: rejected hole · orange: uncertain · green: retained</figcaption></figure></div>' if has_images else '') + '</article>')
    document = ('<!doctype html><meta charset="utf-8"><title>Stone hole lab review</title>'
                '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;background:#f3f5f7;color:#172331;padding:0 20px}'
                'article{background:white;padding:20px;margin:24px 0;border-radius:12px}.pair{display:flex;gap:20px}'
                'figure{flex:1;margin:0;min-width:0}img{width:100%;height:auto}pre{white-space:pre-wrap}h2{font-size:18px}</style>'
                '<h1>Isolated stone-hole experiment</h1><p>Main app and saved production results unchanged. '
                'The narrow verifier removes matching repeated link holes; this is a partial result and is not integrated.</p>'
                '<p>Retained area includes uncertain regions. It is not a verified-only stone-area measurement.</p>'
                f'<details><summary>Validation counts and limitations</summary><pre>{html.escape(json.dumps(summary, indent=2))}</pre></details>'
                + ''.join(cards))
    (OUT / 'review.html').write_text(document, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
