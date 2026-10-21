"""Read-only reflection experiment: compare stones and plain-metal controls.

Highlights are evidence of reflectivity, not material identity or stone area.
This does not modify the stone detector or integrate a production feature.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'StoneDetection')]
import jewel_gem_hsv_report as detection
import stone_area_calculator as area

OUT = ROOT / 'data/audits/reflection-lab'
HOLE_LAB = ROOT / 'data/audits/stone-hole-lab'

CASES = [
    ('20260929_154815_73d25812', 'item_01', 'plain_gold_chain'),
    ('20260925_151711_eecfb85e', 'item_01', 'plain_gold_chain'),
    ('20260923_195643_5e807e2e', 'item_01', 'plain_gold_bracelet'),
    ('20260826_130513_e9835a94', 'item_01', 'faceted_white_stones'),
    ('20260904_112410_79db0245', 'item_02', 'white_stones_and_pearls'),
    ('20261021_134420_a434dc32', 'item_01', 'missed_pale_ornaments'),
    ('live_white_control', 'item_01', 'live_white_and_colored_stones'),
]


def highlight_mask(image, foreground, floor=180, contrast=20, neutral=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    local = cv2.GaussianBlur(value, (0, 0), 3)
    selected = (foreground > 0) & (value >= floor) & (value-local >= contrast)
    if neutral:
        selected &= hsv[:, :, 1] <= 90
    _, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype('uint8'), connectivity=8)
    sizes = stats[:, cv2.CC_STAT_AREA]
    keep = sizes >= 3
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype('uint8')


def veto_gold_context(image, foreground, highlights, window=11, maximum_gold=.4):
    """Ablation only: surrounding gold color is not a material classifier."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold = detection.build_gold_mask(hsv, foreground, strict=True)
    surface = (foreground > 0) & (highlights == 0)
    n = cv2.boxFilter(surface.astype('float32'), -1, (window, window), normalize=False)
    gold_n = cv2.boxFilter(((gold > 0) & surface).astype('float32'), -1, (window, window), normalize=False)
    ratio = gold_n / np.maximum(n, 1)
    return np.where((highlights > 0) & (n >= 10) & (ratio < maximum_gold), 255, 0).astype('uint8')


def sample_masks(session, item, image):
    samples = {}
    if session == 'live_white_control':
        union = np.zeros(image.shape[:2], 'uint8')
        path = HOLE_LAB / 'runs/live_white_control_item_01'
        for jid, rid in [(1, 7), (2, 4), (2, 5)]:
            mask = cv2.imread(str(path / f'jewel_{jid:02d}_region_{rid:03d}.png'), 0)
            if mask is None or mask.shape != union.shape:
                raise RuntimeError('Missing live-control reviewed stone mask')
            union |= mask
        samples['reviewed_true_white_faces'] = union
    if session == '20261021_134420_a434dc32':
        pearls = np.zeros(image.shape[:2], 'uint8')
        # Conservative interior samples of visually reviewed pale ornaments.
        for x, y in [(287, 737), (309, 801), (330, 844), (410, 891),
                     (504, 874), (552, 776), (456, 837)]:
            cv2.circle(pearls, (x, y), 3, 255, -1)
        samples['missed_pale_ornament_interiors'] = pearls
        holes = np.zeros(image.shape[:2], 'uint8')
        path = HOLE_LAB / 'runs' / f'{session}_{item}'
        for rid in range(1, 8):
            mask = cv2.imread(str(path / f'jewel_01_region_{rid:03d}.png'), 0)
            if mask is None or mask.shape != holes.shape:
                raise RuntimeError('Missing reviewed hole mask')
            holes |= mask
        samples['reviewed_background_holes'] = holes
    return samples


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    for session, item, kind in CASES:
        base = HOLE_LAB / session if session == 'live_white_control' else ROOT / 'data/sessions' / session
        image = cv2.imread(str(base / 'items' / f'{item}_raw.png'))
        foreground = cv2.imread(str(base / 'items' / f'{item}_mask.png'), 0)
        if image is None or foreground is None:
            raise RuntimeError(f'Missing input for {session}/{item}')
        foreground = np.where(foreground > 0, 255, 0).astype('uint8')
        samples = sample_masks(session, item, image)
        variants = {'existing_raw_glare': detection.build_specular_glare_mask(image, foreground),
                    'relative_all_colors': highlight_mask(image, foreground),
                    'relative_neutral_only': highlight_mask(image, foreground, neutral=True)}
        variants['relative_neutral_with_gold_veto'] = veto_gold_context(
            image, foreground, variants['relative_neutral_only'])
        row = {'session': session, 'item': item, 'reviewed_kind': kind,
               'foreground_px': int(cv2.countNonZero(foreground)), 'variants': {}, 'sweep': []}
        value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        for name, sample in samples.items():
            row.setdefault('reviewed_samples', {})[name] = {
                'pixels': int(cv2.countNonZero(sample)),
                'value_median': float(np.median(value[sample > 0])),
                'value_p95': float(np.percentile(value[sample > 0], 95))}
        for name, mask in variants.items():
            risk = area.calculate_reflection_risk(mask, foreground)
            row['variants'][name] = {'highlight_px': int(cv2.countNonZero(mask)),
                                     'reflection_risk': risk,
                                     'sample_pixel_coverage_percent': {
                                         label: round(100*cv2.countNonZero(mask & sample)/max(1, cv2.countNonZero(sample)), 3)
                                         for label, sample in samples.items()}}
        for floor in (140, 180, 220):
            for contrast in (12, 20, 30):
                for neutral in (False, True):
                    mask = highlight_mask(image, foreground, floor, contrast, neutral)
                    risk = area.calculate_reflection_risk(mask, foreground)
                    row['sweep'].append({'floor': floor, 'contrast': contrast, 'neutral': neutral,
                                        'flagged': risk['flagged'], 'coverage_percent': risk['coverage_percent'],
                                        'sample_pixel_coverage_percent': {
                                            label: round(100*cv2.countNonZero(mask & sample)/max(1, cv2.countNonZero(sample)), 3)
                                            for label, sample in samples.items()}})
        row['gold_context_sweep'] = []
        sensitive = highlight_mask(image, foreground, 140, 12, True)
        for window in (7, 11, 19):
            for maximum in (.2, .4, .6):
                mask = veto_gold_context(image, foreground, sensitive, window, maximum)
                risk = area.calculate_reflection_risk(mask, foreground)
                row['gold_context_sweep'].append({'window': window, 'maximum_gold_fraction': maximum,
                    'flagged': risk['flagged'], 'highlight_px': int(cv2.countNonZero(mask)),
                    'coverage_percent': risk['coverage_percent'],
                    'sample_pixel_coverage_percent': {
                        label: round(100*cv2.countNonZero(mask & sample)/max(1, cv2.countNonZero(sample)), 3)
                        for label, sample in samples.items()}})
        key = session + '_' + item
        # Display only; these masks are not counted as stone masks.
        overlay = image.copy()
        mask = variants['relative_all_colors']
        overlay[mask > 0] = (0, 100, 255)
        for label, sample in samples.items():
            contours, _ = cv2.findContours(sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 180, 0), 1)
        cv2.imwrite(str(OUT / (key + '_raw.jpg')), image)
        cv2.imwrite(str(OUT / (key + '_highlights.jpg')), overlay)
        cv2.imwrite(str(OUT / (key + '_mask.png')), mask)
        rows.append(row)
        metrics = row['variants']['relative_all_colors']
        cards.append(f'<article><h2>{html.escape(kind)} · {html.escape(key)}</h2>'
                     f'<p>Reflection warning: {metrics["reflection_risk"]["flagged"]}; '
                     f'highlight coverage: {metrics["reflection_risk"]["coverage_percent"]}%.</p>'
                     f'<div class="pair"><img src="{key}_raw.jpg"><img src="{key}_highlights.jpg"></div>'
                     f'<pre>{html.escape(json.dumps(row["variants"], indent=2))}</pre></article>')
    archive = []
    for p in sorted((HOLE_LAB / 'runs').glob('*/baseline_report.json')):
        if p.parent.name.startswith('live_'):
            continue
        report = json.loads(p.read_text())
        archive.append({'key': p.parent.name, 'existing_reflection_flagged': report.get('reflection_flagged', False),
                        'glare_px': sum(j.get('glare_mask_px', 0) for j in report['jewels'])})
    settings = []
    for index, template in enumerate(rows[0]['sweep']):
        positives = [r for r in rows if r['reviewed_kind'] in ('faceted_white_stones', 'white_stones_and_pearls', 'live_white_and_colored_stones')]
        negatives = [r for r in rows if r['reviewed_kind'].startswith('plain_gold')]
        settings.append({k: template[k] for k in ('floor', 'contrast', 'neutral')} | {
            'plain_metal_controls_flagged': sum(r['sweep'][index]['flagged'] for r in negatives),
            'stone_controls_flagged': sum(r['sweep'][index]['flagged'] for r in positives),
            'missed_pale_sample_coverage_percent': next(r for r in rows if r['reviewed_kind']=='missed_pale_ornaments')['sweep'][index]['sample_pixel_coverage_percent']['missed_pale_ornament_interiors']})
    context_settings = []
    for index, template in enumerate(rows[0]['gold_context_sweep']):
        context_settings.append({k: template[k] for k in ('window', 'maximum_gold_fraction')} | {
            'plain_metal_controls_flagged': sum(r['gold_context_sweep'][index]['flagged'] for r in rows if r['reviewed_kind'].startswith('plain_gold')),
            'stone_controls_flagged': sum(r['gold_context_sweep'][index]['flagged'] for r in rows if r['reviewed_kind'] in ('faceted_white_stones', 'white_stones_and_pearls', 'live_white_and_colored_stones'))})
    summary = {'reviewed_controls': len(rows), 'plain_metal_controls': 3, 'white_stone_controls': 3,
               'archived_items_checked_for_existing_reflection': len(archive),
               'existing_reflection_warnings': sum(r['existing_reflection_flagged'] for r in archive),
               'threshold_sweep': settings,
               'gold_context_sweep': context_settings,
               'perfect_control_separator_found': any(
                   s['plain_metal_controls_flagged'] == 0 and s['stone_controls_flagged'] == 3
                   for s in settings + context_settings),
               'conclusion': 'The tested reflection and gold-color-context rules do not identify only stones. Do not promote highlights to confirmed stone instances.',
               'material_identity_validated': False, 'gold_colored_gemstones_validated': False,
               'production_modified': False,
               'limitations': ['Small visually reviewed set; no measured material identities or gem labels.',
                              'Existing foreground masks may exclude missed faces before reflection analysis.',
                              'A highlight mask is not a full stone-face mask and cannot supply stone area/weight.',
                              'The parameter sweep is exploratory and is not an independent test set.']}
    (OUT / 'results.json').write_text(json.dumps({'summary': summary, 'controls': rows, 'existing_archive': archive}, indent=2))
    page = ('<!doctype html><meta charset="utf-8"><title>Reflection experiment</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#f4f6f8;padding:0 20px}'
            'article{background:white;padding:20px;margin:20px 0}.pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain}'
            'pre{white-space:pre-wrap;font-size:12px}h2{font-size:18px}</style>'
            '<h1>Reflection experiment — separate from the main app</h1>'
            '<p><strong>The tested combination does not meet the requirement to identify only stones.</strong> '
            'Sensitive thresholds flag plain gold chains and a bracelet. Gold-color context does not reliably remove them. '
            'The sampled missed pale ornaments produce no highlight evidence in this sweep.</p>'
            '<p>Orange: highlight pixels. Blue: reviewed white-face, pale-ornament or background-hole samples. '
            'Highlights indicate reflective regions; they do not establish stones or complete stone area.</p>'
            f'<details><summary>Summary, parameter sweep and limitations</summary><pre>{html.escape(json.dumps(summary, indent=2))}</pre></details>'
            + ''.join(cards))
    (OUT / 'review.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(r['reviewed_kind'], {k: (v['highlight_px'], v['reflection_risk']['flagged'], v['sample_pixel_coverage_percent']) for k, v in r['variants'].items()})


if __name__ == '__main__':
    main()
