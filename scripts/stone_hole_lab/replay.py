"""Isolated saved-capture lab. No production settings/results are written."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'StoneDetection')]
import audit_stone_sessions as audit
from backend.app.analysis.vision import detect_apriltag

OUT = ROOT / 'data/audits/stone-hole-lab'


def lab(image):
    return cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def aligned_reference(manifest, raw, item):
    reference = cv2.imread(str(OUT / 'reference/empty_tray.png'))
    if reference is None:
        return None, {'valid': False, 'reason': 'missing reference'}
    calibration = manifest.get('calibration') or {}
    if not calibration.get('found'):
        return None, {'valid': False, 'reason': 'missing saved marker geometry'}
    settings = manifest['settings_snapshot']['apriltag']
    detected = detect_apriltag(reference, None, int(settings['id']),
                              float(settings['width_mm']), float(settings['height_mm']))
    a, b = np.array(calibration['corners']), np.array(detected['corners'])
    shift = np.median(a - b, axis=0)
    residual = float(np.max(np.linalg.norm(a - b - shift, axis=1)))
    if residual > 1.5 or np.linalg.norm(shift) > 15:
        return None, {'valid': False, 'reason': 'camera geometry changed',
                      'corner_residual_px': residual, 'shift_px': shift.tolist()}
    registered = cv2.warpAffine(reference, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                                (reference.shape[1], reference.shape[0]))
    processing, bbox = manifest['rois']['processing'], item['bbox']
    x, y = processing['x'] + bbox['x'], processing['y'] + bbox['y']
    h, w = raw.shape[:2]
    patch = registered[y:y+h, x:x+w]
    if patch.shape != raw.shape:
        return None, {'valid': False, 'reason': 'reference dimensions differ'}
    return patch, {'valid': True, 'corner_residual_px': residual, 'shift_px': shift.tolist()}


def metrics(raw, mask, seed, gold, foreground, reference, cache=None):
    cache = {} if cache is None else cache
    core = cv2.erode((mask > 0).astype('uint8'), np.ones((3, 3), 'uint8')) > 0
    core &= gold == 0
    if core.sum() < 4:
        core = (seed > 0) & (gold == 0)
    if core.sum() < 4:
        return {'valid': False, 'reason': 'insufficient core pixels'}
    if 'raw_lab' not in cache:
        cache['raw_lab'] = lab(raw)
        cache['outside'] = cv2.dilate((foreground > 0).astype('uint8'), np.ones((9, 9), 'uint8')) == 0
        cache['outside'] &= (raw.max(axis=2) < 245) & (raw.min(axis=2) > 25)
        if reference is not None:
            cache['ref_lab'] = lab(reference)
    raw_lab = cache['raw_lab']
    x, y, w, h = cv2.boundingRect(mask)
    rx, ry = max(35, w*3), max(35, h*3)
    ys = slice(max(0, int(y+h/2-ry)), min(mask.shape[0], int(y+h/2+ry)))
    xs = slice(max(0, int(x+w/2-rx)), min(mask.shape[1], int(x+w/2+rx)))
    outside = cache['outside'][ys, xs]
    background = raw_lab[ys, xs][outside]
    result = {'valid': True, 'core_pixels': int(core.sum()),
              'raw_lightness_std': float(np.std(raw_lab[:, :, 0][core])),
              'local_background_pixels': int(outside.sum())}
    if outside.sum() >= 30:
        center = np.median(background, axis=0)
        delta = np.linalg.norm(raw_lab[core] - center, axis=1)
        result.update(local_delta_e_median=float(np.median(delta)),
                      local_delta_e_p90=float(np.percentile(delta, 90)))
    if reference is not None and outside.sum() >= 30:
        ref_lab = cache['ref_lab']
        ref_background = ref_lab[ys, xs][outside]
        offset = np.median(background - ref_background, axis=0)
        residual = np.linalg.norm(background - ref_background - offset, axis=1)
        delta = np.linalg.norm(raw_lab[core] - ref_lab[core] - offset, axis=1)
        result.update(reference_delta_e_median=float(np.median(delta)),
                      reference_delta_e_p90=float(np.percentile(delta, 90)),
                      reference_background_p90=float(np.percentile(residual, 90)),
                      reference_lab_offset=offset.tolist())
    return result


def run(task):
    path, detailed, mode = task
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    session = path.parents[3]
    key = session.name + '_' + path.parent.name
    output = OUT / ('runs' if mode == 'baseline' else mode) / key
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((session / 'manifest.json').read_text())
    item = next(i for i in manifest['classification']['items']
                if int(i['index']) == int(path.parent.name.split('_')[-1]))
    raw_path = session / 'items' / (path.parent.name + '_raw.png')
    raw = cv2.imread(str(raw_path)) if raw_path.exists() else None
    if raw is None:
        source = cv2.imread(str(session / 'capture/original.png'))
        processing, bbox = manifest['rois']['processing'], item['bbox']
        x, y = processing['x'] + bbox['x'], processing['y'] + bbox['y']
        raw = source[y:y+bbox['h'], x:x+bbox['w']] if source is not None else None
    if raw is None:
        return {'key': key, 'error': 'raw item is unavailable'}
    try:
        reference, registration = aligned_reference(manifest, raw, item)
    except Exception as exc:
        reference, registration = None, {'valid': False, 'reason': str(exc)}
    detection = audit.stone_detection
    v2 = detection._stone_v2
    originals = {name: getattr(detection, name) for name in
                 ('analyze_jewel_candidate', 'serialize_regions', 'analyze_image_bgr',
                  'remove_enclosed_background_from_jewel_mask')}
    original_generate = v2.generate_stone_candidates
    original_final = v2.build_final_stone_instances
    original_classify = v2.classify_stone_instance_color
    original_warm = v2.is_warm_metal_shadow
    original_neutral = v2.should_reject_neutral_candidate
    original_micro = v2.filter_unsupported_neutral_instances
    context, rows, candidates = {}, [], []
    full_overlay = raw.copy()
    if detailed:
        cv2.imwrite(str(output / 'raw_item.png'), raw)

    def cleanup(image, foreground, *args, **kwargs):
        return originals['remove_enclosed_background_from_jewel_mask'](
            context['raw'] if mode == 'raw-cleanup' else image, foreground, *args, **kwargs)

    def jewel(*args, **kwargs):
        j = args[1] if len(args) > 1 else kwargs['jewel']
        x, y, w, h = j['bbox_global']
        context.update(raw=raw[y:y+h, x:x+w],
                       reference=None if reference is None else reference[y:y+h, x:x+w],
                       foreground=j['crop_mask'], metric_cache={},
                       jewel_id=args[0] if args else kwargs['jewel_index'], jewel_bbox=[x, y, w, h])
        return originals['analyze_jewel_candidate'](*args, **kwargs)

    def generate(image, foreground, seeds, gold, *args, **kwargs):
        cs = original_generate(image, foreground, seeds, gold, *args, **kwargs)
        context.update(gold=gold, cs={c['candidate_id']: c for c in cs})
        if detailed:
            cv2.imwrite(str(output / 'raw.png'), context['raw'])
            cv2.imwrite(str(output / 'prepared.png'), image)
            cv2.imwrite(str(output / 'foreground.png'), foreground * (255 if foreground.max() == 1 else 1))
            cv2.imwrite(str(output / 'gold.png'), gold)
            for c in cs:
                cv2.imwrite(str(output / f"candidate_{c['candidate_id']:03d}_seed.png"), c['seed_mask'])
        return cs

    def final(*args, **kwargs):
        instances, diagnostics = original_final(*args, **kwargs)
        for entry in diagnostics['candidates']:
            entry['jewel_id'] = context['jewel_id']
        context['active_diagnostics'] = diagnostics['candidates']
        candidates.extend(diagnostics['candidates'])
        context['refined_ids'] = {hashlib.sha256(i['mask'].tobytes()).hexdigest(): i['candidate_id'] for i in instances}
        if detailed:
            cv2.imwrite(str(output / 'strict_gold.png'), args[4])
            for i, instance in enumerate(instances):
                cv2.imwrite(str(output / f"candidate_{instance['candidate_id']:03d}_refined_{i:03d}.png"), instance['mask'])
        return instances, diagnostics

    def classify(image, mask, *args, **kwargs):
        result = original_classify(image, mask, *args, **kwargs)
        context['last_id'] = context.get('refined_ids', {}).get(hashlib.sha256(mask.tobytes()).hexdigest())
        for c in context.get('active_diagnostics', []):
            if c['candidate_id'] == context['last_id']:
                c['classified_color'] = result['color']
                c['final_stage'] = 'classified'
        return result

    def mark(reason):
        for c in context.get('active_diagnostics', []):
            if c['candidate_id'] == context.get('last_id'):
                c['final_stage'] = reason

    def warm(*args, **kwargs):
        rejected = original_warm(*args, **kwargs)
        if rejected:
            mark('warm_metal_shadow')
        return rejected

    def neutral(*args, **kwargs):
        rejected = original_neutral(*args, **kwargs)
        if rejected:
            mark('neutral_candidate_filter')
        return rejected

    def micro(instances, **kwargs):
        kept, rejected = original_micro(instances, **kwargs)
        ids = {i['candidate_id'] for i in kept}
        removed = {i['candidate_id'] for i in instances} - ids
        for c in context.get('active_diagnostics', []):
            if c['candidate_id'] in removed:
                c['final_stage'] = 'unsupported_neutral_micro'
        return kept, rejected

    def serialize(regions):
        overlay = context['raw'].copy()
        for r in regions:
            c = context['cs'][r['candidate_id']]
            m = metrics(context['raw'], r['mask'], c['seed_mask'], context['gold'],
                        context['foreground'], context['reference'], context['metric_cache'])
            row = {k: r.get(k) for k in ('region_id', 'candidate_id', 'color', 'bbox', 'area_px', 'source_methods', 'refinement_diagnostics', 'learned_matches')}
            row.update(metrics=m, mask_sha256=hashlib.sha256(r['mask'].tobytes()).hexdigest(), jewel_id=context['jewel_id'])
            rows.append(row)
            contours, _ = cv2.findContours(r['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 220, 0), 1)
            x, y, _, _ = r['bbox']
            cv2.putText(overlay, str(r['region_id']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
            if detailed:
                cv2.imwrite(str(output / f"region_{r['region_id']:03d}.png"), r['mask'])
                gx, gy, gw, gh = context['jewel_bbox']
                full_mask = np.zeros(raw.shape[:2], 'uint8')
                full_mask[gy:gy+gh, gx:gx+gw] = r['mask']
                cv2.imwrite(str(output / f"jewel_{context['jewel_id']:02d}_region_{r['region_id']:03d}.png"), full_mask)
                full_contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(full_overlay, full_contours, -1, (0, 220, 0), 1)
                cv2.putText(full_overlay, f"{context['jewel_id']}.{r['region_id']}", (gx+x, gy+y), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
        if detailed:
            cv2.imwrite(str(output / 'baseline_overlay.jpg'), overlay)
        return originals['serialize_regions'](regions)

    def analyze(*args, **kwargs):
        result = originals['analyze_image_bgr'](*args, **kwargs)
        (output / 'baseline_report.json').write_text(json.dumps(result['report'], indent=2))
        if detailed:
            cv2.imwrite(str(output / 'baseline_overlay.jpg'), full_overlay)
        return result

    try:
        detection.analyze_jewel_candidate = jewel
        detection.serialize_regions = serialize
        detection.analyze_image_bgr = analyze
        detection.remove_enclosed_background_from_jewel_mask = cleanup
        v2.generate_stone_candidates = generate
        v2.build_final_stone_instances = final
        v2.classify_stone_instance_color = classify
        v2.is_warm_metal_shadow = warm
        v2.should_reject_neutral_candidate = neutral
        v2.filter_unsupported_neutral_instances = micro
        audit.analyze_saved_item(session, path.parent)
        final_ids = {(r['jewel_id'], r['candidate_id']) for r in rows}
        for c in candidates:
            if (c['jewel_id'], c['candidate_id']) in final_ids:
                c['final_stage'] = 'accepted_final'
            elif not c['accepted']:
                c['final_stage'] = 'refinement_rejected'
        result = {'key': key, 'registration': registration, 'regions': rows, 'candidates': candidates}
        (output / 'diagnostics.json').write_text(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        return {'key': key, 'error': repr(exc)}
    finally:
        for name, method in originals.items():
            setattr(detection, name, method)
        v2.generate_stone_candidates = original_generate
        v2.build_final_stone_instances = original_final
        v2.classify_stone_instance_color = original_classify
        v2.is_warm_metal_shadow = original_warm
        v2.should_reject_neutral_candidate = original_neutral
        v2.filter_unsupported_neutral_instances = original_micro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='*')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--mode', choices=['baseline', 'raw-cleanup'], default='baseline')
    parser.add_argument('--resume', action='store_true', help='Reuse completed diagnostics')
    args = parser.parse_args()
    paths = sorted((ROOT / 'data/sessions').glob('*/results/stones/item_*/result.json'))
    if args.sessions:
        paths = [p for p in paths if p.parents[3].name in args.sessions]
    results = []
    pending = []
    for p in paths:
        cache = OUT / ('runs' if args.mode == 'baseline' else args.mode) / (p.parents[3].name+'_'+p.parent.name) / 'diagnostics.json'
        if args.resume and cache.exists():
            results.append(json.loads(cache.read_text()))
        else:
            pending.append(p)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run, [(p, args.detailed, args.mode) for p in pending]):
            results.append(r)
            print(f"[{len(results)}/{len(paths)}] {r['key']}: {r.get('error', str(len(r.get('regions', [])))+' regions')}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('replay.json' if args.mode == 'baseline' else args.mode + '.json')).write_text(json.dumps(results, indent=2))
    if any('error' in r for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
