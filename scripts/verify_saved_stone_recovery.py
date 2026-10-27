"""Replay saved stone inputs into testing/, preserving sessions and database."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'StoneDetection'))
import jewel_gem_hsv_report as s


def main():
    cv2.setNumThreads(1)
    output = ROOT.parent / 'testing/pink-stones/replays'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for sid in ['20261024_152502_f448b5c3', '20261024_131129_9886f1a5', '20261024_141233_6fd594a8']:
        session = ROOT / 'data/sessions' / sid
        state = json.loads((session / 'manifest.json').read_text())
        item = state['classification']['items'][0]
        image = cv2.imread(item['paths']['working'])
        mask = cv2.imread(item['paths']['mask'], 0)
        preset = s.build_candidates_from_component_mask(image, (mask > 0).astype(np.uint8),
            min_area=max(80, int(mask.size * .0005)), max_candidates=12,
            reject_border_touching=False, min_area_ratio_to_largest=.01)
        settings = state.get('analysis_settings') or state.get('settings_snapshot', {}).get('analysis', {})
        cal = state.get('calibration') or {}
        scale = {k: cal[k] for k in ['mm_per_pixel_x', 'mm_per_pixel_y']} if cal.get('available') and cal.get('found') else None
        cv2.setRNGSeed(0)
        result = s.analyze_image_bgr(image, zoom_scale=1, preset_candidates=preset, external_mask=mask,
            color_correction=settings.get('color_correction'),
            background_calibration=settings.get('background_calibration'),
            analysis_normalization=settings.get('normalization'), measurement_scale=scale,
            learned_stone_profiles=settings.get('learned_stone_profiles'), stone_v2_debug_dir=output/sid)
        report = result['report']
        (output / (sid + '.json')).write_text(json.dumps(report, indent=2))
        cv2.imwrite(str(output / (sid + '.png')), result['result_gallery_bgr'])
        original = json.loads((session / 'results/stones/item_01/result.json').read_text())
        summary = dict(session=sid, before_count=original['stone_instance_count'],
            after_count=report['stone_instance_count'], before_coverage=original['stone_percentage'],
            after_coverage=report['stone_percentage'], area_mm2=report['stone_measurements'].get('total_area_mm2'))
        if sid == '20261024_152502_f448b5c3':
            # Color-pixel coverage is a reproducible proxy, not annotated bead recall.
            source = cv2.imread(item['paths']['working'])
            x, y, w, h = preset[0]['bbox_global']
            source = source[y:y+h, x:x+w]
            hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            pink = (hsv[:,:,0]>=130)&(hsv[:,:,0]<=178)&(hsv[:,:,1]>=90)&(hsv[:,:,2]>=40)
            final = cv2.imread(str(output/sid/'jewel_01/05_refined_instances.png'),0)
            summary['pink_pixel_coverage'] = float(np.mean(final[pink]>0))
            assert summary['pink_pixel_coverage'] > .90, summary
        else:
            assert report['stone_instance_count'] <= original['stone_instance_count'], summary
        results.append(summary)
        print(json.dumps(summary), flush=True)
    (output/'verification.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
