"""Analyze the user-prepared control photo entirely inside the lab directory."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from backend.app.analysis.testbed_segmentation import TestbedSegmenter
from backend.app.analysis.vision import (build_jewellery_mask, detect_apriltag,
    apriltag_roi_from_detection, marker_ignore_mask, translate_roi_to_crop)
from replay import run, OUT


def main():
    metadata = json.loads((OUT / 'control/metadata.json').read_text())
    settings = metadata['settings']
    frame = cv2.imread(str(OUT / 'control/white_stone_control.png'))
    cv2.setNumThreads(1)
    testbed = TestbedSegmenter(ROOT / 'models/segmentation/testbed.onnx').segment(frame)
    tag = settings['apriltag']
    calibration = detect_apriltag(frame, None, int(tag['id']), float(tag['width_mm']), float(tag['height_mm']))
    calibration['available'] = True
    roi = apriltag_roi_from_detection(frame.shape, calibration)
    local_roi = translate_roi_to_crop(roi, testbed.processing_roi,
                                     testbed.image.shape[1], testbed.image.shape[0])
    ignore = marker_ignore_mask(testbed.image.shape, local_roi)
    raw = testbed.image.copy()
    raw[ignore > 0] = 255
    ignore = cv2.bitwise_or(ignore, np.where(testbed.mask > 0, 0, 255).astype('uint8'))
    prepared, mask = build_jewellery_mask(raw, ignore)
    mask = cv2.bitwise_and(mask, testbed.mask)
    prepared[mask == 0] = 255
    session = OUT / 'live_white_control'
    (session / 'items').mkdir(parents=True, exist_ok=True)
    result_dir = session / 'results/stones/item_01'
    result_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(session / 'items/item_01.png'), prepared)
    cv2.imwrite(str(session / 'items/item_01_raw.png'), raw)
    cv2.imwrite(str(session / 'items/item_01_mask.png'), mask*255)
    manifest = {'id': 'live_white_control', 'laboratory_only': True,
                'settings_snapshot': settings, 'calibration': calibration,
                'rois': {'processing': testbed.processing_roi},
                'classification': {'items': [{'index': 1, 'paths': {},
                    'bbox': {'x': 0, 'y': 0, 'w': raw.shape[1], 'h': raw.shape[0]}}]}}
    (session / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (result_dir / 'result.json').write_text('{}')
    result = run((result_dir / 'result.json', True, 'baseline'))
    (OUT / 'live_control.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'regions': len(result.get('regions', [])),
                      'registration': result.get('registration'), 'error': result.get('error')}, indent=2))
    if 'error' in result:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
