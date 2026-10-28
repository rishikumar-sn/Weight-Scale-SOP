"""Replay complete capture/classification; check scale-specific single-hole policy."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

from offline_runtime import ROOT, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-replay', action='store_true',
                        help='Recheck saved model outputs and run only missing capture fixtures')
    args = parser.parse_args()
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis.testbed_segmentation import TestbedSegmenter
    from backend.app.services.capture_service import CaptureService
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.services.report_service import ReportService
    from backend.app.storage.repository import CaptureRepository
    from backend.app.core.config import TESTBED_MODEL_PATHS

    output = ROOT.parent / 'testing/seven-rings-20261028'
    output.mkdir(exist_ok=True)
    # Only ESSAE may request the physical-hole subtraction, even when the
    # selected ONNX model is shared between different machines.
    for model in TESTBED_MODEL_PATHS:
        service = object.__new__(CaptureService)
        service._testbed_lock = RLock()
        service._testbed_segmenter = None
        service._testbed_model = None
        with patch('backend.app.services.capture_service.TestbedSegmenter') as constructor:
            service._get_testbed_segmenter(model)
            assert constructor.call_args.kwargs['exclude_circle'] == (model == 'essae_jx500')

    binary = np.zeros((500, 500), np.uint8)
    cv2.circle(binary, (100, 100), 18, 1, -1)
    cv2.circle(binary, (300, 300), 8, 1, -1)
    hole = TestbedSegmenter._single_hole_mask(binary)
    assert cv2.connectedComponents(hole)[0] == 2
    assert hole[100, 100] and not hole[300, 300]
    # The highest-scoring circle away from the AprilTag fixture position must
    # not win. There is no unanchored fallback if the proper hole is absent.
    segmenter = object.__new__(TestbedSegmenter)
    segmenter.names = {0: 'test-bed', 1: 'circle'}
    segmenter.circle_class_id = 1
    segmenter.confidence_threshold = .5
    masks = []
    for center in ((300, 300), (93, 158)):
        mask = np.zeros((500, 500), np.uint8)
        cv2.circle(mask, center, 18, 1, -1)
        masks.append(mask)
    segmenter._decode_mask = lambda coeff, *_: masks[int(coeff[0])].copy()
    rows = np.array([[300, 300, 36, 36, 0, .99, 0],
                     [93, 158, 36, 36, 0, .8, 1]], np.float32)
    tag = dict(found=True, corners=[[100, 400], [120, 400], [120, 420], [100, 420]])
    row, _, evidence = segmenter._anchored_circle(rows.T[None], None, None, (500, 500), tag)
    assert row[-1] == 1 and evidence['anchor'] == 'apriltag'
    for prediction, marker in ((rows[:1].T[None], tag), (rows.T[None], {})):
        try:
            segmenter._anchored_circle(prediction, None, None, (500, 500), marker)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Physical hole accepted without AprilTag position evidence')

    fixture_ids = ['20261028_155357_e7a57829', '20261028_154357_839f37cc',
                   '20261028_142449_f707590e', '20261028_121714_b325206d']
    protected = [ROOT / 'data/sessions' / sid / 'manifest.json' for sid in fixture_ids]
    protected += list((ROOT / 'Classification').glob('*.npz'))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    repo = CaptureRepository(output / 'replay.db', output / 'after')
    analysis = AnalysisService(repo)
    results = []
    try:
        for sid in fixture_ids:
            source = json.loads((ROOT / 'data/sessions' / sid / 'manifest.json').read_text())
            raw = cv2.imread(source['paths']['original'])
            settings = SimpleNamespace(get=lambda source=source: copy.deepcopy(source['settings_snapshot']))
            current = None
            if args.reuse_replay:
                for saved in sorted((output / 'after').glob('*/manifest.json'), reverse=True):
                    candidate = json.loads(saved.read_text())
                    if candidate.get('status') != 'awaiting_confirmation':
                        continue
                    previous_raw = cv2.imread(candidate['paths']['original'])
                    if previous_raw is not None and np.array_equal(raw, previous_raw):
                        current = candidate
                        break
            reused = current is not None
            if current is None:
                capture = CaptureService(SimpleNamespace(), SimpleNamespace(), settings, repo)
                reading = dict(source['weight'], connected=True, age_ms=0, unit='g', weight_g=source['weight_g'])
                with patch.object(capture, '_capture_inputs', return_value=(raw, source['camera'], reading)):
                    current = capture.capture()
            # ESSAE still subtracts only its anchored physical hole. Its mask
            # must be unchanged by a fix to the ACZET branch.
            if source['settings_snapshot']['scale']['model'] == 'essae_jx500':
                previous = cv2.imread(source['paths']['testbed_source_mask'], 0)
                fresh = cv2.imread(current['paths']['testbed_source_mask'], 0)
                assert np.array_equal(previous, fresh), sid
                assert current['testbed']['hole_validation']['anchor'] == 'apriltag'
            else:
                assert current['testbed']['hole_validation'] is None
                valid = cv2.imread(current['paths']['testbed_mask'], 0)
                # Continuous platform masks cannot contain interior cutouts.
                contours, hierarchy = cv2.findContours(valid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                assert hierarchy is not None and not np.any(hierarchy[0, :, 3] >= 0), sid
            if not reused:
                current = analysis.classify(current['id'])
            classification = current['classification']
            expected = {'20261028_155357_e7a57829': 7, '20261028_154357_839f37cc': 7,
                        '20261028_142449_f707590e': 7, '20261028_121714_b325206d': 1}[sid]
            row = dict(source_session=sid, replay_session=current['id'],
                       before_count=source.get('classification', {}).get('count'),
                       after_count=classification['count'], status=current['status'],
                       labels=classification['labels'])
            print(json.dumps(row), flush=True)
            assert classification['count'] == expected, row
            if sid in {'20261028_155357_e7a57829', '20261028_154357_839f37cc'}:
                assert classification['labels'] == ['Finger Ring'] * expected, row
                assert current['status'] == 'awaiting_confirmation', row
            if sid == '20261028_142449_f707590e':
                # Raw capture contains six rings plus one earring; the original
                # tray cutouts also hid one ring in this mixed-jewellery control.
                assert classification['labels'].count('Finger Ring') == 6, row
                assert classification['labels'].count('Earrings / Nosepin') == 1, row
            if sid == '20261028_155357_e7a57829':
                # Manually identified full-ring regions in the saved processing
                # coordinates. Match every reference region to one unique crop.
                boxes = [(675, 60, 135, 150), (306, 155, 135, 145), (100, 185, 140, 140),
                         (532, 258, 140, 145), (335, 452, 140, 150),
                         (630, 450, 155, 160), (53, 578, 165, 165)]
                working = cv2.imread(current['paths']['working'])
                mask = cv2.imread(current['paths']['mask'], 0)
                old = cv2.imread(source['paths']['working'])
                hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
                old_valid = cv2.imread(source['paths']['testbed_mask'], 0)
                fresh_valid = cv2.imread(current['paths']['testbed_mask'], 0)
                coverage = []
                owners = set()
                for x, y, w, h in boxes:
                    gold = ((hsv[y:y+h, x:x+w, 0] >= 3) & (hsv[y:y+h, x:x+w, 0] <= 45)
                            & (hsv[y:y+h, x:x+w, 1] > 70) & (hsv[y:y+h, x:x+w, 2] > 35))
                    assert gold.sum() > 100
                    retained = (mask[y:y+h, x:x+w][gold] > 0).mean()
                    assert retained > .96, (x, y, retained)
                    candidates = []
                    for item in classification['items']:
                        b = item['bbox']
                        if b['x'] < x+w/2 < b['x']+b['w'] and b['y'] < y+h/2 < b['y']+b['h']:
                            candidates.append(item['index'])
                    assert len(candidates) == 1 and candidates[0] not in owners, candidates
                    owners.add(candidates[0])
                    coverage.append(float(retained))
                row['gold_pixel_coverage'] = coverage
                row['restored_platform_pixels'] = int(np.count_nonzero((fresh_valid > 0) & (old_valid == 0)))
                # A camera-only reclassification must not silently confirm rings.
                assert all(not item['confirmed'] and item['confirmed_label'] is None for item in classification['items'])
                cv2.imwrite(str(output / 'corrected-working.png'), working)
                cv2.imwrite(str(output / 'corrected-mask.png'), mask)
                cv2.imwrite(str(output / 'corrected-review.jpg'), cv2.imread(current['paths']['evidence']))
                image = ReportService().generate_result_image(current)
                (output / 'corrected-summary.png').write_bytes(image.getvalue())
                (output / 'corrected-session.json').write_text(json.dumps(current, indent=2))
            results.append(row)
    finally:
        analysis.shutdown()
    for path, expected in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    (output / 'verification.json').write_text(json.dumps(results, indent=2))
    print('PASS: scale-specific policy, single anchored ESSAE hole, complete seven-ring segmentation/count/predictions, controls and source preservation', flush=True)


if __name__ == '__main__':
    main()
