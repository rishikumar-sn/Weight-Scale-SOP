"""Check ring sizing and confirmed-only service behavior on captured sessions."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

from offline_runtime import ROOT, configure


def main():
    configure()
    import cv2
    import numpy as np
    from Dimension.finger_ring_detector import visible_ring_ellipses, RingDimensionsUnavailable
    from Dimension.bangle_detector import detect_bangle
    from backend.app.domain.ring_sizes import RING_SIZE_CHART, ring_size
    from backend.app.domain.workflow import route_for_label
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.services.report_service import ReportService

    output = ROOT.parent / 'testing/ring-sizes-20261028'
    output.mkdir(exist_ok=True)
    assert [row[0] for row in RING_SIZE_CHART] == [str(size) for size in range(7, 27)]
    for indian, diameter, circumference in RING_SIZE_CHART:
        size = ring_size(diameter)
        assert (size['indian'], size['reference_circumference_mm']) == (indian, circumference)
        assert 'us' not in size and 'uk' not in size
    for a, b in zip(RING_SIZE_CHART, RING_SIZE_CHART[1:]):
        midpoint = (a[1] + b[1]) / 2
        assert ring_size(midpoint)['indian'] == a[0]
        assert ring_size(midpoint + .0001)['indian'] == b[0]
    for value in (None, 'bad', 0, -1, float('nan'), float('inf')):
        assert ring_size(value)['status'] == 'unavailable'
    for value in (15.299, 23.001, 5, 30):
        assert ring_size(value)['status'] == 'out_of_range'
    assert route_for_label('Finger Ring')['dimension']
    assert not route_for_label('Finger Ring')['stones']
    assert not route_for_label('Earrings / Nosepin')['dimension']

    # Known ground truth: a circular 16.6 mm opening projected into an ellipse.
    for ratio in (1., .8, .65):
        for angle in (0, 35, 90):
            image = np.full((320, 320, 3), 245, np.uint8)
            cv2.ellipse(image, (160, 160), (100, round(100 * ratio)), angle, 0, 360, (30, 130, 170), -1)
            cv2.ellipse(image, (160, 160), (83, round(83 * ratio)), angle, 0, 360, (245, 245, 245), -1)
            path = output / 'synthetic-input.png'
            cv2.imwrite(str(path), image)
            result = detect_bangle(path, scale=.1, jewel_type='Finger Ring')
            assert abs(result['id_mm'] - 16.6) < .2, result
            assert result['opening_visible'] is True
            assert result['diameter_basis'] == 'ellipse_major_axis_assuming_circular_band'
            if ratio == .65 and angle == 35:
                shutil.copy2(result['annotated_path'], output / 'oval-dimension-result.jpg')
    for mode in ('solid', 'obstructed', 'severely_tilted', 'blank'):
        image = np.full((320, 320, 3), 245, np.uint8)
        ratio = .35 if mode == 'severely_tilted' else 1.
        if mode != 'blank':
            cv2.ellipse(image, (160, 160), (100, round(100 * ratio)), 0, 0, 360, (30, 130, 170), -1)
        if mode in ('obstructed', 'severely_tilted'):
            cv2.ellipse(image, (160, 160), (83, round(83 * ratio)), 0, 0, 360, (245, 245, 245), -1)
        if mode == 'obstructed':
            cv2.rectangle(image, (80, 140), (240, 180), (30, 130, 170), -1)
        try:
            visible_ring_ellipses(image, .1)
        except RingDimensionsUnavailable:
            pass
        else:
            raise AssertionError(f'Unsafe ring measurement: {mode}')

    samples = []
    # Fixed captured fixtures keep the replay reproducible as new sessions arrive.
    fixture_sessions = {'20261023_064836_43d263c8', '20261026_135525_5583ef7d',
                        '20261028_142449_f707590e'}
    for manifest in sorted((ROOT / 'data/sessions').glob('*/manifest.json')):
        if manifest.parent.name not in fixture_sessions:
            continue
        state = json.loads(manifest.read_text())
        for item in (state.get('classification') or {}).get('items') or []:
            if item.get('confirmed') is True and item.get('confirmed_label') == 'Finger Ring':
                samples.append((state, item))
    assert len(samples) == 7
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='confirmed-ring-sizes-') as temporary:
        temp = Path(temporary)
        service = object.__new__(AnalysisService)
        service.repository = type('Repo', (), {'session_dir': lambda _, sid: temp / sid})()
        state, item = samples[0]
        # A prediction, a missing item confirmation, or a confirmed different
        # category must not cause a finger-ring measurement.
        for changes in ({'confirmed': False}, {'confirmed': True, 'confirmed_label': None}):
            unconfirmed = dict(item, **changes)
            with patch('backend.app.services.analysis_service.detect_bangle') as detector:
                skipped = service._run_dimension(state, unconfirmed)
                assert skipped['applicable'] is False and not detector.called
        assert not service._route_for_item(dict(item), 'Earrings / Nosepin')['dimension']
        for state, source_item in samples:
            item = copy.deepcopy(source_item)
            destination = temp / f"{state['id']}_{item['index']}.png"
            shutil.copy2(item['paths']['working'], destination)
            item['paths']['working'] = str(destination)
            dimensions = service._run_dimension(state, item)
            name = f"{state['id']}_item_{item['index']:02d}"
            artifact = output / f'{name}.png'
            shutil.copy2(dimensions['image'], artifact)
            dimensions['image'] = str(artifact)
            # A supported open band now receives an explicitly inferred size.
            adjustable = state['id'] == '20261028_142449_f707590e' and item['index'] == 1
            assert dimensions['applicable'] is True, (name, dimensions)
            assert dimensions['is_open_ring_estimate'] is adjustable, (name, dimensions)
            if dimensions['applicable']:
                assert dimensions['ring_size'] == ring_size(dimensions['inner_diameter_mm'])
                expected = {
                    ('20261023_064836_43d263c8', 1): '19',
                    ('20261026_135525_5583ef7d', 1): '11',
                    ('20261028_142449_f707590e', 1): '14',
                    ('20261028_142449_f707590e', 2): '21',
                    ('20261028_142449_f707590e', 3): '20',
                    ('20261028_142449_f707590e', 4): '17',
                    ('20261028_142449_f707590e', 5): '10',
                }
                if (state['id'], item['index']) in expected:
                    assert dimensions['ring_size']['indian'] == expected[state['id'], item['index']]
                assert math.isclose(dimensions['inner_circumference_mm'], math.pi * dimensions['inner_diameter_mm'])
                assert dimensions['bangle_size'] is None
            else:
                assert 'ring_size' not in dimensions
            outcomes.append(dict(session=state['id'], index=item['index'], dimensions=dimensions))
        reports = ReportService()
        base = copy.deepcopy(samples[-1][0])
        same_session = [r for r in outcomes if r['session'] == base['id']]
        base['result'] = dict(count=len(same_session), items=[dict(index=r['index'], label='Finger Ring', dimensions=r['dimensions']) for r in same_session])
        path = output / 'report.pdf'
        path.write_bytes(reports.generate(base).getvalue())
        if shutil.which('pdftotext'):
            text = subprocess.check_output(['pdftotext', str(path), '-'], text=True)
            for expected in ('Indian ring size (approx.)', '10 (approx.)', '21 (approx.)', '17 (approx.)', 'missing arc estimated', 'Current opening only', 'Ellipse long axis'):
                assert expected in text, expected
            assert 'US ' not in text and 'UK ' not in text
        legacy = copy.deepcopy(base)
        legacy['classification'].update(confirmed_label='Finger Ring', confirmed=True, count=1)
        legacy['result'] = dict(count=1, dimensions=outcomes[1]['dimensions'])
        (output / 'legacy-report.pdf').write_bytes(reports.generate(legacy).getvalue())
        missing = copy.deepcopy(samples[0][0]); missing['calibration'] = {}
        skipped = service._run_dimension(missing, samples[0][1])
        assert skipped['applicable'] is False and 'ring_size' not in skipped
    (output / 'confirmed-results.json').write_text(json.dumps(outcomes, indent=2))
    measured = sum(r['dimensions']['applicable'] for r in outcomes)
    matched = sum((r['dimensions'].get('ring_size') or {}).get('status') == 'matched' for r in outcomes)
    print(f'PASS: {len(RING_SIZE_CHART)} chart rows, midpoint/range guards, nine rotated circle/oval fixtures, four unsafe fixtures, confirmed-only gating, missing calibration, {len(outcomes)} confirmed saved rings ({measured} measured, {matched} sized), multi-item and legacy PDFs')


if __name__ == '__main__':
    main()
