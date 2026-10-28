"""Check inferred open-ring sizes against visible arcs and saved confirmed rings."""
from __future__ import annotations

import copy
import hashlib
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
    from Dimension.finger_ring_detector import measure_ring_ellipses, RingDimensionsUnavailable
    from Dimension.bangle_detector import detect_bangle
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.services.report_service import ReportService
    from backend.app.domain.ring_sizes import ring_size
    from backend.app.analysis import annotations

    output = ROOT.parent / 'testing/open-ring-estimate-20261028'
    output.mkdir(exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='open-ring-fit-') as temporary:
        temp = Path(temporary)
        # Known inner diameter is 16.6 mm, regardless of the missing band arc.
        for gap in (20, 40, 60):
            for ratio in (1., .75):
                for angle in (0, 65, 150):
                    image = np.full((320, 320, 3), 245, np.uint8)
                    cv2.ellipse(image, (160, 160), (100, round(100*ratio)), angle,
                                gap/2, 360-gap/2, (30, 130, 170), -1)
                    cv2.ellipse(image, (160, 160), (83, round(83*ratio)), angle,
                                0, 360, (245, 245, 245), -1)
                    path = temp / 'open.png'
                    cv2.imwrite(str(path), image)
                    with patch.object(annotations, 'add_side_panel', wraps=annotations.add_side_panel) as panel:
                        result = detect_bangle(path, scale=.1, jewel_type='Finger Ring')
                        fields = dict(panel.call_args.args[2])
                        assert 'estimated' in fields['RING OPENING']
                        assert 'Dashed outline' in fields['ESTIMATE NOTE']
                    assert result['is_open_ring_estimate'] is True
                    assert abs(result['id_mm'] - 16.6) < .25, (gap, ratio, angle, result)
                    assert result['visible_arc_fraction'] >= .75
                    rows.append(dict(gap_degrees=gap, axis_ratio=ratio, rotation=angle,
                                     estimated_diameter_mm=result['id_mm']))
                    if (gap, ratio, angle) == (40, .75, 65):
                        shutil.copy2(result['annotated_path'], output / 'synthetic-oval-completion.jpg')
        for mode in ('blank', 'solid', 'blocked', 'gap_110', 'gap_160', 'edge_on'):
            image = np.full((320, 320, 3), 245, np.uint8)
            ratio = .35 if mode == 'edge_on' else 1.
            gap = int(mode.split('_')[1]) if mode.startswith('gap_') else 30
            if mode == 'solid':
                cv2.circle(image, (160, 160), 100, (30, 130, 170), -1)
            elif mode != 'blank':
                cv2.ellipse(image, (160, 160), (100, round(100*ratio)), 0,
                            gap/2, 360-gap/2, (30, 130, 170), -1)
                cv2.ellipse(image, (160, 160), (83, round(83*ratio)), 0,
                            0, 360, (245, 245, 245), -1)
            if mode == 'blocked':
                cv2.rectangle(image, (80, 135), (240, 185), (30, 130, 170), -1)
            try:
                measure_ring_ellipses(image, .1)
            except RingDimensionsUnavailable:
                pass
            else:
                raise AssertionError(f'Unsupported open-ring estimate: {mode}')

        service = object.__new__(AnalysisService)
        service.repository = type('Repo', (), {'session_dir': lambda _, sid: temp / sid})()
        samples = []
        for sid, index in [('20261028_155357_e7a57829', 5),
                           ('20261028_154357_839f37cc', 1),
                           ('20261028_142449_f707590e', 1)]:
            manifest = ROOT / 'data/sessions' / sid / 'manifest.json'
            before = hashlib.sha256(manifest.read_bytes()).hexdigest()
            source = json.loads(manifest.read_text())
            original = next(item for item in source['classification']['items'] if item['index'] == index)
            assert original.get('confirmed') is True and original['confirmed_label'] == 'Finger Ring'
            item = copy.deepcopy(original)
            path = temp / f'{sid}_{index}.png'
            shutil.copy2(item['paths']['working'], path)
            item['paths']['working'] = str(path)
            dims = service._run_dimension(source, item)
            assert dims['applicable'] and dims['is_open_ring_estimate'] is True, dims
            assert dims['ring_size'] == ring_size(dims['inner_diameter_mm'])
            assert dims['visible_arc_fraction'] >= .75
            artifact = output / f'{sid}_item_{index:02d}.png'
            shutil.copy2(dims['image'], artifact)
            dims['image'] = str(artifact)
            # Predictions alone still cannot trigger approximate sizing.
            with patch('backend.app.services.analysis_service.detect_bangle') as detector:
                unconfirmed = service._run_dimension(source, dict(item, confirmed=False))
                assert unconfirmed['applicable'] is False and not detector.called
            assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
            samples.append(dict(session=sid, index=index, dimensions=dims))
            if len(samples) == 1:
                replay = copy.deepcopy(source)
                replay['result'] = dict(count=1, items=[dict(index=index, label='Finger Ring', dimensions=dims)])
                pdf = output / 'report.pdf'
                pdf.write_bytes(ReportService().generate(replay).getvalue())
                if shutil.which('pdftotext'):
                    text = subprocess.check_output(['pdftotext', str(pdf), '-'], text=True)
                    for expected in ('Indian ring size', 'missing arc estimated', 'Current opening only', '(approx.)'):
                        assert expected in text, expected
        (output / 'verification.json').write_text(json.dumps(dict(synthetic=rows, saved=samples), indent=2))
    print(f'PASS: {len(rows)} controlled open/oval projections, six rejection fixtures, three confirmed saved open rings, approximation labels, PDF and confirmation guard')
    for sample in samples:
        dims = sample['dimensions']
        print(sample['session'], sample['index'], round(dims['inner_diameter_mm'], 2), dims['ring_size']['display'])


if __name__ == '__main__':
    main()
