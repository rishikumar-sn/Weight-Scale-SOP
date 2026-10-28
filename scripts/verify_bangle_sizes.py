"""Verify supplied bangle chart, dimension image fields, API payload and PDFs."""
from __future__ import annotations

import copy
import json
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
    from backend.app.domain.bangle_sizes import BANGLE_SIZE_CHART, bangle_size, size_for_dimensions
    from backend.app.analysis import annotations
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.services.report_service import ReportService
    from Dimension.bangle_detector import draw_results

    for label, diameter, fit in BANGLE_SIZE_CHART:
        size = bangle_size(diameter)
        assert size['label'] == label and size['fit'] == fit
    for diameter, expected in ((55.49, '2.2'), (55.5, '2.2'), (55.51, '2.4'),
                               (58.5, '2.4'), (58.51, '2.6'), (61.5, '2.6'),
                               (61.51, '2.8'), (69.5, '2.8'), (69.51, '3.0')):
        assert bangle_size(diameter)['label'] == expected
    for diameter in (53.99, 76.01, 20, 100):
        assert bangle_size(diameter)['status'] == 'out_of_range'
    for diameter in (None, 'bad', 0, -1, float('nan'), float('inf')):
        assert bangle_size(diameter)['status'] == 'unavailable'
    assert size_for_dimensions({'applicable': False, 'inner_diameter_mm': 60})['label'] is None

    output = ROOT.parent / 'testing/bangle-sizes-20261028'
    output.mkdir(exist_ok=True)
    circle = dict(center=(200, 200), radius=150, method='test')
    result = dict(outer=circle, inner=dict(circle, radius=130), od_mm=69, id_mm=60,
                  detection_mode='bangle')
    with patch.object(annotations, 'add_side_panel', wraps=annotations.add_side_panel) as panel:
        image = draw_results(np.full((400, 400, 3), 245, np.uint8), result)
        assert ('STANDARD BANGLE SIZE', '2.6 (approx.)') in panel.call_args.args[2]
        assert image.shape == (1024, 1536, 3)
        for mode, diameter, expected in (('finger_ring_edge_ellipse', 60, None),
                                         ('bangle', None, 'Unavailable'),
                                         ('bangle', 80, 'Outside supplied size chart')):
            draw_results(np.full((400, 400, 3), 245, np.uint8), dict(result, detection_mode=mode, id_mm=diameter))
            rows = dict(panel.call_args.args[2])
            assert rows.get('STANDARD BANGLE SIZE') == expected

    # Replay a saved flat bangle through the real calibrated detector, copying
    # inputs so no saved session or its existing evidence is modified.
    state = json.loads((ROOT / 'data/sessions/20261026_140111_d21bbb44/manifest.json').read_text())
    item = copy.deepcopy(state['classification']['items'][0])
    item['confirmed_label'] = 'Bangle'
    with tempfile.TemporaryDirectory(prefix='bangle-size-') as temporary:
        temp = Path(temporary)
        for key in ('working', 'mask'):
            destination = temp / Path(item['paths'][key]).name
            shutil.copy2(item['paths'][key], destination)
            item['paths'][key] = str(destination)
        service = object.__new__(AnalysisService)
        service.repository = type('Repo', (), {'session_dir': lambda _, session_id: temp})()
        dimensions = service._run_dimension(state, item)
        assert dimensions['bangle_size'] == bangle_size(dimensions['inner_diameter_mm'])
        assert dimensions['bangle_size']['label'], dimensions
        shutil.copy2(dimensions['image'], output / 'dimension-result.png')
        dimensions['image'] = str(output / 'dimension-result.png')
        state.update(status='complete')
        state['result'] = {'count': 1, 'items': [dict(index=1, label='Bangle', dimensions=dimensions)]}
        state['classification']['confirmed_label'] = 'Bangle'
        state['classification']['count'] = 1
        reports = ReportService()
        for legacy in (False, True):
            sample = copy.deepcopy(state)
            if legacy:
                sample['result'] = {'count': 1, 'dimensions': dimensions}
            pdf = reports.generate(sample)
            path = output / ('legacy-report.pdf' if legacy else 'report.pdf')
            path.write_bytes(pdf.getvalue())
            if shutil.which('pdftotext'):
                text = subprocess.check_output(['pdftotext', str(path), '-'], text=True)
                for expected in ('Standard bangle size', dimensions['bangle_size']['display'], 'Common fit', 'Nearest listed inner diameter'):
                    assert expected in text, (path, expected)
        (output / 'dimensions.json').write_text(json.dumps(dimensions, indent=2))
    print('PASS: all chart labels, midpoints, invalid/out-of-range diameters, image fields, ring exclusion, real detector payload, per-item and legacy PDFs')
    print(json.dumps(dimensions['bangle_size']))


if __name__ == '__main__':
    main()
