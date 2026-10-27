"""Check dual-resolution raw-frame inference, merging and jewel filtering offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from offline_runtime import BLOCKED_CONNECTIONS, ROOT, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-session-dir", type=Path, help="Optional saved capture directory")
    args = parser.parse_args()
    configure()
    import cv2
    import numpy as np
    from backend.app.services.analysis_service import AnalysisService, OnnxBeadDetector
    from backend.app.storage.repository import CaptureRepository

    detector = OnnxBeadDetector(
        ROOT / 'models/detection/bead_finder.onnx',
        additional_model_paths=[ROOT / 'models/detection/beadfinder_old.onnx'],
    )
    old = detector.additional_detectors[0]
    assert detector.input_size == (960, 960)
    assert old.input_size == (640, 640)
    raw = np.full((400, 800, 3), (20, 90, 180), np.uint8)
    before = raw.copy()
    mask = np.full(raw.shape[:2], 255, np.uint8)
    mask[200:250, 400:450] = 0
    region = dict(x=50, y=50, w=600, h=300)
    # Duplicate large box, unique small/new and large/old, outside jewel,
    # outside testbed mask and below confidence threshold.
    new_boxes = [[100, 100, 200, 200, .9], [300, 100, 310, 110, .8],
                 [680, 100, 710, 130, .99], [400, 200, 440, 240, .99],
                 [350, 100, 370, 120, .1]]
    old_boxes = [[110, 110, 210, 210, .85], [500, 100, 580, 180, .8]]
    calls = []

    class Session:
        def __init__(self, side, boxes):
            self.side, self.boxes = side, boxes

        def run(self, outputs, inputs):
            tensor = next(iter(inputs.values()))
            assert tensor.shape == (1, 3, self.side, self.side)
            # A raw 2:1 image must have letterbox padding and unmasked RGB pixels.
            assert np.allclose(tensor[0, :, 0, 0], 114 / 255)
            assert np.allclose(tensor[0, :, self.side // 2, self.side // 2], [180/255, 90/255, 20/255])
            calls.append(self.side)
            scale, top = self.side / 800, self.side / 4
            rows = [[(x1+x2)/2*scale, (y1+y2)/2*scale+top,
                     (x2-x1)*scale, (y2-y1)*scale, score]
                    for x1, y1, x2, y2, score in self.boxes]
            return [np.array(rows, np.float32).T[None]]

    with patch.object(detector, 'session', Session(960, new_boxes)), patch.object(old, 'session', Session(640, old_boxes)):
        result, _, _ = detector.run(raw, detection_region=region, detection_mask=mask)
        assert result['count'] == 3, result
        assert {tuple(d['bbox']) for d in result['detections']} == {
            (100, 100, 200, 200), (300, 100, 310, 110), (500, 100, 580, 180),
        }
        assert result['decision_source'] == 'onnx_ensemble'
        assert calls == [960, 640]
        assert np.array_equal(raw, before)
        empty, _, _ = detector.run(raw, detection_region=dict(x=0, y=0, w=10, h=10), detection_mask=mask)
        assert empty['count'] == 0
        try:
            detector.run(raw, detection_mask=mask[:10])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid mask must be rejected')

    output = ROOT.parent / 'testing/bead-ensemble'
    output.mkdir(parents=True, exist_ok=True)
    sid = '20261024_141233_6fd594a8'
    session_dir = args.replay_session_dir or ROOT / 'data/sessions' / sid
    if not (session_dir / 'manifest.json').is_file():
        if args.replay_session_dir:
            raise FileNotFoundError(session_dir / 'manifest.json')
        # Deployment bundles may omit saved customer captures. Still execute
        # both real ONNX sessions after the synthetic checks above.
        result, _, _ = detector.run(raw, detection_region=region, detection_mask=mask)
        assert len(result['models']) == 2
        assert not BLOCKED_CONNECTIONS, BLOCKED_CONNECTIONS
        print('Bead ensemble checks passed: synthetic cases and both real ONNX sessions; saved replay unavailable.', flush=True)
        return
    state = json.loads((session_dir / 'manifest.json').read_text())
    sid = session_dir.name
    with tempfile.TemporaryDirectory(prefix='bead-ensemble-') as temp:
        repo = CaptureRepository(Path(temp)/'captures.db', Path(temp)/'sessions')
        service = AnalysisService(repo)
        service._beads = detector
        try:
            result = service._run_beads(state, state['classification']['items'][0])
            assert result['input_source'] == 'full_frame'
            assert result['coordinate_space'] == 'full_frame'
            assert len(result['models']) == 2
            assert {m['model_input_size']['height'] for m in result['models']} == {960, 640}
            raw = cv2.imread(state['paths']['original'])
            mask = cv2.imread(state['paths']['testbed_source_mask'], 0)
            region = result['detection_region']
            for detection in result['detections']:
                x1, y1, x2, y2 = detection['bbox']
                cx, cy = (x1+x2)/2, (y1+y2)/2
                assert region['x'] <= cx < region['x']+region['w']
                assert region['y'] <= cy < region['y']+region['h']
                assert mask[(y1+y2)//2, (x1+x2)//2] > 0
            (output/'replay.png').write_bytes(Path(result['image']).read_bytes())
            result.pop('image')
            result.pop('onnx_image')
        finally:
            service._executor.shutdown(wait=True)
    assert not BLOCKED_CONNECTIONS, BLOCKED_CONNECTIONS
    report = dict(synthetic_checks='passed', replay_session=sid, result=result, external_connections=BLOCKED_CONNECTIONS)
    (output/'verification.json').write_text(json.dumps(report, indent=2))
    print('Bead ensemble checks passed:', result['count'], 'merged beads;', result['models'], flush=True)


if __name__ == '__main__':
    main()
