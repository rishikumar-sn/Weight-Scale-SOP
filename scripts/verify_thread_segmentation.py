"""Replay thread inference and verify pixel exclusion without changing saved captures."""
from pathlib import Path
import json
import sys
from unittest.mock import patch
from offline_runtime import configure, ROOT


def main():
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis.thread_segmentation import ThreadSegmenter, project_thread_mask
    from backend.app.services.analysis_service import AnalysisService, stone_detection

    cv2.setNumThreads(1)
    output = ROOT.parent / 'testing/thread-segmentation'
    output.mkdir(parents=True, exist_ok=True)
    detector = ThreadSegmenter(ROOT / 'models/segmentation/thread.onnx')
    # Rectangular camera frames, negative virtual-square origins, and padding.
    source = np.zeros((70, 120), np.uint8)
    source[5:20, 10:45] = 1
    local = project_thread_mask(source, (50, 70), -10, -5)
    assert np.array_equal(local[5:, 10:], source[:45, :60])
    assert not np.any(project_thread_mask(source, (10, 10), 150, 100))
    # No detections must be a valid empty mask rather than an inference error.
    tensor = np.zeros((1, 37, 1), np.float32)
    proto = np.zeros((1, 32, 240, 240), np.float32)
    with patch.object(detector.session, 'run', return_value=[tensor, proto]):
        empty, _ = detector.detect(np.zeros((70, 120, 3), np.uint8))
        assert empty.shape == (70, 120) and not np.any(empty)

    # Duplicate boxes are suppressed while disconnected thread pieces survive.
    tensor = np.zeros((1, 37, 3), np.float32)
    tensor[0, :5, :] = np.array([[300, 302, 650], [400, 402, 500],
                                [180, 180, 150], [120, 120, 150], [.9, .8, .85]])
    tensor[0, 5, :] = 10
    proto[:, 0] = 1
    with patch.object(detector.session, 'run', return_value=[tensor, proto]):
        union, metadata = detector.detect(np.zeros((600, 900, 3), np.uint8))
        assert len(metadata['confidences']) == 2
        assert cv2.connectedComponents(union)[0] == 3

    class Repository:
        def session_dir(self, capture_id):
            return output / capture_id

    service = AnalysisService(Repository())
    service._thread = detector
    summaries = []
    try:
        for sid in ['20261027_124350_d295df66', '20261027_124653_3a6196d9', '20261027_130309_86bafb21']:
            state = json.loads((ROOT / 'data/sessions' / sid / 'manifest.json').read_text())
            item = state['classification']['items'][0]
            raw = cv2.imread(state['paths']['original'])
            assert raw is not None
            source_mask, metadata = detector.detect(raw)
            origin, bbox = state['rois']['processing'], item['bbox']
            image = cv2.imread(item['paths']['working'])
            foreground = cv2.imread(item['paths']['mask'], 0)
            local = project_thread_mask(source_mask, foreground.shape,
                origin['x'] + bbox['x'], origin['y'] + bbox['y'])
            local &= (foreground > 0).astype(np.uint8)
            target = output / sid
            target.mkdir(exist_ok=True)
            cv2.imwrite(str(target / 'thread-mask.png'), source_mask * 255)
            cv2.imwrite(str(target / 'thread-local-mask.png'), local * 255)
            # Spy on the real pipeline to inspect the final instance union and overlay.
            original_analyze = stone_detection.analyze_image_bgr
            inspected = []
            def analyze(*args, **kwargs):
                kwargs['stone_v2_debug_dir'] = target / 'debug'
                analysis = original_analyze(*args, **kwargs)
                for report, view in zip(analysis['report']['jewels'], analysis['jewel_views']):
                    x, y, w, h = report['bbox_global']
                    excluded = local[y:y+h, x:x+w] > 0
                    # Display pixels are preserved (same correction as normal display).
                    display = stone_detection.apply_color_correction(image[y:y+h, x:x+w],
                        kwargs.get('color_correction'), mask=foreground[y:y+h, x:x+w])
                    assert np.array_equal(view['overlay_bgr'][excluded], display[excluded])
                    final_path = target / 'debug' / f"jewel_{report['jewel_id']:02d}" / '05_refined_instances.png'
                    if final_path.is_file():
                        final = cv2.imread(str(final_path), 0)
                        assert not np.any(final[excluded]), 'Thread pixels leaked into stone instances'
                    inspected.append(int(np.count_nonzero(excluded)))
                return analysis
            with patch.object(stone_detection, 'analyze_image_bgr', side_effect=analyze):
                result = service._run_stones(state, item)
            assert result['thread']['found'] == bool(np.any(local))
            assert result['thread']['excluded_pixels'] == int(np.count_nonzero(local))
            assert result['thread']['input_shape'] == list(raw.shape)
            # Other categories must not invoke the model or receive an exclusion mask.
            other = {**item, 'confirmed_label': 'Earrings / Nosepin'}
            with patch.object(service, '_get_thread', side_effect=AssertionError('Wrong category')), patch.object(
                    stone_detection, 'analyze_image_bgr', wraps=original_analyze) as spy:
                other_result = service._run_stones({**state, 'id': sid + '-other'}, other)
                assert spy.call_args.kwargs['stone_exclusion_mask'] is None
                assert 'thread' not in other_result
            summaries.append({'capture': sid, 'thread': result['thread'],
                              'stones_found': result['found'], 'inspected_candidate_excluded_pixels': inspected})
            print(json.dumps(summaries[-1]), flush=True)
    finally:
        service.shutdown()
    # Only the excluded colorful patch disappears; the neighboring stone survives.
    image = np.full((160, 240, 3), (30, 155, 205), np.uint8)
    image[40:85, 40:80] = (160, 30, 170)
    image[45:90, 140:180] = (160, 30, 170)
    mask = np.full(image.shape[:2], 255, np.uint8)
    exclusion = np.zeros_like(mask)
    exclusion[25:100, 25:100] = 255
    candidate = {'crop_bgr': image, 'crop_mask': mask, 'bbox_global': (0, 0, 240, 160)}
    baseline = stone_detection.analyze_image_bgr(image, zoom_scale=1, preset_candidates=[candidate])
    clipped = stone_detection.analyze_image_bgr(image, zoom_scale=1, preset_candidates=[candidate],
        stone_exclusion_mask=exclusion, stone_v2_debug_dir=output / 'synthetic')
    final = cv2.imread(str(output / 'synthetic/jewel_01/05_refined_instances.png'), 0)
    assert baseline['report']['stone_instance_count'] == 2
    assert clipped['report']['stone_instance_count'] == 1
    assert not np.any(final[exclusion > 0])
    assert np.count_nonzero(final[45:90, 140:180]) > 1700
    assert np.all(clipped['jewel_views'][0]['masked_bgra'][:, :, 3] == 255)
    # All pixels excluded: valid empty stone result while keeping the jewel visible.
    image = np.full((120, 180, 3), (30, 155, 205), np.uint8)
    image[30:90, 60:120] = (180, 30, 160)
    mask = np.full(image.shape[:2], 255, np.uint8)
    result = stone_detection.analyze_image_bgr(image, zoom_scale=1,
        preset_candidates=[{'crop_bgr': image, 'crop_mask': mask, 'bbox_global': (0, 0, 180, 120)}],
        stone_exclusion_mask=mask)
    assert result['report']['stone_instance_count'] == 0
    assert np.all(result['jewel_views'][0]['masked_bgra'][:, :, 3] == 255)
    (output / 'verification.json').write_text(json.dumps(summaries, indent=2))
    print('Passed: full-frame model replay, mask alignment, visible thread, zero thread/stone overlap, category gating, empty predictions and fully excluded jewel.')


if __name__ == '__main__':
    main()
