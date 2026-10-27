"""Replay every image capture in isolated storage; saved sessions remain read-only."""
from __future__ import annotations
import argparse
import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
from offline_runtime import configure, ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,default=ROOT.parent/'testing/capture-improvements-20261026')
    args=parser.parse_args(); output=args.audit_dir;output.mkdir(parents=True,exist_ok=True)
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis.testbed_segmentation import TestbedSegmenter
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    from backend.app.analysis.vision import separate_jewellery_items
    from backend.app.analysis import annotations
    from backend.app.services.capture_service import CaptureService
    from backend.app.storage.repository import CaptureRepository

    manifests=sorted((ROOT/'data/sessions').glob('20*/manifest.json'))
    # Freeze the evidence and database: the replay must never rewrite them.
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/sessions').rglob('*') if p.is_file()}
    hashes[str(ROOT/'data/app.db')]=hashlib.sha256((ROOT/'data/app.db').read_bytes()).hexdigest()
    segmenter=TestbedSegmenter(ROOT/'models/segmentation/essaetestbed.onnx',inset_px=12,outer_inset_ratio=.035)
    detector=ForeignObjectDetector(ROOT/'models/segmentation/anomolies.onnx')
    spec=importlib.util.spec_from_file_location('backend.app.analysis._before_foreign',output/'before/analysis/foreign_objects.py')
    baseline_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline_module)
    baseline=baseline_module.ForeignObjectDetector(ROOT/'models/segmentation/anomolies.onnx')
    native_segmenter_session, native_detector_session=segmenter.session,detector.session
    rows=[]
    with tempfile.TemporaryDirectory(prefix='capture-regression-') as tmp:
        repo=CaptureRepository(Path(tmp)/'app.db',Path(tmp)/'sessions')
        for path in manifests:
            old=json.loads(path.read_text());sid=path.parent.name
            raw=cv2.imread(str(path.parent/'capture/original.png'))
            camera=SimpleNamespace(snapshot_frame=lambda: (raw,old['camera']))
            scale=Mock();scale.snapshot.return_value={**old['weight'],'connected':True,'age_ms':0}
            settings=SimpleNamespace(get=lambda:copy.deepcopy(old['settings_snapshot']))
            service=CaptureService(camera,scale,settings,repo);service._testbed_segmenter=segmenter
            cache=output/'cache'/f'{sid}.npz'
            if cache.exists():
                data=np.load(cache)
                segmenter.session=SimpleNamespace(run=lambda *a: [data['tp'],data['tt']])
                detector.session=SimpleNamespace(run=lambda *a: [data['fp'],data['ft']])
            else:
                tensor,_=segmenter._preprocess(raw);tp,tt=native_segmenter_session.run(None,{segmenter.input_name:tensor})
                tensor,_=detector._preprocess(raw);fp,ft=native_detector_session.run(None,{detector.input_name:tensor})
                data=dict(tp=tp,tt=tt,fp=fp,ft=ft)
                segmenter.session=SimpleNamespace(run=lambda *a:[data['tp'],data['tt']])
                detector.session=SimpleNamespace(run=lambda *a:[data['fp'],data['ft']])
            baseline.session=detector.session
            previous=list(repo.sessions_dir.iterdir())
            try:
                state=service.capture()
            except RuntimeError as exc:
                assert list(repo.sessions_dir.iterdir())==previous, 'Rejected capture left partial artifacts'
                row={'id':sid,'capture_blocked':str(exc)}
                assert sid in {'20261026_105442_721b97c6','20261026_105624_d332e893'},row
                if '105624' in sid:
                    assert str(exc)=='Remove the object on the marker and recapture.'
                    scale.snapshot.assert_not_called()
                rows.append(row);print(sid,'BLOCKED',str(exc),flush=True);continue
            fresh_mask=cv2.imread(state['paths']['testbed_source_mask'],0)
            saved_mask=cv2.imread(str(path.parent/'capture/testbed_source_mask.png'),0)
            same=bool(np.array_equal(fresh_mask,saved_mask))
            if not same:
                assert sid in {'20261026_110519_cc11684c','20261026_111313_37478345'},sid
            working=cv2.imread(state['paths']['working']); mask=cv2.imread(state['paths']['mask'],0)
            separated=separate_jewellery_items(working,mask,min_area_px=350)
            origin=state['rois']['processing']; components=[{'bbox':{**i['bbox'],'x':i['bbox']['x']+origin['x'],'y':i['bbox']['y']+origin['y']},'mask':i['mask']} for i in separated]
            check=detector.detect(raw,fresh_mask,components=components)
            old_check=baseline.detect(raw,saved_mask)
            row={'id':sid,'capture_blocked':None,'tray_mask_unchanged':same,
                 'hole':state['testbed']['hole_validation'],'foreign_check':check,
                 'saved_status':old['status'],'baseline_foreign_passed':old_check.get('passed'), 'saved_foreign_passed':(old.get('foreign_object_check') or {}).get('passed')}
            # Every newly rejected image must have reviewed contamination/merged
            # pieces. Clean necklace+earring sets must never hit the placement gate.
            expected_placement={
                '20261024_175047_cf1b9480','20261024_175104_c560ac42',
                '20261026_110056_a3ab09b5','20261026_110154_ccd8eb1f','20261026_110240_7c054ea6',
                '20261026_110436_2d9dc50c','20261026_110519_cc11684c'}
            assert check['placement_check']['passed'] == (sid not in expected_placement),(sid,check['placement_check'])
            if old_check.get('passed') is False:
                assert not check['passed'],(sid,'Lost a previous contaminant rejection')
            newly_found={'20261026_111313_37478345'}
            if old_check.get('passed') is True and sid not in newly_found:
                assert check['passed'],(sid,'New unexpected foreign-object rejection')
            if sid in newly_found: assert not check['passed'],(sid,'Loose object still hidden as hole')
            # All historical item pixels stay identical except the two repaired
            # holes, so existing jewellery classifier inputs are preserved.
            if same:
                for name in ('working','prepared','mask'):
                    assert np.array_equal(cv2.imread(state['paths'][name],cv2.IMREAD_UNCHANGED),cv2.imread(str(path.parent/'capture'/f'{name}.png'),cv2.IMREAD_UNCHANGED)),(sid,name)
            rows.append(row);print(sid,'same_mask',same,'foreign',check['passed'],'placement',check['placement_check']['passed'],flush=True)

        # Blocking the actual marker must stop before scale access, model inference
        # and creating a session, even when the configured ROI is stale.
        source=next(p for p in manifests if '105213' in p.parent.name)
        old=json.loads(source.read_text());raw=cv2.imread(str(source.parent/'capture/original.png'))
        points=np.array(old['calibration']['corners'],np.int32); x,y,w,h=cv2.boundingRect(points)
        blocked=[]
        for fraction in (0.25,0.5,1.0):
            frame=raw.copy();cv2.rectangle(frame,(x-2,y-2),(x+w+2,y+int(h*fraction)),(150,150,150),-1)
            scale=Mock();settings=SimpleNamespace(get=lambda:copy.deepcopy(old['settings_snapshot']))
            service=CaptureService(SimpleNamespace(snapshot_frame=lambda:(frame,old['camera'])),scale,settings,repo)
            service._get_testbed_segmenter=Mock(side_effect=AssertionError('Model ran on blocked tag'))
            previous=list(repo.sessions_dir.iterdir())
            try: service.capture()
            except RuntimeError as exc: assert str(exc)=='Remove the object on the marker and recapture.'
            else: raise AssertionError(f'Partially blocked marker accepted: {fraction}')
            scale.snapshot.assert_not_called();assert list(repo.sessions_dir.iterdir())==previous
            blocked.append(fraction)

    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items()),'Saved evidence changed'
    assert len(rows)==63
    result={'sessions':rows,'blocked_marker_fractions':blocked,'original_files_unchanged':True,
            'false_positive_placement_rejections_on_reviewed_clean_captures':0,
            'limitations':['No live hardware test.','Jewellery model misses some touching earrings/bangles.','Pen and some edge objects remain missed.','Finite saved-image validation does not guarantee zero future false positives.']}
    (output/'regression.json').write_text(json.dumps(result,indent=2))
    print('PASS: all 63 image captures; saved evidence unchanged; blocked-marker checks passed',flush=True)


if __name__=='__main__': main()
