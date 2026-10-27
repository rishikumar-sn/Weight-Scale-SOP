"""Exercise classification, labels and confirmation guards in temporary sessions."""
from pathlib import Path
import copy,json,tempfile
from types import SimpleNamespace
from unittest.mock import Mock,patch
from offline_runtime import configure,ROOT

def main():
    configure()
    import cv2
    from backend.app.services.capture_service import CaptureService
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository
    from backend.app.analysis.testbed_segmentation import TestbedSegmenter
    from backend.app.analysis import annotations
    out=ROOT.parent/'testing/capture-improvements-20261026';results=[]
    segmenter=TestbedSegmenter(ROOT/'models/segmentation/essaetestbed.onnx',inset_px=12,outer_inset_ratio=.035)
    with tempfile.TemporaryDirectory(prefix='capture-workflows-') as temp:
        repo=CaptureRepository(Path(temp)/'app.db',Path(temp)/'sessions');service=AnalysisService(repo)
        try:
            for short,expected,count in [
                ('105320','rejected',2),('105402','rejected',1),
                ('105708','rejected',None),('105213','awaiting_confirmation',2),
                ('105946','awaiting_confirmation',2),('110056','rejected',0),
                ('110519','rejected',0),('111313','rejected',1)]:
                path=next((ROOT/'data/sessions').glob('20261026_'+short+'*/manifest.json'));old=json.loads(path.read_text())
                raw=cv2.imread(str(path.parent/'capture/original.png'));settings=SimpleNamespace(get=lambda:copy.deepcopy(old['settings_snapshot']))
                scale=SimpleNamespace(snapshot=lambda:{**old['weight'],'connected':True,'age_ms':0})
                capture=CaptureService(SimpleNamespace(snapshot_frame=lambda:(raw,old['camera'])),scale,settings,repo);capture._testbed_segmenter=segmenter
                state=capture.capture();state=service.classify(state['id']);assert state['status']==expected,(short,state['status'])
                classification=state['classification'];assert 'Not Gold Jewelry' not in classification['labels'],(short,classification)
                if count is not None:assert classification['count']==count,(short,classification['count'])
                assert [i['index'] for i in classification['items']]==list(range(1,classification['count']+1))
                if expected=='rejected':
                    try:service.confirm_and_start(state['id'],None,False)
                    except RuntimeError:pass
                    else:raise AssertionError('Rejected capture started analysis')
                image=cv2.imread(state['paths']['evidence']);assert image is not None
                cv2.imwrite(str(out/f'workflow-{short}.jpg'),image)
                row={'source_session':path.parent.name,'status':state['status'],'count':classification['count'],'labels':classification['labels'],'other_objects':state.get('other_objects'),'reason':state.get('rejection_reason')};results.append(row);print(row,flush=True)

            # A semantic rejection must agree with review/API and cannot acquire a
            # J tag or contribute to jewellery count even without anomaly overlap.
            path=next((ROOT/'data/sessions').glob('20261026_105213*/manifest.json'));old=json.loads(path.read_text())
            raw=cv2.imread(str(path.parent/'capture/original.png'))
            state=capture.capture();classifier=service._get_classifier()
            with patch.object(classifier,'check_non_jewelry',side_effect=[(True,'test foreign object'),(False,'clear')]):
                mixed=service.classify(state['id'])
            assert mixed['status']=='rejected' and mixed['classification']['count']==1,mixed['classification']
            results.append({'synthetic_semantic_rejection':True,'count':mixed['classification']['count']})
        finally:service.shutdown()
    (out/'workflows.json').write_text(json.dumps(results,indent=2));print('PASS: workflow and annotation checks',flush=True)

if __name__=='__main__':main()
