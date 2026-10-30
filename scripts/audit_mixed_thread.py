"""Replay the mixed thread failure; save review crops without changing sessions."""
from pathlib import Path
import copy,json,shutil,sys,tempfile
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import cv2,numpy as np
    from PIL import Image
    from backend.app.analysis.thread_classifier import JewelThreadClassifier
    from backend.app.services.analysis_service import AnalysisService,PredictionResult
    from backend.app.storage.repository import CaptureRepository
    output=ROOT.parent/'testing/mixed-thread-20261030';output.mkdir(parents=True,exist_ok=True)
    gate=JewelThreadClassifier(ROOT/'models/classification/jewel-thread-small.onnx')
    mixed=ROOT/'data/sessions/20261030_150638_fca8b158'
    samples=[(mixed/'items/item_01_raw.png','jewel_without_thread','Right-hand jewel in user example'),
             (mixed/'items/item_02_raw.png','thread_only','Left-hand thread-only object identified by user'),
             (ROOT/'data/sessions/20261030_144133_8c1dc894/items/item_01_raw.png','thread_only','Saved tassel-only failure')]
    samples += [(p,'jewel_without_thread','Visually inspected ring') for p in sorted((ROOT/'data/sessions/20261028_162305_dcc85c20/items').glob('*_raw.png'))]
    samples += [(ROOT/f'data/sessions/{session}/items/item_01_raw.png','jewel_with_thread','Saved jewellery with attached thread') for session in ['20261024_111330_a1093e17','20261024_111503_79b46faa']]
    records=[]
    for path,expected,basis in samples:
        image=cv2.imread(str(path));actual=gate.classify(image);height,width=image.shape[:2];side=max(height,width)
        square=np.full((side,side,3),255,np.uint8);y,x=(side-height)//2,(side-width)//2;square[y:y+height,x:x+width]=image
        padded=gate.classify(square)
        destination=output/'review_samples'/expected/(path.parent.parent.name+'_'+path.name)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        records.append({'source':str(path.relative_to(ROOT)),'expected_label':expected,'label_basis':basis,
                        'prediction':actual,'white_square_probe':padded,'review_crop':str(destination.relative_to(output)),
                        'correct_class':actual['label']==expected,'correct_allow_block':actual['passed']==(expected!='thread_only')})
        print(path.parent.parent.name,path.name,expected,actual['label'],actual['confidence'],flush=True)
    source=json.loads((mixed/'manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='mixed-thread-audit-') as temporary:
        repo=CaptureRepository(Path(temporary)/'test.db',Path(temporary)/'sessions');service=AnalysisService(repo)
        try:
            state=copy.deepcopy(source);state.update(id='mixed-replay',status='captured');repo.save(state)
            (repo.session_dir(state['id'])/'capture').mkdir(exist_ok=True);service._jewel_thread=gate
            # Real separation, anomaly model and thread gate; isolate the later
            # jewellery-type classifier since this audit concerns the gate.
            classifier=Mock();classifier.text_model_id='audit-type-stub';classifier.check_non_jewelry.return_value=(False,'')
            photo=Image.new('RGB',(32,32))
            classifier.classify_image.return_value=PredictionResult(label='Chain / Necklace',confidence=.99,scores=[],image_path='',original_image=photo,cropped_image=photo)
            with patch.object(service,'_get_classifier',return_value=classifier):replay=service.classify(state['id'])
            assert replay['foreign_object_check']['passed']
            checks=replay['thread_only_check']['items'];assert len(checks)==2
            # Record the failure; these assertions do not validate accuracy.
            assert checks[0]['label']=='jewel_without_thread' and checks[1]['label']=='jewel_with_thread'
            assert replay['status']=='awaiting_confirmation'
            shutil.copyfile(replay['paths']['evidence'],output/'mixed-current-result.jpg')
            (output/'mixed-replay.json').write_text(json.dumps(replay,indent=2)+'\n')
        finally:service.shutdown()
    result={'status':'audit_complete','model_accuracy_status':'known_failures','mixed_thread_false_accept_reproduced':True,
            'records':records,'correct_allow_block_count':sum(r['correct_allow_block'] for r in records),'sample_count':len(records),
            'note':'Curated failure/regression set, not an accuracy benchmark. Review labels before training. Keep related objects/sessions together in dataset splits.'}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT COMPLETE: mixed false acceptance reproduced; review crops saved; model needs retraining')

if __name__=='__main__':main()
