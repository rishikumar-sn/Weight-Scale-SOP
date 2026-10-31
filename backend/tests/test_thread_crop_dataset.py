import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scripts.build_thread_crop_dataset import DatasetBuilder, thread_assignment


def options(tmp_path):
    model=tmp_path/'thread.onnx';model.write_bytes(b'test-model')
    seeds=tmp_path/'seeds';seeds.mkdir(exist_ok=True)
    return SimpleNamespace(output=tmp_path/'dataset',sessions=tmp_path/'sessions',thread_model=model,
        seed_dir=[seeds],resume=True,min_thread_pixels=24,min_thread_share=.005,
        thread_confidence=.5,min_prediction_confidence=.5)


def test_thread_evidence_is_item_local_and_uncertain_is_review():
    assert thread_assignment(0,400)[0]=='Jewel_without_Thread'
    assert thread_assignment(50,400)[0]=='Jewel_with_Thread'
    assert thread_assignment(2,400)[0]=='review'
    assert thread_assignment(390,400)[0]=='review'


def test_mixed_capture_crops_resume_and_dedup(tmp_path):
    args=options(tmp_path)
    session=args.sessions/'sample';(session/'items').mkdir(parents=True);(session/'capture').mkdir()
    original=np.full((50,100,3),200,np.uint8)
    original[10:30,10:30]=(30,50,170);original[10:30,60:80]=(180,80,20)
    cv2.imwrite(str(session/'capture/original.png'),original)
    items=[]
    for i,x,label in [(1,10,'Finger Ring'),(2,60,'Chain / Necklace')]:
        crop=original[10:30,x:x+20]
        cv2.imwrite(str(session/f'items/item_{i:02d}_raw.png'),crop)
        cv2.imwrite(str(session/f'items/item_{i:02d}_mask.png'),np.full((20,20),255,np.uint8))
        items.append(dict(index=i,bbox=dict(x=x,y=10,w=20,h=20),predicted_label=label,confirmed=True,
            confidence=.9,paths={'raw':str(session/f'items/item_{i:02d}_raw.png'),'mask':str(session/f'items/item_{i:02d}_mask.png')}))
    state={'classification':{'items':items},'paths':{'original':str(session/'capture/original.png')},
        'status':'complete','rois':{'processing':{'x':0,'y':0}}}
    manifest=session/'manifest.json';manifest.write_text(json.dumps(state))
    builder=DatasetBuilder(args)
    thread=np.zeros(original.shape[:2],np.uint8);thread[10:15,60:70]=1
    calls=[]
    def detect(_):calls.append(1);return thread,{}
    builder.segmenter=SimpleNamespace(detect=detect)
    builder.process_session(manifest)
    assert len(calls)==1
    records={r['item_index']:r for r in builder.records}
    assert records[1]['assigned_label']=='Jewel_without_Thread'
    assert records[2]['assigned_label']=='Jewel_with_Thread'
    assert records[2]['thread_overlap_pixels']==50
    assert np.array_equal(cv2.imread(str(args.output/records[1]['output'])),original[10:30,10:30])
    resumed=DatasetBuilder(args)
    assert resumed.process_session(manifest)=='resumed'
    assert len(resumed.records)==2
    resumed.record({'key':'copy','session':'copy','assigned_label':'Jewel_without_Thread'},original[10:30,10:30],'images/Jewel_without_Thread')
    assert 'duplicate_of' in resumed.records[-1]
    resumed.summarize('complete')
    assert (args.output/'review.html').is_file()


def test_reviewed_seeds_override_auto_duplicate_labels(tmp_path):
    args=options(tmp_path)
    seed=args.seed_dir[0]/'Thread-Only'/'session_item_01_raw.png';seed.parent.mkdir()
    pixels=np.full((20,20,3),80,np.uint8);cv2.imwrite(str(seed),pixels)
    builder=DatasetBuilder(args);builder.add_seeds()
    builder.record({'key':'auto','session':'session','assigned_label':'Jewel_without_Thread'},pixels,'images/Jewel_without_Thread')
    assert builder.records[-1]['duplicate_of'].startswith('images/Thread-Only/')
    assert len(list((args.output/'images').rglob('*.png')))==1
