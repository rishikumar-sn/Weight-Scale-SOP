"""Replay foreground ownership and inspect ambiguous SigLIP embeddings, read-only.

Outputs live under data/audits/component_presence_20261031. Source captures and
the production correction gallery are never written. All models are local.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import cv2
import numpy as np
import onnxruntime as ort

OUT = ROOT/'data/audits/component_presence_20261031'


def bounded_factory():
    original = ort.InferenceSession
    def factory(*args, **kwargs):
        options = kwargs.get('sess_options') or ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        kwargs['sess_options'] = options
        return original(*args, **kwargs)
    return factory


def select_sessions():
    all_states = []
    for p in sorted((ROOT/'data/sessions').glob('2026*/manifest.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        if all((p.parent/'capture'/f'{name}.png').is_file() for name in
               ['working','mask','original','testbed_mask','testbed_source_mask']):
            all_states.append((p.parent,s))
    recent = [(p,s) for p,s in all_states if p.name >= '20261015']
    # Deterministic historical coverage across days, in addition to ALL recent captures.
    older = [(p,s) for p,s in all_states if p.name < '20261015' and
             (s.get('classification') or {}).get('confirmed')]
    selected = {p.name:(p,s) for p,s in recent}
    for index in np.linspace(0,len(older)-1,min(64,len(older)),dtype=int):
        p,s = older[index]; selected[p.name]=(p,s)
    return sorted(selected.values(),key=lambda pair:pair[0].name)


def load_components(folder,state):
    from backend.app.analysis.vision import separate_jewellery_items
    image=cv2.imread(str(folder/'capture/working.png'))
    mask=cv2.imread(str(folder/'capture/mask.png'),0)
    valid=cv2.imread(str(folder/'capture/testbed_mask.png'),0)
    settings=((state.get('settings_snapshot') or {}).get('analysis') or {}).get('item_separation') or {}
    return separate_jewellery_items(image,mask,min_area_px=int(settings.get('min_area_px',350)),
        min_area_ratio=float(settings.get('min_area_ratio',.00045)),
        padding_px=int(settings.get('padding_px',18)),valid_area_mask=valid)


def support():
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    with patch.object(ort,'InferenceSession',bounded_factory()):
        detector=ForeignObjectDetector(ROOT/'models/segmentation/anomolies.onnx')
    rows=[]
    for folder,state in select_sessions():
        components=load_components(folder,state);origin=state['rois']['processing']
        projected=[{'bbox':{**i['bbox'],'x':i['bbox']['x']+origin['x'],
                   'y':i['bbox']['y']+origin['y']},'mask':i['mask']} for i in components]
        raw=cv2.imread(str(folder/'capture/original.png'))
        valid=cv2.imread(str(folder/'capture/testbed_source_mask.png'),0)
        check=detector.detect(raw,valid,components=projected)
        detail=[]
        for idx,item in enumerate(components,1):
            shares=[m['component_share'] for d in check['jewellery_instances']
                    for m in d['component_overlaps'] if m['component_index']==idx]
            fg=item['raw_crop_bgr'][item['mask']>0].astype(np.float32)
            detail.append({'component':idx,'bbox':item['bbox'],'area_px':item['area_px'],
                'jewellery_share':max(shares,default=0),
                'median_chroma':float(np.median(np.ptp(fg,axis=1))),
                'median_luma':float(np.median(fg.mean(axis=1)))})
        row={'id':folder.name,'confirmed':bool((state.get('classification') or {}).get('confirmed')),
             'saved_labels':(state.get('classification') or {}).get('confirmed_labels'),
             'status':state.get('status'),'check':check,'components':detail}
        rows.append(row)
        (OUT/'support.json').write_text(json.dumps(rows,indent=2))
        print(folder.name,len(detail),'unsupported',sum(c['jewellery_share']<.2 for c in detail),flush=True)


def panel(image,title):
    tile=np.full((245,320,3),245,np.uint8)
    scale=min(310/image.shape[1],205/image.shape[0])
    image=cv2.resize(image,None,fx=scale,fy=scale)
    tile[35:35+image.shape[0],5:5+image.shape[1]]=image
    cv2.putText(tile,title[:52],(4,18),0,.36,(0,0,0),1)
    return tile


def embeddings():
    import torch
    from PIL import Image
    from Classification.siglip384_classifier import Siglip384Classifier,GALLERY_PATH
    torch.set_num_threads(4)
    copied_gallery=OUT/'gallery_snapshot.npz'
    if GALLERY_PATH.exists(): shutil.copyfile(GALLERY_PATH,copied_gallery)
    with patch.object(ort,'InferenceSession',bounded_factory()):
        classifier=Siglip384Classifier(gallery_path=copied_gallery)
    (OUT/'class_prototypes.json').write_text(json.dumps({
        'model':classifier.text_model_id,'prompt_hash':classifier.prompt_hash,
        'note':'Class/group vectors are normalized averages of expanded prompts; gallery stores indices/labels, not source filenames.',
        'classes':[{'label':d.label,'group':d.group,'prompts':list(d.prompts)}
                   for d in classifier.class_prompt_definitions],
        'groups':classifier.group_prompt_config},indent=2))
    rows=[];sheets=[]
    support_rows=json.loads((OUT/'support.json').read_text())
    for record in support_rows:
        candidates=[c for c in record['components'] if c['jewellery_share']<.2]
        # Also inspect ALL recent multi-component captures, including supported fragments.
        if record['id']>='20261015' and len(record['components'])>1:
            candidates=record['components']
        if not candidates:continue
        folder=ROOT/'data/sessions'/record['id']
        state=json.loads((folder/'manifest.json').read_text())
        components=load_components(folder,state)
        for c in candidates:
            item=components[c['component']-1];row={'id':record['id'],**c,'views':{}}
            for view,key in [('raw','raw_crop_bgr'),('masked','crop_bgr')]:
                pil=Image.fromarray(cv2.cvtColor(item[key],cv2.COLOR_BGR2RGB))
                a=classifier._embed_image(pil);b=classifier._embed_image(classifier._crop_foreground(pil))
                merged=(a+b)/2;merged=merged/max(np.linalg.norm(merged),1e-8)
                cos=merged@classifier.text_embeddings.T
                prob=classifier._compute_class_probabilities(cos,merged)
                groups=merged@classifier.group_embeddings.T
                gallery=classifier.gallery
                neighbors=[]
                if gallery.embeddings is not None:
                    similarities=gallery.embeddings@merged
                    neighbors=[{'index':int(i),'label':gallery.labels[i],'cosine':float(similarities[i])}
                               for i in np.argsort(similarities)[-5:][::-1]]
                row['views'][view]={'class_matches':sorted([
                    {'label':label,'cosine':float(cos[i]),'category_probability':float(prob[i])}
                    for i,label in enumerate(classifier.labels)],key=lambda x:x['category_probability'],reverse=True),
                    'group_matches':sorted([{'label':label,'cosine':float(groups[i])}
                    for i,label in enumerate(classifier.group_labels)],key=lambda x:x['cosine'],reverse=True),
                    'gallery_neighbors':neighbors}
            rows.append(row)
            title=f"{record['id']} C{c['component']} support={c['jewellery_share']:.2f}"
            sheets.append(panel(item['raw_crop_bgr'],title))
            sheets.append(panel(item['crop_bgr'],row['views']['masked']['class_matches'][0]['label']))
            (OUT/'embeddings.json').write_text(json.dumps(rows,indent=2))
            print(title, row['views']['masked']['class_matches'][0]['label'],flush=True)
    for page,start in enumerate(range(0,len(sheets),24)):
        tiles=sheets[start:start+24]
        while len(tiles)%4:tiles.append(np.full((245,320,3),245,np.uint8))
        cv2.imwrite(str(OUT/f'components_{page:02d}.jpg'),np.concatenate([
            np.concatenate(tiles[i:i+4],axis=1) for i in range(0,len(tiles),4)]))


def service_replay():
    """Real production service, models and source pixels; only temporary outputs."""
    from copy import deepcopy
    import tempfile
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    from backend.app.analysis.thread_classifier import JewelThreadClassifier
    from Classification.siglip384_classifier import Siglip384Classifier,GALLERY_PATH
    import torch
    torch.set_num_threads(4)
    cases={
        '20261021_175152_30daafe4':'rejected', # packet glints
        '20261021_175409_a2850223':'rejected', # detected packet
        '20261021_182641_e20ab9ca':'rejected', # detected packet
        '20261021_151923_bec937e6':'rejected', # tray edge
        '20261021_154441_708c5f08':'rejected', # empty tray
        '20261023_115920_78d03b75':'rejected', # detached rim
        '20261022_182620_1c13fa64':'rejected', # marker fragment
        '20261030_181915_f80d25c3':'rejected', # paper edge strips, not authenticity
        '20261027_113142_81835fca':'awaiting_confirmation',
        '20261021_175701_21511494':'rejected', # existing thread-only model false rejection
        '20261022_183109_2247a045':'awaiting_confirmation',
        '20261023_112624_26c35dc9':'rejected', # existing thread-only gate on small genuine items
    }
    with tempfile.TemporaryDirectory(prefix='foreground-presence-replay-') as temp:
        copied_gallery=Path(temp)/'gallery.npz'
        if GALLERY_PATH.exists():shutil.copyfile(GALLERY_PATH,copied_gallery)
        with patch.object(ort,'InferenceSession',bounded_factory()):
            classifier=Siglip384Classifier(gallery_path=copied_gallery)
            foreign=ForeignObjectDetector(ROOT/'models/segmentation/anomolies.onnx')
            thread=JewelThreadClassifier(ROOT/'models/classification/jewel-thread-small.onnx')
        class Repository:
            state=None
            def get(self,_id):return deepcopy(self.state)
            def save(self,state):self.state=deepcopy(state)
            def session_dir(self,sid):return Path(temp)/sid
        repo=Repository();service=AnalysisService(repo)
        service._classifier=classifier;service._foreign_objects=foreign;service._jewel_thread=thread
        rows=[]
        try:
            for sid,expected in cases.items():
                folder=ROOT/'data/sessions'/sid
                (repo.session_dir(sid)/'capture').mkdir(parents=True,exist_ok=True)
                state=json.loads((folder/'manifest.json').read_text())
                for key,path in state['paths'].items():
                    candidate=folder/'capture'/Path(path.replace('\\','/')).name
                    if candidate.is_file():state['paths'][key]=str(candidate)
                state.update(status='captured',classification={},result={},route={})
                for key in ['foreign_object_check','component_presence_check','rejection_kind','rejection_reason']:
                    state.pop(key,None)
                repo.state=state
                result=service.classify(sid)
                assert result['status']==expected,(sid,result['status'],result.get('rejection_reason'))
                if sid in {'20261021_175701_21511494','20261023_112624_26c35dc9'}:
                    assert result['rejection_kind']=='thread_only'
                    assert result['component_presence_check']['passed']
                if expected=='rejected':
                    assert result['classification']['count']==0
                    try:service.confirm_and_start(sid,'Finger Ring',True,save_only=True)
                    except RuntimeError:pass
                    else:raise AssertionError('Rejected capture allowed confirmation/learning')
                rows.append({'id':sid,'status':result['status'],'kind':result.get('rejection_kind'),
                    'labels':result['classification'].get('labels'),
                    'presence':result.get('component_presence_check')})
                (OUT/'service_replay.json').write_text(json.dumps(rows,indent=2))
                print(sid,result['status'],result.get('rejection_kind'),flush=True)
            (OUT/'service_replay.json').write_text(json.dumps(rows,indent=2))
        finally:service.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['support','embeddings','service'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    {'support':support,'embeddings':embeddings,'service':service_replay}[args.mode]()
