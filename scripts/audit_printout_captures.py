"""Read-only capture audit. Writes only to data/audits; never confirms or learns.

Run with .venv/Scripts/python.exe scripts/audit_printout_captures.py depth|siglip
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

OUT = ROOT / 'data/audits/printout_20261030'
HISTORY = ['20261027_113142_81835fca', '20261027_114504_bf860150',
           '20261027_114922_bc70225d', '20261027_120205_3a3ecff1',
           '20261021_182241_10e23459', '20261021_182723_b0d4e371',
           '20261022_183109_2247a045', '20261023_112624_26c35dc9']


def samples():
    return sorted(p.name for p in (ROOT/'data/sessions').glob('20261030_*')) + HISTORY


def load(sid):
    folder = ROOT/'data/sessions'/sid
    state = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    raw = cv2.imread(str(folder/'capture/original.png'))
    r = state['rois']['processing']
    image = raw[r['y']:r['y']+r['h'], r['x']:r['x']+r['w']].copy()
    valid = cv2.imread(str(folder/'capture/testbed_mask.png'), 0)
    foreground = cv2.imread(str(folder/'capture/mask.png'), 0)
    assert valid.shape == foreground.shape == image.shape[:2], sid
    return folder, state, image, valid, foreground


def tile(im, title, width=330, height=350):
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    result = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(result, title[:48], (5, 20), 0, .42, (0, 0, 0), 1, cv2.LINE_AA)
    scale = min((width-10)/im.shape[1], (height-35)/im.shape[0])
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result[30:30+im.shape[0], 5:5+im.shape[1]] = im
    return result


def edge_features(image, valid):
    inner = cv2.erode((valid>0).astype(np.uint8), np.ones((45,45),np.uint8))
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 40, 100)
    edges[inner == 0] = 0
    side = max(image.shape[:2])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=55,
                            minLineLength=int(side*.22), maxLineGap=12)
    lengths = []
    overlay = image.copy()
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            length = float(np.hypot(x2-x1,y2-y1)) / side
            lengths.append(length)
            cv2.line(overlay, (x1,y1),(x2,y2),(0,0,255),2)
    return {'long_line_count':len(lengths), 'longest_line_fraction':max(lengths,default=0)}, overlay


def depth_metrics(depth, valid, foreground, state):
    inside = cv2.erode((valid>0).astype(np.uint8),np.ones((25,25),np.uint8)) > 0
    fg = (foreground>0) & inside
    bg = inside & (cv2.dilate((foreground>0).astype(np.uint8),np.ones((35,35),np.uint8))==0)
    tag = state['rois'].get('working_apriltag')
    if tag:
        x,y,w,h = (int(tag[k]) for k in ['x','y','w','h'])
        bg[max(0,y-20):y+h+20,max(0,x-20):x+w+20] = False
    low, high = np.percentile(depth[inside], [5,95])
    normalized = (depth-low)/max(float(high-low),1e-8)
    yy,xx = np.indices(depth.shape,dtype=np.float32)
    design = np.stack([xx/depth.shape[1], yy/depth.shape[0], np.ones_like(xx)],-1)
    coords = np.flatnonzero(bg)[::8]
    a = design.reshape(-1,3)[coords]; b = normalized.ravel()[coords]
    keep = np.ones(len(b),bool)
    for _ in range(4):
        coeff = np.linalg.lstsq(a[keep],b[keep],rcond=None)[0]
        err = b-a@coeff
        keep = np.abs(err-np.median(err)) <= max(.02, float(np.percentile(np.abs(err),80)))
    residual = normalized-design@coeff
    return {'foreground_pixels':int(fg.sum()),'background_pixels':int(bg.sum()),
            'tray_raw_p95_p05':float(high-low),
            'fg_minus_bg_median':float(np.median(normalized[fg])-np.median(normalized[bg])),
            'fg_depth_p90_p10':float(np.ptp(np.percentile(normalized[fg],[10,90]))),
            'fg_plane_abs_median':float(np.median(np.abs(residual[fg]))),
            'bg_plane_abs_median':float(np.median(np.abs(residual[bg])))}, normalized


def depth_run():
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
    torch.set_num_threads(4)
    model_dir = ROOT/'experiments/midas_depth_lab/models/depth_anything_v2_small_hf'
    processor = AutoImageProcessor.from_pretrained(model_dir,local_files_only=True)
    model = DepthAnythingForDepthEstimation.from_pretrained(model_dir,local_files_only=True).eval()
    results=[]
    for sid in samples():
        folder,state,image,valid,fg=load(sid)
        dest=OUT/sid;dest.mkdir(parents=True,exist_ok=True)
        edge,overlay=edge_features(image,valid)
        row={'id':sid,'kind':'printout' if sid.startswith('20261030') else 'confirmed_real',
             'scale':state.get('settings_snapshot',{}).get('scale',{}).get('model'),
             'edge':edge, 'runs':[]}
        panels=[tile(image,sid), tile(overlay,'Long straight edges / red'),tile(fg,'Saved foreground mask')]
        for context in ['raw_tray_crop','masked_tray']:
            inp=image.copy()
            if context=='masked_tray': inp[valid==0]=255
            batch=processor(images=Image.fromarray(cv2.cvtColor(inp,cv2.COLOR_BGR2RGB)),return_tensors='pt',size={'height':518,'width':518})
            start=time.monotonic()
            with torch.inference_mode():
                prediction=model(**batch).predicted_depth
                depth=torch.nn.functional.interpolate(prediction[:,None],size=image.shape[:2],mode='bicubic',align_corners=False)[0,0].numpy()
            assert np.isfinite(depth).all()
            metrics,norm=depth_metrics(depth,valid,fg,state)
            row['runs'].append({'context':context,'tensor_shape':list(batch['pixel_values'].shape),
                                'seconds':time.monotonic()-start,**metrics})
            np.save(dest/f'{context}_depth.npy',depth)
            heat=cv2.applyColorMap(np.uint8(np.clip(norm,0,1)*255),cv2.COLORMAP_INFERNO);heat[valid==0]=245
            panels.append(tile(heat,context+' / tray P05-P95'))
            cv2.imwrite(str(dest/f'{context}_input.png'),inp)
        cv2.imwrite(str(dest/'comparison.jpg'),np.concatenate(panels,axis=1))
        results.append(row)
        (OUT/'depth_results.json').write_text(json.dumps({'model':str(model_dir),'notes':[
            'Relative monocular depth only; no physical height ground truth.',
            'Each map scaled using only tray P05-P95; hue is not comparable physical height.',
            'Foreground includes segmentation artifacts; statistics are diagnostic, not labelled jewel pixels.',
            'Historical real sessions have differing jewellery and some differing trays; not a paired material study.'],
            'samples':results},indent=2))
        print(sid, row['kind'], 'lines',edge, 'depth',row['runs'][0]['fg_minus_bg_median'],flush=True)


def siglip_run():
    import torch
    from PIL import Image
    from Classification.siglip384_classifier import Siglip384Classifier
    from backend.app.analysis.vision import separate_jewellery_items
    import onnxruntime as ort
    from unittest.mock import patch
    torch.set_num_threads(4)
    # Keep this offline batch from oversubscribing CPU threads. Production is unchanged.
    session_factory = ort.InferenceSession
    def bounded_session(*args, **kwargs):
        options = kwargs.get('sess_options') or ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        kwargs['sess_options'] = options
        return session_factory(*args, **kwargs)
    with patch.object(ort, 'InferenceSession', bounded_session):
        classifier=Siglip384Classifier()
    saved=OUT/'siglip_results.json'
    results=json.loads(saved.read_text()) if saved.exists() else []
    completed={r['id'] for r in results}
    # Avoid repeating the exact raw embedding in both semantic and category calls.
    import hashlib
    embed = classifier._embed_image
    cache = {}
    def cached_embed(image):
        key = (image.size, hashlib.sha256(image.tobytes()).digest())
        if key not in cache:
            cache[key] = embed(image)
        return cache[key]
    classifier._embed_image = cached_embed
    for sid in samples():
        if sid in completed:
            continue
        folder,state,image,valid,fg=load(sid)
        (OUT/sid).mkdir(parents=True,exist_ok=True)
        working=cv2.imread(str(folder/'capture/working.png'))
        components=separate_jewellery_items(working,fg,min_area_px=350,valid_area_mask=valid)
        row={'id':sid,'kind':'printout' if sid.startswith('20261030') else 'confirmed_real','items':[]}
        tiles=[]
        for idx,item in enumerate(components,1):
            info={'component':idx,'bbox':item['bbox'],'area_px':item['area_px']}
            for name,img in [('raw',item['raw_crop_bgr']),('masked',item['crop_bgr'])]:
                pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                strong,reason=classifier.check_non_jewelry(pil)
                pred=classifier.classify_image(pil)
                info[name]={'label':pred.label,'confidence':float(pred.confidence),
                            'model_label':pred.model_label,'decision_source':pred.decision_source,
                            'strong_non_jewellery':strong,'semantic_reason':reason,
                            'gold_reason':pred.gold_verification_reason}
                tiles.append(tile(img,f'C{idx} {name}: {pred.label}',280,270))
            row['items'].append(info)
        if tiles:
            while len(tiles)%4:tiles.append(np.full((270,280,3),245,np.uint8))
            cv2.imwrite(str(OUT/sid/'components.jpg'),np.concatenate([np.concatenate(tiles[i:i+4],axis=1) for i in range(0,len(tiles),4)]))
        results.append(row)
        (OUT/'siglip_results.json').write_text(json.dumps(results,indent=2))
        print(sid,[(i['component'],i['masked']['label']) for i in row['items']],flush=True)


def ownership_run():
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    from backend.app.analysis.vision import separate_jewellery_items
    detector = ForeignObjectDetector(ROOT/'models/segmentation/anomolies.onnx')
    results=[]
    for sid in samples():
        folder,state,image,valid,fg=load(sid)
        working=cv2.imread(str(folder/'capture/working.png'))
        items=separate_jewellery_items(working,fg,min_area_px=350,valid_area_mask=valid)
        origin=state['rois']['processing']
        components=[{'bbox':{**i['bbox'],'x':i['bbox']['x']+origin['x'],
                            'y':i['bbox']['y']+origin['y']},'mask':i['mask']} for i in items]
        raw=cv2.imread(str(folder/'capture/original.png'))
        full_valid=cv2.imread(str(folder/'capture/testbed_source_mask.png'),0)
        check=detector.detect(raw,full_valid,components=components)
        support=[]
        for idx,item in enumerate(items,1):
            matches=[m['component_share'] for d in check['jewellery_instances']
                     for m in d['component_overlaps'] if m['component_index']==idx]
            support.append({'component':idx,'bbox':item['bbox'],
                            'max_jewellery_mask_share':max(matches,default=0)})
        tensor,_=detector._preprocess(raw)
        pred=detector.session.run(None,{detector.input_name:tensor})[0][0].T
        row={'id':sid,'components':support,'fresh_check':check,
             'unfiltered_class_maxima':{name:float(pred[:,4+k].max()) for k,name in detector.names.items()}}
        results.append(row)
        (OUT/'ownership_results.json').write_text(json.dumps(results,indent=2))
        print(sid,[(x['component'],round(x['max_jewellery_mask_share'],3)) for x in support],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['depth','siglip','ownership'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    {'depth':depth_run,'siglip':siglip_run,'ownership':ownership_run}[args.mode]()
