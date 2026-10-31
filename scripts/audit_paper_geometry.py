"""Read-only rectangular printout experiment; outputs only to data/audits.

No production rejection or learning is performed. Run with the project's venv.
"""
from pathlib import Path
import json
import argparse
import itertools
import csv
import hashlib
import html
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/audits/paper_geometry_20261031'


def tile(im, title):
    canvas = np.full((300, 300, 3), 245, np.uint8)
    scale = min(290/im.shape[1], 270/im.shape[0])
    im = cv2.resize(im, None, fx=scale, fy=scale)
    canvas[28:28+im.shape[0], 5:5+im.shape[1]] = im
    cv2.putText(canvas, title, (4, 17), 0, .36, (0, 0, 0), 1)
    return canvas


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []
    rows = []
    for p in sorted((ROOT/'data/sessions').glob('2026103*/manifest.json')):
        s = json.loads(p.read_text(encoding='utf8'))
        im = cv2.imread(str(p.parent/'capture/original.png'))
        if im is None:
            continue
        r = s['rois']['processing']
        im = im[r['y']:r['y']+r['h'], r['x']:r['x']+r['w']]
        panels.append(tile(im, p.parent.name))
        rows.append({'id':p.parent.name, 'classification':s.get('classification'),
                     'status':s.get('status')})
    for page, start in enumerate(range(0, len(panels), 20)):
        batch = panels[start:start+20]
        while len(batch)%4:
            batch.append(np.full((300,300,3),245,np.uint8))
        cv2.imwrite(str(OUT/f'inventory_{page}.jpg'), np.concatenate([
            np.concatenate(batch[i:i+4],axis=1) for i in range(0,len(batch),4)]))
    (OUT/'inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


def detect(image, valid, reference=None):
    """Experimental cues only; coordinates refer to resized processing crop."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    inside = cv2.erode(valid, np.ones((17,17),np.uint8)) > 0
    masks = []
    edges = cv2.Canny(gray, 12, 36)
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    for lo, hi in [(8,24),(20,60),(40,100)]:
        edge = cv2.Canny(gray,lo,hi)
        edge[~inside] = 0
        masks.append((f'canny_{lo}',cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))))
    grad = cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    masks.append(('sobel',np.uint8((grad>35)&inside)*255))
    delta = None
    if reference is not None:
        ref_lab = cv2.cvtColor(reference,cv2.COLOR_BGR2LAB).astype(np.float32)
        # Robust channel offsets accommodate exposure / white balance changes.
        offset = np.median((lab-ref_lab)[inside],axis=0)
        delta = np.linalg.norm(lab-ref_lab-offset,axis=2)
        for threshold in [6,12,20]:
            mask = np.uint8((delta>threshold)&inside)*255
            masks.append((f'reference_{threshold}',cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))))
    # Smooth platform model retains weak paper-vs-tray colour differences.
    smooth = cv2.GaussianBlur(lab,(0,0),35)
    difference = np.linalg.norm(lab-smooth,axis=2)
    masks.append(('local_colour',np.uint8((difference>7)&inside)*255))
    lsd = cv2.createLineSegmentDetector().detect(gray)[0]
    lines = [] if lsd is None else [l[0].tolist() for l in lsd if np.linalg.norm(l[0][:2]-l[0][2:])>35]
    # Join independently detected paper sides even when glare breaks a contour.
    usable=[]
    for line in lines:
        a,b=np.array(line).reshape(2,2)
        xy=np.rint([a,b,(a+b)/2]).astype(int)
        if np.any(xy<0) or np.any(xy[:,0]>=gray.shape[1]) or np.any(xy[:,1]>=gray.shape[0]):continue
        if np.all(inside[xy[:,1],xy[:,0]]):usable.append((a,b))
    usable=sorted(usable,key=lambda ab:-np.linalg.norm(ab[1]-ab[0]))[:70]
    pairs=[]
    for (a,b),(c,d) in itertools.combinations(usable,2):
        v=(b-a)/np.linalg.norm(b-a);u=(d-c)/np.linalg.norm(d-c)
        normal=np.array([-v[1],v[0]])
        separation=abs(np.dot(c-a,normal))
        if abs(np.dot(v,u))>.985 and separation>30:
            pairs.append(((a,b),(c,d),v))
    quads=[]
    def intersect(ab,cd):
        a,b=ab;c,d=cd
        mat=np.stack([b-a,-(d-c)],axis=1)
        if abs(np.linalg.det(mat))<1e-6:return None
        return a+np.linalg.solve(mat,c-a)[0]*(b-a)
    for first,second in itertools.combinations(pairs,2):
        if abs(np.dot(first[2],second[2]))>.25:continue
        q=np.array([intersect(first[0],second[0]),intersect(first[0],second[1]),
                    intersect(first[1],second[1]),intersect(first[1],second[0])])
        if not np.isfinite(q).all() or q.min()<0 or np.any(q[:,0]>=gray.shape[1]) or np.any(q[:,1]>=gray.shape[0]):continue
        # Segments must extend substantially along their proposed rectangle side.
        good=True
        for ab,indices in [(first[0],(0,1)),(second[1],(1,2)),(first[1],(2,3)),(second[0],(3,0))]:
            a,b=q[list(indices)];length=np.linalg.norm(b-a);v=(b-a)/max(length,1)
            positions=sorted([np.dot(ab[0]-a,v),np.dot(ab[1]-a,v)])
            overlap=max(0,min(length,positions[1])-max(0,positions[0]))
            if overlap/max(length,1)<.15:good=False;break
        if good:quads.append(np.rint(q).astype(np.int32).reshape(-1,1,2))
    candidates = []
    masks.append(('line_quadrilateral',None))
    for source,mask in masks:
        if mask is None:contours=quads
        else:contours,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image.shape[0]*image.shape[1]*.002 or area > inside.sum()*.85:
                continue
            for epsilon in [.012,.025,.045]:
                q = cv2.approxPolyDP(contour,epsilon*cv2.arcLength(contour,True),True)
                if len(q)!=4 or not cv2.isContourConvex(q):
                    continue
                q = q[:,0,:]
                lengths = np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
                if min(lengths)<30 or max(lengths)/min(lengths)>8:
                    continue
                region = np.zeros(gray.shape,np.uint8)
                cv2.fillConvexPoly(region,q,255)
                select = region>0
                if np.mean(inside[select])<.995:
                    continue
                supports=[]; jumps=[]; line_support=[]
                center=q.mean(axis=0)
                for a,b in zip(q,np.roll(q,-1,axis=0)):
                    pts = a[None,:]+np.linspace(.08,.92,80)[:,None]*(b-a)[None,:]
                    xy = np.rint(pts).astype(int)
                    supports.append(float(np.mean(distance[xy[:,1],xy[:,0]]<=2.5)))
                    normal=np.array([-(b-a)[1],(b-a)[0]],float)
                    normal/=max(np.linalg.norm(normal),1)
                    pin=np.rint(pts+normal*4).astype(int)
                    pout=np.rint(pts-normal*4).astype(int)
                    pin[:,0]=np.clip(pin[:,0],0,gray.shape[1]-1);pin[:,1]=np.clip(pin[:,1],0,gray.shape[0]-1)
                    pout[:,0]=np.clip(pout[:,0],0,gray.shape[1]-1);pout[:,1]=np.clip(pout[:,1],0,gray.shape[0]-1)
                    jumps.append(float(np.median(np.linalg.norm(lab[pin[:,1],pin[:,0]]-lab[pout[:,1],pout[:,0]],axis=1))))
                    v=(b-a)/np.linalg.norm(b-a); intervals=[]
                    for line in lines:
                        c,d=np.array(line).reshape(2,2)
                        lv=d-c; cosine=abs(np.dot(lv,v)/np.linalg.norm(lv))
                        if cosine>.98 and max(abs(np.dot(c-a,normal)),abs(np.dot(d-a,normal)))<4:
                            bounds=sorted([np.dot(c-a,v),np.dot(d-a,v)])
                            intervals.append((max(0,bounds[0]),min(np.linalg.norm(b-a),bounds[1])))
                    covered=np.zeros(100,bool)
                    for lo,hi in intervals:
                        covered |= (np.linspace(0,np.linalg.norm(b-a),100)>=lo)&(np.linspace(0,np.linalg.norm(b-a),100)<=hi)
                    line_support.append(float(covered.mean()))
                neutral=float(np.mean(hsv[:,:,1][select]<65))
                texture=float(np.median(grad[select]))
                # All four sides, a mostly neutral interior and line evidence.
                score=min(supports)*neutral
                candidate={'source':source,'quad':q.tolist(),'area':float(area),
                    'edge_support':supports,'line_support':line_support,'colour_jump':jumps,
                    'neutral_fraction':neutral,'texture_median':texture,'score':score,
                    'reference_delta':None if delta is None else float(np.median(delta[select]))}
                if not any(np.mean(np.linalg.norm(np.array(c['quad'])-q,axis=1))<8 for c in candidates):
                    candidates.append(candidate)
                break
    candidates.sort(key=lambda c:c['score'],reverse=True)
    return candidates, lines


def run(recent_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    ref_id='20261031_140034_91fbf4bc'
    ref_dir=ROOT/'data/sessions'/ref_id
    ref_state=json.loads((ref_dir/'manifest.json').read_text())
    ref=cv2.imread(str(ref_dir/'capture/original.png'))
    ref_corners=np.array(ref_state['calibration']['corners'],np.float32)
    rows=[];skipped=[]
    for p in sorted((ROOT/'data/sessions').glob('2026*/manifest.json')):
        s=json.loads(p.read_text(encoding='utf8'));sid=p.parent.name
        if sid[:8]>'20261031':continue  # Freeze this audit's visually reviewed capture window.
        recent=sid>='20261030'
        cls=s.get('classification') or {}
        if not recent and (recent_only or not cls.get('confirmed') or 'Not Gold' in str(cls.get('confirmed_label'))):
            continue
        if not (p.parent/'capture/testbed_mask.png').exists():
            skipped.append(sid);continue
        im=cv2.imread(str(p.parent/'capture/original.png'))
        valid=cv2.imread(str(p.parent/'capture/testbed_mask.png'),0)
        r=(s.get('rois') or {}).get('processing')
        if im is None or valid is None or not r:
            continue
        aligned=None;reference_status='unavailable_or_different_setup'
        # A reference from this actual camera/platform setup only. Tag corners
        # carry perspective; never use the image bounding rectangle as calibration.
        corners=(s.get('calibration') or {}).get('corners')
        if recent and corners is not None and im.shape==ref.shape:
            transform=cv2.getPerspectiveTransform(ref_corners,np.array(corners,np.float32))
            aligned=cv2.warpPerspective(ref,transform,(im.shape[1],im.shape[0]))
            aligned=aligned[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']]
            reference_status='tag_homography_exposure_compensated'
        im=im[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']]
        if im.shape[:2]!=valid.shape:
            continue
        scale=min(1,720/max(im.shape[:2]));size=(round(im.shape[1]*scale),round(im.shape[0]*scale))
        im=cv2.resize(im,size);valid=cv2.resize(valid,size,interpolation=cv2.INTER_NEAREST)
        tag=s['rois'].get('working_apriltag')
        if tag:
            x,y,w,h=[round(tag[k]*scale) for k in ['x','y','w','h']]
            cv2.rectangle(valid,(max(0,x-12),max(0,y-12)),(x+w+12,y+h+12),0,-1)
        if aligned is not None:aligned=cv2.resize(aligned,size)
        candidates,lines=detect(im,valid,aligned)
        # Frozen exploratory operating point, evaluated on every session below.
        passed=[c for c in candidates if c['score']>=.50 and sorted(c['line_support'])[1]>=.30 and np.mean(c['line_support'])>=.45 and np.median(c['colour_jump'])>=3]
        row={'id':sid,'recent':recent,'saved_confirmed':cls.get('confirmed',False),
             'saved_label':cls.get('confirmed_label'),'reference':reference_status,
             'anomaly_passed':(s.get('foreign_object_check') or {}).get('passed'),
             'fallback_reject':bool(passed) and (s.get('foreign_object_check') or {}).get('passed') is True,
             'detected':bool(passed),'best_score':max((c['score'] for c in candidates),default=0),
             'candidates':candidates[:15],'detections':passed}
        rows.append(row)
        if recent or passed:
            overlay=im.copy()
            for c in passed:cv2.polylines(overlay,[np.array(c['quad'],np.int32)],True,(0,0,255),2)
            for c in candidates[:1]:cv2.polylines(overlay,[np.array(c['quad'],np.int32)],True,(255,80,0),1)
            cv2.imwrite(str(OUT/f'{sid}.jpg'),overlay)
        print(sid,'PAPER' if passed else 'clear',f"best={row['best_score']:.3f}",flush=True)
        (OUT/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    (OUT/'skipped_missing_testbed_mask.json').write_text(json.dumps(skipped,indent=2))


def summarize(replay_anomalies=False):
    rows=json.loads((OUT/'results.json').read_text())
    if replay_anomalies:
        sys.path.insert(0,str(ROOT))
        import onnxruntime as ort
        from backend.app.analysis.foreign_objects import ForeignObjectDetector
        from unittest.mock import patch
        original=ort.InferenceSession
        def bounded(*args,**kwargs):
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            kwargs['sess_options']=options
            return original(*args,**kwargs)
        model=ROOT/'models/segmentation/anomolies.onnx'
        with patch.object(ort,'InferenceSession',bounded):detector=ForeignObjectDetector(model)
        replay=[]
        for row in rows:
            folder=ROOT/'data/sessions'/row['id']/'capture'
            raw=cv2.imread(str(folder/'original.png'))
            mask=cv2.imread(str(folder/'testbed_source_mask.png'),0)
            check=detector.detect(raw,mask)
            row['anomaly_passed']=check['passed']
            row['fallback_reject']=check['passed'] and row['detected']
            replay.append({'id':row['id'],'check':check})
            print('anomaly replay',row['id'],check['passed'],flush=True)
        (OUT/'anomaly_replay.json').write_text(json.dumps({'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'rows':replay},indent=2))
        (OUT/'results.json').write_text(json.dumps(rows,indent=2))
    summaries=[];labels=[];panels=[]
    for row in rows:
        sid=row['id']
        # Visual review of the inventory; saved classifications are NOT paper labels.
        paper=sid.startswith('20261030') or sid>='20261031_161326'
        mixed=sid.startswith(('20261031_161537','20261031_161545'))
        empty=sid=='20261031_140034_91fbf4bc'
        label='mixed_paper_and_real' if mixed else 'paper' if paper else 'empty' if empty else 'real_jewellery'
        row['visual_label']=label
        labels.append({'id':sid,'label':label,'basis':'visual contact-sheet review; historical controls also have saved human-confirmed jewellery labels'})
        summaries.append({'id':sid,'label':label,'anomaly_passed':row['anomaly_passed'],
                          'geometry_detected':row['detected'],'fallback_reject':row['fallback_reject']})
        folder=ROOT/'data/sessions'/sid
        state=json.loads((folder/'manifest.json').read_text())
        im=cv2.imread(str(folder/'capture/original.png'));r=state['rois']['processing']
        im=im[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']]
        panels.append(tile(im,sid))
        if not (OUT/f'{sid}.jpg').exists():cv2.imwrite(str(OUT/f'{sid}.jpg'),im)
    for page,start in enumerate(range(0,len(panels),20)):
        batch=panels[start:start+20]
        while len(batch)%4:batch.append(np.full((300,300,3),245,np.uint8))
        cv2.imwrite(str(OUT/f'all_cases_{page}.jpg'),np.concatenate([np.concatenate(batch[i:i+4],axis=1) for i in range(0,len(batch),4)]))
    paper=[r for r in summaries if 'paper' in r['label']]
    controls=[r for r in summaries if r['label']=='real_jewellery']
    fallback_paper=[r for r in paper if r['anomaly_passed'] is True]
    fallback_controls=[r for r in controls if r['anomaly_passed'] is True]
    summary={'capture_count':len(rows),'paper_captures':len(paper),'jewellery_controls':len(controls),
             'geometry_paper_detected':sum(r['geometry_detected'] for r in paper),
             'geometry_control_flags':[r['id'] for r in controls if r['geometry_detected']],
             'anomaly_missed_paper':len(fallback_paper),'fallback_paper_caught':sum(r['fallback_reject'] for r in fallback_paper),
             'fallback_jewellery_controls':len(fallback_controls),
             'fallback_control_flags':[r['id'] for r in fallback_controls if r['fallback_reject']],
             'fallback_missed_paper_ids':[r['id'] for r in fallback_paper if not r['fallback_reject']],
             'warning':'Exploratory capture-level evaluation, not held-out or per-object validation. Historical controls are confirmed jewellery, including jewellery in packets. No production gate enabled.'}
    loose=[]
    for row in rows:
        detected=any(c['neutral_fraction']>=.7 and np.mean(c['edge_support'])>=.5
            and sorted(c['line_support'])[1]>=.15 and np.mean(c['line_support'])>=.3
            and np.median(c['colour_jump'])>=2 for c in row['candidates'])
        if detected:loose.append(row)
    summary['looser_diagnostic_on_retained_top15_candidates']={
        'paper_detected':sum('paper' in r['visual_label'] for r in loose),
        'fallback_paper_caught':sum('paper' in r['visual_label'] and r['anomaly_passed'] is True for r in loose),
        'control_flags':[r['id'] for r in loose if r['visual_label']=='real_jewellery'],
        'fallback_control_flags':[r['id'] for r in loose if r['visual_label']=='real_jewellery' and r['anomaly_passed'] is True]}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'visual_labels.json').write_text(json.dumps(labels,indent=2))
    with (OUT/'evaluation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    cards=[]
    for r in summaries:
        sid=r['id'];caption=html.escape(str(r))
        cards.append(f'<article><a href="{sid}.jpg"><img src="{sid}.jpg"></a><p>{caption}</p></article>')
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Paper geometry audit</title><style>body{font:14px sans-serif;background:#eee}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:10px}img{width:100%}p{overflow-wrap:anywhere}</style><h1>Exploratory paper fallback audit</h1><p>Red: proposed paper; blue: strongest candidate. Capture-level labels, not individual object annotations. No production rejection.</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><main>'+''.join(cards)+'</main>',encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--recent-only',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--replay-anomalies',action='store_true')
    args=parser.parse_args()
    if args.summarize:summarize(args.replay_anomalies)
    elif args.inventory:inventory()
    else:run(args.recent_only)
