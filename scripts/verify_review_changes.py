"""Replay review annotations and verify reflection alerts stay removed."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from offline_runtime import configure, ROOT, BLOCKED_CONNECTIONS


def main():
    configure()
    import cv2
    import numpy as np
    from backend.app.analysis import annotations
    from backend.app.analysis.foreign_objects import ForeignObjectDetector
    from backend.app.core.config import FOREIGN_OBJECT_MODEL_PATH
    from backend.app.services.analysis_service import AnalysisService
    from backend.app.storage.repository import CaptureRepository
    from StoneDetection.jewel_gem_hsv_report import surface_reflection_metrics
    from StoneDetection.stone_area_calculator import calculate_reflection_risk

    output = ROOT.parent / 'testing/review-changes'
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    mask = np.full((100, 100), 255, np.uint8)
    for name, points, expected in [
        ('none', [], False), ('one pixel', [(50, 50)], False),
        ('isolated pixels', [(10, 10), (50, 50), (90, 90)], False),
    ]:
        glare = np.zeros_like(mask)
        for y, x in points:
            glare[y, x] = 255
        assert calculate_reflection_risk(glare, mask)['flagged'] is expected
        checks.append(name)
    tiny = np.zeros((500, 500), np.uint8)
    tiny[240:248, 240:248] = 255
    assert not calculate_reflection_risk(tiny, np.full_like(tiny,255))['flagged']
    glare = np.zeros_like(mask)
    glare[44:56,44:56] = 255
    assert not calculate_reflection_risk(glare, mask)['flagged']
    assert not calculate_reflection_risk(mask, mask)['flagged']
    assert not calculate_reflection_risk(mask, np.zeros_like(mask))['flagged']
    for n in (1,3,10):
        m = np.full((n,n),255,np.uint8)
        g = np.zeros_like(m); g[0,0]=255
        assert not calculate_reflection_risk(g,m)['flagged']
    base = np.full((100,100,3),(30,70,110),np.uint8)
    for name, size, expected in [('matte',0,False),('single sparkle',1,False),('reflective region',30,False)]:
        img = base.copy()
        if size: img[30:30+size,30:30+size]=245
        result=surface_reflection_metrics(img,mask)
        assert result['flagged'] is expected, (name,result)
        checks.append(name)
    # Tray pixels outside the jewel never contribute to its reflection notice.
    img = np.full((100,100,3),255,np.uint8); img[30:70,30:70]=base[30:70,30:70]
    m = np.zeros_like(mask);m[30:70,30:70]=255
    assert not surface_reflection_metrics(img,m)['flagged']
    checks.extend(['tiny highlight on large jewel','dense local region','whole surface','empty mask','tiny masks','outside jewel excluded'])

    detector=ForeignObjectDetector(FOREIGN_OBJECT_MODEL_PATH)
    sid='20261023_163520_61a64631'
    state=json.loads((ROOT/'data/sessions'/sid/'manifest.json').read_text())
    raw=cv2.imread(state['paths']['original'])
    tray=cv2.imread(state['paths']['testbed_source_mask'],0)
    check=detector.detect(raw,tray)
    assert not check['passed'] and any(d['label']=='plastic_packet' for d in check['detections'])
    state['foreign_object_check']=check
    # Verify raw-coordinate boxes reach the renderer without the item ROI offset.
    with patch.object(annotations,'label_box',wraps=annotations.label_box) as boxes:
        rendered=annotations.render_capture_review(raw,state,state['classification'].get('items',[]),rejected=True)
        for i,d in enumerate(check['detections'],1):
            b=d['bbox']; expected=(b['x'],b['y'],b['x']+b['w'],b['y']+b['h'])
            assert any(call.args[1]==expected and call.args[2].startswith(f'A{i}:') for call in boxes.call_args_list)
    cv2.imwrite(str(output/'anomaly-review.jpg'),rendered)
    # All model anomaly classes get boxes; jewellery itself does not.
    synthetic=copy.deepcopy(state)
    synthetic['foreign_object_check']['detections']=[dict(label=label,bbox=dict(x=150+i*160,y=550,w=100,h=100)) for i,label in enumerate(['foreign_obj','plastic_packet','paper','jewellery'])]
    with patch.object(annotations,'label_box',wraps=annotations.label_box) as boxes:
        annotations.render_capture_review(raw,synthetic,[],rejected=True)
        assert len(boxes.call_args_list)==3
    checks.extend(['real ONNX plastic bounding box','all anomaly classes','jewellery excluded'])
    replay=[]
    with tempfile.TemporaryDirectory(prefix='review-verification-') as temp:
        repo=CaptureRepository(Path(temp)/'captures.db',Path(temp)/'sessions')
        service=AnalysisService(repo)
        try:
            for sid in ['20261024_141654_cdb8bd7d','20261024_141233_6fd594a8']:
                state=json.loads((ROOT/'data/sessions'/sid/'manifest.json').read_text())
                for item in state['classification']['items'][:2]:
                    item=copy.deepcopy(item)
                    item['confirmed_label']=item.get('confirmed_label') or item['predicted_label']
                    with patch.object(annotations,'add_side_panel',wraps=annotations.add_side_panel) as panels:
                        result=service._run_stones(state,item)
                    assert not any(any(caption=='SURFACE REFLECTION' for caption,value in call.args[2]) for call in panels.call_args_list)
                    report=json.loads(Path(result['report']).read_text())
                    assert result['reflection_flagged'] == bool(report['reflection_flagged'])
                    assert not result['reflection_flagged'], (sid,item['index'],result)
                    assert result['reflection_notice'] is None
                    target=output/f'{sid}-item-{item["index"]}-stones.png'
                    target.write_bytes(Path(result['image']).read_bytes())
                    replay.append(dict(id=sid,item=item['index'],label=item['confirmed_label'],result={k:v for k,v in result.items() if k not in {'image','report'}},reflection=[j['reflection'] for j in report['jewels']]))
                    print('Stone replay:',sid,item['index'],result['reflection_notice'],flush=True)
        finally:
            service._executor.shutdown(wait=True)
    assert not BLOCKED_CONNECTIONS, BLOCKED_CONNECTIONS
    (output/'verification.json').write_text(json.dumps(dict(checks=checks,anomaly=check,replays=replay,external_connections=BLOCKED_CONNECTIONS),indent=2))
    print('All review changes checks passed:',output,flush=True)

if __name__=='__main__':
    main()
