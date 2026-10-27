"""Verify single-capture stone table uses the shared weight calculations."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from offline_runtime import configure, ROOT


def main():
    configure()
    import cv2
    from PIL import ImageDraw
    from backend.app.analysis import annotations
    from backend.app.domain.weights import weight_summary, weight_rows
    from backend.app.services.analysis_service import AnalysisService, stone_detection
    from backend.app.storage.repository import CaptureRepository

    state = json.loads((ROOT/'data/sessions/20261024_141654_cdb8bd7d/manifest.json').read_text())
    item = state['classification']['items'][0]
    image = cv2.imread(item['paths']['working'])
    output = ROOT.parent/'testing/evidence-layout'
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        service = AnalysisService(CaptureRepository(Path(temp)/'db',Path(temp)/'sessions'))
        try:
            for count in (1,2,20):
                for average in (1.25,100.0,None,0.0):
                    source = deepcopy(state)
                    source['classification']['count'] = count
                    source['classification']['items'] = [deepcopy(item) for _ in range(count)]
                    found = average != 0
                    measurements = {'success': average is not None,'estimated_total_average_g': average,'total_area_mm2':71.37}
                    report = {'stone_surface_risk':{'level':'MEDIUM' if found else 'NONE','status':'Medium stones' if found else 'No stones'},'stone_measurements':measurements}
                    view = {'overlay_bgr':image,'jewel_id':1,'detected_colors':['green'],'reflection':{'flagged':True}}
                    text_calls=[]
                    original_text=ImageDraw.ImageDraw.text
                    def record(draw,xy,text,*args,**kwargs):
                        text_calls.append((str(text),draw.textbbox(xy,text,font=kwargs.get('font'),anchor=kwargs.get('anchor'))))
                        return original_text(draw,xy,text,*args,**kwargs)
                    with patch.object(stone_detection,'build_candidates_from_component_mask',return_value=[]), patch.object(stone_detection,'analyze_image_bgr',return_value={'report':report,'jewel_views':[view]}), patch.object(annotations,'add_side_panel',wraps=annotations.add_side_panel) as panels, patch.object(ImageDraw.ImageDraw,'text',record):
                        result = service._run_stones(source,item)
                    rendered_fields = panels.call_args.args[2]
                    captions = [caption for caption, _ in rendered_fields]
                    assert 'SURFACE REFLECTION' not in captions
                    assert not result['reflection_flagged']
                    assert result['reflection_notice'] is None
                    assert ('STONE WEIGHT (ESTIMATED)' in captions) == (count == 1)
                    assert ('STONE SURFACE AREA' in captions) == (count != 1)
                    if count == 1:
                        assert not any('mm²' in text for text, _ in text_calls)
                    weights=panels.call_args.kwargs['weights']
                    assert (weights is not None) == (count == 1)
                    assert any(text=='JEWELLERY WEIGHTS (GRAMS)' for text,_ in text_calls) == (count == 1)
                    if count == 1:
                        expected=weight_summary({'weight_g':state['weight_g'],'result':{'count':1,'items':[{'stones':result}]}})
                        assert weights==expected
                        for row in weight_rows(expected):
                            for text in row:
                                assert any(t==text for t,_ in text_calls),text
                        # Legacy reflection flags must not restore the removed notice.
                        notice=[b for t,b in text_calls if 'reflective' in t or 'possible stones' in t]
                        assert not notice
                        for text,box in text_calls:
                            assert 0<=box[0]<=box[2]<=1536 and 0<=box[1]<=box[3]<=1024,(text,box)
                        if average==1.25:
                            (output/'stone-single-weights.png').write_bytes(Path(result['image']).read_bytes())
        finally:
            service._executor.shutdown(wait=True)
    print('Passed: stone weight in g only for count=1, area for multiple; table only for count=1; identical shared values/ranges/notes; cap, unavailable, no stones; bounds and removed reflection notices.')


if __name__=='__main__':
    main()
