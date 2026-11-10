from copy import deepcopy

import numpy as np
import pytest
from PIL import ImageDraw

from backend.tests.test_foreign_objects import capture_service, detector_for_rows
from backend.app.analysis.foreign_objects import assign_recognized_objects, object_warning
from backend.app.analysis.component_presence import component_presence_check
from backend.app.analysis import annotations
from backend.app.domain.workflow import route_for_label


def object_check(label):
    return {"passed": True, "detections": [], "jewellery_instances": [],
            "recognized_objects": [{"label": label, "confidence": .93,
                "bbox": {"x": 0, "y": 0, "w": 80, "h": 80},
                "mask_contours": [[[0, 0], [79, 0], [79, 79], [0, 79]]],
                "component_overlaps": [{"component_index": 1, "component_share": .95, "pixels": 6080}]}]}


@pytest.mark.parametrize('label,class_id', [('coin', 1), ('bar', 2)])
def test_new_class_order_masks_and_nonblocking_objects(label, class_id):
    # Six classes + a single prototype coefficient, using the candidate order.
    row = [50, 50, 30, 30, 0, 0, 0, 0, 0, 0, 10]
    row[4 + class_id] = .93
    detector = detector_for_rows([row], np.ones((1, 1, 25, 25), np.float32))
    detector.names = dict(enumerate(['jewellery', 'coin', 'bar', 'foreign_obj', 'paper', 'plastic_packet']))
    result = detector.detect(np.full((100, 100, 3), 160, np.uint8), np.ones((100, 100), np.uint8),
                             components=[{'bbox':dict(x=40,y=40,w=20,h=20),'mask':np.ones((20,20),np.uint8)}])
    assert result['passed']
    assert result['detections'] == []
    assert result['recognized_objects'][0]['label'] == label
    assert result['recognized_objects'][0]['mask_contours']
    assert component_presence_check(result, 1)['passed']


@pytest.mark.parametrize('label', ['coin', 'bar'])
def test_coin_bar_bypasses_siglip_keeps_warning_and_confirmation(capture_service, monkeypatch, label):
    service, _ = capture_service
    monkeypatch.setattr(service, '_check_foreign_objects', lambda *_: object_check(label))
    monkeypatch.setattr(service, '_get_classifier', lambda: pytest.fail('Coin/bar must never call SigLIP'))
    monkeypatch.setattr(service, '_get_jewel_thread', lambda: pytest.fail('Coin/bar is not a thread classification input'))
    result = service.classify('capture')
    assert result['status'] == 'awaiting_confirmation'
    item = result['classification']['items'][0]
    assert item['predicted_label'] == label.capitalize()
    assert item['decision_source'] == 'anomaly_segmentation'
    assert 'does not verify gold' in item['gold_warning']
    assert not any(route_for_label(item['predicted_label'])[k] for k in ('dimension','beads','stones'))
    with pytest.raises(RuntimeError, match='segmentation model'):
        service.confirm_and_start('capture', 'Bangle', True, save_only=True)
    service.confirm_and_start('capture', label.capitalize(), True, save_only=True)
    service.confirm_and_start('capture', label.capitalize(), True)
    service.shutdown()
    final = service.repository.get('capture')
    assert final['status'] == 'complete'
    assert final['result']['items'][0]['label'] == label.capitalize()
    assert final['result']['items'][0]['gold_warning'] == item['gold_warning']
    assert not final['classification']['items'][0]['learned']


def test_ambiguous_or_merged_recognition_never_falls_back_to_siglip(capture_service, monkeypatch):
    service, _ = capture_service
    check = object_check('coin')
    check['recognized_objects'].append(deepcopy(check['recognized_objects'][0]))
    monkeypatch.setattr(service, '_check_foreign_objects', lambda *_: check)
    monkeypatch.setattr(service, '_get_classifier', lambda: pytest.fail('Ambiguous objects must require recapture'))
    assert service.classify('capture')['status'] == 'rejected'
    check = object_check('coin')
    check['recognized_objects'][0]['component_overlaps'] = []
    assert not assign_recognized_objects(check, [{}])


def test_mixed_coin_and_jewellery_only_classifies_the_jewellery(capture_service, monkeypatch):
    from backend.app.services import analysis_service as module
    service, _ = capture_service
    components = module.separate_jewellery_items(None, None)
    components.append(deepcopy(components[0]))
    components[1]['bbox']['x'] = 85
    monkeypatch.setattr(module, 'separate_jewellery_items', lambda *_a, **_kw: deepcopy(components))
    check = object_check('coin')
    check['jewellery_instances'] = [{'component_overlaps':[
        {'component_index':2,'component_share':.95,'pixels':6080}]}]
    monkeypatch.setattr(service, '_check_foreign_objects', lambda *_: check)
    classifier = service._get_classifier()
    original = classifier.classify_image
    calls = []
    classifier.classify_image = lambda *a, **kw: calls.append(kw['image_path']) or original(*a, **kw)
    monkeypatch.setattr(service, '_get_classifier', lambda: classifier)
    result = service.classify('capture')
    assert result['status'] == 'awaiting_confirmation'
    assert result['classification']['labels'] == ['Coin','Bangle']
    assert len(calls) == 1 and 'item_02' in calls[0]


def test_review_draws_coin_and_bar_tags_and_gold_warning(monkeypatch):
    values = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, value, *args, **kwargs):
        values.append(value)
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', record)
    items = [dict(index=i, predicted_label=label.capitalize(), gold_warning=object_warning(label),
                  recognized_object=object_check(label)['recognized_objects'][0],
                  bbox=dict(x=50 + i*160, y=100, w=100, h=100)) for i, label in enumerate(['coin','bar'],1)]
    annotations.render_capture_review(np.full((800,800,3),160,np.uint8),{},items)
    assert 'J1 Coin' in values and 'J2 Bar' in values
    assert 'GOLD VERIFICATION REQUIRED' in values


def test_final_result_image_keeps_object_tags_and_gold_warning(tmp_path, monkeypatch):
    from PIL import Image
    from backend.app.services.report_service import ReportService
    path = tmp_path/'original.png'
    Image.new('RGB',(1080,1920),'#eeeeee').save(path)
    values = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, value, *args, **kwargs):
        values.append(value)
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', record)
    items = [dict(index=i, predicted_label=label.capitalize(), confirmed_label=label.capitalize(),
                  recognized_object=object_check(label)['recognized_objects'][0],
                  bbox=dict(x=50+i*200,y=500,w=160,h=180)) for i,label in enumerate(['coin','bar'],1)]
    state = {'paths':{'original':str(path)}, 'classification':{'count':2,'items':items},
             'result':{'count':2,'items':[{'label':'Coin'},{'label':'Bar'}]}}
    ReportService().generate_result_image(state)
    assert 'J1 Coin' in values and 'J2 Bar' in values
    assert values.count('Verify gold content / purity') == 2
