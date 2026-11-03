from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
from PIL import Image
from backend.app.analysis.thread_classifier import JewelThreadClassifier
from backend.app.services.analysis_service import AnalysisService, PredictionResult

NEW_NAMES = {0:'Empty-Test-Bed',1:'Jewel_with_Thread',2:'Jewel_without_Thread',3:'Thread-Only'}
OLD_NAMES = {0:'jewel_with_thread',1:'jewel_without_thread',2:'thread_only'}

@pytest.mark.parametrize('names,index,label,passed',[
    (NEW_NAMES,0,'empty_testbed',False), (NEW_NAMES,1,'jewel_with_thread',True),
    (NEW_NAMES,2,'jewel_without_thread',True), (NEW_NAMES,3,'thread_only',False),
    (OLD_NAMES,0,'jewel_with_thread',True), (OLD_NAMES,1,'jewel_without_thread',True),
    (OLD_NAMES,2,'thread_only',False),
])
def test_export_class_order_and_allow_block(monkeypatch,names,index,label,passed):
    scores=np.zeros((1,len(names)),np.float32); scores[0,index]=1
    session=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images',shape=[1,3,32,32])],
        get_outputs=lambda:[SimpleNamespace(shape=[1,len(names)])],
        get_modelmeta=lambda:SimpleNamespace(custom_metadata_map={'names':str(names)}),
        run=lambda *args:[scores])
    monkeypatch.setattr('backend.app.analysis.thread_classifier.ort.InferenceSession',lambda *args,**kwargs:session)
    gate=JewelThreadClassifier(Path('mock.onnx'))
    result=gate.classify(np.full((50,30,3),200,np.uint8))
    assert result['label']==label
    assert result['passed'] is passed
    assert len(result['scores'])==len(names)
    assert gate.prepare(np.ones((50,30,3),np.uint8)).shape==(1,3,32,32)

@pytest.mark.parametrize('labels,expected_status,kind',[
    (['empty_testbed'],'rejected','empty_testbed'),
    (['thread_only'],'rejected','thread_only'),
    (['jewel_with_thread','empty_testbed'],'rejected','empty_testbed'),
    (['empty_testbed','thread_only'],'rejected','thread_only'),
    (['jewel_with_thread'],'awaiting_confirmation',None),
    (['jewel_without_thread'],'awaiting_confirmation',None),
])
def test_new_model_workflow_preserves_recapture_guards(tmp_path,monkeypatch,labels,expected_status,kind,jewellery_labels=None):
    jewellery_labels = jewellery_labels or ["Chain / Necklace"] * len(labels)
    image=np.full((120,140,3),200,np.uint8)
    mask=np.ones(image.shape[:2],np.uint8)
    paths={}
    for name,pixels in [('original',image),('working',image),('mask',mask*255)]:
        path=tmp_path/f'{name}.png';cv2.imwrite(str(path),pixels);paths[name]=str(path)
    class Repository:
        state={'id':'test','status':'captured','paths':paths,'classification':{},'rois':{'processing':{'x':0,'y':0,'w':140,'h':120}}}
        def get(self,_): return deepcopy(self.state)
        def save(self,state): self.state=deepcopy(state);return deepcopy(state)
        def session_dir(self,_): return tmp_path
    repository=Repository();service=AnalysisService(repository)
    monkeypatch.setattr(service,'_check_real_print',lambda _state,items,_dir:{'passed':True,'items':[{'passed':True,'label':'REAL'} for _ in items]})
    raw_crop=image[10:70,20:100].copy()
    component={'bbox':{'x':20,'y':10,'w':80,'h':60},'area_px':4800,'crop_bgr':np.full_like(raw_crop,255),'raw_crop_bgr':raw_crop,'mask':np.ones((60,80),np.uint8)}
    monkeypatch.setattr('backend.app.services.analysis_service.separate_jewellery_items',lambda *args,**kwargs:[component]*len(labels))
    evidence={'passed':True,'detections':[],'jewellery_instances':[{'component_overlaps':[{'component_index':i,'component_share':.9,'pixels':1000} for i in range(1,len(labels)+1)]}]}
    monkeypatch.setattr(service,'_check_foreign_objects',lambda *args:evidence)
    checks=[{'label':label,'confidence':.99,'passed':label in {'jewel_with_thread','jewel_without_thread'}} for label in labels]
    gate=SimpleNamespace(classify=Mock(side_effect=checks))
    monkeypatch.setattr(service,'_get_jewel_thread',lambda:gate)
    (tmp_path/'capture').mkdir()
    picture=Image.fromarray(image)
    classifier=Mock();classifier.text_model_id='test';classifier.check_non_jewelry.return_value=(False,'')
    classifier.classify_image.side_effect=[PredictionResult(label=jewellery,confidence=.99,scores=[],image_path='',original_image=picture,cropped_image=picture,is_gold_jewelry=True) for jewellery in jewellery_labels]
    get_classifier=Mock(return_value=classifier)
    monkeypatch.setattr(service,'_get_classifier',get_classifier)
    monkeypatch.setattr(service,'_save_rejected_review',Mock())
    try:
        result=service.classify('test')
        assert result['status']==expected_status
        assert gate.classify.call_count == len(checks)
        assert all(item['applicable'] for item in result['thread_only_check']['items'])
        for call in gate.classify.call_args_list:
            np.testing.assert_array_equal(call.args[0],raw_crop)
        if kind:
            assert result['rejection_kind']==kind
            if 'empty_testbed' in labels:
                assert 'empty testbed' in result['rejection_reason']
            get_classifier.assert_not_called()
            assert result['result']==result['route']=={}
            result['status']='awaiting_confirmation';repository.save(result)
            with pytest.raises(RuntimeError,match='recapture'):
                service.confirm_and_start('test','Other Gold Jewellery',False,override_not_gold=True)
            operation=Mock();monkeypatch.setattr(service,'_run_stones',operation)
            service._run('test');operation.assert_not_called()
        else:
            get_classifier.assert_called_once()
    finally:
        service.shutdown()


@pytest.mark.parametrize("jewellery", ["Earrings / Nosepin", "Earrings", "Finger Ring", "Bangle", "Bracelet", "Other Gold Jewellery"])
@pytest.mark.parametrize("count", [1, 3])
def test_small_and_other_jewels_are_checked_as_crops(tmp_path, monkeypatch, jewellery, count):
    test_new_model_workflow_preserves_recapture_guards(
        tmp_path, monkeypatch, ["jewel_without_thread"] * count, "awaiting_confirmation", None,
        [jewellery] * count,
    )


@pytest.mark.parametrize("jewellery_types,labels,status,kind", [
    (["Finger Ring", "Chain / Necklace", "Earrings / Nosepin"],
     ["jewel_without_thread", "jewel_with_thread", "jewel_without_thread"], "awaiting_confirmation", None),
    (["Bangle", "Mattal", "Finger Ring"],
     ["thread_only", "thread_only", "thread_only"], "rejected", "thread_only"),
    (["Mattal"], ["jewel_with_thread"], "awaiting_confirmation", None),
    (["Mattal"], ["empty_testbed"], "rejected", "empty_testbed"),
])
def test_crop_gate_is_independent_per_item(tmp_path, monkeypatch, jewellery_types, labels, status, kind):
    test_new_model_workflow_preserves_recapture_guards(
        tmp_path, monkeypatch, labels, status, kind, jewellery_types,
    )


@pytest.mark.parametrize("label", ["Chain / Necklace", "Mattal", "chain", "necklace", "Haram", "Mangalsutra", "Dollar Chain", "Kasu Malai", "Ear Chain", "Chain Bracelet"])
def test_explicit_chain_aliases_use_thread_classifier(label):
    from backend.app.domain.workflow import uses_thread_classifier
    assert uses_thread_classifier(label)


@pytest.mark.parametrize("jewellery", ["Earrings / Nosepin", "Finger Ring", "Bangle", "Bracelet", "Other Gold Jewellery", "Mattal"])
def test_thread_only_is_blocked_without_jewellery_type_bypass(tmp_path, monkeypatch, jewellery):
    test_new_model_workflow_preserves_recapture_guards(
        tmp_path, monkeypatch, ["thread_only"], "rejected", "thread_only", [jewellery],
    )
