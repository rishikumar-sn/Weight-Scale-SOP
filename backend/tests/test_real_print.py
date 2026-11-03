from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np
import pytest
from backend.app.analysis.real_print import RealPrintClassifier,blocks_real_print,padded_box
from backend.app.analysis.component_presence import blocks_jewellery_labels
from backend.app.analysis import annotations
from backend.app.core.config import PROJECT_ROOT
from test_component_presence import service_case,evidence

MODEL=PROJECT_ROOT/'models/classification/real-print-mobilenetv3.onnx'


@pytest.fixture(scope='module')
def model(): return RealPrintClassifier(MODEL)


@pytest.mark.parametrize('labels,kind',[(('REAL','PRINT'),'print_detected'),(('PRINT','REAL'),'print_detected'),
                                     (('REAL','REVIEW'),'real_print_review')])
def test_any_print_or_review_requires_recapture(service_case,monkeypatch,labels,kind):
    service=service_case
    monkeypatch.setattr(service,'_check_foreign_objects',lambda *_:evidence(.95,.95))
    monkeypatch.setattr(service,'_check_real_print',lambda *_:dict(passed=False,items=[dict(label=v,passed=v=='REAL') for v in labels]))
    result=service.classify('candidate')
    assert result['status']=='rejected' and result['rejection_kind']==kind
    assert result['classification']['count']==0 and result['classification']['items']==[]
    assert result['route']==result['result']=={}
    assert result['real_print_check']['passed'] is False
    for save_only in (False,True):
        with pytest.raises(RuntimeError):
            service.confirm_and_start('candidate','Finger Ring',True,save_only=save_only,override_not_gold=True)
    monkeypatch.setattr(service,'_check_foreign_objects',lambda *_:pytest.fail('Rejected capture must require a new capture'))
    assert service.classify('candidate')['status']=='rejected'


def test_inference_failure_is_persisted_without_category_fallback(service_case,monkeypatch):
    service=service_case
    monkeypatch.setattr(service,'_check_foreign_objects',lambda *_:evidence(.95,.95))
    def fail(*_): raise RuntimeError('model missing')
    monkeypatch.setattr(service,'_check_real_print',fail)
    result=service.classify('candidate')
    assert result['status']=='rejected' and result['rejection_kind']=='real_print_unavailable'
    assert result['real_print_check']['error']=='model missing'
    assert result['classification']['items']==[]


def test_gate_crops_original_pixels_and_maps_virtual_roi(service_case,monkeypatch,model):
    service=service_case;state=service.repository.get('candidate')
    original=np.arange(150*250*3,dtype=np.uint8).reshape(150,250,3)
    cv2.imwrite(state['paths']['original'],original)
    state['rois']['processing']['x']=-10;state['rois']['processing']['y']=15
    mask=np.zeros((40,50),np.uint8);mask[7:28,9:36]=1
    item=dict(bbox=dict(x=30,y=20,w=50,h=40),mask=mask)
    captured=[]
    fake=SimpleNamespace(crops=model.crops,classify=lambda views:(captured.extend(views) or dict(label='REAL',passed=True)))
    monkeypatch.setattr(service,'_get_real_print',lambda:fake)
    from backend.app.services.analysis_service import AnalysisService
    result=AnalysisService._check_real_print(service,state,[item],service.repository.session_dir('candidate'))
    assert result['passed'] and result['items'][0]['source_bbox']==dict(x=29,y=42,w=27,h=21)
    for view,key in zip(captured,('tight','context')):
        x,y,w,h=result['items'][0]['crop_boxes'][key]
        np.testing.assert_array_equal(view,original[y:y+h,x:x+w])
        assert Path(result['items'][0]['paths'][key]).is_file()


def test_checksum_and_model_metadata_must_match(tmp_path):
    copied=tmp_path/'model.onnx';copied.write_bytes(b'wrong model')
    copied.with_suffix('.json').write_text(MODEL.with_suffix('.json').read_text())
    with pytest.raises(ValueError,match='checksum'): RealPrintClassifier(copied)


@pytest.mark.parametrize('p,label,passed',[(.1,'REAL',True),(.46,'REVIEW',False),(.5,'REVIEW',False),(.54,'REVIEW',False),(.9,'PRINT',False)])
def test_decision_threshold_and_uncertainty(model,monkeypatch,p,label,passed):
    logits=np.log(np.array([[1-p,p],[1-p,p]],np.float32))
    monkeypatch.setattr(model,'session',SimpleNamespace(run=lambda *_:[logits]))
    result=model.classify([np.full((30,40,3),200,np.uint8)]*2)
    assert result['label']==label and result['passed'] is passed


def test_invalid_native_output_cannot_accept(model,monkeypatch):
    monkeypatch.setattr(model,'session',SimpleNamespace(run=lambda *_:[np.full((2,2),np.nan,np.float32)]))
    with pytest.raises(RuntimeError,match='Invalid'):model.classify([np.full((30,40,3),200,np.uint8)]*2)


@pytest.mark.parametrize('state',[
    {'real_print_check':{'passed':False}},
    {'real_print_check':{'passed':True,'items':[{'label':'PRINT'}]}},
    {'rejection_kind':'print_detected'},
])
def test_stale_labels_and_queued_work_cannot_bypass_print(service_case,state):
    service=service_case;service.repository.state.update(deepcopy(state),status='awaiting_confirmation')
    before=deepcopy(service.repository.state)
    assert blocks_real_print(before) and blocks_jewellery_labels(before)
    assert service.public_state(before)['classification']['items']==[]
    assert service.repository.state==before
    service._run('candidate')  # Must not start native analysis or correction learning.
    with pytest.raises(RuntimeError): service.confirm_and_start('candidate','Finger Ring',True)


def test_print_evidence_marks_p_not_j(monkeypatch):
    labels=[]
    monkeypatch.setattr(annotations,'label_box',lambda _im,_box,label,**kw:labels.append(label))
    state={'real_print_check':{'passed':False,'items':[{'label':'PRINT','passed':False,'source_bbox':{'x':10,'y':20,'w':40,'h':40}}]}}
    annotations.render_capture_review(np.full((150,150,3),200,np.uint8),state,
        [{'index':1,'predicted_label':'Bangle','bbox':{'x':10,'y':20,'w':40,'h':40}}])
    assert labels==['P1: PRINT']


@pytest.mark.parametrize('handler',['report','result_image'])
def test_api_rejects_old_completed_results_when_print_is_blocked(handler):
    # Compile only the route function: importing main would initialize the
    # real production settings/database and is not appropriate for this test.
    import ast
    from fastapi import HTTPException
    source=ast.parse((PROJECT_ROOT/'backend/app/main.py').read_text(encoding='utf-8'))
    function=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name==handler)
    function.decorator_list=[]
    namespace=dict(_state=lambda _:dict(status='complete',real_print_check={'passed':False}),
        blocks_jewellery_labels=blocks_jewellery_labels,HTTPException=HTTPException)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<isolated route>','exec'),namespace)
    with pytest.raises(HTTPException) as caught: namespace[handler]('printed-capture')
    assert caught.value.status_code==409


def test_export_matches_checkpoint_and_lab_transforms(model):
    # Export validation uses research assets when available; production runtime
    # itself imports no experiment code, torch or torchvision.
    checkpoint=PROJECT_ROOT/'experiment/real_print_lab/checkpoints/run_20261102_enriched_real/best.pt'
    if not checkpoint.is_file(): pytest.skip('Research checkpoint is not part of the production runtime')
    import sys
    sys.path.insert(0,str(checkpoint.parents[2]))
    from training.engine import Predictor
    from app.core import padded_box as lab_box
    import torch
    torch.set_num_threads(2)
    reference=Predictor(checkpoint)
    for row in reference.checkpoint['split']['test']:
        tight=cv2.imread(row['tight']);context=cv2.imread(row['context'])
        actual=model.classify([tight,context]);expected=reference.predict(tight,context,uncertainty=.05)
        assert actual['label']==expected['prediction']
        assert actual['fused_print']==pytest.approx(expected['fused_print'],abs=2e-5)
        for image in (tight,context):
            from PIL import Image
            tensor=reference.tx(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))).numpy()
            np.testing.assert_allclose(model.prepare(image),tensor,atol=2e-6)
    for box in ([0,0,15,7],[230,135,20,15],[20,30,27,33]):
        for padding in (.1,.4): assert padded_box(box,(150,250,3),padding)==lab_box(box,(150,250,3),padding)


@pytest.mark.parametrize("verdict", ["PRINT", "REVIEW", "REAL"])
def test_rim_exclusion_keeps_print_gate_and_component_ownership(service_case, monkeypatch, verdict):
    from backend.app.services import analysis_service as module
    from Classification.jewelry_classifier import PredictionResult
    service = service_case
    monkeypatch.setattr(service, "_check_foreign_objects", lambda *_: evidence(0, .95))
    monkeypatch.setattr(module, "rim_candidates", lambda *_: {
        "items": [{"candidate": True, "source_component_index": 1}],
        "ignored_component_indices": [],
    })
    class Classifier:
        def check_non_jewelry(self, _): return False, "clear"
        def classify_image(self, image, image_path=None):
            return PredictionResult(label="Finger Ring", confidence=.35, scores=[], image_path=image_path,
                                    original_image=image, cropped_image=image)
    monkeypatch.setattr(service, "_get_classifier", lambda: Classifier())
    checked = []
    def print_gate(state, items, _directory):
        checked.extend(items)
        assert len(state["thread_only_check"]["items"]) == 1
        assert state["component_presence_check"]["items"][0]["original_component_index"] == 2
        return {"passed": verdict == "REAL", "items": [{"label": verdict, "passed": verdict == "REAL"}]}
    monkeypatch.setattr(service, "_check_real_print", print_gate)
    result = service.classify("candidate")
    assert len(checked) == 1
    assert result["rim_edge_check"]["ignored_component_indices"] == [1]
    assert result["real_print_check"]["items"][0]["label"] == verdict
    if verdict == "REAL":
        assert result["status"] == "awaiting_confirmation"
        assert result["classification"]["count"] == 1
    else:
        assert result["status"] == "rejected"
        assert result["classification"]["items"] == []
        with pytest.raises(RuntimeError):
            service.confirm_and_start("candidate", "Finger Ring", True)
