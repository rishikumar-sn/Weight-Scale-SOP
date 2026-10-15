from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pytest

from backend.app.services import analysis_service
from Classification.siglip384_classifier import EMBEDDING_SPACE, ModelCorrectionGallery


@pytest.mark.parametrize('selection,expected', [(None,'base384'),('base384','base384'),('patch32','patch32')])
def test_primary_model_selection_and_cached_instance(monkeypatch, selection, expected):
    if selection is None:
        monkeypatch.delenv('JEWELLERY_CLASSIFIER_MODEL',raising=False)
    else:
        monkeypatch.setenv('JEWELLERY_CLASSIFIER_MODEL',selection)
    made=[]
    def factory(name):
        def build(**kwargs):
            value=object();made.append((name,value,kwargs));return value
        return build
    monkeypatch.setattr(analysis_service,'Siglip384Classifier',factory('base384'))
    monkeypatch.setattr(analysis_service,'JewelryZeroShotClassifier',factory('patch32'))
    service=analysis_service.AnalysisService(None)
    try:
        value=service._get_classifier()
        assert service._get_classifier() is value
        assert len(made)==1 and made[0][0]==expected
        if expected=='patch32':
            assert made[0][2]['onnx_model_path'].name=='siglip2-base-patch32-256_vision_encoder.sim.onnx'
    finally:
        service.shutdown()


def test_unknown_model_does_not_silently_change_classifier(monkeypatch):
    monkeypatch.setenv('JEWELLERY_CLASSIFIER_MODEL','typo')
    service=analysis_service.AnalysisService(None)
    try:
        with pytest.raises(ValueError,match='JEWELLERY_CLASSIFIER_MODEL'):
            service._get_classifier()
    finally:
        service.shutdown()


@pytest.mark.parametrize('space',[None,'google/siglip2-base-patch32-256'])
def test_same_dimension_legacy_vectors_cannot_load_into_new_gallery(tmp_path: Path,space):
    path=tmp_path/'gallery.npz'
    fields={'embeddings':np.eye(1,768,dtype=np.float32),'labels':np.array(['Bangle'])}
    if space is not None:fields['embedding_space']=np.array(space)
    np.savez(path,**fields)
    original=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Incompatible base384 gallery'):
        ModelCorrectionGallery(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original


def test_correction_survives_reload_with_model_identity_and_legacy_untouched(tmp_path: Path):
    legacy=tmp_path/'legacy.npz'
    np.savez(legacy,embeddings=np.eye(1,768,dtype=np.float32),labels=np.array(['Bangle']))
    original=legacy.read_bytes()
    path=tmp_path/'base384.npz'
    gallery=ModelCorrectionGallery(path)
    vector=np.eye(1,768,dtype=np.float32)[0]
    gallery.add(vector,'Finger ring')
    reloaded=ModelCorrectionGallery(path)
    result=reloaded.search(vector,expected_label='Bangle')
    assert result.label=='Finger Ring'
    assert result.similarity==pytest.approx(1.0)
    with np.load(path) as data:assert str(data['embedding_space'].item())==EMBEDDING_SPACE
    assert legacy.read_bytes()==original
    assert not path.with_name(path.stem+'.pending.npz').exists()
