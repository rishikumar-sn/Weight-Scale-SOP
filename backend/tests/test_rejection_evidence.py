from copy import deepcopy
import numpy as np
from backend.app.analysis.rejection_evidence import rejection_regions, rejection_summary
from backend.app.analysis import annotations
from test_component_presence import service_case, evidence


def test_presence_failure_marks_only_the_unsupported_component(service_case, monkeypatch):
    service = service_case
    monkeypatch.setattr(service, '_check_foreign_objects', lambda *_: evidence(0, .95))
    result = service.classify('candidate')
    assert result['status'] == 'rejected'
    assert len(result['rejection_regions']) == 1
    region = result['rejection_regions'][0]
    assert region['tag'] == 'U1' and region['label'] == 'Unverified object'
    assert region['bbox']['x'] > 150
    assert 'Could not verify this object' in result['rejection_reason']
    assert not result['classification']['items']
    rendered = []
    monkeypatch.setattr(annotations, 'label_box', lambda image, box, text, **kw: rendered.append((box, text)))
    annotations.render_capture_review(np.full((150,250,3),200,np.uint8),result,[])
    assert len(rendered) == 1 and rendered[0][1] == 'U1: Unverified object'


def test_uncertain_coin_is_not_claimed_to_be_print_or_non_gold():
    state = {'foreign_object_check': {'recognized_objects': [
        {'label':'coin','component_overlaps':[{'component_index':4,'component_share':.9}]}]},
        'real_print_check':{'passed':False,'items':[{'item_index':4,'label':'REVIEW','passed':False,
            'source_bbox':dict(x=520,y=1008,w=118,h=119)}]}}
    saved = deepcopy(state)
    regions = rejection_regions(state)
    assert regions[0]['annotation'] == 'P4 Coin: Verification uncertain'
    assert 'does not establish whether the object is gold' in regions[0]['reason']
    assert state == saved


def test_remapped_foreground_uses_original_source_location():
    state = {'foreground_components':[{'source_component_index':3,'source_bbox':dict(x=500,y=900,w=70,h=100)}],
             'component_presence_check':{'items':[{'source_component_index':2,'original_component_index':3,'passed':False}]}}
    region = rejection_regions(state)[0]
    assert region['tag'] == 'U2' and region['bbox']['x'] == 500


def test_no_location_is_fabricated_when_verification_is_unavailable():
    state = {'rejection_reason':'Object verification unavailable','foreign_object_check':{'error':'model unavailable'}}
    assert rejection_regions(state) == []
    assert rejection_summary(state,[]) == state['rejection_reason']


def test_thread_failure_projects_processing_coordinates():
    state={'rois':{'processing':dict(x=100,y=200)},'thread_only_check':{'items':[
        {'item_index':2,'label':'thread_only','passed':False,'bbox':dict(x=20,y=30,w=40,h=50)}]}}
    region=rejection_regions(state)[0]
    assert region['tag']=='T2' and region['bbox']==dict(x=120,y=230,w=40,h=50)


def test_semantic_rejection_marks_the_failed_object():
    state={'rejection_kind':'non_jewellery','rois':{'processing':dict(x=20,y=40)},
           'other_objects':[{'kind':'non_jewellery','bbox':dict(x=10,y=20,w=80,h=90)}]}
    region=rejection_regions(state)[0]
    assert region['tag']=='O1' and region['bbox']==dict(x=30,y=60,w=80,h=90)


def test_ambiguous_mask_does_not_mark_a_separate_valid_coin():
    from backend.app.analysis.foreign_objects import assign_recognized_objects
    check={'recognized_objects':[
        {'label':'coin','bbox':dict(x=10,y=20,w=30,h=40),'component_overlaps':[
            {'component_index':1,'component_share':.9}]},
        {'label':'bar','bbox':dict(x=50,y=80,w=40,h=90),'component_overlaps':[]}]}
    assert not assign_recognized_objects(check,[{}])
    assert check['object_assignment_failure_indices']==[2]
    regions=rejection_regions({'foreign_object_check':check,'object_assignment_check':{'passed':False}})
    assert len(regions)==1 and regions[0]['bbox']['x']==50
