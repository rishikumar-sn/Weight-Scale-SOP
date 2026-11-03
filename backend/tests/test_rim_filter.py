from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock
import cv2
import numpy as np

from backend.app.analysis.rim_filter import rim_candidates, remap_component_check
from backend.app.analysis.component_presence import component_presence_check
from backend.app.services.analysis_service import AnalysisService


def item(x=200, y=13, w=51, h=6, colour=(180,180,180)):
    return {'bbox':dict(x=x,y=y,w=w,h=h), 'mask':np.ones((h,w),np.uint8),
            'raw_crop_bgr': np.full((h,w,3),colour,np.uint8), 'area_px':w*h}


class RimFilterTests(unittest.TestCase):
    def setUp(self):
        self.items = [item(), item(200,200,40,40)]
        self.check = {'passed':True,'detections':[], 'jewellery_instances':[
            {'component_overlaps':[{'component_index':2,'component_share':.9,'pixels':1440}]}],
            'placement_check':{'passed':True}}
        self.valid = np.ones((500,500),np.uint8)
        self.cal = dict(available=True,found=True,mm_per_pixel_x=.2,mm_per_pixel_y=.2)
        self.testbed = dict(confidence=.8)

    def run_rule(self):
        return rim_candidates(self.items,self.check,self.valid,self.cal,self.testbed)

    def test_neutral_strip_at_outer_border_is_candidate(self):
        self.assertEqual(self.run_rule()['candidate_indices'],[1])

    def test_partial_jewellery_overlap_protects_edge_item(self):
        self.check['jewellery_instances'][0]['component_overlaps'].append(
            dict(component_index=1,component_share=.001,pixels=1))
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_interior_object_is_retained(self):
        self.items[0] = item(y=100)
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_crop_padding_does_not_make_interior_object_a_rim(self):
        mask=np.zeros((70,100),np.uint8);mask[45:51,20:71]=1
        self.items[0]=dict(bbox=dict(x=180,y=0,w=100,h=70),mask=mask,
                           raw_crop_bgr=np.full((70,100,3),180,np.uint8))
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_tag_or_fixture_hole_is_not_outer_boundary(self):
        self.items[0]=item(x=200,y=200)
        cv2.rectangle(self.valid,(195,190),(260,195),0,-1)
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_gold_or_coloured_strip_is_retained(self):
        self.items[0]=item(colour=(0,180,220))
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_closed_opening_is_retained(self):
        self.items[0]=item(h=8,y=10)
        self.items[0]['mask'][2:6,2:-2]=0
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_compact_border_object_is_retained(self):
        self.items[0]=item(w=12,h=10,y=5)
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_thin_object_pointing_into_tray_is_retained(self):
        self.items[0]=item(w=4,h=18,y=1)
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_no_calibration_no_exclusion(self):
        self.cal['found']=False
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_invalid_testbed_confidence_no_exclusion(self):
        self.testbed['confidence']=float('nan')
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_missing_or_ambiguous_testbed_no_exclusion(self):
        self.valid=None
        self.assertEqual(self.run_rule()['candidate_indices'],[])
        self.valid=np.zeros((500,500),np.uint8)
        self.valid[1:80,1:80]=1;self.valid[100:200,100:200]=1
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_no_jewel_does_not_accept_empty_tray(self):
        self.check['jewellery_instances']=[]
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_explicit_anomalies_cannot_be_removed(self):
        self.check['passed']=False
        self.check['detections']=[{'label':'paper'}]
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_placement_rejection_cannot_be_removed(self):
        self.check['placement_check']['passed']=False
        self.assertEqual(self.run_rule()['candidate_indices'],[])

    def test_check_remapping_preserves_ownership_without_mutating_original(self):
        old=deepcopy(self.check)
        new=remap_component_check(self.check,[2])
        self.assertEqual(self.check,old)
        self.assertEqual(new['jewellery_instances'][0]['component_overlaps'][0]['component_index'],1)
        self.assertEqual(new['jewellery_instances'][0]['component_overlaps'][0]['source_component_index'],2)
        self.assertTrue(component_presence_check(new,1)['passed'])

    def service_filter(self,prediction=None,error=None):
        service=AnalysisService(None)
        classifier=Mock()
        classifier.classify_image=Mock(return_value=prediction,side_effect=error)
        service._get_classifier=lambda:classifier
        state={'calibration':self.cal,'testbed':self.testbed}
        try:
            result=service._exclude_verified_rim_strips(state,self.items,self.check,self.valid)
        finally:
            service.shutdown()
        return result,state

    def test_weak_category_guess_allows_geometry_rule_preserves_crop_identity(self):
        prediction=SimpleNamespace(label='Finger Ring',confidence=.35,gallery_match=False,is_gold_jewelry=True)
        (kept,check),state=self.service_filter(prediction)
        self.assertEqual(len(kept),1)
        self.assertIs(kept[0],self.items[1])
        self.assertTrue(component_presence_check(check,1)['passed'])
        self.assertEqual(state['rim_edge_check']['ignored_component_indices'],[1])

    def test_confident_siglip_jewel_vetoes_exclusion(self):
        p=SimpleNamespace(label='Finger Ring',confidence=.8,gallery_match=False,is_gold_jewelry=True)
        (kept,_),_=self.service_filter(p)
        self.assertEqual(len(kept),2)

    def test_positive_gallery_vetoes_exclusion(self):
        p=SimpleNamespace(label='Chain / Necklace',confidence=.4,gallery_match=True,is_gold_jewelry=True)
        (kept,_),_=self.service_filter(p)
        self.assertEqual(len(kept),2)

    def test_invalid_semantic_confidence_retains_component(self):
        p=SimpleNamespace(label='Finger Ring',confidence=float('nan'),gallery_match=False,is_gold_jewelry=True)
        (kept,_),_=self.service_filter(p)
        self.assertEqual(len(kept),2)

    def test_semantic_failure_keeps_existing_rejection(self):
        (kept,check),_=self.service_filter(error=RuntimeError('model unavailable'))
        self.assertEqual(len(kept),2)
        self.assertFalse(component_presence_check(check,2)['passed'])


if __name__=='__main__': unittest.main(verbosity=2)
