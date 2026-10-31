from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from StoneDetection.gemstone_colors import (predict_gemstone_color, describe_regions, stone_description, original_color_crop, SWATCHES)
from backend.app.core.config import SettingsStore
from backend.app.domain.weights import stone_estimation_enabled, weight_summary
from backend.app.services.report_service import ReportService

@pytest.mark.parametrize('name',list(SWATCHES))
def test_reference_colors(name):
    image=np.full((30,30,3),SWATCHES[name][0][::-1],np.uint8)
    result=predict_gemstone_color(image,np.ones((30,30),np.uint8))
    assert result['color']==name
    assert result['confidence']>.45

@pytest.mark.parametrize('rgb,expected',[((110,15,30),'Red'),((245,150,185),'Pink')])
def test_shadows_and_glare_do_not_override_chromatic_body(rgb,expected):
    image=np.full((60,60,3),(20,20,20),np.uint8)
    image[:25]=rgb[::-1]
    image[50:]=255
    result=predict_gemstone_color(image,np.ones((60,60),np.uint8))
    assert result['color']==expected

@pytest.mark.parametrize('value,expected',[(0,'Uncertain'),(255,'Uncertain'),(220,'White/colorless'),(35,'Black'),(120,'Grey')])
def test_neutral_and_clipped_evidence(value,expected):
    result=predict_gemstone_color(np.full((30,30,3),value,np.uint8),np.ones((30,30),np.uint8))
    assert result['color']==expected

def test_mixed_evidence_is_uncertain():
    image=np.full((40,40,3),(40,25,190),np.uint8)
    image[20:]=(65,130,20)
    assert predict_gemstone_color(image,np.ones((40,40),np.uint8))['color']=='Uncertain'

def test_source_mapping_and_zoom_mask_use_native_color():
    image=np.full((40,60,3),(40,25,190),np.uint8)
    crop=original_color_crop(image,-5,-10,(60,70))
    assert np.array_equal(crop[10:50,5:65],image)
    assert np.all(crop[:10]==0)
    regions=[{'mask':np.ones((120,180),np.uint8)}]
    entries=describe_regions(image,regions,np.ones((120,180),np.uint8),{'mm_per_pixel_x':.1,'mm_per_pixel_y':.2})
    assert entries[0]['color']=='Red'
    assert entries[0]['area_mm2']==48
    assert entries[0]['size']=='large'
    assert entries[0]['continuous_area']
    assert 'continuous' in stone_description(entries,'necklace')['summary']

@pytest.mark.parametrize('model',['embsys_wm','essae_jx500','accet_cg1003l'])
def test_setting_shared_across_scales_and_platforms(tmp_path,model):
    path=tmp_path/'settings.json'
    store=SettingsStore(path,platform='windows')
    assert not store.get()['stone_analysis']['enable_weight_estimation']
    store.update({'stone_analysis':{'enable_weight_estimation':True}})
    store.select_scale(model)
    assert store.get()['stone_analysis']['enable_weight_estimation']
    assert SettingsStore(path,platform='linux').get()['stone_analysis']['enable_weight_estimation']

@pytest.mark.parametrize('enabled',[False,True])
def test_print_and_image_do_not_mention_estimates_when_disabled(tmp_path,monkeypatch,enabled):
    original=tmp_path/'original.png'
    Image.new('RGB',(1080,1920),'white').save(original)
    stones=dict(found=True,estimated_weight_g=2,weight_min_g=2,weight_max_g=3,summary='This necklace has 2 detected stone regions: 2 small red.',color_note='Red includes maroon.')
    state=dict(id='test',paths={'original':str(original)},weight_g=20,captured_at='2026-10-31T12:00:00',settings_snapshot={'stone_analysis':{'enable_weight_estimation':enabled}},result={'count':1,'items':[{'index':1,'label':'Necklace','stones':stones}]})
    assert stone_estimation_enabled(state)==enabled
    if not enabled:
        assert weight_summary(state)=={'gross_g':20,'enabled':False}
    texts=[]
    original_draw=ImageDraw.ImageDraw.text
    def record(self,xy,text,*args,**kwargs):
        texts.append(text)
        return original_draw(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',record)
    service=ReportService()
    service.generate_result_image(state)
    assert any('estimated' in str(t).lower() for t in texts)==enabled
    assert '2 detected stone regions' not in ' '.join(str(t) for t in texts)
    assert 'STONE ANALYSIS' not in texts
    # Verify actual exported PDF text, not just the helper's return value.
    from backend.app.services import report_service
    original_document = report_service.SimpleDocTemplate
    def uncompressed_document(*args, **kwargs):
        kwargs['pageCompression'] = 0
        return original_document(*args, **kwargs)
    monkeypatch.setattr(report_service, 'SimpleDocTemplate', uncompressed_document)
    text=service.generate(state).getvalue().decode('latin1')
    assert ('Estimated stone weight' in text)==enabled
    assert '2 detected stone regions' in text
    assert ('Red includes maroon' in text)==enabled


def test_summary_omits_uncertain_tiny_and_minor_colors():
    def entry(color, area, confidence=.9, reportable=True):
        return dict(color=color, confidence=confidence, area_px=area, area_mm2=area/100,
                    size="small", continuous_area=False, reportable=reportable)
    entries = [entry("Green", 700), entry("Red", 250), entry("Blue", 30),
               entry("Uncertain", 600), entry("Pink", 1000, reportable=False)]
    result = stone_description(entries, "necklace")
    assert result["region_count"] == 4
    assert "green and red" in result["summary"]
    assert "uncertain" not in result["summary"].lower()
    assert "blue" not in result["summary"].lower()
    assert "pink" not in result["summary"].lower()
    assert stone_description(entries, found=False)["summary"] == ""
    assert stone_description([entry("Red", 10, reportable=False)])["summary"] == ""
    assert "color" not in stone_description([entry("Uncertain", 50)])["summary"].lower()


def test_color_source_does_not_change_detection_or_measurement():
    from StoneDetection import jewel_gem_hsv_report as detection
    image = np.full((180,180,3), 180, np.uint8)
    mask = np.zeros((180,180), np.uint8)
    cv2.circle(image, (90,90), 65, (25,125,180), -1)
    cv2.circle(mask, (90,90), 65, 1, -1)
    cv2.circle(image, (90,90), 17, (25,25,130), -1)
    candidates = detection.build_candidates_from_component_mask(image, mask, min_area=80)
    kwargs = dict(zoom_scale=1, preset_candidates=candidates, external_mask=mask,
                  measurement_scale={"mm_per_pixel_x":.18, "mm_per_pixel_y":.18},
                  fastsam_model=None, fastsam_lock=None)
    first = detection.analyze_image_bgr(image, original_color_bgr=image, **kwargs)["report"]
    green = np.full_like(image, (65,130,20))
    second = detection.analyze_image_bgr(image, original_color_bgr=green, **kwargs)["report"]
    assert first["stone_instance_count"] > 0
    assert first["stone_instance_count"] == second["stone_instance_count"]
    assert first["stone_measurements"] == second["stone_measurements"]
    assert first["jewels"][0]["regions"] == second["jewels"][0]["regions"]
    assert second["jewels"][0]["color_predictions"][0]["color"] == "Green"


def test_color_prediction_preserves_segmentation_random_state():
    cv2.setRNGSeed(419)
    expected=np.zeros((10,10),np.float32)
    cv2.randu(expected,0,1)
    cv2.setRNGSeed(419)
    image=np.full((40,40,3),(30,25,190),np.uint8)
    image[10:20]=(255,255,255)
    image[20:30]=(20,20,20)
    predict_gemstone_color(image,np.ones((40,40),np.uint8))
    actual=np.zeros_like(expected)
    cv2.randu(actual,0,1)
    assert np.array_equal(actual,expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_gallery_sentence_only_mode(tmp_path, monkeypatch, enabled):
    from StoneDetection import jewel_gem_hsv_report as detection
    texts=[]
    draw_text=ImageDraw.ImageDraw.text
    def record(self, xy, text, *args, **kwargs):
        texts.append(str(text))
        return draw_text(self, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    view=dict(jewel_id=1, overlay_bgr=np.full((100,100,3), 220, np.uint8), detected_colors=[{"color":"Red"}])
    fields=[("STONE INSIGHTS", "This necklace has 2 detected stone regions. Most present stone colors: red.")]
    weights=weight_summary(dict(weight_g=20, result={"count":1,"items":[{"stones":{"found":True,"estimated_weight_g":2,"weight_min_g":2,"weight_max_g":3}}]})) if enabled else None
    detection.build_result_gallery([view], fields, weights=weights, description_only=not enabled)
    assert "STONE INSIGHTS" in texts
    if not enabled:
        assert not any("AREA" in t or "WEIGHT" in t or "estimated" in t.lower() for t in texts)
    else:
        assert any("estimated" in t.lower() for t in texts)
