from copy import deepcopy

import cv2
import numpy as np
import pytest
from PIL import ImageDraw

# Initialize the optional stone pipeline imports exactly as the app does.
from backend.app.services.analysis_service import AnalysisService
from backend.app.analysis import annotations
from Dimension.bangle_detector import draw_results
from StoneDetection import jewel_gem_hsv_report as stones


@pytest.mark.parametrize("count", [2, 4, 13, 24])
def test_multi_review_keeps_every_item_and_combined_weight(monkeypatch, count):
    texts = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, value, *args, **kwargs):
        texts.append((xy, value))
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    source = np.full((1920, 1080, 3), 180, np.uint8)
    items = [dict(index=2 * i + 1, predicted_label="Finger Ring",
                  bbox=dict(x=50 + i % 4 * 220, y=100 + i // 4 * 220, w=100, h=100))
             for i in range(count)]
    state = dict(weight_g=40.81)
    saved = deepcopy((state, items))
    result = annotations.render_capture_review(source, state, items)
    values = [value for _, value in texts]
    assert "TOTAL CAPTURED WEIGHT" in values
    assert values.count("40.81 g") == 1
    assert values.count("Finger Ring") == count
    for item in items:
        assert f"J{item['index']}" in values
    assert all(0 <= xy[1] < result.shape[0] for xy, _ in texts)
    assert result.shape[1] == 1536
    assert result.shape[0] > 1024 if count >= 13 else result.shape[0] == 1024
    assert (state, items) == saved
    assert np.all(source == 180)


def test_stone_regions_keep_outer_capture_identity(monkeypatch):
    references = []
    original = annotations.identify_result_image
    def record(image, reference):
        references.append(reference)
        return original(image, reference)
    monkeypatch.setattr(annotations, "identify_result_image", record)
    source = np.full((100, 100, 3), 180, np.uint8)
    views = [dict(jewel_id=i, overlay_bgr=source, detected_colors=[]) for i in (1, 2)]
    reference = annotations.jewel_reference(dict(index=7, predicted_label="Bangle", confirmed_label="Finger Ring"))
    stones.build_result_gallery(views, item_reference=reference)
    assert references == ["J7 / Finger Ring / Region 1", "J7 / Finger Ring / Region 2"]
    stones.build_result_gallery([], item_reference=reference)
    assert references[-1] == "J7 / Finger Ring"
    assert np.all(source == 180)


def test_result_identity_is_added_only_to_export_copy():
    source = np.full((1024, 1536, 3), 180, np.uint8)
    output = annotations.identify_result_image(source, "J13 / Bangle")
    assert not np.array_equal(output, source)
    assert np.all(source == 180)
    np.testing.assert_array_equal(output[144:], source[144:])


@pytest.mark.parametrize("labels, expected", [
    (["Bangle"], "Image OK"),
    (["Chain / Necklace", "Bangle", "Earrings / Nosepin"], "Image OK"),
    (["Bangle", "Not Gold Jewelry"], "Image is not proper"),
    ([], "Presence not checked"),
])
def test_presence_badge_uses_all_existing_classifications(monkeypatch, labels, expected):
    text = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, value, *args, **kwargs):
        text.append(value)
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    photo = np.full((1920, 1080, 3), 170, np.uint8)
    items = [dict(index=i + 1, predicted_label=label, bbox=dict(x=100, y=100 + i * 150, w=80, h=100)) for i, label in enumerate(labels)]
    state = dict(rois={"processing": dict(x=40, y=250, w=950, h=950)}, weight_g=20)
    before = deepcopy((state, items))
    result = annotations.render_capture_review(photo, state, items)
    assert expected in text
    assert result.shape[1] == 1536
    assert (state, items) == before
    assert np.all(photo == 170) and photo.shape == (1920, 1080, 3)
    if expected != "Image OK":
        assert "Image OK" not in text


def test_review_translates_boxes_and_keeps_tag_in_display_crop(monkeypatch):
    boxes = []
    original = annotations.label_box
    def record(image, box, text, **kwargs):
        boxes.append((box, text))
        original(image, box, text, **kwargs)
    monkeypatch.setattr(annotations, "label_box", record)
    source = np.full((300, 400, 3), 200, np.uint8)
    state = {"rois": {"processing": dict(x=-30, y=20, w=400, h=250)},
             "calibration": {"corners": [[5, 5], [15, 5], [15, 15], [5, 15]]}}
    items = [dict(index=3, predicted_label="Bangle", bbox=dict(x=80, y=70, w=100, h=120))]
    annotations.render_capture_review(source, state, items)
    assert boxes == [((50, 90, 150, 210), "J3")]
    assert np.all(source == 200)


def test_display_crop_clamps_virtual_padding_and_returns_independent_pixels():
    photo = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    copy = photo.copy()
    cropped = annotations.display_crop(photo, dict(x=-40, y=-20, w=100, h=70), padding=0)
    np.testing.assert_array_equal(cropped, photo[:54, :64])
    cropped[:] = 0
    np.testing.assert_array_equal(photo, copy)


def test_bead_percentages_and_colors_are_from_saved_counts(monkeypatch):
    texts, colors = [], []
    original_text, original_rectangle = ImageDraw.ImageDraw.text, cv2.rectangle
    def record_text(self, xy, value, *args, **kwargs):
        texts.append(value)
        return original_text(self, xy, value, *args, **kwargs)
    def record_rectangle(image, p1, p2, color, *args, **kwargs):
        colors.append(color)
        return original_rectangle(image, p1, p2, color, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    monkeypatch.setattr(cv2, "rectangle", record_rectangle)
    detections = [dict(bbox=[10, 10, 20, 20], size_category=category) for category, count in [("tiny", 111), ("small", 61), ("large", 3)] for _ in range(count)]
    analysis = {"size": {"available": True, "counts": dict(tiny=111, small=61, large=3)}}
    saved = deepcopy((detections, analysis))
    source = np.full((200, 300, 3), 180, np.uint8)
    rendered = annotations.render_bead_evidence(source, detections, analysis, dict(x=0, y=0, w=100, h=100))
    assert rendered.shape == (1024, 1536, 3)
    assert {"175", "111", "61", "3", "63.4%", "34.9%", "1.7%"}.issubset(texts)
    assert set(colors) == set(annotations.BEAD_SIZE_COLORS_BGR.values())
    assert (detections, analysis) == saved
    assert np.all(source == 180)


def test_dimension_render_does_not_modify_fitted_geometry_or_input():
    photo = np.full((400, 400, 3), 255, np.uint8)
    circle = dict(center=[200, 200], radius=150, method="fixture")
    result = dict(outer=circle, inner={**circle, "radius": 110}, od_mm=60., id_mm=44., od_px=300., id_px=220.)
    saved = deepcopy(result)
    output = draw_results(photo, result)
    assert output.shape == (1024, 1536, 3)
    assert result == saved and np.all(photo == 255)


def test_small_jewel_zoom_is_capped_and_neighbour_is_hidden():
    source = np.full((100, 100, 3), 255, np.uint8)
    source[30:70, 30:70] = (20, 90, 160)
    source[10:25, 75:95] = (255, 0, 0)  # Another object outside this jewel's mask.
    alpha = np.zeros((100, 100, 4), np.uint8)
    alpha[30:70, 30:70, 3] = 255
    saved = source.copy()
    output = stones.build_result_gallery([dict(
        jewel_id=1, overlay_bgr=source, masked_bgra=alpha, detected_colors=[]
    )])
    photo = output[156:1000, 12:852]
    jewel_pixels = np.all(photo == (20, 90, 160), axis=2)
    points = cv2.findNonZero(jewel_pixels.astype(np.uint8))
    _, _, width, height = cv2.boundingRect(points)
    assert 70 <= width <= 80 and 70 <= height <= 80
    assert not np.any(np.all(photo == (255, 0, 0), axis=2))
    np.testing.assert_array_equal(source, saved)


def test_stone_note_is_small_neutral_footer_and_empty_results_have_none(monkeypatch):
    texts = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, value, *args, **kwargs):
        texts.append((xy, value, kwargs))
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    view = dict(jewel_id=1, overlay_bgr=np.full((100, 100, 3), 255, np.uint8), detected_colors=[{"color": "Green"}])
    output = stones.build_result_gallery([view])
    notes = [(xy, value, kwargs) for xy, value, kwargs in texts if value.startswith("Note:")]
    assert len(notes) == 1
    xy, value, style = notes[0]
    assert "green regions" in value
    assert xy[1] >= output.shape[0] - 80
    assert style["font"].size == 20 and style["fill"] == "#617083"
    assert not np.any(np.all(output == (243, 240, 255), axis=2))  # Old pink note card.
    texts.clear()
    stones.build_result_gallery([{**view, "detected_colors": []}])
    assert not any("Note:" in value or "NOTE" == value for _, value, _ in texts)


def test_dimension_export_omits_notes_even_when_supplied(monkeypatch):
    texts = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, value, *args, **kwargs):
        texts.append(value)
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    annotations.add_side_panel(np.full((100, 100, 3), 255, np.uint8),
                               "DIMENSION ANALYSIS", [("NOTE", "Old note"), ("OUTER", "60 mm")],
                               note="Geometry explanation")
    assert "60 mm" in texts
    assert not any("note" in value.lower() or "Geometry" in value for value in texts)


def test_stone_mask_fill_is_52_percent_and_contour_one_pixel():
    source = np.full((100, 100, 3), (70, 55, 150), np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    _, overlay, _ = stones.render_jewel_outputs(source, np.ones_like(mask), [dict(mask=mask, contour=contour, learned_matches=[])])
    expected = source.copy()
    expected[mask > 0] = (source[mask > 0] * .48 + np.array((118, 155, 36)) * .52).astype(np.uint8)
    cv2.drawContours(expected, [contour], -1, (118, 155, 36), 1, cv2.LINE_AA)
    np.testing.assert_array_equal(overlay, expected)
    np.testing.assert_array_equal(source[50, 50], (70, 55, 150))
    np.testing.assert_array_equal(overlay[50, 50], (94, 107, 90))
    _, empty, _ = stones.render_jewel_outputs(source, np.ones_like(mask), [])
    np.testing.assert_array_equal(empty, source)


def test_rejected_review_only_updates_display_path_and_preserves_original(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = tmp_path / "original.png"
    cv2.imwrite(str(path), np.full((200, 300, 3), 180, np.uint8))
    original_bytes = path.read_bytes()
    state = dict(id="rejected", status="captured", weight_g=8.5,
                 classification={}, paths={"original": str(path)})
    saved = []
    repository = SimpleNamespace(session_dir=lambda _: tmp_path,
                                 save=lambda value: saved.append(deepcopy(value)))
    service = AnalysisService(repository)
    try:
        service._save_rejected_review(state)
        assert saved[0]["status"] == "captured"
        assert saved[0]["classification"] == {}
        assert saved[0]["weight_g"] == 8.5
        assert cv2.imread(state["paths"]["evidence"]).shape == (1024, 1536, 3)
        assert path.read_bytes() == original_bytes
        def fail(*args, **kwargs):
            raise TypeError("Rendering failure")
        monkeypatch.setattr("backend.app.services.analysis_service.render_capture_review", fail)
        # A rendering failure must leave the original presence rejection in control.
        service._save_rejected_review(state)
        assert len(saved) == 1
    finally:
        service.shutdown()
