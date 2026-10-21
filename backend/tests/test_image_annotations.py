import numpy as np
from PIL import ImageDraw

from backend.app.analysis.annotations import add_side_panel, label_box
from Dimension.bangle_detector import draw_results


def test_side_panel_wraps_values_and_preserves_entire_photo(monkeypatch):
    image = np.full((520, 520, 3), 183, dtype=np.uint8)
    rendered = []
    original = ImageDraw.ImageDraw.text

    def track(self, xy, text, *args, **kwargs):
        rendered.append((xy, text, self.textbbox(xy, text, font=kwargs["font"], anchor=kwargs["anchor"])))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track)
    value = "Chain / Necklace, Earrings, Finger Ring, Bangle, Bracelet, Pendant"
    result = add_side_panel(image, "CAPTURE REVIEW", [("JEWELLERY TYPE", value)])
    assert result.shape == (1024, 1536, 3)
    assert np.array_equal(image, np.full((520, 520, 3), 183, dtype=np.uint8))
    assert len(rendered) > 3  # The value has wrapped instead of shrinking.
    for _, _, box in rendered:
        assert 0 <= box[0] < box[2] <= result.shape[1]
        assert 0 <= box[1] < box[3] < result.shape[0]


def test_jewel_label_is_not_squeezed_to_small_detection_width():
    image = np.full((400, 520, 3), 255, dtype=np.uint8)
    label_box(image, (250, 250, 280, 280), "Jewel 1 | Finger Ring")
    # The label expands beyond the 30px object instead of becoming unreadable.
    assert np.any(image[200:250, 285:350] != 255)


def test_portrait_side_panel_uses_medium_type_at_card_size(monkeypatch):
    rendered = []
    original = ImageDraw.ImageDraw.text

    def track(self, xy, text, *args, **kwargs):
        rendered.append((text, kwargs["font"].size))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track)
    photo = np.full((1920, 1080, 3), 183, dtype=np.uint8)
    result = add_side_panel(photo, "BEAD ANALYSIS", [
        ("DETECTED", "227"), ("TINY", "126"), ("SMALL", "100"), ("LARGE", "1"),
    ])
    assert result.shape == (1024, 1536, 3)
    # A portrait source is contained in the landscape display export, not resized
    # in place. Production callers select the tray/jewel crop on a copy first.
    assert photo.shape == (1920, 1080, 3)
    assert np.all(photo == 183)
    assert min(size for text, size in rendered if text != "i") * 768 / result.shape[1] >= 14



def test_dimension_labels_keep_mm_and_uncalibrated_px_distinct(monkeypatch):
    texts = []
    original = ImageDraw.ImageDraw.text

    def track(self, xy, text, *args, **kwargs):
        texts.append(text)
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track)
    circle = {"center": (90, 90), "radius": 60, "method": "test"}
    result = {"outer": circle, "inner": {**circle, "radius": 45},
              "od_mm": 25.0549, "id_mm": None, "od_px": 120, "id_px": 90}
    image = np.full((180, 180, 3), 255, dtype=np.uint8)
    output = draw_results(image, result)
    assert "25.05 mm" in texts
    assert "90.0 px (uncalibrated)" in texts
    assert image.shape == (180, 180, 3)
    assert output.shape[0] == 1024
    assert output.shape[1] > 520
