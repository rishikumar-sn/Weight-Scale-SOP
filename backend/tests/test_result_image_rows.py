import pytest
from PIL import Image, ImageDraw

from backend.app.services.report_service import ReportService


@pytest.mark.parametrize("count", [6, 12, 24])
def test_result_image_renders_every_jewel_above_footer(tmp_path, monkeypatch, count):
    original = tmp_path / "original.png"
    Image.new("RGB", (1080, 1920), "#777777").save(original)
    labels = [f"Jewellery type {index}" for index in range(1, count + 1)]
    rendered = []
    draw_text = ImageDraw.ImageDraw.text

    def track_text(self, position, text, *args, **kwargs):
        if text in labels:
            rendered.append((text, position))
        return draw_text(self, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track_text)
    report = ReportService()
    state = {"paths": {"original": str(original)}, "weight_g": 10,
             "captured_at": "2026-10-16T12:00:00+05:30",
             "result": {"count": count, "items": [{"label": label} for label in labels]}}
    with Image.open(report.generate_result_image(state)) as output:
        assert [label for label, _ in rendered] == labels
        assert output.height >= 1920
        if count >= 12:
            assert output.height > 1920
        if report.client_logo.is_file():
            # Every label stays above the footer divider (250 scaled pixels).
            scale = min(583 / 580, 1920 / 1920)
            assert max(y for _, (_, y) in rendered) < output.height - round(250 * scale)
