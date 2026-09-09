import pytest

from backend.app.domain.weights import stone_weight_fields, weight_summary, weight_rows


def capture(average=1.7, gross=10.0, count=1):
    stones = {"found": True, **stone_weight_fields(
        {"success": True, "estimated_total_average_g": average}, True)}
    return {"weight_g": gross, "result": {"count": count, "items": [{"stones": stones}]}}


def test_requested_range_and_reversed_net_endpoints():
    result = weight_summary(capture())
    assert (result["stone_g"], result["stone_min_g"], result["stone_max_g"]) == (1.7, 1.7, 2.7)
    assert (result["net_g"], result["net_min_g"], result["net_max_g"]) == (8.3, 7.3, 8.3)
    assert weight_rows(result)[2] == ["Stone (estimated)", "1.70 g", "1.70 - 2.70 g"]


@pytest.mark.parametrize("gross", [None, -1, float("nan"), float("inf")])
def test_invalid_scale_cannot_produce_net(gross):
    result = weight_summary(capture(gross=gross))
    assert result["net_g"] is None
    assert result["stone_g"] == 1.7


def test_multiple_jewels_never_use_combined_gross_for_individual_net():
    assert weight_summary(capture(count=2))["net_g"] is None


def test_missing_calibration_is_not_zero_stone_weight():
    assert stone_weight_fields({"success": False, "estimated_total_average_g": 0}, True)["estimated_weight_g"] is None
    assert weight_summary(capture(average=None))["net_g"] is None


def test_no_stones_has_no_one_gram_deduction():
    state = capture()
    state["result"]["items"][0]["stones"] = {"found": False, **stone_weight_fields({}, False)}
    result = weight_summary(state)
    assert result["stone_max_g"] == 0
    assert result["net_min_g"] == result["net_max_g"] == 10


def test_impossible_estimate_is_flagged_and_net_never_negative():
    result = weight_summary(capture(gross=1))
    assert result["net_g"] is None
    assert "exceeds gross" in result["note"]
    result = weight_summary(capture(gross=2))
    assert result["net_min_g"] == 0
    assert result["net_g"] == 0.3


def test_failed_or_skipped_stone_analysis_is_unavailable():
    state = capture()
    state["result"]["items"] = [{}]
    assert weight_summary(state)["net_g"] is None


def test_pdf_and_result_image_render_weights_and_beads(tmp_path):
    from PIL import Image
    from backend.app.services.report_service import ReportService

    original = tmp_path / "capture.png"
    Image.new("RGB", (1440, 1920), "white").save(original)
    state = capture()
    state.update(paths={"original": str(original)}, captured_at="2026-09-06T12:30:00",
                 classification={"count": 1, "confirmed_label": "Necklace"})
    state["result"]["items"][0].update(index=1, label="Necklace", beads={"count": 12, "beads_detected": True})
    service = ReportService()
    pdf = service.generate(state).getvalue()
    assert pdf.startswith(b"%PDF")
    assert b"/FlateDecode" in pdf
    image = Image.open(service.generate_result_image(state))
    assert image.size == (2218, 1920)
