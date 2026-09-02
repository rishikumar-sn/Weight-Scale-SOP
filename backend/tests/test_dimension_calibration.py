import math

import pytest

from backend.app.services.analysis_service import _apriltag_scale_mm_per_pixel


def test_apriltag_scale_uses_both_calibrated_axes():
    calibration = {
        "available": True,
        "found": True,
        "mm_per_pixel_x": 0.16,
        "mm_per_pixel_y": 0.25,
    }

    assert _apriltag_scale_mm_per_pixel(calibration) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "calibration",
    [
        None,
        {"available": False, "message": "AprilTag was not found"},
        {"available": True, "found": True, "mm_per_pixel_x": 0.1},
        {"available": True, "found": True, "mm_per_pixel_x": math.nan, "mm_per_pixel_y": 0.1},
    ],
)
def test_dimension_calibration_rejects_missing_or_invalid_tag(calibration):
    with pytest.raises(RuntimeError, match="AprilTag calibration|valid X/Y scales|finite and positive"):
        _apriltag_scale_mm_per_pixel(calibration)
