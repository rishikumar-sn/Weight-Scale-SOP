import cv2
import numpy as np

from backend.app.analysis.vision import apriltag_roi_from_detection, detect_apriltag, roi_to_normalized


def test_apriltag_detection_retries_full_frame_when_configured_roi_misses_tag():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 1, 160)
    image = np.full((500, 700, 3), 255, dtype=np.uint8)
    image[170:330, 370:530] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    result = detect_apriltag(
        image,
        roi={"x": 20, "y": 20, "w": 200, "h": 200},
        marker_id=1,
        width_mm=20.0,
        height_mm=20.0,
    )

    assert result["found"] is True
    assert result["id"] == 1
    assert result["detection_scope"] == "full_frame_fallback"
    assert 0.12 < result["mm_per_pixel_x"] < 0.13
    assert 0.12 < result["mm_per_pixel_y"] < 0.13


def test_apriltag_detection_builds_a_padded_clipped_roi():
    detection = {
        "corners": [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]],
    }

    roi = apriltag_roi_from_detection((100, 200, 3), detection)

    assert roi == {"x": 0, "y": 6, "w": 65, "h": 69}
    assert roi_to_normalized(roi, 200, 100) == {
        "x": 0.0,
        "y": 0.06,
        "width": 0.325,
        "height": 0.69,
    }
