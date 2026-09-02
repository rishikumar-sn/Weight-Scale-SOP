import cv2
import numpy as np

from backend.app.analysis.vision import detect_apriltag


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
