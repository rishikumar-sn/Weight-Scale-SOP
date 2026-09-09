import cv2
import numpy as np
import pytest

from Dimension.bangle_detector import detect_bangle, finger_ring_circle_detection


def ring_image(axes=(50, 50), center=(260, 210), decoration=False):
    image = np.full((400, 500, 3), 170, np.uint8)
    cv2.ellipse(image, center, axes, 25, 0, 360, (40, 70, 95), -1)
    cv2.ellipse(image, center, (axes[0] - 6, axes[1] - 6),
                25, 0, 360, (170, 170, 170), -1)
    if decoration:
        # A larger gold ornament with numerous small circular details must not
        # displace the actual opening (the previous Hough scorer preferred it).
        cv2.circle(image, (90, 105), 70, (35, 100, 140), -1)
        for x in range(50, 135, 15):
            for y in range(70, 140, 15):
                cv2.circle(image, (x, y), 5, (150, 185, 205), 2)
    return image


@pytest.mark.parametrize('axes,center', [((50, 50), (260, 210)), ((40, 55), (335, 245))])
def test_boundaries_follow_ring_at_original_coordinates(axes, center):
    outer, inner, _ = finger_ring_circle_detection(ring_image(axes, center, True), .18)
    assert np.linalg.norm(np.array(outer.center) - center) < 1
    assert np.linalg.norm(np.array(inner.center) - center) < 1
    assert sorted(outer.ellipse[1]) == pytest.approx(sorted(2 * v for v in axes), abs=1.5)
    assert sorted(inner.ellipse[1]) == pytest.approx(sorted(2 * (v - 6) for v in axes), abs=1.5)


def test_hidden_opening_is_rejected():
    image = ring_image(decoration=True)
    cv2.circle(image, (260, 210), 54, (35, 100, 140), -1)
    with pytest.raises(RuntimeError, match='not clearly visible'):
        finger_ring_circle_detection(image, .18)


def test_apriltag_is_not_a_ring():
    image = np.full((400, 500, 3), 170, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, 1, 120)
    image[80:200, 160:280] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    with pytest.raises(RuntimeError, match='not clearly visible'):
        finger_ring_circle_detection(image, .18)


def test_app_entry_point_exports_fitted_ellipses(tmp_path):
    path = tmp_path / 'ring.png'
    cv2.imwrite(str(path), ring_image((40, 55)))
    result = detect_bangle(path, scale=.18, jewel_type='Finger Ring')
    assert result['detection_mode'] == 'finger_ring_edge_ellipse'
    assert result['od_mm'] == pytest.approx(95 * .18, abs=.3)
    assert result['id_mm'] == pytest.approx(83 * .18, abs=.3)
    assert result['outer']['ellipse'] is not None
    assert cv2.imread(result['annotated_path']).shape == (400, 500, 3)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match='finite and positive'):
        finger_ring_circle_detection(ring_image(), scale)
