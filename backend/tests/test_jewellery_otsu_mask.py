from __future__ import annotations

import cv2
import numpy as np

from backend.app.analysis import vision


def test_erase_region_is_removed_before_threshold_candidate_scoring(monkeypatch) -> None:
    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    erase = np.zeros((100, 100), dtype=np.uint8)
    erase[:, :2] = 255

    whole_background = np.full((100, 100), 255, dtype=np.uint8)
    boundary_connected_jewel = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(boundary_connected_jewel, (0, 45), (42, 55), 255, -1)
    cv2.circle(boundary_connected_jewel, (45, 50), 14, 255, -1)

    monkeypatch.setattr(
        vision,
        "_threshold",
        lambda _image, white_background: (
            whole_background.copy()
            if white_background
            else boundary_connected_jewel.copy()
        ),
    )

    _prepared, mask = vision.build_jewellery_mask(image, erase)

    assert mask[50, 45] == 1
    assert mask[50, 80] == 0
    assert cv2.countNonZero(mask) < 1500


def test_crop_exterior_erases_tray_rim_when_testbed_reaches_crop_edge(monkeypatch) -> None:
    image = np.full((240, 240, 3), 180, dtype=np.uint8)
    erase = np.zeros((240, 240), dtype=np.uint8)
    erase[:20, :20] = 255  # Marker; testbed otherwise fills the crop.
    foreground = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(foreground, (120, 120), 40, 255, 8)
    foreground[-10:, 75:165] = 255  # Bottom rim incorrectly inside the platform mask.
    monkeypatch.setattr(vision, "_threshold", lambda *_args: foreground.copy())

    prepared, mask = vision.build_jewellery_mask(image, erase)
    items = vision.separate_jewellery_items(image, mask, min_area_px=350)

    assert not np.any(mask[-12:, :])
    assert np.all(prepared[-12:, :] == 255)
    np.testing.assert_array_equal(mask[60:180, 60:180], foreground[60:180, 60:180] > 0)
    assert len(items) == 1
