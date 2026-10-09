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
