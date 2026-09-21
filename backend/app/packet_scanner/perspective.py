import cv2
import numpy as np
from .obb_utils import order_points


def warp_perspective(image, points):
    """Warp from original pixels and put the long label axis horizontally."""
    tl, tr, br, bl = order_points(points)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    w, h = int(round(width)), int(round(height))
    if w < 4 or h < 4:
        raise ValueError("Detected target is too small for rectification")
    dest = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dest)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped
