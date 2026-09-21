import cv2
import numpy as np


def letterbox(image, size=640):
    """Preserve aspect ratio; remember exact resize and left/top padding."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, (new_width / width, new_height / height), (left, top)


def xywhr_to_four_points(box):
    """Ultralytics OBB angle is radians; OpenCV boxPoints uses degrees."""
    cx, cy, width, height, angle = map(float, box)
    return cv2.boxPoints(((cx, cy), (width, height), np.degrees(angle))).astype(np.float32)


def scale_points_to_original(points, scale, padding, original_shape):
    result = np.asarray(points, dtype=np.float32).copy()
    result[:, 0] = (result[:, 0] - padding[0]) / scale[0]
    result[:, 1] = (result[:, 1] - padding[1]) / scale[1]
    h, w = original_shape[:2]
    result[:, 0] = np.clip(result[:, 0], 0, w - 1)
    result[:, 1] = np.clip(result[:, 1], 0, h - 1)
    return result


def order_points(points):
    """Clockwise TL, TR, BR, BL for a convex quadrilateral."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -int(start), axis=0).astype(np.float32)


def rotated_nms(boxes, scores, iou_threshold):
    """Suppress overlapping rotated rectangles in detector pixel coordinates."""
    if not boxes:
        return []
    rects = [((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(np.degrees(b[4]))) for b in boxes]
    indices = cv2.dnn.NMSBoxesRotated(rects, list(map(float, scores)), 0.0, float(iou_threshold))
    return np.asarray(indices).reshape(-1).astype(int).tolist() if len(indices) else []
