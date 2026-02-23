from typing import List, Optional, Tuple

import numpy as np

from .config import CROP_EMA_ALPHA, CROP_MARGIN_X, CROP_MARGIN_Y, CROP_MIN_POINTS, MIN_VISIBILITY


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def _bbox_from_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    min_xy = np.nanmin(points_xy, axis=0)
    max_xy = np.nanmax(points_xy, axis=0)
    x1 = (min_xy[0] - CROP_MARGIN_X) * width
    x2 = (max_xy[0] + CROP_MARGIN_X) * width
    y1 = (min_xy[1] - CROP_MARGIN_Y) * height
    y2 = (max_xy[1] + CROP_MARGIN_Y) * height
    return _clip_box(np.array([x1, y1, x2, y2], dtype=np.float32), width, height)


def compute_crop_boxes(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    frame_shape: Tuple[int, int, int],
    initial_box: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    height, width = frame_shape[:2]
    boxes: List[np.ndarray] = []
    if initial_box is not None:
        prev = _clip_box(initial_box.astype(np.float32), width, height).astype(np.float32)
    else:
        prev = np.array([0, 0, width, height], dtype=np.float32)

    for i in range(keypoints.shape[0]):
        vis = visibility[i] >= MIN_VISIBILITY
        pts = keypoints[i, vis, :2]
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] >= CROP_MIN_POINTS:
            current = _bbox_from_points(pts, width, height).astype(np.float32)
        else:
            current = prev.copy()
        smooth = (CROP_EMA_ALPHA * current) + ((1.0 - CROP_EMA_ALPHA) * prev)
        smooth = _clip_box(smooth, width, height).astype(np.float32)
        boxes.append(smooth.astype(np.int32))
        prev = smooth
    return boxes


def crop_frame(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.int32)
    return frame[y1:y2, x1:x2]
