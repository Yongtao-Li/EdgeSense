from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MIN_VISIBILITY


def _draw_line(frame: np.ndarray, p1: np.ndarray, p2: np.ndarray, color: Tuple[int, int, int]):
    if np.isnan(p1).any() or np.isnan(p2).any():
        return
    h, w = frame.shape[:2]
    pt1 = (int(p1[0] * w), int(p1[1] * h))
    pt2 = (int(p2[0] * w), int(p2[1] * h))
    cv2.line(frame, pt1, pt2, color, 2)


def _draw_point(frame: np.ndarray, p: np.ndarray, color: Tuple[int, int, int]):
    if np.isnan(p).any():
        return
    h, w = frame.shape[:2]
    pt = (int(p[0] * w), int(p[1] * h))
    cv2.circle(frame, pt, 4, color, -1)


def _draw_legend(frame: np.ndarray, phase: Optional[str], turn_id: Optional[int]):
    h, w = frame.shape[:2]
    box_w = int(w * 0.40)
    box_h = 122
    x0, y0 = 18, 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    lines = [
        ("Hip line", (0, 255, 0)),
        ("Knee line", (0, 200, 255)),
        ("COM vertical", (255, 0, 0)),
    ]
    for i, (label, color) in enumerate(lines):
        y = y0 + 22 + i * 22
        cv2.line(frame, (x0 + 10, y - 7), (x0 + 40, y - 7), color, 2)
        cv2.putText(frame, label, (x0 + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if phase and turn_id is not None:
        cv2.putText(
            frame,
            f"Turn {turn_id + 1} - {phase}",
            (x0 + 10, y0 + box_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _safe_midpoint(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    a_ok = np.all(np.isfinite(a[:2]))
    b_ok = np.all(np.isfinite(b[:2]))
    if a_ok and b_ok:
        return 0.5 * (a[:2] + b[:2])
    if a_ok:
        return a[:2]
    if b_ok:
        return b[:2]
    return None


def overlay_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    phase: Optional[str] = None,
    turn_id: Optional[int] = None,
) -> np.ndarray:
    drawn = frame.copy()
    kp = keypoints.copy()
    kp[visibility < MIN_VISIBILITY] = np.nan

    left_hip, right_hip = kp[23], kp[24]
    left_knee, right_knee = kp[25], kp[26]
    com = _safe_midpoint(left_hip, right_hip)
    if com is None:
        com = _safe_midpoint(kp[11], kp[12])

    _draw_line(drawn, left_hip[:2], right_hip[:2], (0, 255, 0))
    _draw_line(drawn, left_knee[:2], right_knee[:2], (0, 200, 255))
    if com is not None:
        _draw_line(drawn, np.array([com[0], 0.0]), np.array([com[0], 1.0]), (255, 0, 0))

    for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
        _draw_point(drawn, kp[idx, :2], (220, 220, 220))
    if com is not None:
        _draw_point(drawn, com[:2], (255, 0, 0))
    _draw_legend(drawn, phase=phase, turn_id=turn_id)
    return drawn
