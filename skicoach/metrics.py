from dataclasses import dataclass
from typing import List

import numpy as np

from .config import COM_HIP_WEIGHT
from .turn_detection import TurnSegment


@dataclass
class TurnMetrics:
    turn_index: int
    direction: str
    outside_ski_offset: float
    centered_ratio: float
    angulation_deg: float
    leg_tilt_deg: float
    com_move_ratio: float
    outside_ski_score: float
    angulation_score: float
    timing_score: float


def _line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    vec = b - a
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def _angle_to_vertical_deg(a: np.ndarray, b: np.ndarray) -> float:
    vec = b - a
    vertical = np.array([0.0, -1.0], dtype=np.float32)
    denom = (np.linalg.norm(vec) * np.linalg.norm(vertical)) or 1.0
    cos_theta = float(np.clip(np.dot(vec, vertical) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _normalize_score(raw: float) -> float:
    return float(np.clip(1.0 - raw, 0.0, 1.0))


def compute_turn_metrics(keypoints: np.ndarray, turns: List[TurnSegment]) -> List[TurnMetrics]:
    metrics: List[TurnMetrics] = []

    left_ankle = keypoints[:, 27, :2]
    right_ankle = keypoints[:, 28, :2]
    left_hip = keypoints[:, 23, :2]
    right_hip = keypoints[:, 24, :2]
    left_shoulder = keypoints[:, 11, :2]
    right_shoulder = keypoints[:, 12, :2]
    mid_hip = 0.5 * (left_hip + right_hip)
    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
    com_xy = (COM_HIP_WEIGHT * mid_hip) + ((1.0 - COM_HIP_WEIGHT) * mid_shoulder)

    for idx, turn in enumerate(turns):
        mid = turn.apex
        direction = "right" if turn.direction > 0 else "left"
        outside = left_ankle[mid] if turn.direction < 0 else right_ankle[mid]
        stance_w = abs(left_ankle[mid, 0] - right_ankle[mid, 0]) or 1e-6
        com = com_xy[mid]

        outside_ski_offset = abs((com[0] - outside[0]) / stance_w)

        turn_slice = slice(turn.start, turn.end + 1)
        base_center_x = 0.5 * (left_ankle[turn_slice, 0] + right_ankle[turn_slice, 0])
        centered = np.abs(com_xy[turn_slice, 0] - base_center_x) < (0.12 * stance_w)
        centered_ratio = float(np.nanmean(centered.astype(np.float32)))

        hip_angle = _line_angle_deg(left_hip[mid], right_hip[mid])
        shoulder_angle = _line_angle_deg(left_shoulder[mid], right_shoulder[mid])
        angulation_deg = abs(shoulder_angle - hip_angle)

        outside_leg_vec_start = left_ankle[mid] if turn.direction < 0 else right_ankle[mid]
        outside_leg_vec_end = left_hip[mid] if turn.direction < 0 else right_hip[mid]
        leg_tilt_deg = _angle_to_vertical_deg(outside_leg_vec_start, outside_leg_vec_end)

        onset = None
        target_sign = 1 if turn.direction > 0 else -1
        for i in range(turn.start, turn.apex + 1):
            pelvis_shift = com_xy[i, 0] - 0.5 * (left_ankle[i, 0] + right_ankle[i, 0])
            if np.sign(pelvis_shift) == target_sign and abs(pelvis_shift) > 0.02:
                onset = i
                break
        if onset is None:
            com_move_ratio = 1.0
        else:
            com_move_ratio = (onset - turn.start) / max(1, (turn.apex - turn.start))

        outside_ski_score = _normalize_score(max(0.0, outside_ski_offset - 0.35))
        angulation_score = float(np.clip(angulation_deg / 20.0, 0.0, 1.0))
        timing_score = _normalize_score(max(0.0, com_move_ratio - 0.7))

        metrics.append(
            TurnMetrics(
                turn_index=idx,
                direction=direction,
                outside_ski_offset=outside_ski_offset,
                centered_ratio=centered_ratio,
                angulation_deg=angulation_deg,
                leg_tilt_deg=leg_tilt_deg,
                com_move_ratio=com_move_ratio,
                outside_ski_score=outside_ski_score,
                angulation_score=angulation_score,
                timing_score=timing_score,
            )
        )
    return metrics
