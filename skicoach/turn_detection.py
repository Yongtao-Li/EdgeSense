from dataclasses import dataclass
from typing import List

import numpy as np

from .config import TURN_MIN_FRAMES


@dataclass
class TurnSegment:
    start: int
    apex: int
    end: int
    direction: int  # +1 right, -1 left
    phases: List[str]


def compute_com_x(keypoints: np.ndarray) -> np.ndarray:
    left_hip = keypoints[:, 23, 0]
    right_hip = keypoints[:, 24, 0]
    left_shoulder = keypoints[:, 11, 0]
    right_shoulder = keypoints[:, 12, 0]
    com_x = np.nanmean(
        np.stack([left_hip, right_hip, left_shoulder, right_shoulder], axis=1),
        axis=1,
    )
    return com_x


def _turn_boundaries_from_velocity(com_x: np.ndarray) -> List[int]:
    vel = np.diff(com_x)
    boundaries = [0]
    last = 0
    for i in range(1, len(vel)):
        if np.isnan(vel[i - 1]) or np.isnan(vel[i]):
            continue
        if vel[i - 1] == 0.0 or vel[i] == 0.0:
            continue
        if np.sign(vel[i - 1]) != np.sign(vel[i]):
            idx = i
            if idx - last >= TURN_MIN_FRAMES // 2:
                boundaries.append(idx)
                last = idx
    if boundaries[-1] != len(com_x) - 1:
        boundaries.append(len(com_x) - 1)
    return boundaries


def _find_apex(com_x: np.ndarray, start: int, end: int) -> int:
    segment = com_x[start : end + 1]
    if len(segment) == 0 or np.isnan(segment).all():
        return (start + end) // 2
    baseline = np.nanmean(segment)
    rel = np.abs(segment - baseline)
    apex_local = int(np.nanargmax(rel))
    return start + apex_local


def detect_turns(keypoints: np.ndarray) -> List[TurnSegment]:
    com_x = compute_com_x(keypoints)
    boundaries = _turn_boundaries_from_velocity(com_x)
    turns: List[TurnSegment] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < TURN_MIN_FRAMES:
            continue
        apex = _find_apex(com_x, start, end)
        direction = 1 if com_x[end] >= com_x[start] else -1
        phases = _assign_phases(start, apex, end)
        turns.append(TurnSegment(start=start, apex=apex, end=end, direction=direction, phases=phases))
    return turns


def _assign_phases(start: int, apex: int, end: int) -> List[str]:
    total = end - start + 1
    phases = []
    for i in range(total):
        idx = start + i
        if idx < start + total // 3:
            phases.append("initiation")
        elif idx < start + 2 * total // 3:
            phases.append("apex")
        else:
            phases.append("finish")
    return phases
