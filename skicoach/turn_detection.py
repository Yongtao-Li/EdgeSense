from dataclasses import dataclass
from typing import List

import numpy as np

from .config import TURN_MIN_FRAMES


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    filled = values.astype(np.float32).copy()
    if filled.size == 0:
        return filled

    valid_idx = np.where(np.isfinite(filled))[0]
    if valid_idx.size == 0:
        return np.zeros_like(filled)

    first = int(valid_idx[0])
    last = int(valid_idx[-1])
    filled[:first] = filled[first]
    filled[last + 1 :] = filled[last]

    for idx in range(first + 1, last + 1):
        if not np.isfinite(filled[idx]):
            filled[idx] = filled[idx - 1]
    return filled


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
    stacked = np.stack([left_hip, right_hip, left_shoulder, right_shoulder], axis=1)
    valid_counts = np.sum(np.isfinite(stacked), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        summed = np.nansum(stacked, axis=1)
        com_x = summed / np.maximum(valid_counts, 1)
    com_x[valid_counts == 0] = np.nan
    return _fill_nan_1d(com_x)


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
