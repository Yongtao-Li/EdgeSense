from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

import numpy as np

from .config import (
    TRACK_DETECT_INTERVAL,
    TRACK_HIT_STREAK_TO_LOCK,
    TRACK_MATCH_SCORE_THRESHOLD,
    TRACK_MISS_STREAK_TO_LOST,
)
from .pose_provider import PersonCandidate, detect_person_candidates_in_frame


@dataclass
class TrackStep:
    box_xyxy: np.ndarray
    state: str
    confidence: float


class TrackingSummary(TypedDict):
    lock_ratio: Optional[float]
    reacquire_count: int
    max_lost_streak: int
    mean_confidence: Optional[float]


class TrackResult(TypedDict):
    boxes: List[np.ndarray]
    states: List[str]
    confidences: List[float]
    summary: TrackingSummary


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(x1 + 1.0, min(float(width), float(x2)))
    y2 = max(y1 + 1.0, min(float(height), float(y2)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _histogram(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.int32)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((24,), dtype=np.float32)
    hist_list = []
    for channel in range(3):
        hist, _ = np.histogram(patch[:, :, channel], bins=8, range=(0, 256), density=False)
        hist_list.append(hist.astype(np.float32))
    hist_vec = np.concatenate(hist_list, axis=0)
    norm = float(np.linalg.norm(hist_vec) + 1e-6)
    return hist_vec / norm


def _appearance_similarity(target_hist: np.ndarray, candidate_hist: np.ndarray) -> float:
    if target_hist.shape != candidate_hist.shape:
        return 0.0
    return float(np.clip(np.dot(target_hist, candidate_hist), 0.0, 1.0))


def _predict_box(prev_box: np.ndarray, velocity: np.ndarray, width: int, height: int) -> np.ndarray:
    predicted = prev_box.copy()
    predicted[[0, 2]] += velocity[0]
    predicted[[1, 3]] += velocity[1]
    return _clip_box(predicted, width, height)


def _choose_candidate(
    frame: np.ndarray,
    candidates: List[PersonCandidate],
    predicted_box: np.ndarray,
    target_hist: np.ndarray,
    allow_loose_gate: bool,
) -> Tuple[Optional[np.ndarray], float]:
    if not candidates:
        return None, 0.0

    ph, pw = frame.shape[:2]
    diag = float(np.hypot(pw, ph) + 1e-6)
    pred_center = _center(predicted_box)
    pred_area = max(1.0, float((predicted_box[2] - predicted_box[0]) * (predicted_box[3] - predicted_box[1])))

    best_box = None
    best_score = -1.0
    gate_norm = 0.55 if allow_loose_gate else 0.35

    for candidate in candidates:
        c_box = np.asarray(candidate.bbox_xyxy, dtype=np.float32)
        c_center = _center(c_box)
        c_area = max(1.0, float((c_box[2] - c_box[0]) * (c_box[3] - c_box[1])))
        dist_norm = float(np.linalg.norm(c_center - pred_center) / diag)
        if dist_norm > gate_norm:
            continue

        iou = _iou(predicted_box, c_box)
        area_score = float(np.exp(-abs(np.log(c_area / pred_area))))
        app_score = _appearance_similarity(target_hist, _histogram(frame, c_box))
        center_score = max(0.0, 1.0 - dist_norm)

        score = (0.42 * app_score) + (0.30 * iou) + (0.18 * center_score) + (0.10 * area_score)
        if score > best_score:
            best_score = score
            best_box = c_box

    if best_box is None:
        return None, 0.0
    return best_box, float(best_score)


def track_target_boxes(
    frames: List[np.ndarray],
    initial_box_xyxy: np.ndarray,
    detect_interval: int = TRACK_DETECT_INTERVAL,
) -> TrackResult:
    if not frames:
        return {
            "boxes": [],
            "states": [],
            "confidences": [],
            "summary": {
                "lock_ratio": 0.0,
                "reacquire_count": 0,
                "max_lost_streak": 0,
                "mean_confidence": 0.0,
            },
        }

    h, w = frames[0].shape[:2]
    prev_box = _clip_box(initial_box_xyxy.astype(np.float32), w, h)
    velocity = np.zeros((2,), dtype=np.float32)
    target_hist = _histogram(frames[0], prev_box)

    steps: List[TrackStep] = [TrackStep(box_xyxy=prev_box.copy(), state="locked", confidence=1.0)]
    miss_streak = 0
    hit_streak = 1
    max_lost_streak = 0
    reacquire_count = 0
    state = "locked"

    for frame_idx in range(1, len(frames)):
        frame = frames[frame_idx]
        predicted = _predict_box(prev_box, velocity, w, h)

        should_detect = (frame_idx % max(1, detect_interval) == 0) or miss_streak > 0 or state in {"lost", "lost_hold"}
        candidates = detect_person_candidates_in_frame(frame, max_candidates=8) if should_detect else []
        allow_loose_gate = miss_streak >= 2
        matched_box, match_score = _choose_candidate(
            frame,
            candidates,
            predicted,
            target_hist,
            allow_loose_gate=allow_loose_gate,
        )

        if not should_detect:
            current_box = predicted
            hit_streak = max(1, hit_streak)
            if state in {"lost", "lost_hold"}:
                state = "lost_hold"
                confidence = max(0.12, steps[-1].confidence * 0.98)
            else:
                state = "locked"
                confidence = max(0.35, steps[-1].confidence * 0.995)
        elif matched_box is not None and match_score >= TRACK_MATCH_SCORE_THRESHOLD:
            current_box = _clip_box(matched_box, w, h)
            prev_center = _center(prev_box)
            current_center = _center(current_box)
            velocity = 0.75 * velocity + 0.25 * (current_center - prev_center)

            was_lost = miss_streak > 0 or state in {"lost", "lost_hold"}
            miss_streak = 0
            hit_streak += 1
            if was_lost and hit_streak >= max(1, TRACK_HIT_STREAK_TO_LOCK):
                state = "reacquired"
                reacquire_count += 1
            elif was_lost:
                state = "lost_hold"
            else:
                state = "locked"

            confidence = float(min(1.0, max(0.2, match_score)))
            alpha = 0.1
            target_hist = ((1.0 - alpha) * target_hist) + (alpha * _histogram(frame, current_box))
            target_hist = target_hist / (float(np.linalg.norm(target_hist)) + 1e-6)
        else:
            current_box = predicted
            miss_streak += 1
            hit_streak = 0
            max_lost_streak = max(max_lost_streak, miss_streak)
            state = "lost" if miss_streak >= max(1, TRACK_MISS_STREAK_TO_LOST) else "lost_hold"
            confidence = max(0.05, 0.45 - 0.06 * miss_streak)
            velocity *= 0.95

        steps.append(TrackStep(box_xyxy=current_box.copy(), state=state, confidence=float(confidence)))
        prev_box = current_box

    boxes = [step.box_xyxy.astype(np.int32) for step in steps]
    states = [step.state for step in steps]
    confidences = [round(float(step.confidence), 4) for step in steps]

    locked_frames = sum(1 for state in states if state in {"locked", "reacquired"})
    lock_ratio = float(locked_frames / len(states)) if states else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "boxes": boxes,
        "states": states,
        "confidences": confidences,
        "summary": {
            "lock_ratio": round(lock_ratio, 4),
            "reacquire_count": int(reacquire_count),
            "max_lost_streak": int(max_lost_streak),
            "mean_confidence": round(mean_conf, 4),
        },
    }
