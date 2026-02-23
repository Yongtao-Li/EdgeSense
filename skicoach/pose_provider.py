from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    MIN_VISIBILITY,
    PERSON_DETECTOR_BACKEND,
    POSE_DRIFT_HOLD_FRAMES,
    POSE_DRIFT_MAX_CENTER_DIST_NORM,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL,
)


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


@dataclass
class PoseSequence:
    fps: float
    frames: List[np.ndarray]
    keypoints: np.ndarray
    visibility: np.ndarray


@dataclass
class PersonCandidate:
    person_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    center_xy: Tuple[float, float]


@dataclass
class _DetectionResult:
    boxes: List[np.ndarray]
    scores: List[float]
    backend: str


_PERSON_DETECTOR_BACKEND = "unknown"
_YOLO_MODEL = None


def read_video_frames(video_path: str) -> Tuple[float, List[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return float(fps), frames


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
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


def _detect_person_candidates_hog(frame: np.ndarray) -> _DetectionResult:
    height, width = frame.shape[:2]
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    if len(rects) == 0:
        return _DetectionResult(boxes=[], scores=[], backend="hog")

    confidences = np.asarray(weights, dtype=np.float32).reshape(-1)
    boxes: List[np.ndarray] = []
    scores: List[float] = []
    min_area = max(1.0, float(width * height) * 0.01)

    for i, (x, y, w, h) in enumerate(rects):
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(width, int(x + w))
        y2 = min(height, int(y + h))
        area = (x2 - x1) * (y2 - y1)
        if x2 <= x1 or y2 <= y1 or area < min_area:
            continue
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        score = float(confidences[i]) if i < confidences.shape[0] else 0.0
        scores.append(score)

    return _DetectionResult(boxes=boxes, scores=scores, backend="hog")


def _load_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    from ultralytics import YOLO  # type: ignore

    _YOLO_MODEL = YOLO(YOLO_MODEL)
    return _YOLO_MODEL


def _detect_person_candidates_yolo(frame: np.ndarray, max_candidates: int) -> _DetectionResult:
    model = _load_yolo_model()
    pred = model.predict(
        source=frame,
        conf=YOLO_CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        classes=[0],
        max_det=max(1, max_candidates * 4),
        verbose=False,
    )
    if not pred:
        return _DetectionResult(boxes=[], scores=[], backend="yolo")

    boxes_data = pred[0].boxes
    if boxes_data is None or boxes_data.xyxy is None:
        return _DetectionResult(boxes=[], scores=[], backend="yolo")

    height, width = frame.shape[:2]
    boxes: List[np.ndarray] = []
    scores: List[float] = []
    xyxy = boxes_data.xyxy.cpu().numpy()
    conf = boxes_data.conf.cpu().numpy() if boxes_data.conf is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)

    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i].tolist()
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(x1 + 1, min(width, int(round(x2))))
        y2 = max(y1 + 1, min(height, int(round(y2))))
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        scores.append(float(conf[i]) if i < conf.shape[0] else 0.0)

    return _DetectionResult(boxes=boxes, scores=scores, backend="yolo")


def _detect_person_candidates(frame: np.ndarray, max_candidates: int = 6) -> List[PersonCandidate]:
    global _PERSON_DETECTOR_BACKEND

    backend = PERSON_DETECTOR_BACKEND.strip().lower()
    result: Optional[_DetectionResult] = None

    try:
        if backend in {"auto", "yolo"}:
            result = _detect_person_candidates_yolo(frame, max_candidates=max_candidates)
    except Exception:
        result = None

    if result is None:
        result = _detect_person_candidates_hog(frame)

    _PERSON_DETECTOR_BACKEND = result.backend

    boxes = result.boxes
    scores = result.scores
    if not boxes:
        return []
    height, width = frame.shape[:2]

    order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
    keep: List[np.ndarray] = []
    kept_scores: List[float] = []
    iou_threshold = 0.4

    for idx in order:
        candidate = boxes[int(idx)]
        if any(_bbox_iou(candidate, existing) > iou_threshold for existing in keep):
            continue
        keep.append(candidate)
        kept_scores.append(float(scores[int(idx)]))
        if len(keep) >= max_candidates:
            break

    sortable = []
    for box, score in zip(keep, kept_scores):
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) * 0.5
        area_norm = ((x2 - x1) * (y2 - y1)) / float(max(1, width * height))
        bottom_norm = y2 / float(max(1, height))
        rank_score = (0.5 * area_norm) + (0.35 * bottom_norm) + (0.15 * max(0.0, score))
        sortable.append((rank_score, cx, (int(x1), int(y1), int(x2), int(y2)), score))

    sortable.sort(key=lambda item: (-item[0], item[1]))

    candidates: List[PersonCandidate] = []
    for person_id, (_, _, bbox, score) in enumerate(sortable):
        x1, y1, x2, y2 = bbox
        center_xy = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        candidates.append(
            PersonCandidate(
                person_id=person_id,
                bbox_xyxy=bbox,
                confidence=float(score),
                center_xy=center_xy,
            )
        )
    return candidates


def active_person_detector_backend() -> str:
    return _PERSON_DETECTOR_BACKEND


def detect_person_candidates_in_frame(frame: np.ndarray, max_candidates: int = 6) -> List[PersonCandidate]:
    if frame is None or frame.size == 0:
        return []
    return _detect_person_candidates(frame, max_candidates=max_candidates)


def detect_person_candidates_first_frame(video_path: str, max_candidates: int = 6) -> List[PersonCandidate]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return []
    return _detect_person_candidates(frame, max_candidates=max_candidates)


class MediaPipePoseProvider:
    def __init__(self, static_image_mode: bool = False):
        pose_mod = getattr(mp, "solutions", None)
        if pose_mod is None:
            from mediapipe.python import solutions as mp_solutions

            pose_api = mp_solutions.pose
        else:
            pose_api = pose_mod.pose

        self._pose = pose_api.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_infer_stats: dict[str, object] = {
            "frames": 0,
            "drift_frames": 0,
            "fallback_frames": 0,
        }

    def infer_frames(
        self,
        frames: List[np.ndarray],
        fps: float,
        crop_boxes: Optional[List[np.ndarray]] = None,
        expected_boxes: Optional[List[np.ndarray]] = None,
    ) -> PoseSequence:
        keypoints: List[np.ndarray] = []
        visibility: List[np.ndarray] = []
        drift_frames = 0
        fallback_frames = 0
        previous_pts: Optional[np.ndarray] = None
        previous_vis: Optional[np.ndarray] = None
        fallback_streak = 0

        for i, frame in enumerate(frames):
            frame_h, frame_w = frame.shape[:2]
            crop_box = None
            if crop_boxes is not None and i < len(crop_boxes):
                raw = crop_boxes[i].astype(np.int32)
                x1 = max(0, min(frame_w - 1, int(raw[0])))
                y1 = max(0, min(frame_h - 1, int(raw[1])))
                x2 = max(x1 + 1, min(frame_w, int(raw[2])))
                y2 = max(y1 + 1, min(frame_h, int(raw[3])))
                if x2 > x1 and y2 > y1:
                    crop_box = (x1, y1, x2, y2)

            pose_input = frame
            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                pose_input = frame[y1:y2, x1:x2]
                if pose_input.size == 0:
                    pose_input = frame
                    crop_box = None

            rgb = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            is_drifted = False
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if crop_box is not None:
                    x1, y1, x2, y2 = crop_box
                    crop_w = max(1, x2 - x1)
                    crop_h = max(1, y2 - y1)
                    pts = np.array(
                        [
                            [
                                (x1 + (lm.x * crop_w)) / max(1, frame_w),
                                (y1 + (lm.y * crop_h)) / max(1, frame_h),
                                lm.z * (crop_w / max(1.0, float(frame_w))),
                            ]
                            for lm in landmarks
                        ],
                        dtype=np.float32,
                    )
                else:
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                vis = np.array([lm.visibility for lm in landmarks], dtype=np.float32)

                if expected_boxes is not None and i < len(expected_boxes):
                    exp = expected_boxes[i].astype(np.float32)
                    ex1, ey1, ex2, ey2 = exp.tolist()
                    expected_center = np.array(
                        [((ex1 + ex2) * 0.5) / max(1, frame_w), ((ey1 + ey2) * 0.5) / max(1, frame_h)],
                        dtype=np.float32,
                    )

                    mid_hip = 0.5 * (pts[23, :2] + pts[24, :2])
                    mid_shoulder = 0.5 * (pts[11, :2] + pts[12, :2])
                    if np.all(np.isfinite(mid_hip)):
                        pose_center = mid_hip
                    elif np.all(np.isfinite(mid_shoulder)):
                        pose_center = mid_shoulder
                    else:
                        pose_center = np.array([np.nan, np.nan], dtype=np.float32)

                    box_diag = float(np.hypot(max(1.0, ex2 - ex1), max(1.0, ey2 - ey1)))
                    if np.all(np.isfinite(pose_center)) and box_diag > 1.0:
                        dx = (pose_center[0] - expected_center[0]) * frame_w
                        dy = (pose_center[1] - expected_center[1]) * frame_h
                        center_dist_norm = float(np.hypot(dx, dy) / box_diag)
                        is_drifted = center_dist_norm > float(POSE_DRIFT_MAX_CENTER_DIST_NORM)
            else:
                pts = np.zeros((len(LANDMARK_NAMES), 3), dtype=np.float32)
                vis = np.zeros((len(LANDMARK_NAMES),), dtype=np.float32)

            if is_drifted:
                drift_frames += 1

            has_landmarks = bool(np.any(vis > 0.0))
            should_fallback = (not has_landmarks) or is_drifted
            if should_fallback and previous_pts is not None and previous_vis is not None and fallback_streak < max(0, int(POSE_DRIFT_HOLD_FRAMES)):
                pts = previous_pts.copy()
                vis = np.clip(previous_vis * 0.9, 0.0, 1.0)
                fallback_streak += 1
                fallback_frames += 1
            elif has_landmarks and not is_drifted:
                previous_pts = pts.copy()
                previous_vis = vis.copy()
                fallback_streak = 0

            keypoints.append(pts)
            visibility.append(vis)

        self.last_infer_stats = {
            "frames": len(frames),
            "drift_frames": drift_frames,
            "fallback_frames": fallback_frames,
        }

        return PoseSequence(
            fps=fps,
            frames=frames,
            keypoints=np.stack(keypoints, axis=0),
            visibility=np.stack(visibility, axis=0),
        )

    def read_video(self, video_path: str, crop_boxes: Optional[List[np.ndarray]] = None) -> PoseSequence:
        fps, frames = read_video_frames(video_path)
        return self.infer_frames(frames=frames, fps=fps, crop_boxes=crop_boxes)


def mask_low_visibility(keypoints: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    masked = keypoints.copy()
    low = visibility < MIN_VISIBILITY
    masked[low] = np.nan
    return masked
