from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np

from .config import MIN_VISIBILITY


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

    def read_video(self, video_path: str) -> PoseSequence:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[np.ndarray] = []
        keypoints: List[np.ndarray] = []
        visibility: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                vis = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
            else:
                pts = np.zeros((len(LANDMARK_NAMES), 3), dtype=np.float32)
                vis = np.zeros((len(LANDMARK_NAMES),), dtype=np.float32)
            keypoints.append(pts)
            visibility.append(vis)

        cap.release()
        return PoseSequence(
            fps=fps,
            frames=frames,
            keypoints=np.stack(keypoints, axis=0),
            visibility=np.stack(visibility, axis=0),
        )


def mask_low_visibility(keypoints: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    masked = keypoints.copy()
    low = visibility < MIN_VISIBILITY
    masked[low] = np.nan
    return masked
