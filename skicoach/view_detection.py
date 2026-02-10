import numpy as np


def detect_view(keypoints: np.ndarray) -> str:
    left_shoulder = keypoints[:, 11, 0]
    right_shoulder = keypoints[:, 12, 0]
    diff = np.nanmean(left_shoulder - right_shoulder)
    if diff > 0:
        return "right"
    return "left"
