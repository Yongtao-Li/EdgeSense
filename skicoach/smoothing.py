from typing import Optional

import numpy as np


def smooth_keypoints(keypoints: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return keypoints
    pad = window // 2
    padded = np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(keypoints)
    for i in range(keypoints.shape[0]):
        out[i] = np.nanmean(padded[i : i + window], axis=0)
    return out
