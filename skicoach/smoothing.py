from typing import Optional

import numpy as np


def smooth_keypoints(keypoints: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return keypoints
    pad = window // 2
    padded = np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(keypoints)
    last_valid = np.full(keypoints.shape[1:], np.nan, dtype=keypoints.dtype)
    for i in range(keypoints.shape[0]):
        window_slice = padded[i : i + window]
        valid_counts = np.sum(np.isfinite(window_slice), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            summed = np.nansum(window_slice, axis=0)
            mean = summed / np.maximum(valid_counts, 1)
        mean[valid_counts == 0] = np.nan

        if np.isfinite(mean).any():
            merged = mean.copy()
            missing = ~np.isfinite(merged)
            merged[missing] = last_valid[missing]
            out[i] = merged
            last_valid = out[i].copy()
        else:
            out[i] = last_valid
    return out
