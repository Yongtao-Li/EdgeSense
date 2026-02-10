import numpy as np

from skicoach.metrics import compute_turn_metrics
from skicoach.turn_detection import TurnSegment


def test_metrics_outside_ski_offset():
    frames = 60
    keypoints = np.zeros((frames, 33, 3), dtype=np.float32)
    keypoints[:, 23, :2] = [0.5, 0.6]
    keypoints[:, 24, :2] = [0.6, 0.6]
    keypoints[:, 27, :2] = [0.4, 0.9]
    keypoints[:, 28, :2] = [0.8, 0.9]
    keypoints[:, 11, :2] = [0.5, 0.4]
    keypoints[:, 12, :2] = [0.6, 0.4]

    turn = TurnSegment(start=0, apex=30, end=59, direction=1, phases=[])
    metrics = compute_turn_metrics(keypoints, [turn])
    assert metrics[0].outside_ski_offset > 0.1


def test_metrics_angulation():
    frames = 30
    keypoints = np.zeros((frames, 33, 3), dtype=np.float32)
    keypoints[:, 23, :2] = [0.4, 0.6]
    keypoints[:, 24, :2] = [0.6, 0.6]
    keypoints[:, 11, :2] = [0.45, 0.4]
    keypoints[:, 12, :2] = [0.55, 0.5]
    keypoints[:, 27, :2] = [0.4, 0.9]
    keypoints[:, 28, :2] = [0.6, 0.9]

    turn = TurnSegment(start=0, apex=15, end=29, direction=1, phases=[])
    metrics = compute_turn_metrics(keypoints, [turn])
    assert metrics[0].angulation_deg >= 0.0


def test_metrics_timing_ratio_in_range():
    frames = 40
    keypoints = np.zeros((frames, 33, 3), dtype=np.float32)
    t = np.linspace(-0.1, 0.2, frames)
    keypoints[:, 23, 0] = 0.5 + t
    keypoints[:, 24, 0] = 0.52 + t
    keypoints[:, 11, 0] = 0.5 + t
    keypoints[:, 12, 0] = 0.52 + t
    keypoints[:, 27, :2] = [0.4, 0.9]
    keypoints[:, 28, :2] = [0.65, 0.9]
    keypoints[:, 23, 1] = 0.6
    keypoints[:, 24, 1] = 0.6
    keypoints[:, 11, 1] = 0.4
    keypoints[:, 12, 1] = 0.4

    turn = TurnSegment(start=0, apex=20, end=39, direction=1, phases=[])
    metric = compute_turn_metrics(keypoints, [turn])[0]
    assert 0.0 <= metric.com_move_ratio <= 1.0
