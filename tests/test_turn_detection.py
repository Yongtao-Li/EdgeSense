import numpy as np

from skicoach.turn_detection import detect_turns


def test_detect_turns_synthetic():
    frames = 120
    keypoints = np.zeros((frames, 33, 3), dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, frames)
    com_x = 0.5 + 0.1 * np.sin(t)
    keypoints[:, 23, 0] = com_x
    keypoints[:, 24, 0] = com_x
    keypoints[:, 11, 0] = com_x
    keypoints[:, 12, 0] = com_x

    turns = detect_turns(keypoints)
    assert len(turns) >= 2
    for turn in turns:
        assert turn.start < turn.apex < turn.end
