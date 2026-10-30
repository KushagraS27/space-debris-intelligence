import pytest
import numpy as np
import cv2
from src.optical.detect_streaks import detect_streaks


def test_streak_detection_execution():
    """Smoke test: the deep learning pipeline executes without crashing."""
    frame = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(frame, (100, 100), (220, 220), 255, 2)

    detections, mask = detect_streaks(frame, min_length=30)

    assert isinstance(detections, list)
    assert mask.shape == (512, 512)


def test_streak_detection_returns_dict_entries():
    """Each detection should be a dict with start, end, length_px, angle_deg."""
    frame = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(frame, (50, 50), (400, 400), 200, 3)

    detections, _ = detect_streaks(frame, min_length=20)

    for d in detections:
        assert isinstance(d, dict)
        assert "start" in d
        assert "end" in d
        assert "length_px" in d
        assert "angle_deg" in d


def test_blank_frame_returns_no_detections():
    """A completely blank (black) frame should yield zero detections."""
    frame = np.zeros((256, 256), dtype=np.uint8)
    detections, mask = detect_streaks(frame, min_length=30)
    assert detections == []


def test_colour_input_is_handled():
    """detect_streaks should accept 3-channel BGR images."""
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.line(frame, (10, 10), (200, 200), (255, 255, 255), 2)
    detections, mask = detect_streaks(frame, min_length=20)
    assert isinstance(detections, list)
    assert mask.shape == (256, 256)
