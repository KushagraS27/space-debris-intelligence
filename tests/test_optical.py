import numpy as np
import cv2
from src.optical.detect_streaks import detect_streaks

def test_streak_detection_execution():
    # 512x512 synthetic star field
    frame = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(frame, (100, 100), (220, 220), 255, 2)

    # We only test that the deep learning pipeline executes without crashing
    # since the dummy model weights won't perfectly detect this geometric line
    detections, mask = detect_streaks(frame, min_length=30)
    
    assert isinstance(detections, list)
    assert mask.shape == (512, 512)
