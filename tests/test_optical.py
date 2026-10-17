import numpy as np
import cv2
from src.optical.detect_streaks import detect_streaks

def test_streak_detection_synthetic():
    # 512x512 synthetic star field
    frame = np.zeros((512, 512), dtype=np.uint8)
    
    # Add random point stars
    np.random.seed(42)
    star_x = np.random.randint(0, 512, 100)
    star_y = np.random.randint(0, 512, 100)
    for x, y in zip(star_x, star_y):
        cv2.circle(frame, (x, y), 1, 200, -1)
        
    # Inject 1 linear debris streak
    cv2.line(frame, (100, 100), (220, 220), 255, 2)

    detections, _ = detect_streaks(frame, min_length=30)
    
    assert len(detections) >= 1
    streak = detections[0]
    assert streak["length_px"] > 50.0
    assert 40.0 <= abs(streak["angle_deg"]) <= 50.0
