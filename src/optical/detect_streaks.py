import cv2
import numpy as np

def detect_streaks(image_array: np.ndarray, min_length: int = 40, max_gap: int = 5):
    """
    Detects linear streaks from high-velocity orbital debris or satellites
    against stationary background stars.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array.copy()

    # 1. Morphological filtering to isolate elongated structures from point-like stars
    kernel_star = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background_stars = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_star)
    streak_isolated = cv2.subtract(gray, background_stars)

    # 2. Adaptive binary thresholding
    _, thresh = cv2.threshold(streak_isolated, 50, 255, cv2.THRESH_BINARY)

    # 3. Probabilistic Hough Transform
    lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=min_length, maxLineGap=max_gap)

    detections = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            detections.append({
                "start": (int(x1), int(y1)),
                "end": (int(x2), int(y2)),
                "length_px": round(length, 2),
                "angle_deg": round(angle, 2)
            })

    return detections, thresh
