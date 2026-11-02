import cv2
import numpy as np
from pathlib import Path

def generate_sample_dataset():
    out_dir = Path("data/optical_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 1. Real-style Starlink Mega-Constellation Multi-Streak Frame
    h, w = 512, 512
    frame1 = np.full((h, w), 18, dtype=np.uint8)
    # Add starfield
    for _ in range(160):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(frame1, (x, y), int(np.random.choice([1, 1, 2])), int(np.random.randint(90, 255)), -1)
    # Add multiple parallel Starlink debris/satellite trails
    for offset in [-45, 0, 40, 85]:
        streak = np.zeros((h, w), dtype=np.uint8)
        cv2.line(streak, (30, 480 + offset), (480, 50 + offset), 220, 2)
        streak = cv2.GaussianBlur(streak, (3, 3), 0)
        frame1 = cv2.add(frame1, streak)
    cv2.imwrite(str(out_dir / "real_starlink_multiple_streaks.jpg"), frame1)

    # 2. Tumbling Debris Fragment (Faint broken streak)
    frame2 = np.full((h, w), 12, dtype=np.uint8)
    noise = np.random.normal(0, 8, (h, w)).astype(np.uint8)
    frame2 = cv2.add(frame2, noise)
    for _ in range(120):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(frame2, (x, y), 1, int(np.random.randint(80, 220)), -1)
    # Intermittent/tumbling flash streak
    cv2.line(frame2, (100, 150), (250, 220), 190, 2)
    cv2.line(frame2, (290, 240), (430, 310), 230, 2)
    cv2.imwrite(str(out_dir / "tumbling_debris_fragment.jpg"), frame2)

    # 3. High-Velocity Crossing Debris (Two intersecting streaks)
    frame3 = np.full((h, w), 22, dtype=np.uint8)
    for _ in range(180):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(frame3, (x, y), int(np.random.choice([1, 2])), int(np.random.randint(100, 255)), -1)
    cv2.line(frame3, (40, 60), (470, 440), 240, 3)
    cv2.line(frame3, (460, 50), (80, 460), 210, 2)
    cv2.imwrite(str(out_dir / "crossing_debris_orbits.jpg"), frame3)

    # 4. Clean Deep-Space Sky (Baseline - Zero Streaks)
    frame4 = np.full((h, w), 10, dtype=np.uint8)
    noise2 = np.random.normal(0, 5, (h, w)).astype(np.uint8)
    frame4 = cv2.add(frame4, noise2)
    for _ in range(220):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(frame4, (x, y), 1, int(np.random.randint(80, 255)), -1)
    cv2.imwrite(str(out_dir / "clean_background_zero_streaks.jpg"), frame4)

    print("Generated multiple optical frames successfully.")

if __name__ == "__main__":
    generate_sample_dataset()
