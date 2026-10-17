import cv2
import numpy as np
from pathlib import Path

def create_realistic_frame():
    out_dir = Path("data/optical_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulated_real_streak.jpg"
    
    # Base dark sky with light pollution gradient
    frame = np.linspace(10, 30, 512).reshape(1, -1).repeat(512, axis=0).astype(np.uint8)
    
    # Add real sensor noise (Gaussian)
    noise = np.random.normal(0, 15, (512, 512))
    frame = cv2.add(frame, noise.astype(np.uint8))
    
    # Add varying sized background stars
    np.random.seed(42)
    for _ in range(150):
        x, y = np.random.randint(0, 512, 2)
        brightness = np.random.randint(100, 255)
        radius = np.random.choice([1, 1, 2])
        cv2.circle(frame, (x, y), radius, int(brightness), -1)
        
    # Add a satellite streak (with blur to simulate atmospheric distortion)
    streak = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(streak, (50, 450), (450, 80), 200, 2)
    streak = cv2.GaussianBlur(streak, (5, 5), 0)
    
    # Combine everything
    final_frame = cv2.addWeighted(frame, 1.0, streak, 0.8, 0)
    
    cv2.imwrite(str(out_path), final_frame)
    print(f"✅ Generated realistic noisy telescope image at: {out_path}")

if __name__ == "__main__":
    create_realistic_frame()
