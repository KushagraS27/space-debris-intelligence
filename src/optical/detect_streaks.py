import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

def detect_streaks(image_array: np.ndarray, min_length: int = 30, max_gap: int = 5):
    """Uses the Kaggle-trained AutoEncoder to detect debris anomalies."""
    model_path = Path("src/models/optical_autoencoder.keras")
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array.copy()

    # Resize to match our Kaggle neural network input (128x128)
    resized = cv2.resize(gray, (128, 128))
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))

    if not model_path.exists():
        return [], np.zeros_like(gray)

    # Load DL model and predict background
    autoencoder = tf.keras.models.load_model(model_path)
    reconstructed = autoencoder.predict(input_tensor, verbose=0)[0, :, :, 0]
    
    # The anomaly (streak) is the difference between original and reconstructed background
    diff = np.abs(normalized - reconstructed)
    anomaly_mask = (diff > 0.1).astype(np.uint8) * 255
    
    # Scale back up to original image size
    anomaly_mask_full = cv2.resize(anomaly_mask, (gray.shape[1], gray.shape[0]))
    
    # Extract line vectors using the dashboard's min_length parameter
    lines = cv2.HoughLinesP(anomaly_mask_full, 1, np.pi/180, 20, minLineLength=min_length, maxLineGap=max_gap)
    
    detections = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            length = float(np.hypot(x2 - x1, y2 - y1))
            detections.append({
                "start": (int(x1), int(y1)), "end": (int(x2), int(y2)),
                "length_px": round(length, 2), "angle_deg": round(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))), 2)
            })

    return detections, anomaly_mask_full
