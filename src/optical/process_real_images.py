import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

def process_real_images(image_dir: str):
    print("🔭 Starting Real Optical Image Analysis...")
    model_path = Path("src/models/optical_autoencoder.keras")
    if not model_path.exists(): return

    autoencoder = tf.keras.models.load_model(model_path)
    image_files = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))

    for img_path in image_files:
        frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if frame is None: continue
            
        original_shape = frame.shape
        resized = cv2.resize(frame, (128, 128))
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized, axis=(0, -1))

        reconstructed = autoencoder.predict(input_tensor, verbose=0)[0, :, :, 0]
        diff = np.abs(normalized - reconstructed)
        
        # Binary thresholding
        anomaly_mask = (diff > 0.25).astype(np.uint8) * 255
        anomaly_mask_full = cv2.resize(anomaly_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Morphological opening to erase residual tiny star noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(anomaly_mask_full, cv2.MORPH_OPEN, kernel)
        
        # Find unique connected pixel blobs (contours) instead of overlapping lines
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        true_streaks = 0
        for cnt in contours:
            # Calculate the bounding box diagonal to approximate streak length
            x, y, w, h = cv2.boundingRect(cnt)
            length = np.hypot(w, h)
            
            # If the blob is exceptionally long (e.g., > 100 pixels), count it as 1 streak
            if length > 100:
                true_streaks += 1
                
        print(f" - Analyzed {img_path.name}: Found {true_streaks} true debris streak(s).")

if __name__ == "__main__":
    process_real_images("data/optical_frames")
