"""Train a convolutional autoencoder for optical debris streak detection.

The autoencoder learns to reconstruct *clean* starfield patches.  At inference
time, debris streaks produce high reconstruction error and are segmented out.

Usage:
    python -m src.optical.train_autoencoder
    python -m src.optical.train_autoencoder --epochs 30 --samples 2000
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# TensorFlow is imported lazily so the module can be inspected without a GPU.


def _generate_starfield_patches(
    n_samples: int = 1500,
    patch_size: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic starfield patches (no streaks) for training."""
    rng = np.random.RandomState(seed)
    patches = []
    for _ in range(n_samples):
        # Dark sky with slight gradient + Gaussian noise
        base_level = rng.randint(5, 25)
        patch = np.full((patch_size, patch_size), base_level, dtype=np.float32)
        noise = rng.normal(0, 8, (patch_size, patch_size)).astype(np.float32)
        patch = patch + noise

        # Scatter 30-120 point-like stars
        n_stars = rng.randint(30, 120)
        for _ in range(n_stars):
            sx, sy = rng.randint(0, patch_size, 2)
            brightness = rng.uniform(80, 255)
            radius = rng.choice([1, 1, 2])
            cv2.circle(patch, (int(sx), int(sy)), radius, float(brightness), -1)

        patches.append(patch)

    data = np.array(patches, dtype=np.float32) / 255.0
    return data[..., np.newaxis]  # (N, H, W, 1)


def build_autoencoder(patch_size: int = 128):
    """Build a small Conv-Deconv autoencoder."""
    import tensorflow as tf

    encoder_input = tf.keras.Input(shape=(patch_size, patch_size, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(
        encoder_input
    )
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = tf.keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(epochs: int = 20, n_samples: int = 1500) -> Path:
    """Generate data, train the model, and save to disk."""
    model_path = Path("src/models/optical_autoencoder.keras")

    print(f"Generating {n_samples} synthetic starfield patches …")
    data = _generate_starfield_patches(n_samples=n_samples)

    # 80/20 train-val split
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]

    print("Building autoencoder …")
    autoencoder = build_autoencoder()
    autoencoder.summary()

    print(f"Training for {epochs} epochs …")
    autoencoder.fit(
        train_data,
        train_data,
        epochs=epochs,
        batch_size=32,
        shuffle=True,
        validation_data=(val_data, val_data),
        verbose=1,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    autoencoder.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1500)
    args = parser.parse_args()
    train_autoencoder(epochs=args.epochs, n_samples=args.samples)
