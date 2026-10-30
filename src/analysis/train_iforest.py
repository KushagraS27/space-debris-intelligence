"""Train (or retrain) the Isolation Forest anomaly detector.

Usage:
    python -m src.analysis.train_iforest          # uses scored_candidates.csv
    python -m src.analysis.train_iforest --data path/to/data.csv
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score


FEATURE_COLS = [
    "miss_distance_km",
    "time_to_approach_hours",
    "data_age_hours",
    "relative_velocity_kms",
]

DEFAULT_DATA = Path("data/processed/scored_candidates.csv")
MODEL_PATH = Path("src/models/tuned_iforest.joblib")


def train_iforest(data_path: Path = DEFAULT_DATA) -> Path:
    """Train an Isolation Forest and save the model to disk."""
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]

    print(f"Training Isolation Forest on {len(X)} samples …")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # Quick sanity check: how many anomalies does it flag?
    preds = model.predict(X)
    n_anomalies = (preds == -1).sum()
    print(f"  Flagged {n_anomalies}/{len(X)} as anomalies ({n_anomalies/len(X)*100:.1f}%)")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    train_iforest(args.data)
