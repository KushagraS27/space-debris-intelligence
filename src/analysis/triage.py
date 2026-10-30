import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "miss_distance_km",
    "time_to_approach_hours",
    "data_age_hours",
    "relative_velocity_kms",
]


def score_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Score candidates with anomaly detection and ADR prioritisation.

    Changes from original:
    - Isolation Forest is trained and saved if no serialised model exists.
    - ADR target uses normalised proximity factor so scores never go negative.
    - RF / GB regressors are evaluated on a held-out split; predictions are
      still made for the full dataset so every row gets a score.
    """
    X = candidates_df[FEATURE_COLS].copy()

    # ------------------------------------------------------------------ #
    # 1. Anomaly Detection (Isolation Forest)                             #
    # ------------------------------------------------------------------ #
    model_path = Path("src/models/tuned_iforest.joblib")
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        # Train a new Isolation Forest and persist it for reproducibility
        model = IsolationForest(
            n_estimators=100, contamination=0.15, random_state=42
        )
        model.fit(X)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    candidates_df["anomaly_score"] = model.decision_function(X)
    candidates_df["is_anomaly"] = model.predict(X)
    candidates_df["anomaly_consensus"] = (
        (candidates_df["is_anomaly"] == -1).astype(int) * 3
    )

    # ------------------------------------------------------------------ #
    # 2. Active Debris Removal (ADR) ML Scoring                           #
    # ------------------------------------------------------------------ #
    # Normalised proximity: approaches 1 when miss distance is small,
    # decays toward 0 for large separations — prevents negative scores.
    proximity_factor = 1.0 / (1.0 + candidates_df["miss_distance_km"] / 1000.0)

    vel_range = X["relative_velocity_kms"].max() - X["relative_velocity_kms"].min()
    vel_norm = (
        (X["relative_velocity_kms"] - X["relative_velocity_kms"].min())
        / (vel_range + 1e-9)
    )

    # Base mass-risk proxy from object classification
    base_mass_risk = np.where(
        candidates_df["object_classification"].isin([0, 1]), 75.0, 10.0
    )

    # Composite target: base risk weighted by velocity & proximity
    target_adr_score = (
        base_mass_risk * (0.5 + 0.3 * vel_norm + 0.2 * proximity_factor)
    ).clip(0, 100)

    # Train / test split for honest evaluation
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, target_adr_score, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X, target_adr_score

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Predict on the *full* dataset so every row gets a score
    candidates_df["adr_rf_score"] = np.round(rf.predict(X), 2)
    candidates_df["adr_gb_score"] = np.round(gb.predict(X), 2)
    candidates_df["adr_priority_score"] = np.round(
        ((candidates_df["adr_rf_score"] + candidates_df["adr_gb_score"]) / 2).clip(
            0, 100
        ),
        2,
    )

    return candidates_df.sort_values("adr_priority_score", ascending=False)
