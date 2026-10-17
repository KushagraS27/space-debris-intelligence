import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def score_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    features = ['miss_distance_km', 'time_to_approach_hours', 'data_age_hours', 'relative_velocity_kms']
    X = candidates_df[features]
    
    # 1. Existing Anomaly Triage
    model_path = Path("src/models/tuned_iforest.joblib")
    if model_path.exists():
        model = joblib.load(model_path)
        candidates_df['anomaly_score'] = model.decision_function(X)
        candidates_df['is_anomaly'] = model.predict(X)
        candidates_df['anomaly_consensus'] = (candidates_df['is_anomaly'] == -1).astype(int) * 3
    else:
        candidates_df['anomaly_score'] = -1.0
        candidates_df['anomaly_consensus'] = 0

    # 2. Active Debris Removal (ADR) ML Scoring
    # ADR targets massive objects (Class 0: Payload, Class 1: Rocket Body)
    # Synthesize training targets based on mass proxy + collision kinematics
    base_mass_risk = np.where(candidates_df['object_classification'].isin([0, 1]), 75.0, 10.0)
    target_adr_score = base_mass_risk + (candidates_df['relative_velocity_kms'] * 1.5) - (candidates_df['miss_distance_km'] / 100.0)
    
    # Train regressor models to predict removal priority
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
    
    rf.fit(X, target_adr_score)
    gb.fit(X, target_adr_score)
    
    # Predict and compare models
    candidates_df['adr_rf_score'] = np.round(rf.predict(X), 2)
    candidates_df['adr_gb_score'] = np.round(gb.predict(X), 2)
    candidates_df['adr_priority_score'] = np.round(((candidates_df['adr_rf_score'] + candidates_df['adr_gb_score']) / 2).clip(0, 100), 2)
    
    return candidates_df.sort_values('adr_priority_score', ascending=False)
