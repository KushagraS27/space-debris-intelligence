import pandas as pd
import numpy as np
from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def score_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    df = candidates_df.copy()

    features = ['miss_distance_km', 'time_to_approach_hours',
                'data_age_hours', 'relative_velocity_kms']
    X = df[features].fillna(0.0)

    # 1. Anomaly triage: three unsupervised models fitted on the current candidates
    if len(X) >= 5:
        Xs = StandardScaler().fit_transform(X)

        iso = IsolationForest(contamination=0.1, random_state=42).fit(Xs)
        svm = OneClassSVM(nu=0.1, gamma="scale").fit(Xs)
        lof = LocalOutlierFactor(n_neighbors=min(20, len(Xs) - 1), contamination=0.1)

        df['anomaly_score'] = iso.decision_function(Xs)
        df['iforest_anomaly'] = (iso.predict(Xs) == -1).astype(int)
        df['svm_anomaly'] = (svm.predict(Xs) == -1).astype(int)
        df['lof_anomaly'] = (lof.fit_predict(Xs) == -1).astype(int)
    else:
        df['anomaly_score'] = 0.0
        df['iforest_anomaly'] = 0
        df['svm_anomaly'] = 0
        df['lof_anomaly'] = 0

    # Consensus = how many of the 3 models flag this pair (0 to 3)
    df['anomaly_consensus'] = (df['iforest_anomaly']
                               + df['svm_anomaly']
                               + df['lof_anomaly'])

    # 2. Active Debris Removal (ADR) priority score
    # NOTE: the target below is a hand-written heuristic (mass proxy + kinematics),
    # so this ranking is a heuristic, not a measured ground truth.
    adr_features = features + ['object_classification']
    X_adr = df[adr_features].fillna(0.0)

    base_mass_risk = np.where(df['object_classification'].isin([0, 1]), 75.0, 10.0)
    proximity = 1.0 - (df['miss_distance_km'] / 15000.0).clip(0, 1)   # 1 = very close
    target_adr_score = base_mass_risk + df['relative_velocity_kms'] * 1.5 + 25.0 * proximity

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
    rf.fit(X_adr, target_adr_score)
    gb.fit(X_adr, target_adr_score)

    df['adr_rf_score'] = np.round(rf.predict(X_adr), 2)
    df['adr_gb_score'] = np.round(gb.predict(X_adr), 2)
    df['adr_priority_score'] = np.round(
        ((df['adr_rf_score'] + df['adr_gb_score']) / 2).clip(0, 100), 2)

    return df.sort_values('adr_priority_score', ascending=False)