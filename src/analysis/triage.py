import pandas as pd
import joblib
from pathlib import Path

def score_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    features = ['miss_distance_km', 'time_to_approach_hours', 'data_age_hours', 'relative_velocity_kms', 'object_classification']
    X = candidates_df[features]
    
    model_path = Path("src/models/ensemble_classification_models.joblib")
    if not model_path.exists():
        print("ERROR: Ensemble model not found.")
        return candidates_df
        
    ensemble = joblib.load(model_path)
    
    # Score with all three models (-1 is anomaly, 1 is normal)
    candidates_df['iforest_anomaly'] = ensemble['isolation_forest'].predict(X)
    candidates_df['svm_anomaly'] = ensemble['oc_svm'].predict(X)
    candidates_df['lof_anomaly'] = ensemble['lof'].predict(X)
    
    # Calculate an agreement score (0 to 3 models flagging it)
    candidates_df['anomaly_consensus'] = (candidates_df['iforest_anomaly'] == -1).astype(int) + \
                                         (candidates_df['svm_anomaly'] == -1).astype(int) + \
                                         (candidates_df['lof_anomaly'] == -1).astype(int)
    
    return candidates_df.sort_values('anomaly_consensus', ascending=False)
