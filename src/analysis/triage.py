import pandas as pd
from sklearn.ensemble import IsolationForest

def score_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    features = ['miss_distance_km', 'time_to_approach_hours', 'data_age_hours']
    
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    X = candidates_df[features]
    
    # Train the model first
    model.fit(X)
    
    candidates_df['anomaly_score'] = model.decision_function(X)
    candidates_df['is_anomaly'] = model.predict(X)
    
    return candidates_df.sort_values('anomaly_score')

if __name__ == "__main__":
    dummy_data = pd.DataFrame([
        {"obj_a": "123", "obj_b": "456", "miss_distance_km": 5.0, "time_to_approach_hours": 12, "data_age_hours": 2},
        {"obj_a": "123", "obj_b": "789", "miss_distance_km": 0.5, "time_to_approach_hours": 24, "data_age_hours": 48}, 
        {"obj_a": "456", "obj_b": "789", "miss_distance_km": 15.0, "time_to_approach_hours": 4, "data_age_hours": 1},
    ])
    
    scored = score_candidates(dummy_data)
    print("\n--- ML Triage Results ---")
    print(scored[['obj_a', 'obj_b', 'miss_distance_km', 'anomaly_score', 'is_anomaly']])
