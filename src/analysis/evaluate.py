import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path

def precision_at_k(y_true, y_scores, k):
    idx = np.argsort(y_scores)[::-1]
    top_k_labels = np.array(y_true)[idx][:k]
    return np.sum(top_k_labels) / k

def evaluate_models():
    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists():
        print("Data not found. Run main.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Ground Truth: Class 3 are injected Foreign/Unknown anomalies
    y_true = (df['object_classification'] == 3).astype(int)
    
    if y_true.sum() == 0:
        print("No foreign anomalies found in dataset to evaluate against.")
        return

    # Baseline: Inverse of miss distance (smaller distance = higher priority)
    baseline_scores = 1.0 / (df['miss_distance_km'] + 1e-6)
    
    # ML Ensemble Priority: Multi-model consensus (0 to 3) + kinematic velocity weight
    ml_scores = df['anomaly_consensus'].astype(float) + (df['relative_velocity_kms'] / df['relative_velocity_kms'].max())

    print("\n" + "="*55)
    print("🔬 RESEARCH METRICS: ML ENSEMBLE vs DISTANCE BASELINE")
    print("="*55)

    # 1. PR-AUC Calculation
    base_prec, base_rec, _ = precision_recall_curve(y_true, baseline_scores)
    base_pr_auc = auc(base_rec, base_prec)
    
    ml_prec, ml_rec, _ = precision_recall_curve(y_true, ml_scores)
    ml_pr_auc = auc(ml_rec, ml_prec)

    print(f"\n1. Precision-Recall AUC (Higher is better, Max 1.0):")
    print(f"   - Distance-Only Baseline: {base_pr_auc:.4f}")
    print(f"   - ML Ensemble Pipeline:   {ml_pr_auc:.4f}")

    # 2. Precision@K Calculation
    print("\n2. Precision@K (Simulated Observation Budget):")
    total_samples = len(df)
    k_values = [5, 10, min(25, total_samples)]
    for k in k_values:
        p_base = precision_at_k(y_true, baseline_scores, k)
        p_ml = precision_at_k(y_true, ml_scores, k)
        print(f"   - Top {k} Scheduled Slots:")
        print(f"       * Distance-Only caught: {int(p_base * k)}/{k} ({p_base*100:.1f}%)")
        print(f"       * ML Ensemble caught:   {int(p_ml * k)}/{k} ({p_ml*100:.1f}%)")
    
    print("\n" + "="*55)

if __name__ == "__main__":
    evaluate_models()
