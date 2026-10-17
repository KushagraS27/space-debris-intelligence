import pandas as pd
from pathlib import Path

def evaluate_budgets():
    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists(): return
    df = pd.read_csv(data_path)
    
    # Priority Queue 1: Baseline Distance-First
    baseline_df = df.sort_values("miss_distance_km", ascending=True)
    
    # Priority Queue 2: ML Uncertainty Consensus
    df["triage_rank_score"] = (df["anomaly_consensus"] * 100.0) + df["relative_velocity_kms"] - (df["miss_distance_km"] / 1000.0)
    ml_df = df.sort_values("triage_rank_score", ascending=False)
    
    budgets = [2.0, 4.0, 8.0] # Hours
    slot_mins = 15
    
    print("\n" + "="*70)
    print("🔭 TELESCOPE OBSERVING-TIME BUDGET SIMULATION (MULTI-SCENARIO)")
    print("="*70)
    
    for b in budgets:
        slots = int((b * 60) // slot_mins)
        b_queue = baseline_df.head(slots)
        m_queue = ml_df.head(slots)
        
        b_crit = len(b_queue[(b_queue["object_classification"] == 3) | (b_queue["anomaly_consensus"] == 3)])
        m_crit = len(m_queue[(m_queue["object_classification"] == 3) | (m_queue["anomaly_consensus"] == 3)])
        
        print(f"\nScenario: {b} Hour Budget ({slots} total observations possible)")
        print(f"  - Distance Baseline Caught: {b_crit}/{slots} critical targets ({(b_crit/slots)*100:.1f}%)")
        print(f"  - ML Ensemble Caught:       {m_crit}/{slots} critical targets ({(m_crit/slots)*100:.1f}%)")
        print(f"  --> ML Improvement: +{m_crit - b_crit} critical targets intercepted.")

if __name__ == "__main__":
    evaluate_budgets()
