import pandas as pd
import numpy as np
from pathlib import Path

def run_schedule_simulation(budget_hours: float = 4.0, slot_minutes: int = 15):
    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists():
        print("Data not found. Run main.py first.")
        return

    df = pd.read_csv(data_path)
    total_slots = int((budget_hours * 60) // slot_minutes)
    
    # Priority Queue 1: Baseline Distance-First
    baseline_queue = df.sort_values("miss_distance_km", ascending=True).head(total_slots)
    
    # Priority Queue 2: ML Uncertainty & Risk Consensus
    df["triage_rank_score"] = (
        df["anomaly_consensus"] * 100.0 
        + df["relative_velocity_kms"] 
        - (df["miss_distance_km"] / 1000.0)
    )
    ml_queue = df.sort_values("triage_rank_score", ascending=False).head(total_slots)
    
    # Evaluation metric: Count of high-risk items (Class 3 foreign/unknown or consensus == 3)
    def evaluate_queue(queue_df):
        critical_caught = len(queue_df[(queue_df["object_classification"] == 3) | (queue_df["anomaly_consensus"] == 3)])
        debris_caught = len(queue_df[queue_df["object_classification"] == 2])
        payloads_screened = len(queue_df[queue_df["object_classification"] == 0])
        return critical_caught, debris_caught, payloads_screened

    b_crit, b_deb, b_pay = evaluate_queue(baseline_queue)
    m_crit, m_deb, m_pay = evaluate_queue(ml_queue)

    print("\n" + "="*60)
    print("🔭 TELESCOPE OBSERVING-TIME BUDGET SIMULATION")
    print(f"Total Observing Budget: {budget_hours} hours | Available Slots: {total_slots} ({slot_minutes} min/pass)")
    print("="*60)
    print(f"\n1. Baseline Strategy (Nearest Distance First):")
    print(f"   - Critical Anomalies / Unknowns Intercepted: {b_crit} / {total_slots} ({(b_crit/total_slots)*100:.1f}%)")
    print(f"   - Standard Catalog Debris Observed:         {b_deb}")
    print(f"   - Active Payloads Tracked:                  {b_pay}")

    print(f"\n2. Uncertainty-Aware ML Strategy (Consensus Triage):")
    print(f"   - Critical Anomalies / Unknowns Intercepted: {m_crit} / {total_slots} ({(m_crit/total_slots)*100:.1f}%)")
    print(f"   - Standard Catalog Debris Observed:         {m_deb}")
    print(f"   - Active Payloads Tracked:                  {m_pay}")

    gain = m_crit - b_crit
    print(f"\nObservation Yield Delta: {gain:+d} high-priority targets under equal operational budget.")
    print("="*60)

if __name__ == "__main__":
    run_schedule_simulation()
