import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from src.orbits.propagate import propagate_omm
from src.orbits.screen import separation_km

def analyze_uncertainty():
    snapshots = sorted(Path("data/raw/snapshots").glob("*/records.json"))
    older_records = {r["NORAD_CAT_ID"]: r for r in json.loads(snapshots[0].read_text())}
    newer_records = {r["NORAD_CAT_ID"]: r for r in json.loads(snapshots[-1].read_text())}

    errors = []
    for norad_id, newer_record in newer_records.items():
        if norad_id in older_records:
            older_record = older_records[norad_id]
            # SGP4 strict fields fallback for raw data
            for rec in (older_record, newer_record):
                rec.setdefault("MEAN_MOTION_DOT", 0.0)
                rec.setdefault("MEAN_MOTION_DDOT", 0.0)
                if "OBJECT_ID" not in rec: rec["OBJECT_ID"] = "UNKNOWN"

            target_time = datetime.fromisoformat(newer_record["EPOCH"].replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
            older_epoch = datetime.fromisoformat(older_record["EPOCH"].replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
            
            try:
                state_ref = propagate_omm(newer_record, target_time)
                state_pred = propagate_omm(older_record, target_time)
                err_km = separation_km(state_ref.position_km, state_pred.position_km)
                horizon_hours = (target_time - older_epoch).total_seconds() / 3600.0
                
                if horizon_hours > 0:
                    errors.append({"norad_id": norad_id, "horizon_hours": horizon_hours, "error_km": err_km})
            except Exception:
                continue

    df = pd.DataFrame(errors)
    if not df.empty:
        print("\n--- Empirical Uncertainty Summary ---")
        print(f"Objects analyzed: {len(df)}")
        print(f"Average Horizon (hours): {df['horizon_hours'].mean():.2f}")
        print(f"Median Error (km): {df['error_km'].median():.4f}")
        print(f"95th Percentile Error (km): {df['error_km'].quantile(0.95):.4f}")
    else:
        print("No matching objects found or horizons were zero.")

if __name__ == "__main__":
    analyze_uncertainty()
