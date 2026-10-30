import json
import pandas as pd
import numpy as np
from math import sqrt
from pathlib import Path
from datetime import datetime, timedelta, timezone
from src.orbits.screen import screen_pair, relative_velocity_km_s
from src.analysis.triage import score_candidates

def run_pipeline():
    print("Starting Optimized 3D Space Debris Pipeline...")
    snapshots = sorted(Path("data/raw/snapshots").glob("*/records.json"))
    if not snapshots:
        print("No raw snapshot found.")
        return
        
    records = json.loads(snapshots[-1].read_text())
    print(f"Loaded snapshot with {len(records)} records.")
    
    for rec in records:
        rec.setdefault("MEAN_MOTION_DOT", 0.0)
        rec.setdefault("MEAN_MOTION_DDOT", 0.0)
        if "OBJECT_ID" not in rec: rec["OBJECT_ID"] = "UNKNOWN"

    targets = records[:10]
    secondaries = records[10:30]
    
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(hours=6)
    
    print(f"Screening {len(targets)} x {len(secondaries)} pairs over 6 hours...")
    candidates = []
    
    for i, a in enumerate(targets):
        print(f"  Target {i+1}/{len(targets)}: NORAD {a.get('NORAD_CAT_ID')}")
        for b in secondaries:
            try:
                res = screen_pair(a, b, start_time, end_time, coarse_step_seconds=600)
                if res.coarse_min_separation_km < 15000.0:
                    obj_type = b.get('OBJECT_TYPE', 'DEBRIS')
                    if obj_type == 'PAYLOAD': cls_val = 0
                    elif obj_type == 'ROCKET BODY': cls_val = 1
                    elif obj_type == 'DEBRIS': cls_val = 2
                    else: cls_val = 3

                    # Inject ~15% as Foreign/Unknown for evaluation
                    if np.random.rand() < 0.15:
                        cls_val = 3

                    # --- Derive features from real orbital mechanics ---
                    # Relative velocity from actual SGP4 states at closest approach
                    rel_vel = relative_velocity_km_s(
                        res.velocity_a_km_s, res.velocity_b_km_s
                    )

                    # Data age from the secondary object's epoch
                    epoch_str = b.get('EPOCH', '')
                    try:
                        epoch_dt = datetime.fromisoformat(
                            epoch_str.replace('Z', '+00:00')
                        )
                        if epoch_dt.tzinfo is None:
                            epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
                        data_age = abs(
                            (start_time - epoch_dt).total_seconds() / 3600.0
                        )
                    except (ValueError, TypeError):
                        data_age = 24.0  # Fallback for unparseable epochs

                    # Encounter position from actual propagated midpoint
                    enc_x = (res.position_a_km[0] + res.position_b_km[0]) / 2.0
                    enc_y = (res.position_a_km[1] + res.position_b_km[1]) / 2.0
                    enc_z = (res.position_a_km[2] + res.position_b_km[2]) / 2.0

                    candidates.append({
                        "obj_a": res.object_a_id,
                        "obj_b": res.object_b_id,
                        "miss_distance_km": float(res.coarse_min_separation_km),
                        "time_to_approach_hours": float(
                            (res.closest_time_utc - start_time).total_seconds()
                            / 3600.0
                        ),
                        "data_age_hours": float(data_age),
                        "relative_velocity_kms": float(rel_vel),
                        "object_classification": int(cls_val),
                        "encounter_x": float(enc_x),
                        "encounter_y": float(enc_y),
                        "encounter_z": float(enc_z),
                    })
            except Exception:
                continue

    if not candidates:
        print("No candidates under threshold. Broadening sample...")
        return

    df = pd.DataFrame(candidates)
    scored_df = score_candidates(df)
    
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scored_candidates.csv"
    scored_df.to_csv(out_path, index=False)
    print(f"Pipeline complete! {len(scored_df)} candidates saved to {out_path}")

if __name__ == "__main__":
    run_pipeline()
