import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from src.orbits.screen import screen_pair
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
        if "OBJECT_ID" not in rec:
            rec["OBJECT_ID"] = "UNKNOWN"

    targets = records[:10]
    secondaries = records[10:30]

    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(hours=6)

    print(f"Screening {len(targets)} x {len(secondaries)} pairs over 6 hours...")
    candidates = []
    epoch_failures = 0

    for i, a in enumerate(targets):
        print(f"  Target {i+1}/{len(targets)}: NORAD {a.get('NORAD_CAT_ID')}")
        for b in secondaries:
            try:
                res = screen_pair(a, b, start_time, end_time, coarse_step_seconds=600)
                if res.coarse_min_separation_km < 15000.0:
                    obj_type = b.get('OBJECT_TYPE', 'DEBRIS')
                    name = str(b.get("OBJECT_NAME", "")).upper()
                    if "R/B" in name:
                        cls_val = 1      # rocket body
                    elif " DEB" in name:
                        cls_val = 2      # debris
                    elif "TBA" in name or name == "":
                        cls_val = 3      # unknown / unassigned
                    else:
                        cls_val = 0      # payload (satellite)
                    

                    # Real data age: hours since this object's orbit record (EPOCH)
                    try:
                        epoch = datetime.fromisoformat(b["EPOCH"]).replace(tzinfo=timezone.utc)
                        age_h = (start_time - epoch).total_seconds() / 3600.0
                    except Exception:
                        age_h = 0.0
                        epoch_failures += 1

                    candidates.append({
                        "obj_a": res.object_a_id,
                        "obj_b": res.object_b_id,
                        "miss_distance_km": float(res.coarse_min_separation_km),
                        "time_to_approach_hours": float((res.closest_time_utc - start_time).total_seconds() / 3600.0),
                        "data_age_hours": float(age_h),
                        # SIMULATED values below (not computed from real orbits yet)
                        "relative_velocity_kms": float(np.random.normal(10.0, 2.0)),
                        "object_classification": int(cls_val),
                        "encounter_x": float(np.random.uniform(-7000, 7000)),
                        "encounter_y": float(np.random.uniform(-7000, 7000)),
                        "encounter_z": float(np.random.uniform(-7000, 7000)),
                    })
            except Exception:
                continue

    if epoch_failures:
        print(f"WARNING: could not read EPOCH for {epoch_failures} candidates (age set to 0.0).")

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
