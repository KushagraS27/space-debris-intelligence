import json
from pathlib import Path

from pydantic import ValidationError

from src.data.schema import OrbitalRecord


RAW_SNAPSHOTS_DIR = Path("data/raw/snapshots")
PROCESSED_DIR = Path("data/processed")


def latest_snapshot_dir() -> Path:
    snapshots = sorted(
        path for path in RAW_SNAPSHOTS_DIR.iterdir() if path.is_dir()
    )
    if not snapshots:
        raise FileNotFoundError("No raw snapshots found. Run fetch_gp.py first.")
    return snapshots[-1]


def validate_latest_snapshot() -> dict:
    snapshot_dir = latest_snapshot_dir()
    records = json.loads((snapshot_dir / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))

    valid_count = 0
    invalid_records = []

    for index, raw_record in enumerate(records):
        record = dict(raw_record)
        record["NORAD_CAT_ID"] = str(record.get("NORAD_CAT_ID", ""))
        record["source"] = manifest["source"]
        record["retrieved_at"] = manifest["retrieved_at_utc"]

        try:
            OrbitalRecord(**record)
            valid_count += 1
        except ValidationError as error:
            invalid_records.append(
                {
                    "index": index,
                    "object_id": record["NORAD_CAT_ID"],
                    "errors": error.errors(),
                }
            )

    report = {
        "snapshot_id": snapshot_dir.name,
        "source": manifest["source"],
        "record_count": len(records),
        "valid_count": valid_count,
        "invalid_count": len(invalid_records),
        "invalid_records": invalid_records,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROCESSED_DIR / f"validation_report_{snapshot_dir.name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    validation_report = validate_latest_snapshot()
    print(
        f"Validated {validation_report['valid_count']} of "
        f"{validation_report['record_count']} records. "
        f"Invalid: {validation_report['invalid_count']}"
    )