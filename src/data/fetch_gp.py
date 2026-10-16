import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
RAW_DATA_DIR = Path("data/raw")


def fetch_gp_snapshot(group: str) -> Path:
    """Download one versioned OMM JSON snapshot from CelesTrak."""
    retrieved_at = datetime.now(timezone.utc)
    response = requests.get(
        BASE_URL,
        params={"GROUP": group.upper(), "FORMAT": "JSON"},
        headers={"User-Agent": "space-debris-intelligence-student-project/0.1"},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"CelesTrak request failed with HTTP {response.status_code}. "
            "Do not retry repeatedly; inspect the response first."
        )

    try:
        records = response.json()
    except requests.JSONDecodeError as error:
        raise RuntimeError("CelesTrak did not return valid JSON.") from error

    if not isinstance(records, list) or not records:
        raise RuntimeError("CelesTrak returned no orbital records.")

    snapshot_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = RAW_DATA_DIR / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    raw_path = snapshot_dir / "records.json"
    raw_bytes = response.content
    raw_path.write_bytes(raw_bytes)

    manifest = {
        "source": "CelesTrak",
        "url": response.url,
        "retrieved_at_utc": retrieved_at.isoformat(),
        "group": group.upper(),
        "format": "OMM_JSON",
        "record_count": len(records),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return snapshot_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        default="stations",
        help="CelesTrak GP group to download, for example stations.",
    )
    args = parser.parse_args()

    saved_to = fetch_gp_snapshot(args.group)
    print(f"Snapshot saved to: {saved_to}")