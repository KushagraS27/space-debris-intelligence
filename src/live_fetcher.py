import time
import requests
import json
from datetime import datetime, timezone
from pathlib import Path

def fetch_live_data():
    # Pulls the entire active satellite catalog
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching real-time active catalog from CelesTrak...")
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(f"data/raw/snapshots/{timestamp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_file = out_dir / "records.json"
        out_file.write_text(json.dumps(data))
        print(f"Saved {len(data)} objects to {out_file}")
        
    except Exception as e:
        print(f"Fetch failed: {e}")

if __name__ == "__main__":
    while True:
        fetch_live_data()
        print("Waiting 4 hours for the next CelesTrak update. Leave this terminal open...")
        time.sleep(4 * 3600)
