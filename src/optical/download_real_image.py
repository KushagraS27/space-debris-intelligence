import urllib.request
from pathlib import Path
import os

def download_real_telescope_image():
    out_dir = Path("data/optical_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_starlink_streak.jpg"
    
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Starlink_Trails_over_the_Uruguayan_sky.jpg/800px-Starlink_Trails_over_the_Uruguayan_sky.jpg"
    
    print("🔭 Downloading real telescope image of satellite streaks...")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"✅ Real image saved to {out_path}")
    except Exception as e:
        print(f"Failed to download image: {e}")

if __name__ == "__main__":
    download_real_telescope_image()
