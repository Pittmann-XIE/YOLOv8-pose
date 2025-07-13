#!/usr/bin/env python3
import os
import requests
import concurrent.futures
import signal
import sys
from pathlib import Path
from urllib.parse import urlparse

# Configuration
BASE_DIR = Path("./Dataset")  # Change this
TMP_DIR = BASE_DIR / "temp_downloads"
MAX_WORKERS = 30

# URLs to download
LABEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip"
IMAGE_URLS = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip", 
    "http://images.cocodataset.org/zips/test2017.zip"
]

# Global flag for interruption
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\nInterrupt received, cleaning up...")
    sys.exit(1)

def download_file(url, destination):
    if interrupted:
        return False

    temp_path = TMP_DIR / (Path(urlparse(url).path).name + ".tmp")
    final_path = destination / Path(urlparse(url).path).name

    # Skip if already downloaded
    if final_path.exists():
        print(f"Already exists: {final_path}")
        return True

    try:
        # Create temp directory if needed
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with progress
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0

            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if interrupted:
                        return False
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Downloading {temp_path.name}: {percent:.1f}%", end='\r')

        # Move to final location
        temp_path.rename(final_path)
        print(f"Downloaded: {final_path}")
        return True

    except Exception as e:
        print(f"\nError downloading {url}: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create directories
    (BASE_DIR / "images").mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    # Download labels
    print("Downloading labels...")
    if not download_file(LABEL_URL, BASE_DIR.parent):
        sys.exit(1)

    # Download images in parallel
    print(f"Downloading images with {MAX_WORKERS} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, url, BASE_DIR / "images"): url 
            for url in IMAGE_URLS
        }
        
        try:
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                try:
                    success = future.result()
                    if not success and not interrupted:
                        print(f"Failed to download {url}")
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    if not interrupted:
                        interrupted = True
                        executor.shutdown(wait=False)
                        break

        except KeyboardInterrupt:
            print("\nInterrupt received, stopping downloads...")
            interrupted = True
            executor.shutdown(wait=False)
            sys.exit(1)

    # Cleanup
    try:
        for temp_file in TMP_DIR.glob("*.tmp"):
            temp_file.unlink()
        TMP_DIR.rmdir()
    except:
        pass

    if not interrupted:
        print("All downloads completed successfully!")

if __name__ == "__main__":
    main()