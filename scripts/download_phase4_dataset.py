#!/usr/bin/env python3
"""
TRUSTTRACE: Download and Extract Phase 4 Official Dataset ("Find it again!").

Dataset: "Find it again! — Receipt Dataset for Document Forgery Detection"
Official URL: https://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip
Destination Zip: C:\\Users\\jay\\Downloads\\findit2.zip
Destination Dir: C:\\Users\\jay\\Downloads\\finditagain
"""

import sys
import os
import time
import zipfile
import urllib.request
import ssl
from pathlib import Path

# UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

URL = "https://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip"
EXPECTED_SIZE = 673974679  # 673.97 MB
DOWNLOAD_DIR = Path("C:/Users/jay/Downloads")
ZIP_PATH = DOWNLOAD_DIR / "findit2.zip"
EXTRACT_DIR = DOWNLOAD_DIR / "finditagain"


def download_dataset() -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.is_file() and ZIP_PATH.stat().st_size == EXPECTED_SIZE:
        print(f"[FOUND] Existing complete archive at: {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")
        return ZIP_PATH

    print(f"Downloading official dataset from:\n  {URL}\nDestination:\n  {ZIP_PATH}")
    print(f"Expected Size: {EXPECTED_SIZE / (1024 * 1024):.2f} MB")

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrustTrace/1.0"})

    t0 = time.time()
    last_report_time = t0
    downloaded = 0
    chunk_size = 2 * 1024 * 1024  # 2MB chunks

    # Temporary file for atomic write
    temp_zip = ZIP_PATH.with_suffix(".zip.tmp")

    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(temp_zip, "wb") as f_out:
        content_len = int(resp.headers.get("Content-Length", EXPECTED_SIZE))
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f_out.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_report_time >= 5.0 or downloaded == content_len:
                pct = downloaded / content_len * 100.0
                mb_down = downloaded / (1024 * 1024)
                mb_total = content_len / (1024 * 1024)
                speed = mb_down / max(now - t0, 0.001)
                eta = (mb_total - mb_down) / max(speed, 0.001)
                print(f"  Progress: {pct:5.1f}% [{mb_down:6.1f} / {mb_total:6.1f} MB] Speed: {speed:4.2f} MB/s ETA: {eta:4.1f}s", flush=True)
                last_report_time = now

    temp_zip.replace(ZIP_PATH)
    elapsed = time.time() - t0
    print(f"[DONE] Download completed in {elapsed:.1f}s ({EXPECTED_SIZE / (1024 * 1024 * elapsed):.2f} MB/s avg).", flush=True)
    return ZIP_PATH


def extract_dataset(zip_path: Path) -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting archive {zip_path.name} to {EXTRACT_DIR}...", flush=True)

    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as z:
        total_files = len(z.infolist())
        print(f"Archive contains {total_files} files.", flush=True)
        z.extractall(EXTRACT_DIR)

    elapsed = time.time() - t0
    print(f"[DONE] Extraction completed in {elapsed:.1f}s.", flush=True)
    return EXTRACT_DIR


if __name__ == "__main__":
    archive = download_dataset()
    extracted = extract_dataset(archive)
    print(f"\nDataset ready at: {extracted}")
