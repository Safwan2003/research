"""
Download and extract OpenI (Indiana University Chest X-ray Collection) dataset.
"""

import os
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "openi"
REPORTS_DIR = DATA_DIR / "reports"
IMAGES_DIR = DATA_DIR / "images"

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"{dest_path.name} already exists, skipping download.")
        return
    print(f"Downloading {url} to {dest_path}...")
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / (total_size + 1e-8))
        print(f"\rProgress: {percent}%", end="")
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print("\nDownload complete.")


def extract_tgz(tgz_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tgz_path.name} to {extract_to}...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extraction complete for {tgz_path.name}.")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reports_tgz = DATA_DIR / "NLMCXR_reports.tgz"
    images_tgz = DATA_DIR / "NLMCXR_png.tgz"

    download_file(REPORTS_URL, reports_tgz)
    extract_tgz(reports_tgz, REPORTS_DIR)

    download_file(IMAGES_URL, images_tgz)
    extract_tgz(images_tgz, IMAGES_DIR)

    print("OpenI dataset ready.")
