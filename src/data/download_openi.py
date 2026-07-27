"""
Download and unpack OpenI (Indiana University Chest X-ray) dataset into data/openi/
matching config.py's OPENI_REPORTS_DIR and OPENI_IMAGES_DIR.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"


def download_and_extract():
    reports_dir = Path(config.OPENI_REPORTS_DIR)
    images_dir = Path(config.OPENI_IMAGES_DIR)

    reports_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    data_root = reports_dir.parent.parent  # data/openi -> data/

    # 1. Download & Extract Reports
    reports_tgz = data_root / "NLMCXR_reports.tgz"
    if not any(reports_dir.glob("*.xml")):
        if not reports_tgz.exists():
            print(f"Downloading OpenI reports from {REPORTS_URL}...")
            urllib.request.urlretrieve(REPORTS_URL, reports_tgz)
            print(f"Downloaded {reports_tgz.name} ({reports_tgz.stat().st_size / (1024*1024):.1f} MB)")

        print(f"Extracting reports to {reports_dir}...")
        with tarfile.open(reports_tgz, "r:gz") as tar:
            # Handle tar member paths (some archives extract into ecgen-radiology/ or directly)
            for member in tar.getmembers():
                if member.name.endswith(".xml"):
                    member.name = Path(member.name).name  # flatten filename into reports_dir
                    tar.extract(member, path=reports_dir)
        print(f"Extracted {len(list(reports_dir.glob('*.xml')))} XML report files.")

    # 2. Download & Extract Images
    images_tgz = data_root / "NLMCXR_png.tgz"
    if not any(images_dir.glob("*.png")):
        if not images_tgz.exists():
            print(f"Downloading OpenI images from {IMAGES_URL}...")
            urllib.request.urlretrieve(IMAGES_URL, images_tgz)
            print(f"Downloaded {images_tgz.name} ({images_tgz.stat().st_size / (1024*1024):.1f} MB)")

        print(f"Extracting images to {images_dir}...")
        with tarfile.open(images_tgz, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.lower().endswith(".png"):
                    member.name = Path(member.name).name
                    tar.extract(member, path=images_dir)
        print(f"Extracted {len(list(images_dir.glob('*.png')))} PNG image files.")


if __name__ == "__main__":
    download_and_extract()
