"""
Download CheXpert dataset batches from Stanford Azure Blob Storage link into data/chexpert/
"""

import os
import zipfile
import urllib.request
import urllib.parse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

BASE_BLOB_URL = "https://aimistanforddatasets01.blob.core.windows.net/chexpertchestxrays-u20210408"
SAS_TOKEN = "sv=2019-02-02&sr=c&sig=EfKePbHmlzi4voPAdUKcBBYZmgd3G1MgMWrAsDEvjnU%3D&st=2026-07-22T07%3A44%3A14Z&se=2026-08-21T07%3A49%3A14Z&sp=rl"

BLOBS_TO_DOWNLOAD = [
    "CheXpert-v1.0 batch 1 (validate & csv).zip",
    "CheXpert-v1.0 batch 2 (train 1).zip",
]


def download_blob(blob_name: str, target_dir: Path):
    encoded_name = urllib.parse.quote(blob_name)
    download_url = f"{BASE_BLOB_URL}/{encoded_name}?{SAS_TOKEN}"
    zip_path = target_dir / blob_name

    if not zip_path.exists():
        print(f"\n--- Downloading: '{blob_name}' ---")
        print(f"Target path: {zip_path}")
        
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        last_percent = [-1]

        def report_hook(blocknum, blocksize, totalsize):
            read_so_far = blocknum * blocksize
            if totalsize > 0:
                percent = int(read_so_far * 100 / totalsize)
                if percent % 10 == 0 and percent != last_percent[0]:
                    last_percent[0] = percent
                    print(f"[{blob_name}] Downloaded {read_so_far / (1024*1024):.1f} MB / {totalsize / (1024*1024):.1f} MB ({percent}%)")

        urllib.request.urlretrieve(download_url, zip_path, reporthook=report_hook)
        print(f"Successfully downloaded '{blob_name}' ({zip_path.stat().st_size / (1024*1024):.1f} MB)")

    print(f"Extracting '{blob_name}' into {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extracted '{blob_name}' successfully.")


def download_chexpert():
    chexpert_dir = Path(config.CHEXPERT_IMAGES_ROOT)
    chexpert_dir.mkdir(parents=True, exist_ok=True)

    for blob_name in BLOBS_TO_DOWNLOAD:
        download_blob(blob_name, chexpert_dir)

    print("\nCheXpert download and extraction complete!")
    print(f"CSV Files found: {list(chexpert_dir.glob('**/*.csv'))}")


if __name__ == "__main__":
    download_chexpert()
