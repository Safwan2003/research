"""
Download CheXpert dataset batches from Stanford Azure Blob Storage link into data/chexpert/
"""

import zipfile
import urllib.request
import urllib.parse
from pathlib import Path
from urllib.parse import urlsplit
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

BLOBS_TO_DOWNLOAD = [
    "CheXpert-v1.0 batch 1 (validate & csv).zip",
    "CheXpert-v1.0 batch 2 (train 1).zip",
]


def _blob_url(sas_url: str, blob_name: str) -> str:
    """
    Same URL-building shape as scripts/download_chexpert_subset.py's
    _blob_url() -- both scripts must accept the identical full-SAS-URL
    format stored in .env / secrets/chexpert_sas_url.txt.
    """
    parts = urlsplit(sas_url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return f"{base}/{urllib.parse.quote(blob_name)}?{parts.query}"


def download_blob(blob_name: str, target_dir: Path, sas_url: str):
    download_url = _blob_url(sas_url, blob_name)
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
    if not config.CHEXPERT_SAS_URL:
        sys.exit(
            "Missing CHEXPERT_SAS_URL -- set it in .env (copy .env.example to .env and fill it "
            "in) or in secrets/chexpert_sas_url.txt. Get the SAS URL from your professor / "
            "Stanford AIMI's CheXpert access page; never hardcode it into a tracked .py file."
        )

    chexpert_dir = Path(config.CHEXPERT_IMAGES_ROOT)
    chexpert_dir.mkdir(parents=True, exist_ok=True)

    for blob_name in BLOBS_TO_DOWNLOAD:
        download_blob(blob_name, chexpert_dir, config.CHEXPERT_SAS_URL)

    print("\nCheXpert download and extraction complete!")
    print(f"CSV Files found: {list(chexpert_dir.glob('**/*.csv'))}")


if __name__ == "__main__":
    download_chexpert()
