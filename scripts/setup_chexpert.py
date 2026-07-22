"""
Helper script to extract and set up CheXpert dataset inside data/chexpert/
matching config.py paths.
"""

import sys
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "chexpert"


def setup_chexpert_from_zip(zip_path: str | Path):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"Error: {zip_path} does not exist.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} into {DATA_DIR}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("CheXpert extraction complete.")
    # Check if files were extracted inside a nested subfolder like CheXpert-v1.0-small
    nested = list(DATA_DIR.glob("CheXpert*"))
    if nested and nested[0].is_dir():
        print(f"Found nested directory {nested[0].name}, linking/organizing files...")
        for item in nested[0].iterdir():
            target = DATA_DIR / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
    print(f"CheXpert setup verified at {DATA_DIR}.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        setup_chexpert_from_zip(sys.argv[1])
    else:
        print(
            "Usage: python scripts/setup_chexpert.py <path_to_CheXpert-v1.0-small.zip>\n"
            "CheXpert requires registration under Stanford Research Use Agreement:\n"
            "    https://stanfordmlgroup.github.io/competitions/chexpert/\n"
            "Pass the downloaded zip file path to this script to extract and format automatically."
        )
