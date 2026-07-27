"""
Extract train.csv, valid.csv, and the real validation-set images out of the
downloaded CheXpert "batch 1 (validate & csv)" zip into the flat
data/chexpert/ layout config.py expects (CHEXPERT_IMAGES_ROOT / row["Path"]
must resolve to a real file). Idempotent: already-extracted files are skipped,
so re-running after an interruption just picks up the rest.
"""

import shutil
import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/chexpert/batch1.zip")
DEST_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/chexpert")


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        root = names[0]  # e.g. "CheXpert-v1.0 batch 1 (validate & csv)/"

        extracted, skipped = 0, 0
        for name in names:
            if name.endswith("/"):
                continue
            rel = name[len(root):]
            if rel in ("train.csv", "valid.csv"):
                dest = DEST_ROOT / rel
            elif rel.startswith("valid/"):
                dest = DEST_ROOT / "CheXpert-v1.0" / rel
            else:
                continue

            if dest.exists():
                skipped += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1

    print(f"Extracted {extracted} new file(s), {skipped} already present, into {DEST_ROOT}")


if __name__ == "__main__":
    main()
