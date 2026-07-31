"""
Smart partial downloader for CheXpert's training batches.

The AIMI Azure container that ships CheXpert stores the ~223k training images
as three giant zip blobs (~162GB / ~185GB / ~91GB -- see README.md's note on
how run_master.sh's dataset step works). Downloading a whole batch just to
get the first N studies needed for config.N_STUDIES_ABLATION/N_STUDIES_AGENTIC
is wasteful and, on machines without ~450GB free, impossible.

Instead this uses `remotezip` to read the zip's central directory over HTTP
range requests and fetch ONLY the specific image files a given row range of
train.csv needs -- typically a few hundred MB instead of ~162GB.

NOTE: Azure Blob Storage doesn't support the HTTP suffix-range shorthand
("Range: bytes=-N") that remotezip uses by default to locate the end-of-
central-directory record, so `support_suffix_range=False` is required here --
without it every RemoteZip() call raises RangeNotSupported.

Usage:
    python3 scripts/download_chexpert_subset.py --n-studies 1000
    python3 scripts/download_chexpert_subset.py --n-studies 2000 --batch "CheXpert-v1.0 batch 3 (train 2).zip"

Safe to re-run / interrupt: already-downloaded files are skipped, so this is
the "download batch-by-batch while training runs in another terminal" loop --
just re-run with a bigger --n-studies whenever you want more data.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

PROJECT_ROOT = Path(__file__).parent.parent
SAS_PATH = PROJECT_ROOT / "secrets" / "chexpert_sas_url.txt"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "chexpert"

CSV_TRAIN_PREFIX = "CheXpert-v1.0/train/"


def _blob_url(sas_url: str, blob_name: str) -> str:
    parts = urlsplit(sas_url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return f"{base}/{quote(blob_name)}?{parts.query}"


def _progress_bar(done: int, total: int, start_time: float, current: str, width: int = 30) -> str:
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    return (
        f"\r[{bar}] {done}/{total} ({frac*100:5.1f}%)  "
        f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  {current[:40]:<40}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-studies", type=int, required=True,
                         help="Ensure images for the first N rows of train.csv are present locally.")
    parser.add_argument("--batch", default="CheXpert-v1.0 batch 2 (train 1).zip",
                         help="Which training batch blob to pull from (default covers the lowest patient IDs).")
    parser.add_argument(
        "--data-root", default=None,
        help="Where train.csv lives and images get written, e.g. to target a different drive "
             "for a large download without touching the default data/chexpert/ (default: "
             f"{DEFAULT_DATA_ROOT}). Must already contain train.csv (copy it there first -- "
             "it doesn't need the images, just the CSV of paths/labels).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    train_csv = data_root / "train.csv"
    images_root = data_root

    if not SAS_PATH.exists():
        sys.exit(f"Missing {SAS_PATH} -- create it first (see antigravity-prompt.txt).")
    if not train_csv.exists():
        sys.exit(f"Missing {train_csv} -- run the batch-1 (validate & csv) download first (or copy "
                  f"train.csv from {DEFAULT_DATA_ROOT} if --data-root points somewhere new).")

    sas_url = SAS_PATH.read_text().strip()

    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(args.n_studies), reader)]

    # batch zip's top-level folder name mirrors the blob's own filename minus ".zip"
    batch_root = args.batch[:-len(".zip")] + "/" if args.batch.endswith(".zip") else args.batch + "/"

    missing = []
    for row in rows:
        rel = row["Path"]
        if not rel.startswith(CSV_TRAIN_PREFIX):
            continue  # not a train-split row, skip
        local_path = images_root / rel
        if not local_path.exists():
            missing.append(rel)

    print(f"Requested {len(rows)} studies from train.csv; {len(rows) - len(missing)} already present locally, "
          f"{len(missing)} to fetch from '{args.batch}'.")
    if not missing:
        print("Nothing to download -- all requested studies are already local.")
        return

    from remotezip import RemoteZip

    url = _blob_url(sas_url, args.batch)
    print(f"Opening remote zip index for '{args.batch}' (one-time cost, reads the central directory)...")
    t_index_start = time.time()
    with RemoteZip(url, support_suffix_range=False) as z:
        print(f"Index ready in {time.time() - t_index_start:.1f}s ({len(z.namelist())} entries in this batch).")

        start_time = time.time()
        fetched, skipped_not_found, retried = 0, 0, 0
        max_retries = 5
        for i, rel in enumerate(missing, 1):
            entry_name = batch_root + rel[len(CSV_TRAIN_PREFIX):]
            local_path = images_root / rel
            sys.stdout.write(_progress_bar(i, len(missing), start_time, rel.split("/")[-2] + "/" + rel.split("/")[-1]))
            sys.stdout.flush()

            for attempt in range(1, max_retries + 1):
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    data = z.read(entry_name)
                    local_path.write_bytes(data)
                    fetched += 1
                    break
                except KeyError:
                    skipped_not_found += 1
                    break
                except Exception as e:
                    # Azure resets long-lived connections occasionally over a
                    # run this long (hundreds of individual HTTPS requests) --
                    # transient, not a bug. Back off and retry a few times
                    # before giving up on this one file and moving on.
                    retried += 1
                    if attempt == max_retries:
                        print(f"\n  WARNING: giving up on {rel} after {max_retries} attempts ({e}); continuing.")
                        break
                    time.sleep(min(2 ** attempt, 30))

        print()  # newline after the progress bar

    print(f"Done: fetched {fetched} images, {skipped_not_found} not found in this batch, "
          f"{retried} transient retries "
          f"(try a different --batch if 'not found' is unexpectedly high -- patient ranges differ per batch).")


if __name__ == "__main__":
    main()
