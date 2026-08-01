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
    python3 scripts/download_chexpert_subset.py --n-studies 1000 --sample-strategy random
    python3 scripts/download_chexpert_subset.py --n-studies 300 --sample-strategy random --offset 300  # held-out slice

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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
import config

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "chexpert"

CSV_TRAIN_PREFIX = "CheXpert-v1.0/train/"

# CheXpert's ~223k train images are split across these two blobs by
# patient-ID range (we don't know the exact split point) -- tried in order,
# per-file, with a fallback to the next one on a KeyError (not found in that
# blob, not a transient error). --sample-strategy random can select patients
# from anywhere in train.csv, so both need to be tried; --batch (singular)
# still pins exactly one blob with no fallback, unchanged, for anyone who
# wants that.
DEFAULT_TRAIN_BATCHES = [
    "CheXpert-v1.0 batch 2 (train 1).zip",
    "CheXpert-v1.0 batch 3 (train 2).zip",
]


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
                         help="With --sample-strategy sequential (default): ensure images for the "
                              "first N rows of train.csv are present locally. With random: target "
                              "row count for a patient-level stratified-random sample (see "
                              "src/data/dataset.py's load_chexpert_dataset docstring -- this counts "
                              "rows/images, not patients, satisfied by including whole patients).")
    parser.add_argument("--batch", default=None,
                         help="Pin exactly one training batch blob to pull from, no fallback "
                              "(default: try both known batches per file, see DEFAULT_TRAIN_BATCHES).")
    parser.add_argument(
        "--data-root", default=None,
        help="Where train.csv lives and images get written, e.g. to target a different drive "
             "for a large download without touching the default data/chexpert/ (default: "
             f"{DEFAULT_DATA_ROOT}). Must already contain train.csv (copy it there first -- "
             "it doesn't need the images, just the CSV of paths/labels).",
    )
    parser.add_argument("--sample-strategy", choices=["sequential", "random"], default="sequential",
                         help="'sequential' (default): first N rows of train.csv -- narrow, "
                              "non-representative, single-patient-ID-range. 'random': patient-level "
                              "stratified-random sample matching "
                              "load_chexpert_dataset(sample_strategy='random') exactly (reuses the "
                              "same sample_chexpert_patients() call) -- fixes that bias.")
    parser.add_argument("--offset", type=int, default=0,
                         help="Row-count offset into the canonical random order (only used with "
                              "--sample-strategy random) -- pass a previous call's --n-studies here "
                              "to fetch the disjoint next slice (e.g. a held-out set after a "
                              "training set). See load_chexpert_dataset's docstring.")
    parser.add_argument("--seed", type=int, default=None, help="Default: config.RANDOM_SEED.")
    parser.add_argument("--target-finding", default=None, help="Default: config.CHEXPERT_TARGET_FINDING.")
    parser.add_argument("--no-stratify", action="store_true",
                         help="With --sample-strategy random, plain-shuffle patients instead of "
                              "stratifying by --target-finding.")
    parser.add_argument("--uncertain-policy", choices=["u_zeros", "u_ones"], default="u_zeros",
                         help="Must match whatever load_chexpert_dataset call this is feeding, or "
                              "the canonical patient order can diverge from what that call expects.")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    train_csv = data_root / "train.csv"
    images_root = data_root

    if not config.CHEXPERT_SAS_URL:
        sys.exit("Missing CHEXPERT_SAS_URL -- set it in .env (copy .env.example to .env) or in "
                  "secrets/chexpert_sas_url.txt (see antigravity-prompt.txt).")
    if not train_csv.exists():
        sys.exit(f"Missing {train_csv} -- run the batch-1 (validate & csv) download first (or copy "
                  f"train.csv from {DEFAULT_DATA_ROOT} if --data-root points somewhere new).")

    sas_url = config.CHEXPERT_SAS_URL
    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    target_finding = args.target_finding if args.target_finding is not None else config.CHEXPERT_TARGET_FINDING

    if args.sample_strategy == "sequential":
        with open(train_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for _, row in zip(range(args.n_studies), reader)]
    else:
        import pandas as pd
        from dataset import sample_chexpert_patients, _patient_id_column, _select_patients_by_row_budget

        df = pd.read_csv(train_csv)
        # Reuse dataset.py's vectorized extractor (drops/warns on unparseable rows
        # instead of raising) rather than a second, independently-written one --
        # a hand-rolled version here previously called extract_patient_id() (which
        # raises ValueError on no match) on any row containing the substring
        # "patient", so a single malformed Path would crash the whole download
        # instead of just being skipped.
        patient_col = _patient_id_column(df)
        df = df[patient_col.notna()].copy()
        df["_patient_id"] = patient_col[patient_col.notna()]
        row_counts = df["_patient_id"].value_counts().to_dict()

        order = sample_chexpert_patients(
            csv_path=str(train_csv), seed=seed,
            stratify_by=None if args.no_stratify else target_finding,
            uncertain_policy=args.uncertain_policy,
        )
        selected = set(_select_patients_by_row_budget(order, row_counts, args.offset, args.n_studies))
        rows = df[df["_patient_id"].isin(selected)].to_dict("records")

    missing = []
    for row in rows:
        rel = row["Path"]
        if not rel.startswith(CSV_TRAIN_PREFIX):
            continue  # not a train-split row, skip
        local_path = images_root / rel
        if not local_path.exists():
            missing.append(rel)

    candidate_batches = [args.batch] if args.batch else DEFAULT_TRAIN_BATCHES
    print(f"Requested {len(rows)} studies from train.csv; {len(rows) - len(missing)} already present locally, "
          f"{len(missing)} to fetch from {candidate_batches}.")
    if not missing:
        print("Nothing to download -- all requested studies are already local.")
        return

    from contextlib import ExitStack
    from remotezip import RemoteZip

    def _batch_root(batch_name: str) -> str:
        return batch_name[:-len(".zip")] + "/" if batch_name.endswith(".zip") else batch_name + "/"

    with ExitStack() as stack:
        open_zips = []
        for batch_name in candidate_batches:
            url = _blob_url(sas_url, batch_name)
            print(f"Opening remote zip index for '{batch_name}' (one-time cost, reads the central directory)...")
            t_index_start = time.time()
            z = stack.enter_context(RemoteZip(url, support_suffix_range=False))
            print(f"  Index ready in {time.time() - t_index_start:.1f}s ({len(z.namelist())} entries).")
            open_zips.append((batch_name, z))

        start_time = time.time()
        fetched, skipped_not_found, retried = 0, 0, 0
        fetched_per_batch = {b: 0 for b, _ in open_zips}
        max_retries = 5

        for i, rel in enumerate(missing, 1):
            local_path = images_root / rel
            sys.stdout.write(_progress_bar(i, len(missing), start_time, rel.split("/")[-2] + "/" + rel.split("/")[-1]))
            sys.stdout.flush()

            fetched_this_file = False
            for batch_name, z in open_zips:
                entry_name = _batch_root(batch_name) + rel[len(CSV_TRAIN_PREFIX):]
                for attempt in range(1, max_retries + 1):
                    try:
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        data = z.read(entry_name)
                        local_path.write_bytes(data)
                        fetched += 1
                        fetched_per_batch[batch_name] += 1
                        fetched_this_file = True
                        break
                    except KeyError:
                        break  # not in THIS batch -- not transient, try the next one, don't retry
                    except Exception as e:
                        # Azure resets long-lived connections occasionally over a
                        # run this long (hundreds of individual HTTPS requests) --
                        # transient, not a bug. Back off and retry a few times
                        # before giving up on this file in this batch.
                        retried += 1
                        if attempt == max_retries:
                            print(f"\n  WARNING: giving up on {rel} in '{batch_name}' after "
                                  f"{max_retries} attempts ({e}).")
                            break
                        time.sleep(min(2 ** attempt, 30))
                if fetched_this_file:
                    break
            if not fetched_this_file:
                skipped_not_found += 1

        print()  # newline after the progress bar

    per_batch_str = ", ".join(f"{n} from '{b}'" for b, n in fetched_per_batch.items())
    print(f"Done: fetched {fetched} images ({per_batch_str}), {skipped_not_found} not found in any "
          f"candidate batch, {retried} transient retries.")


if __name__ == "__main__":
    main()
