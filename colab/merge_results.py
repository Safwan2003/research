"""
Merge a pulled results/experiment_log.json (from a colab/run_remote_pipeline.sh
run) into the local one, de-duplicated by run_id.

results/experiment_log.json is append-only (src/evaluation/results_store.py)
-- runs are never edited after being written, so a run_id present in both
files is identical in both, and it's always safe to keep just one copy.

Usage:
    python3 colab/merge_results.py --remote /path/to/pulled/experiment_log.json --local results/experiment_log.json

Bare `python3 colab/merge_results.py` (no args) runs a self-test with
synthetic data -- no network/Colab/GPU needed, per this project's
convention (see CLAUDE.md) of every new module being sanity-checkable
standalone.
"""

import argparse
import json
from pathlib import Path


def merge_logs(remote_path: str, local_path: str) -> int:
    """Merge remote_path's runs into local_path (in place, sorted by timestamp). Returns the count of newly added runs."""
    remote = json.loads(Path(remote_path).read_text())
    local_file = Path(local_path)
    local = (
        json.loads(local_file.read_text())
        if local_file.exists()
        else {"schema_version": remote.get("schema_version", 1), "runs": []}
    )

    existing_ids = {r["run_id"] for r in local["runs"]}
    new_runs = [r for r in remote.get("runs", []) if r["run_id"] not in existing_ids]
    local["runs"].extend(new_runs)
    local["runs"].sort(key=lambda r: r["timestamp_utc"])

    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(json.dumps(local, indent=2))
    return len(new_runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote", required=True, help="experiment_log.json pulled from the Colab runtime.")
    parser.add_argument("--local", required=True, help="Local results/experiment_log.json to merge into.")
    args = parser.parse_args()
    n_new = merge_logs(args.remote, args.local)
    print(f"Merged {n_new} new run(s) from {args.remote} into {args.local}.")


def _self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        local_path = Path(d) / "local.json"
        remote_path = Path(d) / "remote.json"
        local_path.write_text(json.dumps({
            "schema_version": 1,
            "runs": [{"run_id": "a", "timestamp_utc": "2026-01-01T00:00:00+00:00"}],
        }))
        remote_path.write_text(json.dumps({
            "schema_version": 1,
            "runs": [
                {"run_id": "a", "timestamp_utc": "2026-01-01T00:00:00+00:00"},  # duplicate, already local
                {"run_id": "b", "timestamp_utc": "2026-01-02T00:00:00+00:00"},  # new
            ],
        }))

        n_new = merge_logs(str(remote_path), str(local_path))
        assert n_new == 1, f"expected 1 new run, got {n_new}"
        merged = json.loads(local_path.read_text())
        assert {r["run_id"] for r in merged["runs"]} == {"a", "b"}
        assert len(merged["runs"]) == 2, "duplicate run_id got double-counted"

        # Merging the same remote file again must be a no-op (idempotent).
        n_new_second = merge_logs(str(remote_path), str(local_path))
        assert n_new_second == 0, "re-merging the same remote file should add nothing"

    print("OK: merge_logs() dedupes by run_id, merges new runs, and is idempotent on re-merge.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_test()
    else:
        main()
