from __future__ import annotations

"""
Append-only JSON experiment log.

Every pipeline run (ablation / hallucination / text_metrics /
agentic_reasoning / single) gets appended as one record to
`results/experiment_log.json`. Records are never overwritten -- this is
what lets you accumulate a history of results across machines (this dev
box and the lab PC) instead of each run erasing the last one.

See docs/superpowers/specs/2026-07-22-results-persistence-design.md for
the full schema.
"""

import json
import os
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def load_results_log(path: str | Path) -> dict:
    """Returns the existing log, or a fresh empty one if the file doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "runs": []}
    with open(path, "r") as f:
        return json.load(f)


def new_run_record(mode: str, config_snapshot: dict, results: dict, notes: str = "") -> dict:
    """
    Build one run record. `results` should only contain the keys the run
    actually produced, e.g. {"ablation": {...}} or {"hallucination": {...}}
    -- see the spec doc for the full set of possible keys.
    """
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "mode": mode,
        "config_snapshot": config_snapshot,
        "results": results,
        "notes": notes,
    }


def append_run(path: str | Path, run_record: dict) -> None:
    """Load, append, write atomically (temp file + os.replace) so a crash mid-write can't corrupt history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    log = load_results_log(path)
    log["runs"].append(run_record)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "experiment_log.json"

        run1 = new_run_record(
            mode="ablation",
            config_snapshot={"seed": 42, "n_studies": 300, "dataset": "synthetic"},
            results={"ablation": {"Text only": 0.918, "Radiomics + XAI + Text": 0.925}},
        )
        append_run(log_path, run1)

        run2 = new_run_record(
            mode="hallucination",
            config_snapshot={"seed": 42, "n_studies": 300, "dataset": "synthetic"},
            results={"hallucination": {"Image only": 1.14, "Image + Text": 0.25}},
        )
        append_run(log_path, run2)

        log = load_results_log(log_path)
        assert len(log["runs"]) == 2, "expected both runs to survive -- append must not overwrite"
        assert log["runs"][0]["run_id"] != log["runs"][1]["run_id"]

        print(f"Appended {len(log['runs'])} runs to {log_path}, both preserved:")
        for run in log["runs"]:
            print(f"  - {run['mode']}: {run['results']}")
