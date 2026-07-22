# Results Persistence Layer (Append-Only JSON Experiment Log)

## Problem

`run_pipeline.py` and the evaluation modules (`ablation_study.py`,
`hallucination.py`, `text_metrics.py`) currently only print results to
stdout. Nothing is saved. Every time the pipeline is re-run — especially
across two machines (this dev sandbox and the lab PC running Windows +
Ubuntu WSL with Antigravity CLI) — there is no record of what a prior run
produced, so results are effectively erased rather than accumulated.

## Goal

A single append-only JSON file that accumulates one record per pipeline
run, covering every metric the paper (arXiv:2604.08815) reports across
Tables 1–4, so results from any machine/run can be compared over time
without ever being overwritten.

## Storage format

`results/experiment_log.json`:

```json
{
  "schema_version": 1,
  "runs": [ { "...": "one run record" } ]
}
```

Appending means: load the file (or initialize this structure if missing),
push a new object onto `runs`, write atomically (temp file + `os.replace`)
so a crash mid-write can't corrupt prior history. Plain JSON was chosen
over SQLite/CSV per explicit preference — no dependency, human-readable,
identical behavior on Linux and Windows/WSL.

## Run record schema

```json
{
  "run_id": "uuid4 string",
  "timestamp_utc": "ISO 8601",
  "git_commit": "40-char sha or null if not in a git repo",
  "host": "hostname",
  "mode": "ablation | hallucination | text_metrics | agentic_reasoning | single",
  "config_snapshot": {"vlm_model_name": "...", "seed": 42, "n_studies": 1000, "dataset": "openi|chexpert"},
  "results": {
    "ablation": {"Radiomics only": 0.555, "XAI only": 0.519, "Text only": 0.918,
                 "Radiomics + Text": 0.921, "XAI + Text": 0.918,
                 "Radiomics + XAI + Text": 0.925},
    "hallucination": {"Image only": 1.14, "Image + Radiomics": 1.11,
                       "Image + XAI": 1.01, "Image + Radiomics + XAI": 0.60,
                       "Image + Text": 0.25, "Image + Text + Radiomics + XAI": 0.28},
    "agentic_reasoning": {"uncertainty_mean": 0.68, "uncertainty_std": 0.14,
                           "evidence_length_mean": 15.3, "evidence_length_std": 18.0,
                           "presence_limitations": 0.8, "presence_safety_notes": 0.8},
    "text_metrics": {"<variant_name>": {"n": 217, "rouge1": 0.091, "rouge2": 0.011,
                      "rougeL": 0.053, "bertscore_f1": 0.778, "phi_pct": 0.0,
                      "unsafe_pct": 0.0, "unc_rate": 0.168}}
  },
  "notes": ""
}
```

Only the keys under `results` that a given run actually produced are
populated; the rest are omitted (not null-padded) to keep records honest
about what was actually measured.

## Components

1. **`src/evaluation/results_store.py`** (new)
   - `load_results_log(path) -> dict` — returns `{"schema_version": 1, "runs": []}` if the file doesn't exist yet.
   - `new_run_record(mode, config_snapshot, results, notes="") -> dict` — fills `run_id` (uuid4), `timestamp_utc`, `git_commit` (via `git rev-parse HEAD`, `None` on failure), `host` (`socket.gethostname()`).
   - `append_run(path, run_record) -> None` — load, append, atomic write.
   - Local imports only where needed (`subprocess` for git sha) — no heavy deps, importable with zero setup, matching repo convention.
   - `if __name__ == "__main__":` self-test using synthetic run records, verifying two appended runs both survive in the file.

2. **`src/evaluation/responsible_ai_metrics.py`** (new)
   - Fills the one real gap in metric coverage: Table 3's agentic-reasoning stats and Table 4's Responsible-AI columns (PHI %, unsafe %, uncertainty marker rate) are not computed anywhere in the current codebase — they require parsing the structured VLM JSON output (`impression`, `evidence`, `uncertainty`, `limitations`, `safety_note` — see `RESPONSE_SCHEMA_INSTRUCTIONS` in `src/vlm/prompts.py`).
   - `compute_responsible_ai_metrics(structured_outputs: list[dict]) -> dict` returns exactly the `agentic_reasoning`-shaped dict above (mean/std of `uncertainty`, mean/std of word count of `evidence`, fraction with non-empty `limitations`, fraction with non-empty `safety_note`).
   - Heuristic PHI/unsafe detection kept intentionally simple (regex over generated text for the same categories the paper describes — numeric identifiers, unsafe-content keyword list) since the paper itself calls these "heuristic Responsible-AI indicators."
   - Self-test block using synthetic structured outputs mirroring the schema.

3. **`config.py`** — one new line: `RESULTS_LOG_PATH = PROJECT_ROOT / "results" / "experiment_log.json"`.

4. **`run_pipeline.py`** — the `ablation` mode currently only prints a TODO. Add a `--synthetic` flag that runs the existing synthetic self-test from `ablation_study.py` end-to-end and calls `append_run`, so the full log path is verifiably runnable today without a GPU or real dataset. The TODO instructions for wiring real OpenI features stay as-is (that remains "Known gap #4", a lab-PC/real-data job).

## Non-goals

- Not building a query/reporting layer on top of the JSON (analysis can load it into pandas ad hoc — `pd.json_normalize(json.load(open(...))["runs"])`).
- Not touching `hallucination.py` / `text_metrics.py` internals — their existing return dicts already slot directly into `results["hallucination"]` / `results["text_metrics"]` as-is; a caller just wraps them in `new_run_record(...)`.
- Not adding SQLite/CSV backends — explicitly ruled out per user preference.

## Testing

Each new module gets a synthetic `__main__` self-test (repo convention),
runnable with zero setup:

```
python src/evaluation/results_store.py
python src/evaluation/responsible_ai_metrics.py
python run_pipeline.py ablation --synthetic
```

The last command should produce a `results/experiment_log.json` with one
run appended; running it again should produce a second run record without
touching the first.
