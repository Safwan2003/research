# Remote pipeline via the Colab CLI

Runs the exact same fine-tuning comparison pipeline (`run_finetune_pipeline.sh`)
on a remote Colab GPU instead of your laptop, then pulls the results back and
pushes them to GitHub as a backup. Kept deliberately separate from
`run_master.sh` / `run_finetune_pipeline.sh` — this folder is the only place
that knows Colab exists; the WSL/lab-machine workflow is untouched.

Why this exists: CheXpert's real training images are ~223k files across
~450GB of Azure blob storage, too large to keep on a laptop. The download
already happens the "dynamic" way, streamed directly from Azure —
`scripts/download_chexpert_subset.py` uses `remotezip` to fetch only the
specific files a sample needs via HTTP range requests, never the whole
91-185GB batch blobs. Combined with the sampling fix in `src/data/dataset.py`
(patient-level stratified random sampling instead of "first N rows of
train.csv" — see the root `CLAUDE.md`), any sample size stays representative.
Crucially, **all of this happens on the Colab VM's own disk, never your
laptop's** — your laptop only ever receives the small results files
(`experiment_log.json`, a markdown report) at the end.

## Setup (one-time)

1. Install the **official** Google Colab CLI — there are several unrelated
   packages with similar names on PyPI/GitHub; use this one (version 0.6.0
   as of writing):
   ```bash
   pip install --user google-colab-cli
   # or: uv tool install google-colab-cli
   colab --help    # full command list
   colab readme    # the CLI's own bundled docs, worth reading once
   ```
   **Known bug as of 0.6.0**: `google-colab-cli` depends on `jupyter-kernel-client`
   with no version upper bound, and pip resolves that to `1.0.0` by default —
   which renamed the class the CLI actually imports, so `colab exec` crashes
   with `AttributeError: module 'jupyter_kernel_client' has no attribute
   'KernelClient'`. Fix (confirmed working):
   ```bash
   pip install --user "jupyter_kernel_client==0.15.0"
   ```
   Check this is still needed with a harmless throwaway command before
   assuming it's still broken — `google-colab-cli` may have shipped a real
   fix by the time you read this:
   ```bash
   colab new -s smoketest && echo "print(1)" | colab exec -s smoketest && colab stop -s smoketest
   ```

2. The first `colab new` (or the first `colab exec`/`colab download`/etc.
   this script runs) triggers Google OAuth sign-in — cached afterward to
   `~/.config/colab-cli/`, so this part only happens once. **Confirmed by
   testing: `colab drivemount` (used for crash-safe checkpointing, see
   below) is different — it needs a *fresh* interactive Drive-access grant
   on every new session**, not just once ever. Net effect: **always run
   `bash colab/run_remote_pipeline.sh` itself in a real, separate terminal
   window**, not through an AI coding tool's command interface (which can't
   hold open the "visit this URL, then press Enter" prompt either step
   needs) — the script will pause on its own mid-run asking for this when a
   new session gets created (including automatically, if a previous session
   died and you're re-running to resume).

3. From the repo root, copy `.env.example` to `.env` and fill in the two
   values yourself, directly in a text editor (see the root README's
   "Getting the datasets" section):
   ```
   CHEXPERT_SAS_URL=...   # Stanford AIMI's CheXpert Azure SAS link
   GITHUB_TOKEN=...       # GitHub PAT (repo scope) -- only used for step 5's backup push
   ```
   `.env` is gitignored. It gets uploaded to the remote runtime (never
   committed anywhere) so the remote pipeline can download CheXpert too.

## Usage

```bash
bash colab/run_remote_pipeline.sh
```

Override sizes the same way `run_finetune_pipeline.sh` supports. For a large,
representative run with only frozen/LoRA/QLoRA (no full fine-tuning):
```bash
N_ABLATION=20000 N_AGENTIC=200 N_FINETUNE_STUDIES=5000 EPOCHS=1 RUN_FULL_FT=0 \
  bash colab/run_remote_pipeline.sh
```
If a session gets killed partway (quota, timeout, closed laptop), **run the
exact same command again** — see "Crash/quota recovery" below.

## What it does

1. Creates (or reuses) a named Colab session (`research` by default) with a
   GPU attached (`T4` by default — matches the paper's own setup, see the
   root `CLAUDE.md`).
2. Mounts Google Drive (`colab drivemount`) — this is what makes checkpoints
   survive a killed session, see "Crash/quota recovery" below.
3. Uploads your local `.env` to the session's filesystem (`colab upload`
   — secrets never touch git or the remote repo).
4. Runs a small generated Python driver via `colab exec -f ...` that
   clones/pulls this repo, creates a venv, symlinks `models/`,
   `.setup_markers/`, and `results/` into the mounted Drive, and launches
   `run_finetune_pipeline.sh` — the *same* script, unmodified — **in the
   background** on the VM, writing progress to `colab_pipeline.log` and a
   `colab_pipeline.status` marker (`DONE`/`FAILED`) when it finishes. This
   avoids needing one `colab exec` call to stay blocked open for the
   multi-hour duration of a real fine-tuning run (`colab exec`'s own
   default timeout is only 30s).
5. Polls every `POLL_INTERVAL` seconds (default 120) by attempting to
   `colab download` the status marker — small and cheap, just fails
   harmlessly until the file exists — printing the tail of the remote log
   each time so you can watch progress. If the session itself dies while
   waiting, this is detected and reported (see below) instead of polling
   forever.
6. Once done, pulls `results/experiment_log.json` and
   `results/finetune_comparison_report.md` back to
   `colab/pulled_results/<timestamp>/` and merges the log into your local
   one via `colab/merge_results.py` (append-only, de-duplicated by
   `run_id` — safe to run repeatedly). Model checkpoints in `models/` are
   *not* pulled automatically (can be several GB) — the script prints the
   exact `colab download` command to grab one yourself if you want it.
7. Commits and pushes the updated `results/` to GitHub as a backup, using
   `GITHUB_TOKEN` only for that one push (never written to `.git/config`,
   never printed).
8. **Always stops the session on exit** (success, failure, or Ctrl-C) so
   you don't leave a billed GPU running by accident — set `KEEP_SESSION=1`
   to skip this if you want to inspect the VM afterward (then
   `colab stop -s research` yourself when done).

## Crash/quota recovery

`run_finetune_pipeline.sh` and `src/vlm/finetune_qwen.py` already have
layered resumability built in (`.setup_markers/*.done` step-skipping,
`--resume` reloading `models/<regime>_chexpert_2b_last/training_state.pt`)
— the gap this pipeline closes is that a Colab session's local disk is
**wiped** when the session stops, which would otherwise erase all of that.

Fix: `models/`, `.setup_markers/`, and `results/` are symlinked into
Google Drive (confirmed working on Drive's FUSE mount by an actual test —
writes through the symlink are immediately visible via the real Drive path,
and survive a full session teardown + fresh session + remount). `data/`
(the downloaded CheXpert images) is deliberately **not** persisted — free
Drive is only 15GB, and re-downloading is fast/cheap compared to re-doing
GPU training.

So: if a session dies mid-run, **just re-run the same command in a real
terminal**. It creates a new session, and the pipeline's own resumability
picks the *work* up automatically from the last checkpoint. The one
non-automatic part (confirmed by testing): Drive authorization is
per-session, so the new run will pause once, asking you to visit a URL and
grant Drive access again (~30 seconds) before it can continue.

## Design notes / things confirmed by an actual test run

Live-tested against `google-colab-cli` 0.6.0 on a real (free, CPU-only)
Colab session, not just read from docs:

- `colab exec` runs *Python code* in a remote Jupyter kernel, not arbitrary
  shell commands (confirmed: it's the reason the driver script shells out
  via `subprocess` rather than sending bash directly), and its kernel's
  cwd is always `/content`.
- `colab upload`/`colab download` require an **absolute** remote path
  (`/content/research/...`) — a bare relative path (`research/...`, which
  the CLI's own doc examples use and which `colab exec` code happily
  accepts) fails for these two commands specifically. `REMOTE_PATH` in the
  script exists specifically to get this right in one place.
- **The background-launch-then-poll design works**: a
  `subprocess.Popen(..., start_new_session=True)`-launched background
  process genuinely keeps running and writing files after the `colab exec`
  call that launched it returns, independently checkable via later,
  separate `colab exec`/`colab download` calls. Tested end-to-end with a
  ~30s background job polled across 7 separate `colab download` calls —
  it kept running and completed correctly. This was the design assumption
  most likely to be wrong (unclear whether the service might kill child
  processes when an exec "completes"); it wasn't.
- **`os.symlink` works fine on Drive's FUSE-mounted filesystem**, and data
  written through the symlink genuinely persists in Drive and survives a
  full session teardown + new session + fresh mount — tested directly, not
  assumed (FUSE filesystems don't always support symlinks; this one does).
- **Drive authorization is per-session, not cached** — tested directly by
  killing a session and mounting Drive again on a fresh one; it required a
  brand-new interactive grant. See "Crash/quota recovery" above.
- Re-running the script reuses an existing session (`colab status -s
  research`) rather than erroring — if a previous run is still in
  progress, re-running just resumes polling instead of relaunching.

## Files

- `run_remote_pipeline.sh` — the orchestrator described above (run from your laptop).
- `merge_results.py` — de-duplicated append-only merge of a pulled
  `experiment_log.json` into the local one. Torch-free, self-testable:
  `python3 colab/merge_results.py` (no args) runs its self-test.
- `pulled_results/` — timestamped local copies of what got pulled back from
  each Colab run (gitignored — this is a local cache, `results/` at the repo
  root is the source of truth once merged).
