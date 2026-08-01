#!/usr/bin/env bash
# Remote fine-tuning pipeline over Google's official Colab CLI
# (https://github.com/googlecolab/google-colab-cli, package name
# `google-colab-cli` -- NOT one of the other similarly-named unofficial
# "colab-cli" packages on PyPI). Runs FROM YOUR LAPTOP, not inside Colab: it
# provisions a remote GPU session, runs the existing run_finetune_pipeline.sh
# pipeline there (same script WSL/the lab machine uses -- nothing about it
# is duplicated or reimplemented here), pulls results back to your laptop,
# and pushes them to GitHub as a backup.
#
# Deliberately separate from run_master.sh / run_finetune_pipeline.sh (per
# request) so the WSL/lab-machine workflow is untouched -- this file is the
# only thing that knows about Colab.
#
# ============================================================================
# How this maps onto the real `colab` CLI (verified against `colab --help`,
# `colab readme`, and each subcommand's --help on google-colab-cli 0.6.0)
# ============================================================================
# `colab exec` runs PYTHON code in a remote Jupyter kernel -- it is NOT a
# generic "run any shell command" tool, and its default --timeout is only
# 30s. So instead of sending shell commands directly, this script generates
# a small Python driver (see _make_driver_py below) that shells out via
# subprocess, and launches the actual (multi-hour) run_finetune_pipeline.sh
# call IN THE BACKGROUND on the VM, writing a status-marker file. This
# script then polls for that marker via `colab download` (small, fast,
# either succeeds once the file exists or fails harmlessly if it doesn't
# yet) instead of holding one `colab exec` call open for hours -- more
# robust against any exec-side timeout than trying to guess a "big enough"
# --timeout value.
#
# ============================================================================
# Crash/quota-exhaustion safety (Google Drive persistence)
# ============================================================================
# A Colab session's local disk (/content) is WIPED when the session is
# stopped or killed (quota exhaustion, idle timeout, you closing your
# laptop). run_finetune_pipeline.sh and src/vlm/finetune_qwen.py already
# have their own layered resumability (.setup_markers/*.done step-skipping,
# --resume reloading models/<regime>_chexpert_2b_last/training_state.pt) --
# but that only helps if those files survive the VM being torn down. So
# this script mounts Google Drive and symlinks exactly those three
# directories (models/, .setup_markers/, results/ -- all small: LoRA/QLoRA
# adapters are a few MB each, markers/results are tiny) into it. Re-running
# this script after a killed session creates a NEW VM and picks the WORK up
# automatically from the last checkpoint -- but NOT fully hands-off (tested,
# see colab/README.md's Design notes): Drive authorization is per-session,
# not cached, so each new session pauses once needing you to visit a URL
# and grant Drive access again (~30s), same as the very first `colab new`
# sign-in. Run this from a real terminal (see the auth note in the README)
# so that prompt has somewhere to go.
#
# Deliberately NOT persisted to Drive: data/chexpert/ (the downloaded
# CheXpert images). Free Google Drive is only 15GB, too small for a large
# sample, and losing downloaded images just costs re-download time (fast,
# cloud-to-cloud, and download_chexpert_subset.py already skips
# already-present files) -- not lost GPU training time, which is the actual
# expensive thing this protects.
#
# Usage (run from a real terminal, not through an AI coding tool's command
# interface -- both drivemount and, on a first-ever run, the CLI's own
# sign-in need an interactive browser+terminal round trip):
#   cp .env.example .env   # fill in CHEXPERT_SAS_URL and GITHUB_TOKEN yourself
#   bash colab/run_remote_pipeline.sh
#
# Override via env vars, same convention as run_finetune_pipeline.sh:
#   COLAB_GPU=A100 N_FINETUNE_STUDIES=5000 RUN_FULL_FT=0 bash colab/run_remote_pipeline.sh
#
# If a session dies mid-run (quota/timeout), just re-run the same command --
# it detects the dead session, tells you so, and the only manual step on the
# way back up is the Drive re-authorization above. Nothing is lost except
# whatever wasn't checkpointed yet within the current training step.
#
# COST NOTE: a GPU session keeps billing/using your Colab compute units
# until stopped. This script stops it automatically on exit (success,
# error, or Ctrl-C) unless you set KEEP_SESSION=1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
banner() { echo; echo -e "${BOLD}${CYAN}==> $1${RESET}"; }
ok()     { echo -e "  ${GREEN}[OK]${RESET} $1"; }
warn()   { echo -e "  ${YELLOW}[!]${RESET} $1"; }
# `colab status -s <name>` always exits 0, even when <name> doesn't exist --
# it just prints "[colab] Session '<name>' not found." to stdout (confirmed
# by reading colab_cli/commands/session.py's status(): the "not found" branch
# never raises typer.Exit(1)). Checking only the exit code (what this used to
# do) always looks like success, so a session that was never created (or
# already died) gets silently treated as alive, and the first command that
# actually needs real session state -- e.g. drivemount -- crashes with
# AttributeError: 'NoneType' object has no attribute 'url'. The text is the
# only reliable signal the CLI gives; parse it instead.
_session_exists() {
  local output
  output="$(colab status -s "$1" 2>&1 || true)"
  [[ "$output" != *"not found"* ]]
}
fail()   { echo -e "  ${RED}[FAIL]${RESET} $1" >&2; }

COLAB_GPU="${COLAB_GPU:-T4}"                          # paper's own setup uses a Tesla T4 -- see CLAUDE.md
SESSION="${COLAB_SESSION:-research}"
N_ABLATION="${N_ABLATION:-1000}"
N_AGENTIC="${N_AGENTIC:-50}"
N_FINETUNE_STUDIES="${N_FINETUNE_STUDIES:-300}"
EPOCHS="${EPOCHS:-1}"
RUN_FULL_FT="${RUN_FULL_FT:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"                  # seconds between status checks
KEEP_SESSION="${KEEP_SESSION:-0}"                      # 1 = don't auto-stop the VM on exit (costs money while idle)
REMOTE_DIR="research"                                  # working dir name on the VM
# `colab exec`'s kernel always runs with cwd=/content (Colab's standard working
# dir -- confirmed by testing, see colab/README.md's Design notes), and
# `colab upload`/`colab download` require an ABSOLUTE remote path (a bare
# relative path like "research/foo" 500s -- also confirmed by testing, this
# isn't documented). REMOTE_PATH is the one place that combines the two.
REMOTE_PATH="/content/${REMOTE_DIR}"
DRIVE_STATE="/content/drive/MyDrive/${DRIVE_STATE_DIR_NAME:-context_aligned_medical_vlm_state}"

PULL_DIR="colab/pulled_results/$(date +%Y%m%d_%H%M%S)"
LOCAL_TMP="$(mktemp -d)"
trap 'rm -rf "$LOCAL_TMP"' EXIT

# --- Preflight -----------------------------------------------------------------
banner "Preflight checks"

if ! command -v colab >/dev/null 2>&1; then
  fail "The 'colab' CLI isn't on PATH."
  echo "  Install the OFFICIAL Google Colab CLI (not an unofficial same-named package):" >&2
  echo "    pip install google-colab-cli" >&2
  echo "  or:" >&2
  echo "    uv tool install google-colab-cli" >&2
  exit 1
fi
ok "'colab' CLI found: $(command -v colab) ($(colab version 2>&1))"

if [ ! -f .env ]; then
  fail ".env not found -- copy .env.example to .env and fill in CHEXPERT_SAS_URL / GITHUB_TOKEN first."
  exit 1
fi
# shellcheck disable=SC1091
set -a; source .env; set +a

# Falls back to secrets/chexpert_sas_url.txt -- the older convention
# config.py's _load_chexpert_sas_url() already treats as equivalent to
# CHEXPERT_SAS_URL in .env -- so this preflight (and what actually gets
# uploaded below) doesn't fail for anyone who only ever set up the secret
# that way and never duplicated it into .env.
if [ -z "${CHEXPERT_SAS_URL:-}" ] && [ -f secrets/chexpert_sas_url.txt ]; then
  CHEXPERT_SAS_URL="$(tr -d '[:space:]' < secrets/chexpert_sas_url.txt)"
fi
if [ -z "${CHEXPERT_SAS_URL:-}" ]; then
  fail "CHEXPERT_SAS_URL is empty in both .env and secrets/chexpert_sas_url.txt -- the remote runtime needs it to download CheXpert."
  exit 1
fi
export CHEXPERT_SAS_URL
if [ -z "${GITHUB_TOKEN:-}" ]; then
  warn "GITHUB_TOKEN is empty in .env -- results will still be pulled to your laptop, but the "
  warn "final 'push to GitHub as backup' step will be skipped."
fi
ok ".env loaded (values not printed)."

# What actually gets uploaded to the VM (see step 3 below) -- built from the
# resolved values above rather than uploading the local .env file verbatim,
# so the secrets/ fallback above actually reaches the remote runtime too
# (a raw file upload wouldn't see values that only got resolved into this
# shell's environment).
printf 'CHEXPERT_SAS_URL=%s\nGITHUB_TOKEN=%s\n' "$CHEXPERT_SAS_URL" "${GITHUB_TOKEN:-}" > "$LOCAL_TMP/.env.resolved"

GIT_REMOTE_URL="$(git remote get-url origin)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ok "Will clone/pull ${GIT_REMOTE_URL} (branch ${GIT_BRANCH}) on the remote runtime."

mkdir -p "$PULL_DIR"

# --- 1. Provision (or reuse) the remote GPU session -----------------------------
banner "1/7 Provisioning Colab session '${SESSION}' (${COLAB_GPU})"
if _session_exists "$SESSION"; then
  ok "Session '${SESSION}' already exists, reusing it."
else
  colab new -s "$SESSION" --gpu "$COLAB_GPU"
  ok "Session '${SESSION}' created."
fi

if [ "$KEEP_SESSION" != "1" ]; then
  trap 'echo; echo -e "  ${DIM}Stopping Colab session '"'"'${SESSION}'"'"' (KEEP_SESSION=1 to skip this)...${RESET}"; colab stop -s "'"$SESSION"'" || true; rm -rf "'"$LOCAL_TMP"'"' EXIT
else
  warn "KEEP_SESSION=1 -- the VM will keep running (and billing) after this script exits."
  warn "Stop it yourself when done: colab stop -s ${SESSION}"
fi

# --- 2. Mount Google Drive for crash/quota-safe persistence ---------------------
banner "2/7 Mounting Google Drive (checkpoints/results survive a killed session)"
# NOTE (confirmed by testing): Drive authorization is PER-SESSION, not cached
# like the CLI's own login -- every new session needs a fresh interactive
# grant here (visit a URL, approve, press Enter), same as the original
# `colab new` sign-in. This is why 'just re-run the script and it silently
# resumes' isn't quite true -- it resumes the WORK automatically, but this
# one step still needs ~30s of your attention each time a session had to be
# recreated. If this hangs, it's waiting on you in THIS terminal, not stuck.
colab drivemount -s "$SESSION"
ok "Drive mounted; persistent state lives under ${DRIVE_STATE}"

# --- 3. Set up the repo + secrets on the VM, then launch the pipeline in the background ---
banner "3/7 Setting up the repo and launching run_finetune_pipeline.sh on the VM"

_make_driver_py() {
  # Generates the Python driver colab exec runs. Config is baked in as
  # literals (there's no `colab exec --env` flag) rather than passed via
  # environment, since each exec call is a fresh kernel invocation.
  cat <<PYEOF
import os, shutil, subprocess, sys

repo_dir = "${REMOTE_PATH}"
drive_state = "${DRIVE_STATE}"
log_path = os.path.join(repo_dir, "colab_pipeline.log")
status_path = os.path.join(repo_dir, "colab_pipeline.status")

def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

if not os.path.isdir(os.path.join(repo_dir, ".git")):
    # NOT `git clone url repo_dir` -- repo_dir already exists and is
    # non-empty by this point (.env was uploaded into it just before this
    # driver runs, see step 3 below), and `git clone` refuses to clone into
    # an existing non-empty directory (fatal: destination path ... already
    # exists and is not an empty directory, exit 128). git init + remote add
    # works on an existing non-empty directory, since it only touches .git/.
    run(["git", "init", repo_dir])
    run(["git", "-C", repo_dir, "remote", "add", "origin", "${GIT_REMOTE_URL}"])
run(["git", "-C", repo_dir, "fetch", "origin"])
run(["git", "-C", repo_dir, "checkout", "${GIT_BRANCH}"])
run(["git", "-C", repo_dir, "pull", "origin", "${GIT_BRANCH}"])
run(["python3", "-m", "venv", os.path.join(repo_dir, ".venv")])

# Symlink the small, checkpoint-bearing dirs into Drive so a killed/quota-exhausted
# session doesn't lose progress -- run_finetune_pipeline.sh and finetune_qwen.py's
# own resumability (.setup_markers/*.done, --resume reloading
# <output_dir>_last/training_state.pt), plus run_chexpert_eval.py's per-study
# ablation/agentic checkpoints (.eval_checkpoints/*.jsonl -- see that script's
# _ablation_checkpoint_path/_agentic_checkpoint_path), then just work across VM
# teardowns, since these paths transparently point at Drive-backed storage instead
# of /content's ephemeral disk. data/chexpert/ (the bulk images) is deliberately
# NOT included -- it's cheap/fast to re-download (Azure-to-Colab, not laptop
# bandwidth) and would blow past free Drive's 15GB quota for a large sample.
for sub in ("models", ".setup_markers", ".eval_checkpoints", "results"):
    local_path = os.path.join(repo_dir, sub)
    drive_path = os.path.join(drive_state, sub)
    os.makedirs(drive_path, exist_ok=True)
    if os.path.islink(local_path):
        continue
    if os.path.exists(local_path):
        # git clone creates this (results/ is git-tracked -- experiment_log.json's
        # committed baseline). Seed Drive with it on the very first run, without
        # clobbering anything a prior interrupted run already put there.
        for name in os.listdir(local_path):
            src, dst = os.path.join(local_path, name), os.path.join(drive_path, name)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(local_path)
    os.symlink(drive_path, local_path)
    print(f"symlinked {local_path} -> {drive_path}")

if os.path.exists(status_path):
    os.remove(status_path)

launch = (
    f"cd {repo_dir} && source .venv/bin/activate && "
    f"(N_ABLATION=${N_ABLATION} N_AGENTIC=${N_AGENTIC} "
    f"N_FINETUNE_STUDIES=${N_FINETUNE_STUDIES} EPOCHS=${EPOCHS} "
    f"RUN_FULL_FT=${RUN_FULL_FT} bash run_finetune_pipeline.sh "
    f"> {log_path} 2>&1 && echo DONE > {status_path} "
    f"|| echo FAILED > {status_path}) &"
)
subprocess.Popen(
    ["bash", "-c", launch], cwd=repo_dir,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print("LAUNCHED")
PYEOF
}

_make_driver_py > "$LOCAL_TMP/driver.py"

# ${REMOTE_PATH} doesn't exist yet on a first run -- create it before uploading .env into it.
colab exec -s "$SESSION" --timeout 60 <<< "import os; os.makedirs('${REMOTE_PATH}', exist_ok=True)"
colab upload -s "$SESSION" "$LOCAL_TMP/.env.resolved" "${REMOTE_PATH}/.env"
ok "Uploaded resolved CHEXPERT_SAS_URL/GITHUB_TOKEN to ${REMOTE_PATH}/.env on the VM (not printed, not committed)."

colab exec -s "$SESSION" -f "$LOCAL_TMP/driver.py" --timeout 900
ok "Setup complete, pipeline launched in the background on the VM."

# --- 4. Poll until the background pipeline finishes ------------------------------
banner "4/7 Waiting for the remote pipeline to finish (polling every ${POLL_INTERVAL}s)"
echo -e "  ${DIM}This can take a long time for real fine-tuning runs -- safe to Ctrl-C and re-run${RESET}"
echo -e "  ${DIM}this script later; it reuses the existing session and re-polls instead of${RESET}"
echo -e "  ${DIM}relaunching (Ctrl-C now, then re-run with the same COLAB_SESSION, to resume watching).${RESET}"

STATUS=""
while true; do
  if colab download -s "$SESSION" "${REMOTE_PATH}/colab_pipeline.status" "$LOCAL_TMP/status" >/dev/null 2>&1; then
    STATUS="$(tr -d '[:space:]' < "$LOCAL_TMP/status")"
    break
  fi
  if ! _session_exists "$SESSION"; then
    fail "Session '${SESSION}' is no longer alive (likely stopped by Colab -- quota/timeout)."
    echo "  Nothing important is lost: models/, .setup_markers/, and results/ are symlinked"
    echo "  into Google Drive (${DRIVE_STATE}) and survive the VM being torn down."
    echo "  Just re-run the exact same command from a real terminal -- it creates a new"
    echo "  session and run_finetune_pipeline.sh / finetune_qwen.py --resume WILL pick up"
    echo "  from the last checkpoint automatically. The one thing that ISN'T automatic:"
    echo "  Drive authorization is per-session, so the new run will pause once asking you"
    echo "  to visit a URL and grant Drive access again (~30s) before it can continue."
    echo "  (Also: data/chexpert/ needs re-downloading on the new VM, which is fast and"
    echo "  already skips files that are still present.)"
    exit 1
  fi
  echo -e "  ${DIM}... still running ($(date -u +%H:%M:%SZ)); tailing remote log:${RESET}"
  colab download -s "$SESSION" "${REMOTE_PATH}/colab_pipeline.log" "$LOCAL_TMP/log" >/dev/null 2>&1 \
    && tail -n 5 "$LOCAL_TMP/log" | sed 's/^/    /' \
    || echo "    (log not written yet)"
  sleep "$POLL_INTERVAL"
done

if [ "$STATUS" != "DONE" ]; then
  fail "Remote pipeline reported '$STATUS'. Full log:"
  colab download -s "$SESSION" "${REMOTE_PATH}/colab_pipeline.log" "$LOCAL_TMP/log" >/dev/null 2>&1 && cat "$LOCAL_TMP/log"
  exit 1
fi
ok "Remote pipeline finished successfully."

# --- 4. Pull results back to the laptop -----------------------------------------
banner "5/7 Pulling results back to ${PULL_DIR}"
mkdir -p "$PULL_DIR/results"
colab download -s "$SESSION" "${REMOTE_PATH}/results/experiment_log.json" "$PULL_DIR/results/experiment_log.json"
colab download -s "$SESSION" "${REMOTE_PATH}/results/finetune_comparison_report.md" "$PULL_DIR/results/finetune_comparison_report.md" \
  || warn "finetune_comparison_report.md not found (compile_finetune_comparison.py step may have failed upstream)."
ok "Pulled results into $PULL_DIR"
echo "  (Model checkpoints in ${REMOTE_PATH}/models/ are NOT auto-pulled -- can be several GB."
echo "   Pull one yourself if you want it, e.g.:"
echo "     colab download -s ${SESSION} ${REMOTE_PATH}/models/lora_chexpert_2b/adapter_model.safetensors ./adapter.safetensors"
echo "   or use 'colab console -s ${SESSION}' to browse ${REMOTE_PATH}/models/ interactively.)"

# --- 5. Merge into the local results log -----------------------------------------
banner "6/7 Merging into local results/experiment_log.json"
python3 colab/merge_results.py --remote "$PULL_DIR/results/experiment_log.json" --local results/experiment_log.json
[ -f "$PULL_DIR/results/finetune_comparison_report.md" ] && cp "$PULL_DIR/results/finetune_comparison_report.md" results/

# --- 6. Push to GitHub as backup --------------------------------------------------
banner "7/7 Pushing results to GitHub as backup"
if [ -z "${GITHUB_TOKEN:-}" ]; then
  warn "Skipping push -- GITHUB_TOKEN not set in .env. Commit/push results/ yourself when ready."
else
  git add results/
  if git diff --cached --quiet; then
    ok "No new results to commit."
  else
    git commit -m "Colab run: merge remote results (${COLAB_GPU}, $(date -u +%Y-%m-%dT%H:%M:%SZ))"
    PUSH_URL="$(echo "$GIT_REMOTE_URL" | sed -E "s#https://#https://x-access-token:${GITHUB_TOKEN}@#")"
    # Push over a URL carrying the token rather than `git remote set-url`, so the
    # token is never written into .git/config (set-url would persist it on disk).
    git push "$PUSH_URL" "HEAD:${GIT_BRANCH}"
    unset PUSH_URL
    ok "Pushed to ${GIT_REMOTE_URL} (${GIT_BRANCH})."
  fi
fi

echo
ok "Done. results/experiment_log.json and results/finetune_comparison_report.md are up to date locally."
