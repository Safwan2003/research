#!/usr/bin/env bash
# Master pipeline: GPU check -> Python env + libraries -> download CheXpert
# -> pick a model (Qwen2-VL or LLaVA-1.5-7B) -> run the CheXpert evaluation
# -> results appended to results/experiment_log.json.
#
# Safe to re-run: each step is skipped if already done (markers in
# .setup_markers/), so re-running after an interruption picks up where it
# left off instead of redoing everything.
#
# Run from inside WSL Ubuntu:  bash run_master.sh
# Override defaults via env vars, e.g.:
#   MODEL=llava-1.5-7b N_ABLATION=200 N_AGENTIC=20 bash run_master.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MARKERS_DIR=".setup_markers"
mkdir -p "$MARKERS_DIR"

# --- visual helpers -----------------------------------------------------
BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'

STEP_START=0
banner()  { STEP_START=$(date +%s); echo; echo -e "${BOLD}${CYAN}==> $1${RESET}"; }
ok()      { echo -e "  ${GREEN}[OK]${RESET} $1"; }
info()    { echo -e "  ${DIM}...${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}[!]${RESET} $1"; }
fail()    { echo -e "  ${RED}[FAIL]${RESET} $1" >&2; }
step_done() {
  local elapsed=$(( $(date +%s) - STEP_START ))
  echo -e "  ${GREEN}[DONE]${RESET} ${1:-step complete} ${DIM}(${elapsed}s)${RESET}"
}
is_done()   { [ -f "$MARKERS_DIR/$1.done" ]; }
mark_done() { touch "$MARKERS_DIR/$1.done"; }

trap 'fail "run_master.sh aborted (line $LINENO) -- re-run to resume from where it left off."' ERR

echo -e "${BOLD}Context-Aligned Medical VLM -- master pipeline${RESET}"
echo -e "${DIM}$(date)${RESET}"

# --- 1. GPU check ------------------------------------------------------------
banner "1/6 Checking NVIDIA GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found."
  echo "  If you're in WSL: this usually means the WINDOWS-side NVIDIA driver" >&2
  echo "  needs updating (CUDA-on-WSL support requires a recent driver) -- fix" >&2
  echo "  that on the Windows side, don't try to install a Linux GPU driver here." >&2
  exit 1
fi
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)"
ok "GPU visible: $GPU_INFO"
step_done

# --- 2. Python environment + libraries ---------------------------------------
banner "2/6 Python environment + libraries"
if ! is_done venv; then
  info "Creating .venv ..."
  python3 -m venv .venv
  mark_done venv
else
  info ".venv already created, skipping creation."
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Skip the (slow) pip install if requirements.txt hasn't changed since the
# last successful run -- pip itself would no-op too, but hashing is faster
# than letting pip re-resolve/re-check ~20 packages every single run.
REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
REQ_HASH_FILE="$MARKERS_DIR/requirements.sha256"
if [ -f "$REQ_HASH_FILE" ] && [ "$(cat "$REQ_HASH_FILE")" = "$REQ_HASH" ]; then
  info "requirements.txt unchanged since last install -- skipping pip install."
else
  info "Installing/upgrading pip + requirements.txt ..."
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
  ok "All packages from requirements.txt installed."
fi

info "Checking torch sees the GPU ..."
python3 -c "
import torch, sys
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else 'no GPU visible to torch'
print(('  [OK] ' if ok else '  [FAIL] ') + f'CUDA available: {ok} - {name}')
sys.exit(0 if ok else 1)
" || {
  fail "torch has no CUDA GPU available. Reinstall torch with a CUDA build matching"
  echo "  your driver (see pytorch.org's install selector) rather than continuing on CPU." >&2
  exit 1
}
step_done "environment ready"

# --- 3. CheXpert dataset (smart, disk-aware download) ------------------------
# NOTE: this deliberately does NOT `azcopy sync` the full ~471GB CheXpert
# release -- only batch 1 (~486MB, csvs + validation images) is fetched here.
# Growing beyond that is scripts/download_chexpert_subset.py's job (HTTP range
# requests into the remote zip, fetching only the N studies you ask for) --
# see step 3's N_TRAIN_EXTRA handling below and COMMANDS.txt section 2b.
banner "3/6 CheXpert dataset"
mkdir -p data/chexpert

if [ ! -f secrets/chexpert_sas_url.txt ]; then
  fail "Missing secrets/chexpert_sas_url.txt."
  echo "  Create it first: mkdir -p secrets && printf '%s' 'YOUR_SAS_URL' > secrets/chexpert_sas_url.txt" >&2
  exit 1
fi
ok "SAS URL file found."

# The AIMI container ships CheXpert as a handful of zip blobs, not per-patient
# folders -- "batch 1 (validate & csv)" is small (~486MB) and contains BOTH
# csv files plus every real validation-set image (234 studies), so it's
# downloaded in full. The training batches are 91-185GB each; those are
# fetched on demand, study-by-study, via scripts/download_chexpert_subset.py
# (HTTP range requests into the remote zip -- never downloaded whole).
BATCH1_BLOB_ENCODED="CheXpert-v1.0%20batch%201%20%28validate%20%26%20csv%29.zip"

if is_done chexpert_batch1; then
  ok "batch-1 (validate & csv) already downloaded+extracted, skipping."
else
  SAS_URL="$(tr -d '\r\n\000\377\376' < secrets/chexpert_sas_url.txt | xargs)"
  BASE_URL="${SAS_URL%%\?*}"
  QUERY="${SAS_URL#*\?}"
  BLOB_URL="${BASE_URL}/${BATCH1_BLOB_ENCODED}?${QUERY}"

  info "Downloading CheXpert-v1.0 batch 1 (validate & csv).zip (~486MB) ..."
  azcopy copy "$BLOB_URL" "data/chexpert/batch1.zip"
  ok "Downloaded."

  info "Extracting train.csv, valid.csv, and validation images ..."
  python3 scripts/extract_chexpert_batch1.py
  mark_done chexpert_batch1
fi

N_VALID=$(($(wc -l < data/chexpert/valid.csv) - 1))
ok "valid.csv ready: $N_VALID real studies with images on disk (config.py points here by default)."

if [ "${N_TRAIN_EXTRA:-0}" -gt 0 ]; then
  info "N_TRAIN_EXTRA=$N_TRAIN_EXTRA set -- fetching that many train.csv studies via smart partial download ..."
  info "(This can take a while -- roughly a few seconds per image. Safe to Ctrl-C and re-run to resume.)"
  python3 scripts/download_chexpert_subset.py --n-studies "$N_TRAIN_EXTRA"
  warn "Remember: switch CHEXPERT_CSV_PATH in config.py to data/chexpert/train.csv to actually use these."
else
  info "N_TRAIN_EXTRA not set (or 0) -- staying on the $N_VALID-study valid.csv subset."
  info "To grow toward the paper's 1000-study target later, run e.g.:"
  info "  python3 scripts/download_chexpert_subset.py --n-studies 1000"
  info "(run that in a SEPARATE terminal/venv shell while this one evaluates -- see safwan-prompt.txt)"
fi
step_done "dataset ready"

# --- 4. Model selection -------------------------------------------------------
banner "4/6 Model selection"
MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
  echo "  Which model?"
  echo "    [1] Qwen2-VL"
  echo "    [2] LLaVA-1.5-7B"
  read -rp "  > " choice
  case "$choice" in
    1) MODEL="qwen2-vl" ;;
    2) MODEL="llava-1.5-7b" ;;
    *) fail "Invalid choice: $choice"; exit 1 ;;
  esac
fi
ok "Using model: $MODEL"
step_done

# --- 5. Run the CheXpert evaluation pipeline ---------------------------------
banner "5/6 Running CheXpert evaluation ($MODEL)"
info "n-ablation=${N_ABLATION:-1000}  n-agentic=${N_AGENTIC:-50}"
python3 run_chexpert_eval.py \
  --model "$MODEL" \
  --n-ablation "${N_ABLATION:-1000}" \
  --n-agentic "${N_AGENTIC:-50}"
step_done "evaluation complete"

# --- 6. Done -------------------------------------------------------------------
banner "6/6 Done"
ok "Results appended to results/experiment_log.json"
echo
echo "Next: git pull origin main (pick up anything the other person logged)," \
     "then commit + push results/experiment_log.json and any code changes."
