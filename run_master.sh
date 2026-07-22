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

step()       { echo; echo "=== $1 ==="; }
is_done()    { [ -f "$MARKERS_DIR/$1.done" ]; }
mark_done()  { touch "$MARKERS_DIR/$1.done"; }

# --- 1. GPU check ------------------------------------------------------------
step "1/6 Checking NVIDIA GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found." >&2
  echo "If you're in WSL: this usually means the WINDOWS-side NVIDIA driver" >&2
  echo "needs updating (CUDA-on-WSL support requires a recent driver) -- fix" >&2
  echo "that on the Windows side, don't try to install a Linux GPU driver here." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# --- 2. Python environment + libraries ---------------------------------------
step "2/6 Python environment + libraries"
if ! is_done venv; then
  python3 -m venv .venv
  mark_done venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
REQ_HASH_FILE="$MARKERS_DIR/requirements.sha256"
if [ -f "$REQ_HASH_FILE" ] && [ "$(cat "$REQ_HASH_FILE")" = "$REQ_HASH" ]; then
  echo "requirements.txt unchanged since last install -- skipping pip install."
else
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
fi

python3 -c "
import torch, sys
ok = torch.cuda.is_available()
print('CUDA available:', ok, '-', torch.cuda.get_device_name(0) if ok else 'no GPU visible to torch')
sys.exit(0 if ok else 1)
" || {
  echo "torch has no CUDA GPU available. Reinstall torch with a CUDA build matching" >&2
  echo "your driver (see pytorch.org's install selector) rather than continuing on CPU." >&2
  exit 1
}

# --- 3. CheXpert dataset (idempotent, resumable) -----------------------------
step "3/6 CheXpert dataset"
if is_done chexpert_download; then
  echo "Already fully downloaded (per $MARKERS_DIR/chexpert_download.done) --" \
       "skipping entirely, no network call made."
  echo "Delete that marker file to force a re-check."
else
  if [ ! -f secrets/chexpert_sas_url.txt ]; then
    echo "Missing secrets/chexpert_sas_url.txt." >&2
    echo "Create it first: mkdir -p secrets && echo 'YOUR_SAS_URL' > secrets/chexpert_sas_url.txt" >&2
    exit 1
  fi

  if ! command -v azcopy >/dev/null 2>&1; then
    echo "Installing azcopy..."
    curl -sL https://aka.ms/downloadazcopy-v10-linux | tar -xz -C /tmp
    sudo cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
  fi

  mkdir -p data/chexpert
  AVAIL_GB=$(df --output=avail -B1G "$(pwd)/data" | tail -1 | tr -d ' ')
  if [ "$AVAIL_GB" -lt 480 ]; then
    echo "WARNING: only ${AVAIL_GB}GB free on this drive -- the full CheXpert" >&2
    echo "release is ~471GB. Free up space or point data/chexpert at a bigger" >&2
    echo "disk (e.g. a symlink to a Windows drive mounted under /mnt) before" >&2
    echo "continuing." >&2
    read -rp "Continue anyway? [y/N] " confirm
    [ "$confirm" = "y" ] || exit 1
  fi

  # `sync` (not `copy`) is what makes this genuinely resumable: if this step
  # gets interrupted partway (closed terminal, network drop, laptop sleep),
  # re-running the whole script re-enters this branch (marker was never
  # written) and `sync` compares size/last-modified against what's already in
  # data/chexpert, re-fetching only files that are missing or incomplete --
  # it does NOT restart the whole 471GB from zero. --delete-destination=false
  # guarantees it will never remove anything already downloaded.
  echo "Syncing CheXpert into data/chexpert (resumable -- already-downloaded" \
       "files are left alone; only new/missing/incomplete ones are transferred)..."
  azcopy sync "$(cat secrets/chexpert_sas_url.txt)" "data/chexpert" \
    --recursive=true --delete-destination=false
  mark_done chexpert_download
fi

# --- 4. Model selection -------------------------------------------------------
step "4/6 Model selection"
MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
  echo "Which model?"
  echo "  [1] Qwen2-VL"
  echo "  [2] LLaVA-1.5-7B"
  read -rp "> " choice
  case "$choice" in
    1) MODEL="qwen2-vl" ;;
    2) MODEL="llava-1.5-7b" ;;
    *) echo "Invalid choice: $choice" >&2; exit 1 ;;
  esac
fi
echo "Using model: $MODEL"

# --- 5. Run the CheXpert evaluation pipeline ---------------------------------
step "5/6 Running CheXpert evaluation ($MODEL)"
python3 run_chexpert_eval.py \
  --model "$MODEL" \
  --n-ablation "${N_ABLATION:-1000}" \
  --n-agentic "${N_AGENTIC:-50}"

# --- 6. Done -------------------------------------------------------------------
step "6/6 Done"
echo "Results appended to results/experiment_log.json"
echo
echo "Next: git pull origin main (pick up anything the other person logged)," \
     "then commit + push results/experiment_log.json and any code changes."
