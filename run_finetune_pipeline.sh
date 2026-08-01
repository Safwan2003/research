#!/usr/bin/env bash
# Fine-tuning comparison pipeline: download a representative random sample
# (patient-level stratified, not the old "first N rows" bias -- see
# src/data/dataset.py / CLAUDE.md) -> rerun the frozen CheXpert eval (post
# label-leakage fix) -> train+evaluate LoRA -> train+evaluate QLoRA ->
# train+evaluate full fine-tuning -> compile one comparison report across
# all four regimes. Assumes run_master.sh has already been run at least once
# on this machine (venv + CheXpert data already present) -- this script only
# adds the peft/bitsandbytes deps and the fine-tuning steps on top.
#
# Safe to re-run: each step is skipped if already done (markers in
# .setup_markers/), same convention as run_master.sh.
#
# Run from inside WSL Ubuntu:  bash run_finetune_pipeline.sh
# Override defaults via env vars, e.g.:
#   N_ABLATION=1000 N_AGENTIC=50 N_FINETUNE_STUDIES=300 EPOCHS=1 bash run_finetune_pipeline.sh
#
# Full fine-tuning (RUN_FULL_FT=1 by default) trains EVERY parameter of the
# 2B model -- the most VRAM-hungry regime by far, even with gradient
# checkpointing + 8-bit Adam (see src/vlm/finetune_qwen.py's run_finetune
# docstring). If it OOMs on your GPU, re-run with RUN_FULL_FT=0 to skip it
# and still get the frozen/LoRA/QLoRA comparison:
#   RUN_FULL_FT=0 bash run_finetune_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MARKERS_DIR=".setup_markers"
mkdir -p "$MARKERS_DIR"

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

trap 'fail "run_finetune_pipeline.sh aborted (line $LINENO) -- re-run to resume from where it left off."' ERR

N_ABLATION="${N_ABLATION:-1000}"
N_AGENTIC="${N_AGENTIC:-50}"
N_FINETUNE_STUDIES="${N_FINETUNE_STUDIES:-300}"
EPOCHS="${EPOCHS:-1}"
RUN_FULL_FT="${RUN_FULL_FT:-1}"
LORA_DIR="models/lora_chexpert_2b"
QLORA_DIR="models/qlora_chexpert_2b"
FULL_FT_DIR="models/full_chexpert_2b"
# Fine-tuning trains on a patient-level random sample covering row-budget
# [0, N_FINETUNE_STUDIES) of the canonical sample order (see
# src/data/dataset.py's sample_chexpert_patients/load_chexpert_dataset).
# run_chexpert_eval.py's LoRA/QLoRA/full eval steps read the
# train_patient_ids.json sidecar each training run writes and exclude
# EXACTLY those patients -- no offset guess/fudge-factor needed anymore
# (this used to be "+100" to approximate a margin; that's gone). This value
# is still passed as a fallback offset for the rare case no sidecar is found.
AGENTIC_OFFSET="$N_FINETUNE_STUDIES"

echo -e "${BOLD}Context-Aligned Medical VLM -- fine-tuning comparison pipeline${RESET}"
echo -e "${DIM}$(date)${RESET}"
echo -e "${DIM}n_ablation=$N_ABLATION n_agentic=$N_AGENTIC n_finetune_studies=$N_FINETUNE_STUDIES epochs=$EPOCHS${RESET}"

if [ ! -d .venv ]; then
  fail ".venv not found -- run run_master.sh (or run_master.bat from PowerShell) first."
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 1. Install fine-tuning deps (peft/bitsandbytes) --------------------------
banner "1/8 Installing peft/bitsandbytes"
pip install -r requirements.txt
ok "requirements.txt (incl. peft/bitsandbytes) installed."
step_done

# --- 2. Download a representative random sample --------------------------------
# Patient-level stratified-random (see src/data/dataset.py), not the old
# "first N rows of train.csv" bug -- that gave a narrow, single-patient-ID-
# range slice and let a patient's images straddle the train/held-out split.
# Covers: the ablation sample [0, N_ABLATION), the fine-tuning training
# slice [0, N_FINETUNE_STUDIES), and the held-out eval slice
# [N_FINETUNE_STUDIES, N_FINETUNE_STUDIES+N_AGENTIC) -- same seed/target-
# finding as config.py defaults throughout, so all three calls agree on one
# canonical patient order.
banner "2/8 Downloading representative random CheXpert sample"
if is_done finetune_download_sample; then
  ok "Random sample already downloaded, skipping."
else
  python3 scripts/download_chexpert_subset.py --n-studies "$N_ABLATION" --sample-strategy random
  python3 scripts/download_chexpert_subset.py --n-studies "$N_FINETUNE_STUDIES" --sample-strategy random
  python3 scripts/download_chexpert_subset.py --n-studies "$N_AGENTIC" --sample-strategy random --offset "$AGENTIC_OFFSET"
  mark_done finetune_download_sample
fi
step_done "sample present locally"

# --- 3. Rerun the FROZEN baseline (post label-leakage fix) --------------------
banner "3/8 Frozen baseline (post leakage-fix rerun)"
if is_done finetune_frozen_rerun; then
  ok "Already reran the frozen baseline after the fix, skipping."
else
  python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
    --sample-strategy random
  mark_done finetune_frozen_rerun
fi
step_done "frozen baseline logged"

# --- 4. Train LoRA adapter -----------------------------------------------------
banner "4/8 Training LoRA adapter"
if is_done finetune_lora_train; then
  ok "LoRA adapter already trained at $LORA_DIR, skipping."
else
  python3 src/vlm/finetune_qwen.py --regime lora --n-studies "$N_FINETUNE_STUDIES" --epochs "$EPOCHS" \
    --output-dir "$LORA_DIR" --sample-strategy random --resume
  mark_done finetune_lora_train
fi
step_done "LoRA adapter ready"

# --- 5. Evaluate LoRA -----------------------------------------------------------
banner "5/8 Evaluating LoRA adapter"
if is_done finetune_lora_eval; then
  ok "LoRA already evaluated, skipping."
else
  python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
    --adapter-path "$LORA_DIR" --finetune-regime lora --sample-strategy random --agentic-offset "$AGENTIC_OFFSET"
  mark_done finetune_lora_eval
fi
step_done "LoRA evaluated"

# --- 6. Train + evaluate QLoRA ---------------------------------------------------
banner "6/8 Training + evaluating QLoRA adapter"
if is_done finetune_qlora_train; then
  ok "QLoRA adapter already trained at $QLORA_DIR, skipping."
else
  python3 src/vlm/finetune_qwen.py --regime qlora --n-studies "$N_FINETUNE_STUDIES" --epochs "$EPOCHS" \
    --output-dir "$QLORA_DIR" --sample-strategy random --resume
  mark_done finetune_qlora_train
fi
if is_done finetune_qlora_eval; then
  ok "QLoRA already evaluated, skipping."
else
  python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
    --adapter-path "$QLORA_DIR" --finetune-regime qlora --sample-strategy random --agentic-offset "$AGENTIC_OFFSET"
  mark_done finetune_qlora_eval
fi
step_done "QLoRA trained + evaluated"

# --- 7. Train + evaluate FULL fine-tuning -----------------------------------------
banner "7/8 Training + evaluating full fine-tuning"
if [ "$RUN_FULL_FT" != "1" ]; then
  warn "RUN_FULL_FT=$RUN_FULL_FT -- skipping full fine-tuning. The comparison report will only"
  warn "have frozen/LoRA/QLoRA rows. Set RUN_FULL_FT=1 (default) to include it."
else
  warn "Full fine-tuning trains ALL ~2B parameters -- by far the most VRAM-hungry regime here."
  warn "If this OOMs, re-run with RUN_FULL_FT=0 to skip it and keep the other three regimes."
  if is_done finetune_full_train; then
    ok "Full-FT checkpoint already trained at $FULL_FT_DIR, skipping."
  else
    python3 src/vlm/finetune_qwen.py --regime full --n-studies "$N_FINETUNE_STUDIES" --epochs "$EPOCHS" \
      --output-dir "$FULL_FT_DIR" --sample-strategy random --resume
    mark_done finetune_full_train
  fi
  if is_done finetune_full_eval; then
    ok "Full-FT already evaluated, skipping."
  else
    python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
      --model-path "$FULL_FT_DIR" --finetune-regime full --sample-strategy random --agentic-offset "$AGENTIC_OFFSET"
    mark_done finetune_full_eval
  fi
fi
step_done "full fine-tuning stage complete"

# --- 8. Compile the comparison report -------------------------------------------
banner "8/8 Compiling comparison report"
python3 scripts/compile_finetune_comparison.py
step_done "report written"

echo
ok "Done. results/finetune_comparison_report.md has the frozen/LoRA/QLoRA/full comparison table."
echo "Send that file (or results/experiment_log.json for the raw numbers) to your professor."
