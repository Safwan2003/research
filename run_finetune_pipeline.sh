#!/usr/bin/env bash
# Fine-tuning comparison pipeline: rerun the frozen CheXpert eval (post
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
# Fine-tuning always trains on rows [0, N_FINETUNE_STUDIES). Evaluating a
# fine-tuned regime on the same rows would score it on its own training data,
# inflating every downstream quality metric -- so LoRA/QLoRA/full eval reads
# from just past that range instead (frozen eval, already done, is unaffected
# since it was never trained on anything and correctly used offset 0).
AGENTIC_OFFSET="$((N_FINETUNE_STUDIES + 100))"

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
banner "1/7 Installing peft/bitsandbytes"
pip install -r requirements.txt
ok "requirements.txt (incl. peft/bitsandbytes) installed."
step_done

# --- 2. Rerun the FROZEN baseline (post label-leakage fix) --------------------
banner "2/7 Frozen baseline (post leakage-fix rerun)"
if is_done finetune_frozen_rerun; then
  ok "Already reran the frozen baseline after the fix, skipping."
else
  python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC"
  mark_done finetune_frozen_rerun
fi
step_done "frozen baseline logged"

# --- 3. Train LoRA adapter -----------------------------------------------------
banner "3/7 Training LoRA adapter"
if is_done finetune_lora_train; then
  ok "LoRA adapter already trained at $LORA_DIR, skipping."
else
  python3 src/vlm/finetune_qwen.py --regime lora --n-studies "$N_FINETUNE_STUDIES" --epochs "$EPOCHS" \
    --output-dir "$LORA_DIR" --resume
  mark_done finetune_lora_train
fi
step_done "LoRA adapter ready"

# --- 4. Evaluate LoRA -----------------------------------------------------------
banner "4/7 Evaluating LoRA adapter"
if is_done finetune_lora_eval; then
  ok "LoRA already evaluated, skipping."
else
  python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
    --adapter-path "$LORA_DIR" --finetune-regime lora --agentic-offset "$AGENTIC_OFFSET"
  mark_done finetune_lora_eval
fi
step_done "LoRA evaluated"

# --- 5. Train + evaluate QLoRA ---------------------------------------------------
banner "5/7 Training + evaluating QLoRA adapter"
if is_done finetune_qlora_train; then
  ok "QLoRA adapter already trained at $QLORA_DIR, skipping."
else
  python3 src/vlm/finetune_qwen.py --regime qlora --n-studies "$N_FINETUNE_STUDIES" --epochs "$EPOCHS" \
    --output-dir "$QLORA_DIR" --resume
  mark_done finetune_qlora_train
fi
if is_done finetune_qlora_eval; then
  ok "QLoRA already evaluated, skipping."
else
  python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
    --adapter-path "$QLORA_DIR" --finetune-regime qlora --agentic-offset "$AGENTIC_OFFSET"
  mark_done finetune_qlora_eval
fi
step_done "QLoRA trained + evaluated"

# --- 6. Train + evaluate FULL fine-tuning -----------------------------------------
banner "6/7 Training + evaluating full fine-tuning"
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
      --output-dir "$FULL_FT_DIR" --resume
    mark_done finetune_full_train
  fi
  if is_done finetune_full_eval; then
    ok "Full-FT already evaluated, skipping."
  else
    python3 run_chexpert_eval.py --model qwen2-vl --n-ablation "$N_ABLATION" --n-agentic "$N_AGENTIC" \
      --model-path "$FULL_FT_DIR" --finetune-regime full --agentic-offset "$AGENTIC_OFFSET"
    mark_done finetune_full_eval
  fi
fi
step_done "full fine-tuning stage complete"

# --- 7. Compile the comparison report -------------------------------------------
banner "7/7 Compiling comparison report"
python3 scripts/compile_finetune_comparison.py
step_done "report written"

echo
ok "Done. results/finetune_comparison_report.md has the frozen/LoRA/QLoRA/full comparison table."
echo "Send that file (or results/experiment_log.json for the raw numbers) to your professor."
