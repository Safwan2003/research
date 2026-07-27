"""
One-off helper: pre-download every model the paper actually used, into the
local Hugging Face cache, so later runs never need to download mid-run.

- Qwen2-VL-7B-Instruct: the paper's larger backbone, used for its Table 4 row.
  config.py's VLM_MODEL_NAME can be pointed at this to reproduce that row.
- llava-hf/llava-1.5-7b-hf: the paper's other VLM (Ammar's track,
  reasoning_llava.py already targets this checkpoint).
- Salesforce/blip-itm-base-coco: a BLIP checkpoint with a text encoder, as a
  best-effort match for the paper's "BLIP" text-embedding column in Table 1.
  NOTE: the paper doesn't name an exact BLIP checkpoint, and nothing in this
  codebase's ablation_study.py/run_chexpert_eval.py currently uses BLIP for
  embeddings (it always uses sentence-transformers) -- downloading this only
  caches the weights; wiring it into the ablation as an actual alternative
  text-embedding source is separate follow-up work, not done here.

Pure disk download via huggingface_hub -- does NOT load anything onto the
GPU, so safe to run alongside an active training/eval process.
"""

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "Salesforce/blip-itm-base-coco",
]

for repo_id in MODELS:
    print(f"\n=== Downloading {repo_id} ===")
    path = snapshot_download(repo_id=repo_id)
    print(f"Cached at: {path}")

print("\nAll done.")
