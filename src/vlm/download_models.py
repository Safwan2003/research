"""
Pre-download HuggingFace model weights for Qwen2-VL into local cache (~.cache/huggingface).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def download_qwen_model(model_name: str = None):
    if model_name is None:
        model_name = config.VLM_MODEL_NAME

    print(f"Pre-downloading Qwen model weights for: '{model_name}'...")
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=model_name)
        print(f"Successfully cached '{model_name}' to: {path}")
    except ImportError:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        print(f"Downloading processor for '{model_name}'...")
        AutoProcessor.from_pretrained(model_name)
        print(f"Downloading model weights for '{model_name}'...")
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="cpu"
        )
        print(f"Successfully downloaded '{model_name}'.")


if __name__ == "__main__":
    download_qwen_model("Qwen/Qwen2-VL-2B-Instruct")
