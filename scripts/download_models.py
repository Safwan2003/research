"""
Script to pre-download VLM models and chest X-ray classifier weights into local Hugging Face / Torch caches.
"""

import sys
from pathlib import Path


def download_models():
    print("Pre-downloading model weights into local cache...")

    # 1. Download huggingface hub models
    try:
        from huggingface_hub import snapshot_download
        print("\n--- Downloading LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) ---")
        snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf")
        print("LLaVA-1.5-7B download complete.")

        print("\n--- Downloading Qwen2-VL-2B-Instruct ---")
        snapshot_download(repo_id="Qwen/Qwen2-VL-2B-Instruct")
        print("Qwen2-VL-2B-Instruct download complete.")
    except Exception as e:
        print(f"HuggingFace model download error: {e}")

    # 2. Download torchxrayvision model weights
    try:
        import torchxrayvision as xrv
        print("\n--- Downloading TorchXRayVision DenseNet weights ---")
        _ = xrv.models.DenseNet(weights="densenet121-res224-all")
        print("TorchXRayVision DenseNet weights download complete.")
    except Exception as e:
        print(f"TorchXRayVision download error: {e}")


if __name__ == "__main__":
    download_models()
