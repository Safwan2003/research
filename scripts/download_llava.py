"""
Dedicated script to download LLaVA-1.5-7B weights into local Hugging Face cache for Ammar's track.
"""

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    print("Downloading LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) model weights into local Hugging Face cache...")
    snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf")
    print("LLaVA-1.5-7B model weights successfully cached locally.")
