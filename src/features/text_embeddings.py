"""
Text embedding extraction for report text -- the "text" modality in the
paper's Table 1 ablation (radiomics vs XAI vs text-embeddings vs combinations).

The paper compares "Qwen" vs "BLIP" embedding sources for this column, but
building bespoke embedding extraction from each VLM's own encoder is a much
bigger undertaking than this project needs to get a working ablation
pipeline. CLAUDE.md's own "Known gaps" section suggests sentence-transformers
as a stand-in ("e.g. sentence-transformers") -- that's what this module uses.
Tag config_snapshot["text_embedding_source"] with whichever you actually use
so it's clear this is an approximation of the paper's exact Qwen/BLIP columns.
"""

import numpy as np


def extract_text_embeddings(texts: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode a list of report strings into fixed-size sentence embeddings.

    Requires: pip install sentence-transformers (downloads the model on
    first use -- needs internet).

    Args:
        texts: list of report strings (or CheXpert pseudo-reports).
        model_name: any sentence-transformers model name.

    Returns:
        (N, D) numpy array of embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


if __name__ == "__main__":
    print(
        "This module needs `sentence-transformers` + internet to download the "
        "model on first use, so it can't self-test in a no-internet sandbox.\n"
        "Run it in your GPU/internet environment, e.g.:\n\n"
        "    from text_embeddings import extract_text_embeddings\n"
        "    embeddings = extract_text_embeddings([\n"
        "        'No focal consolidation, pleural effusion, or pneumothorax.',\n"
        "        'Cardiomegaly present. Pleural Effusion uncertain.',\n"
        "    ])\n"
        "    print(embeddings.shape)  # (2, 384) for all-MiniLM-L6-v2\n"
    )
