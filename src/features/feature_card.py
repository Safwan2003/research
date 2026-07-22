"""
Stage 2: Context Serialization and Multimodal Fusion.

Implements Eq. (8): F_tool = Serialize(F_rad, F_xai, F_voc)

This module turns the three separate feature dicts (radiomics, XAI, vocabulary)
into ONE compact "feature card" -- a JSON object that gets injected into the
VLM's prompt as extra, structured context. This is literally what makes the
model "context-driven" instead of just seeing a raw image and report.
"""

import json


def build_feature_card(radiomics: dict, xai: dict, vocabulary: dict) -> dict:
    """
    Combine the three tool outputs into a single structured feature card.

    Args:
        radiomics: output of features.radiomics.extract_radiomics()
        xai: output of features.xai_gradcam.derive_spatial_statistics()
        vocabulary: output of features.vocabulary.extract_vocabulary_features()

    Returns:
        A nested dict, ready to be dumped to JSON and inserted into a prompt.
    """
    return {
        "radiomics": {
            "mean_intensity": round(radiomics.get("mean", 0.0), 3),
            "intensity_variance": round(radiomics.get("variance", 0.0), 3),
            "texture_contrast": round(radiomics.get("glcm_contrast", 0.0), 3),
            "texture_homogeneity": round(radiomics.get("glcm_homogeneity", 0.0), 3),
            "lbp_entropy": round(radiomics.get("lbp_entropy", 0.0), 3),
        },
        "explainability": {
            "attention_mean": round(xai.get("xai_mean", 0.0), 3),
            "attention_max": round(xai.get("xai_max", 0.0), 3),
            "attention_entropy": round(xai.get("xai_entropy", 0.0), 3),
            "attention_top10pct_mass": round(xai.get("xai_top10pct_mass", 0.0), 3),
        },
        "vocabulary": {
            "matched_clinical_terms": vocabulary.get("matched_terms", []),
            "num_matched_terms": vocabulary.get("num_matched_terms", 0),
        },
    }


def feature_card_to_prompt_text(feature_card: dict) -> str:
    """
    Render the feature card as a compact JSON string suitable for insertion
    into a VLM chat prompt (see vlm/prompts.py).
    """
    return json.dumps(feature_card, indent=2)


if __name__ == "__main__":
    # Small end-to-end self-test wiring together all three feature modules.
    import numpy as np
    from radiomics import extract_radiomics
    from xai_gradcam import derive_spatial_statistics
    from vocabulary import extract_vocabulary_features

    rng = np.random.default_rng(1)
    synthetic_xray = rng.normal(120, 30, size=(256, 256)).clip(0, 255)
    synthetic_heatmap = rng.uniform(0.2, 0.9, size=(64, 64))
    sample_report = (
        "No focal consolidation, pleural effusion, or pneumothorax. "
        "Cardiac silhouette within normal limits."
    )

    rad = extract_radiomics(synthetic_xray)
    xai = derive_spatial_statistics(synthetic_heatmap)
    voc = extract_vocabulary_features(sample_report)

    card = build_feature_card(rad, xai, voc)
    print("Feature card (F_tool):")
    print(feature_card_to_prompt_text(card))
