"""
Section 5.1 / Table 1: Multimodal ablation study.

Trains a simple logistic regression classifier on different combinations of
feature vectors (radiomics-only, XAI-only, text-only, and combinations) and
reports AUC for each -- reproducing Table 1 and Figure 2 of the paper.

This is pure scikit-learn and has NO dependency on the VLM at all -- it's a
classical ML experiment that quantifies "how much predictive signal does
each evidence source carry on its own, and does combining them help?"
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_ablation(
    radiomics_features: np.ndarray,
    xai_features: np.ndarray,
    text_embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Reproduce the modality-combination AUC table.

    Args:
        radiomics_features: (N, D_rad) array -- one radiomics feature vector per study.
        xai_features: (N, D_xai) array -- one XAI feature vector per study.
        text_embeddings: (N, D_text) array -- one text embedding per study
                         (e.g. sentence-transformers embedding of the report).
        labels: (N,) binary array -- ground truth label (e.g. "abnormal" vs "normal").
        n_folds: number of cross-validation folds.
        random_state: for reproducibility (paper uses a fixed seed too).

    Returns:
        dict mapping configuration name -> mean AUC across folds, e.g.
        {"Radiomics only": 0.55, "Text only": 0.92, ...}
    """
    configs = {
        "Radiomics only": radiomics_features,
        "XAI only": xai_features,
        "Text only": text_embeddings,
        "Radiomics + Text": np.concatenate([radiomics_features, text_embeddings], axis=1),
        "XAI + Text": np.concatenate([xai_features, text_embeddings], axis=1),
        "Radiomics + XAI + Text": np.concatenate(
            [radiomics_features, xai_features, text_embeddings], axis=1
        ),
    }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {}

    for name, X in configs.items():
        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        scores = cross_val_score(clf, X, labels, cv=cv, scoring="roc_auc")
        results[name] = float(np.mean(scores))

    return results


def print_ablation_table(results: dict):
    """Pretty-print results in the same shape as Table 1 in the paper."""
    print(f"{'Feature Configuration':<28} {'AUC':>6}")
    print("-" * 36)
    for name, auc in results.items():
        print(f"{name:<28} {auc:>6.3f}")


if __name__ == "__main__":
    # Synthetic self-test: builds fake feature vectors where text is the
    # strongest signal, radiomics/XAI are weak alone but slightly helpful
    # combined -- mirroring the qualitative pattern reported in Table 1.
    rng = np.random.default_rng(42)
    n = 300

    labels = rng.integers(0, 2, size=n)

    # Text embeddings: strongly correlated with the label (simulate signal).
    text_embeddings = rng.normal(0, 1, size=(n, 16)) + labels[:, None] * 2.0

    # Radiomics/XAI: mostly noise, weakly correlated with label.
    radiomics_features = rng.normal(0, 1, size=(n, 8)) + labels[:, None] * 0.3
    xai_features = rng.normal(0, 1, size=(n, 4)) + labels[:, None] * 0.2

    results = run_ablation(radiomics_features, xai_features, text_embeddings, labels)
    print_ablation_table(results)
