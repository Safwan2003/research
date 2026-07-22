"""
Vocabulary-grounded semantic feature extraction (F_voc in the paper).

Implements Section 3.5 "Vocabulary-Grounded Semantic Features":
  Given a curated set of radiology terms V, and a report R, extract
  F_voc = {v in V | v appears in R}   (Eq. 7)

These matched terms act as structured semantic anchors -- e.g. if the report
says "no pleural effusion", the model should be told explicitly that
"pleural effusion" is a recognized, negated clinical concept in this study,
rather than relying on the VLM to parse that out of free text on its own.
"""

import re
from pathlib import Path


DEFAULT_VOCAB_PATH = Path(__file__).parent.parent.parent / "vocabulary_list.txt"


def load_vocabulary(path: Path = DEFAULT_VOCAB_PATH) -> list:
    """Load the curated radiology vocabulary, one term per line."""
    with open(path, "r") as f:
        terms = [line.strip().lower() for line in f if line.strip()]
    # Sort longest-first so multi-word terms (e.g. "pleural effusion") are
    # matched before their substrings (e.g. "effusion") get a chance to.
    return sorted(set(terms), key=len, reverse=True)


def extract_vocabulary_features(report: str, vocabulary: list = None) -> dict:
    """
    Match curated radiology vocabulary terms against a report's free text.

    Args:
        report: free-text radiology report.
        vocabulary: list of terms to match against. If None, loads the
                    default curated list from vocabulary_list.txt.

    Returns:
        dict with:
          - "matched_terms": list of vocabulary terms found in the report
          - "num_matched_terms": count (a simple scalar feature for the
             ablation study's logistic regression classifier)
    """
    if vocabulary is None:
        vocabulary = load_vocabulary()

    text = report.lower()
    matched = []
    remaining = text

    for term in vocabulary:
        # Word-boundary match so "mass" doesn't match inside "massive".
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, remaining):
            matched.append(term)

    return {
        "matched_terms": matched,
        "num_matched_terms": len(matched),
    }


if __name__ == "__main__":
    sample_report = (
        "The lungs are clear without focal consolidation, pleural effusion, "
        "or pneumothorax. The cardiac silhouette is within normal limits. "
        "No acute osseous abnormality. Mild cardiomegaly is noted."
    )

    result = extract_vocabulary_features(sample_report)
    print(f"Matched {result['num_matched_terms']} vocabulary terms:")
    for term in result["matched_terms"]:
        print(f"  - {term}")
