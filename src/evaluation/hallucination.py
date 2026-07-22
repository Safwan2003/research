"""
Section 4.4 / Table 2: Hallucination Rate (HR) evaluation.

HR = average number of hallucinated clinical keywords per study, where a
"hallucinated keyword" is a clinical vocabulary term that appears in the
model's GENERATED output but does NOT appear in the ground-truth report.

This gives you a cheap, reproducible proxy for "did the model invent a
finding that isn't actually supported by the report" -- exactly what the
paper uses to show context alignment reduces hallucination (1.14 -> 0.25).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))
from vocabulary import load_vocabulary  # noqa: E402


def _find_terms(text: str, vocabulary: list) -> set:
    text = text.lower()
    found = set()
    for term in vocabulary:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            found.add(term)
    return found


def hallucination_rate(generated_text: str, ground_truth_report: str, vocabulary: list = None) -> dict:
    """
    Compute hallucinated keyword count for a single study.

    Args:
        generated_text: the model's generated impression/evidence text.
        ground_truth_report: the actual radiology report for that study.
        vocabulary: curated term list; loads the default if None.

    Returns:
        dict with:
          - "hallucinated_terms": terms mentioned by the model but not in ground truth
          - "hallucination_count": len(hallucinated_terms)  <- this is what gets averaged for HR
    """
    if vocabulary is None:
        vocabulary = load_vocabulary()

    generated_terms = _find_terms(generated_text, vocabulary)
    ground_truth_terms = _find_terms(ground_truth_report, vocabulary)

    hallucinated = generated_terms - ground_truth_terms

    return {
        "hallucinated_terms": sorted(hallucinated),
        "hallucination_count": len(hallucinated),
    }


def average_hallucination_rate(generated_texts: list, ground_truth_reports: list, vocabulary: list = None) -> float:
    """
    HR across a whole dataset/study set -- this is the number you compare
    against Table 2's 1.14 (image-only) vs 0.25 (image+text) etc.
    """
    if vocabulary is None:
        vocabulary = load_vocabulary()

    counts = [
        hallucination_rate(gen, gt, vocabulary)["hallucination_count"]
        for gen, gt in zip(generated_texts, ground_truth_reports)
    ]
    return sum(counts) / len(counts) if counts else 0.0


if __name__ == "__main__":
    ground_truth = "No focal consolidation, pleural effusion, or pneumothorax."

    # A cautious, well-grounded generated output -- should have ~0 hallucinations.
    good_output = "No acute cardiopulmonary abnormality. Lungs are clear."

    # An overconfident generated output that invents unsupported findings.
    bad_output = "Findings consistent with pneumonia and mild cardiomegaly with pulmonary edema."

    print("Good (grounded) output:", hallucination_rate(good_output, ground_truth))
    print("Bad (hallucinated) output:", hallucination_rate(bad_output, ground_truth))
