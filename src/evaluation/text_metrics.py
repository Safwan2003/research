"""
Section 4.4 / Table 4: text generation quality metrics.

Computes ROUGE-1/2/L, SacreBLEU, and BERTScore-F1 comparing generated
impressions against ground-truth reports.

NOTE: ROUGE and SacreBLEU are lightweight and pip-installable with no model
download. BERTScore downloads a BERT model on first use (needs internet) --
everything else in this module works without it.

Install with:
    pip install rouge-score sacrebleu bert-score
"""


def compute_rouge(generated: str, reference: str) -> dict:
    """ROUGE-1 / ROUGE-2 / ROUGE-L F-measure between one generated/reference pair."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_sacrebleu(generated: str, reference: str) -> float:
    """SacreBLEU score between one generated/reference pair."""
    import sacrebleu

    bleu = sacrebleu.sentence_bleu(generated, [reference])
    return bleu.score


def compute_bertscore(generated_list: list, reference_list: list, model_type: str = "distilbert-base-uncased") -> dict:
    """
    BERTScore precision/recall/F1, batched over a list of generated/reference
    pairs (BERTScore is much more efficient run in batches than one at a time).
    Requires internet access to download the underlying BERT model.
    """
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(generated_list, reference_list, model_type=model_type, lang="en")
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def evaluate_batch(generated_list: list, reference_list: list, run_bertscore: bool = True) -> dict:
    """
    Full Table-4-style evaluation over a batch of (generated, reference) pairs.

    Returns dict with mean rouge1/rouge2/rougeL, mean sacrebleu, and
    (optionally) bertscore f1.
    """
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    bleu_total = 0.0
    n = len(generated_list)

    for gen, ref in zip(generated_list, reference_list):
        r = compute_rouge(gen, ref)
        for k in rouge_totals:
            rouge_totals[k] += r[k]
        bleu_total += compute_sacrebleu(gen, ref)

    results = {k: v / n for k, v in rouge_totals.items()}
    results["sacrebleu"] = bleu_total / n

    if run_bertscore:
        bert_results = compute_bertscore(generated_list, reference_list)
        results["bertscore_f1"] = bert_results["f1"]

    return results


if __name__ == "__main__":
    generated = ["No acute cardiopulmonary abnormality identified."]
    reference = ["No focal consolidation, pleural effusion, or pneumothorax."]

    print("ROUGE:", compute_rouge(generated[0], reference[0]))
    print("SacreBLEU:", compute_sacrebleu(generated[0], reference[0]))
    print("\n(Skipping BERTScore in this self-test -- it needs to download a "
          "model, run evaluate_batch(..., run_bertscore=True) once you have "
          "internet access in your actual environment.)")
