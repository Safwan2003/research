"""
CheXpert evaluation entry point (current project focus, see CLAUDE.md Known
Gap #6 and the antigravity-prompt.txt / safwan-prompt.txt / ammar-prompt.txt
"CURRENT FOCUS" sections).

Runs, against real CheXpert data:
  1. Table-1-style ablation (radiomics / XAI / text-embeddings / combinations).
  2. Stepwise agentic reasoning (Y(0)/Y(1)/Y(2)) over a study subset using
     the chosen frozen VLM backbone (Qwen2-VL or LLaVA-1.5-7B -- weights are
     NEVER updated, matching the paper's f_theta; only the small logistic
     regression classifier in step 1 is actually "trained").
  3. Hallucination rate (Table 2), agentic-reasoning stats (Table 3), and
     text-quality + Responsible-AI metrics (Table 4) on that same subset.

Every result is appended to the shared results/experiment_log.json via
src/evaluation/results_store.py -- never overwritten.

Usage:
    python run_chexpert_eval.py --model qwen2-vl --n-ablation 1000 --n-agentic 50
    python run_chexpert_eval.py --model llava-1.5-7b
    python run_chexpert_eval.py --synthetic   # dry run, no GPU/data needed --
                                               # verifies the whole plumbing
                                               # + logging path end to end.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "data"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "features"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "vlm"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "ablation"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "evaluation"))

import config
from feature_card import build_feature_card, feature_card_to_prompt_text
from results_store import append_run, new_run_record

QUESTION = "Is there evidence of active cardiopulmonary abnormality?"


class _SyntheticReasoner:
    """
    Stand-in for QwenVLReasoner/LlavaVLReasoner used only in --synthetic mode,
    so the metrics-computation and results-logging plumbing can be verified
    end to end without a GPU or downloaded model weights.
    """

    def __init__(self, seed: int = 0):
        import numpy as np
        self._rng = np.random.default_rng(seed)

    def _fake_structured_output(self, uncertainty_bias: float) -> dict:
        u = float(min(1.0, max(0.0, self._rng.normal(uncertainty_bias, 0.1))))
        return {
            "impression": "No acute cardiopulmonary abnormality.",
            "evidence": "Lungs clear. Cardiac silhouette normal.",
            "uncertainty": u,
            "limitations": "Single frontal radiograph without prior comparison.",
            "safety_note": "For research use only; not a substitute for expert interpretation.",
        }

    def stepwise_agentic_reasoning(self, image_path, question, report_text, radiomics_json, feature_card_json) -> dict:
        return {
            "step0": "Findings consistent with pneumonia and mild cardiomegaly.",  # deliberately hallucination-prone
            "step1": self._fake_structured_output(uncertainty_bias=0.70),
            "step2": self._fake_structured_output(uncertainty_bias=0.68),
        }


def _get_reasoner(model_family: str, synthetic: bool):
    if synthetic:
        return _SyntheticReasoner()
    if model_family == "qwen2-vl":
        from reasoning import QwenVLReasoner
        return QwenVLReasoner(model_name=config.VLM_MODEL_NAME, device=config.VLM_DEVICE)
    elif model_family == "llava-1.5-7b":
        from reasoning_llava import LlavaVLReasoner
        return LlavaVLReasoner(device=config.VLM_DEVICE)
    raise ValueError(f"Unknown model_family: {model_family}")


def _load_real_studies(n: int, uncertain_policy: str):
    from dataset import load_chexpert_dataset

    csv_path = config.CHEXPERT_CSV_PATH
    valid_csv = config.CHEXPERT_IMAGES_ROOT / "valid.csv"
    if valid_csv.exists():
        csv_path = valid_csv

    return load_chexpert_dataset(
        csv_path=str(csv_path),
        images_root=str(config.CHEXPERT_IMAGES_ROOT),
        limit=n,
        uncertain_policy=uncertain_policy,
    )


def _extract_features_real(study):
    """Radiomics + XAI + vocabulary for one real study. Returns (radiomics, xai, vocab, image_array)."""
    import numpy as np
    from PIL import Image
    from radiomics import extract_radiomics
    from vocabulary import extract_vocabulary_features

    image = np.array(Image.open(study.frontal_image_path).convert("L"))
    radiomics = extract_radiomics(image)
    vocab = extract_vocabulary_features(study.report_text)

    try:
        from xai_gradcam import (
            GradCAM, derive_spatial_statistics,
            load_torchxrayvision_classifier, preprocess_for_torchxrayvision,
        )
        model, target_layer = load_torchxrayvision_classifier()
        cam = GradCAM(model, target_layer)
        tensor = preprocess_for_torchxrayvision(image)
        heatmap = cam(tensor)
        xai = derive_spatial_statistics(heatmap)
        xai["_placeholder"] = False
    except Exception as e:
        print(f"  WARNING: real GradCAM unavailable ({e}); using placeholder XAI stats. "
              f"Install torchxrayvision (`pip install torchxrayvision`) to fix this.")
        rng = np.random.default_rng(0)
        xai = derive_spatial_statistics(rng.uniform(0.2, 0.8, size=(32, 32)))
        xai["_placeholder"] = True

    return radiomics, xai, vocab, image


def run_ablation_on_chexpert(studies: list, model_family: str, synthetic: bool):
    import numpy as np
    from ablation_study import run_ablation, print_ablation_table

    if synthetic:
        rng = np.random.default_rng(config.RANDOM_SEED)
        n = len(studies)
        labels = np.array([s["label"] for s in studies])
        text_embeddings = rng.normal(0, 1, size=(n, 16)) + labels[:, None] * 2.0
        radiomics_features = rng.normal(0, 1, size=(n, 8)) + labels[:, None] * 0.3
        xai_features = rng.normal(0, 1, size=(n, 4)) + labels[:, None] * 0.2
        text_embedding_source = "synthetic"
    else:
        from text_embeddings import extract_text_embeddings

        radiomics_list, xai_list, report_texts, labels = [], [], [], []
        for study in studies:
            radiomics, xai, _vocab, _image = _extract_features_real(study)
            radiomics_list.append(list({k: v for k, v in radiomics.items() if isinstance(v, (int, float))}.values()))
            xai_list.append([xai["xai_mean"], xai["xai_max"], xai["xai_entropy"], xai["xai_top10pct_mass"]])
            report_texts.append(study.report_text)
            labels.append(study.label)

        radiomics_features = np.array(radiomics_list)
        xai_features = np.array(xai_list)
        text_embeddings = extract_text_embeddings(report_texts)
        labels = np.array(labels)
        text_embedding_source = "sentence-transformers/all-MiniLM-L6-v2"

    results = run_ablation(radiomics_features, xai_features, text_embeddings, labels, random_state=config.RANDOM_SEED)
    print_ablation_table(results)

    run_record = new_run_record(
        mode="ablation",
        config_snapshot={
            "seed": config.RANDOM_SEED, "n_studies": len(studies), "dataset": "chexpert",
            "model_family": model_family, "text_embedding_source": text_embedding_source,
            "synthetic": synthetic,
        },
        results={"ablation": results},
    )
    append_run(config.RESULTS_LOG_PATH, run_record)
    print(f"Logged ablation run: {run_record['run_id']}")


def run_agentic_on_chexpert(studies: list, model_family: str, synthetic: bool):
    import sys as _sys
    sys.path.insert(0, str(Path(__file__).parent / "src" / "evaluation"))
    from hallucination import average_hallucination_rate
    from text_metrics import evaluate_batch
    from responsible_ai_metrics import compute_agentic_reasoning_stats, compute_responsible_ai_indicators

    reasoner = _get_reasoner(model_family, synthetic)

    step0_texts, step1_outputs, step2_outputs, ground_truths = [], [], [], []

    for study in studies:
        if synthetic:
            radiomics, xai, vocab, image_path = {"mean": 0.0}, {"xai_mean": 0.0, "xai_max": 0.0, "xai_entropy": 0.0, "xai_top10pct_mass": 0.0}, {"matched_terms": [], "num_matched_terms": 0}, "synthetic.png"
            report_text = study["report_text"]
        else:
            radiomics, xai, vocab, _image = _extract_features_real(study)
            image_path = study.frontal_image_path
            report_text = study.report_text

        card = build_feature_card(radiomics, xai, vocab)
        card_json = feature_card_to_prompt_text(card)
        radiomics_json = json.dumps({k: v for k, v in radiomics.items() if isinstance(v, (int, float))})

        result = reasoner.stepwise_agentic_reasoning(image_path, QUESTION, report_text, radiomics_json, card_json)

        step0_texts.append(result["step0"])
        step1_outputs.append(result["step1"])
        step2_outputs.append(result["step2"])
        ground_truths.append(report_text)

    def _to_text(structured_or_str):
        if isinstance(structured_or_str, str):
            return structured_or_str
        return f"{structured_or_str.get('impression', '')} {structured_or_str.get('evidence', '')}"

    hr_baseline = average_hallucination_rate(step0_texts, ground_truths)
    hr_context = average_hallucination_rate([_to_text(o) for o in step2_outputs], ground_truths)

    valid_step1 = [o for o in step1_outputs if isinstance(o, dict) and "uncertainty" in o]
    valid_step2 = [o for o in step2_outputs if isinstance(o, dict) and "uncertainty" in o]
    n_parse_failures = (len(step1_outputs) - len(valid_step1)) + (len(step2_outputs) - len(valid_step2))
    if n_parse_failures:
        print(f"  WARNING: {n_parse_failures} structured outputs failed to parse as JSON and were excluded from Table 3/4 stats.")

    agentic_stats = {
        "step0_vs_step1": {
            "step1": compute_agentic_reasoning_stats(valid_step1) if valid_step1 else None,
        },
        "step2": compute_agentic_reasoning_stats(valid_step2) if valid_step2 else None,
    }

    text_metrics_results = None
    responsible_ai_results = None
    if valid_step2:
        text_metrics_results = evaluate_batch(
            [_to_text(o) for o in step2_outputs], ground_truths, run_bertscore=not synthetic,
        )
        responsible_ai_results = compute_responsible_ai_indicators(valid_step2)

    base_config = {
        "seed": config.RANDOM_SEED, "n_studies": len(studies), "dataset": "chexpert",
        "model_family": model_family, "synthetic": synthetic,
    }

    append_run(config.RESULTS_LOG_PATH, new_run_record(
        mode="hallucination",
        config_snapshot=base_config,
        results={"hallucination": {"Image + Text (baseline Y0)": hr_baseline, "Image + Text + Radiomics + XAI (Y2)": hr_context}},
    ))
    append_run(config.RESULTS_LOG_PATH, new_run_record(
        mode="agentic_reasoning", config_snapshot=base_config, results={"agentic_reasoning": agentic_stats},
    ))
    if text_metrics_results is not None:
        append_run(config.RESULTS_LOG_PATH, new_run_record(
            mode="text_metrics", config_snapshot=base_config,
            results={"text_metrics": {**text_metrics_results, **responsible_ai_results}},
        ))

    print(f"Hallucination rate: baseline={hr_baseline:.3f}, context-driven={hr_context:.3f}")
    print(f"Agentic reasoning stats: {agentic_stats}")
    if text_metrics_results is not None:
        print(f"Text metrics + Responsible-AI indicators: {text_metrics_results}, {responsible_ai_results}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["qwen2-vl", "llava-1.5-7b"], help="Frozen VLM backbone to use")
    parser.add_argument("--n-ablation", type=int, default=config.N_STUDIES_ABLATION)
    parser.add_argument("--n-agentic", type=int, default=config.N_STUDIES_AGENTIC)
    parser.add_argument("--uncertain-policy", choices=["u_zeros", "u_ones"], default="u_zeros")
    parser.add_argument("--synthetic", action="store_true", help="Dry run with synthetic data, no GPU/CheXpert download needed")
    args = parser.parse_args()

    if not args.synthetic and not args.model:
        parser.error("--model is required unless --synthetic is set")

    model_family = args.model or "qwen2-vl"

    print(f"=== CheXpert evaluation: model={model_family}, synthetic={args.synthetic} ===\n")

    if args.synthetic:
        import numpy as np
        rng = np.random.default_rng(config.RANDOM_SEED)
        n = max(args.n_ablation, args.n_agentic)
        studies = [
            {"study_id": str(i), "label": int(rng.integers(0, 2)), "report_text": "No Finding." if i % 2 == 0 else "Cardiomegaly present."}
            for i in range(n)
        ]
        ablation_studies = studies[: args.n_ablation]
        agentic_studies = studies[: args.n_agentic]
    else:
        studies = _load_real_studies(n=max(args.n_ablation, args.n_agentic), uncertain_policy=args.uncertain_policy)
        ablation_studies = studies[: args.n_ablation]
        agentic_studies = studies[: args.n_agentic]

    print(f"\n--- Ablation study ({len(ablation_studies)} studies) ---")
    run_ablation_on_chexpert(ablation_studies, model_family, args.synthetic)

    print(f"\n--- Agentic reasoning / hallucination / text metrics ({len(agentic_studies)} studies) ---")
    run_agentic_on_chexpert(agentic_studies, model_family, args.synthetic)

    print(f"\nAll results appended to {config.RESULTS_LOG_PATH}")


if __name__ == "__main__":
    main()
