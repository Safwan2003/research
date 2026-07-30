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


def _get_reasoner(
    model_family: str, synthetic: bool, adapter_path: str = None,
    finetune_regime: str = "frozen", model_path: str = None,
):
    if synthetic:
        return _SyntheticReasoner()
    if model_family == "qwen2-vl":
        from reasoning import QwenVLReasoner
        # For "full" fine-tuning there's no separate frozen base + adapter --
        # model_path points straight at the standalone checkpoint
        # src/vlm/finetune_qwen.py saved (full model + processor). For
        # frozen/lora/qlora the base is always the paper's own checkpoint.
        base_model_name = model_path if finetune_regime == "full" else config.VLM_MODEL_NAME
        return QwenVLReasoner(
            model_name=base_model_name, device=config.VLM_DEVICE,
            adapter_path=adapter_path, load_in_4bit=(finetune_regime == "qlora"),
        )
    elif model_family == "llava-1.5-7b":
        if adapter_path or model_path:
            raise ValueError("LoRA/QLoRA/full fine-tuning are only wired up for qwen2-vl so far.")
        from reasoning_llava import LlavaVLReasoner
        return LlavaVLReasoner(device=config.VLM_DEVICE)
    raise ValueError(f"Unknown model_family: {model_family}")


def _load_real_studies(n: int, uncertain_policy: str, offset: int = 0):
    from dataset import load_chexpert_dataset

    # NOTE: this must NOT silently fall back to valid.csv (234 studies) --
    # an earlier version did exactly that whenever valid.csv happened to
    # exist on disk (which it always does once run_master.bat has run once),
    # so --n-ablation/--n-agentic were silently ignored and every real run
    # used 234 studies regardless of what was actually requested.
    # config.CHEXPERT_CSV_PATH is the single source of truth -- point it at
    # valid.csv yourself in config.py if that's genuinely what you want.
    return load_chexpert_dataset(
        csv_path=str(config.CHEXPERT_CSV_PATH),
        images_root=str(config.CHEXPERT_IMAGES_ROOT),
        limit=n,
        uncertain_policy=uncertain_policy,
        target_finding=config.CHEXPERT_TARGET_FINDING,
        offset=offset,
    )


def _progress(done: int, total: int, start_time, label: str = "", width: int = 30) -> None:
    """
    Live progress line for long per-study loops (feature extraction over
    potentially thousands of studies, VLM inference over the agentic subset).
    Same elapsed/eta style as scripts/download_chexpert_subset.py so output
    looks consistent across the pipeline. Overwrites the same terminal line
    (\r) rather than spamming one line per study.
    """
    import time
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    end = "\n" if done == total else ""
    print(
        f"\r  [{bar}] {done}/{total} ({frac*100:5.1f}%)  "
        f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  {label[:40]:<40}",
        end=end, flush=True,
    )


def _extract_features_real(study, cam):
    """
    Radiomics + XAI + vocabulary for one real study. Returns
    (radiomics, xai, vocab, image_array), or None if the image couldn't be
    loaded (missing/corrupt file -- caller should skip this study rather
    than crash a multi-thousand-study run over one bad file).

    `cam`: a GradCAM instance from xai_gradcam.load_xai_backend(), loaded
    ONCE by the caller and reused across all studies -- loading the
    torchxrayvision classifier fresh for every single study (as an earlier
    version of this function did) turns a few-second one-time cost into
    thousands of redundant reloads.
    """
    import numpy as np
    from PIL import Image
    from radiomics import extract_radiomics
    from vocabulary import extract_vocabulary_features
    from xai_gradcam import compute_xai_stats

    try:
        image = np.array(Image.open(study.frontal_image_path).convert("L"))
    except (FileNotFoundError, OSError) as e:
        print(f"\n  WARNING: skipping study {study.study_id!r} ({study.frontal_image_path}): {e}")
        return None

    radiomics = extract_radiomics(image)
    vocab = extract_vocabulary_features(study.report_text)
    xai = compute_xai_stats(image, cam)

    return radiomics, xai, vocab, image


def run_ablation_on_chexpert(studies: list, model_family: str, synthetic: bool, finetune_regime: str = "frozen"):
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
        import time
        from text_embeddings import extract_text_embeddings
        from xai_gradcam import load_xai_backend

        cam = load_xai_backend(device=config.VLM_DEVICE)  # loaded ONCE, reused for every study below
        start_time = time.time()

        radiomics_list, xai_list, report_texts, labels = [], [], [], []
        n_skipped = 0
        for i, study in enumerate(studies, 1):
            extracted = _extract_features_real(study, cam)
            if extracted is None:
                n_skipped += 1
                continue
            radiomics, xai, _vocab, _image = extracted
            radiomics_list.append(list({k: v for k, v in radiomics.items() if isinstance(v, (int, float))}.values()))
            xai_list.append([xai["xai_mean"], xai["xai_max"], xai["xai_entropy"], xai["xai_top10pct_mass"]])
            report_texts.append(study.report_text)
            labels.append(study.label)
            _progress(i, len(studies), start_time, label=f"study {study.study_id}")

        if n_skipped:
            print(f"  WARNING: skipped {n_skipped}/{len(studies)} studies with missing/corrupt images.")

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
            "seed": config.RANDOM_SEED, "n_studies": len(labels), "n_studies_requested": len(studies),
            "dataset": "chexpert", "model_family": model_family, "text_embedding_source": text_embedding_source,
            "synthetic": synthetic,
            # Ablation is pure sklearn over radiomics/XAI/text-embedding features and
            # never calls the VLM at all, so it's identical regardless of which
            # finetune_regime's agentic run it's paired with in a comparison report --
            # tagged "frozen" here just so the report generator has a consistent key.
            "finetune_regime": finetune_regime,
            "target_finding": config.CHEXPERT_TARGET_FINDING,
        },
        results={"ablation": results},
    )
    append_run(config.RESULTS_LOG_PATH, run_record)
    print(f"Logged ablation run: {run_record['run_id']}")


def run_agentic_on_chexpert(
    studies: list, model_family: str, synthetic: bool, adapter_path: str = None,
    finetune_regime: str = "frozen", model_path: str = None,
):
    import sys as _sys
    sys.path.insert(0, str(Path(__file__).parent / "src" / "evaluation"))
    from hallucination import average_hallucination_rate
    from text_metrics import evaluate_batch
    from responsible_ai_metrics import compute_agentic_reasoning_stats, compute_responsible_ai_indicators

    reasoner = _get_reasoner(
        model_family, synthetic, adapter_path=adapter_path,
        finetune_regime=finetune_regime, model_path=model_path,
    )

    cam = None
    if not synthetic:
        import time
        from xai_gradcam import load_xai_backend
        cam = load_xai_backend(device=config.VLM_DEVICE)  # loaded ONCE, reused for every study below
        start_time = time.time()

    step0_texts, step1_outputs, step2_outputs, ground_truths = [], [], [], []
    n_skipped = 0

    for i, study in enumerate(studies, 1):
        if synthetic:
            radiomics, xai, vocab, image_path = {"mean": 0.0}, {"xai_mean": 0.0, "xai_max": 0.0, "xai_entropy": 0.0, "xai_top10pct_mass": 0.0}, {"matched_terms": [], "num_matched_terms": 0}, "synthetic.png"
            report_text = study["report_text"]
        else:
            extracted = _extract_features_real(study, cam)
            if extracted is None:
                n_skipped += 1
                continue
            radiomics, xai, vocab, _image = extracted
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

        if not synthetic:
            _progress(i, len(studies), start_time, label=f"study {study.study_id}")

    if n_skipped:
        print(f"  WARNING: skipped {n_skipped}/{len(studies)} studies with missing/corrupt images.")

    def _to_text(structured_or_str):
        if isinstance(structured_or_str, str):
            return structured_or_str
        return f"{structured_or_str.get('impression', '')} {structured_or_str.get('evidence', '')}"

    def _has_valid_uncertainty(o):
        # Algorithm 1 Phase 4 requires y_unc in [0,1] -- a VLM occasionally
        # ignores the schema and writes a sentence into "uncertainty" instead
        # of a number (small models like Qwen2-VL-2B do this sometimes). That
        # counts as a schema-constraint failure, same as unparseable JSON.
        if not isinstance(o, dict) or "uncertainty" not in o:
            return False
        try:
            return 0.0 <= float(o["uncertainty"]) <= 1.0
        except (TypeError, ValueError):
            return False

    hr_baseline = average_hallucination_rate(step0_texts, ground_truths)
    hr_context = average_hallucination_rate([_to_text(o) for o in step2_outputs], ground_truths)

    valid_step1 = [o for o in step1_outputs if _has_valid_uncertainty(o)]
    valid_step2 = [o for o in step2_outputs if _has_valid_uncertainty(o)]
    n_parse_failures = (len(step1_outputs) - len(valid_step1)) + (len(step2_outputs) - len(valid_step2))
    if n_parse_failures:
        print(f"  WARNING: {n_parse_failures} structured outputs failed schema validation "
              f"(bad JSON or non-numeric/out-of-range uncertainty) and were excluded from Table 3/4 stats.")

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
        "seed": config.RANDOM_SEED, "n_studies": len(ground_truths), "n_studies_requested": len(studies),
        "dataset": "chexpert", "model_family": model_family, "synthetic": synthetic,
        "finetune_regime": finetune_regime, "adapter_path": adapter_path, "model_path": model_path,
        "target_finding": config.CHEXPERT_TARGET_FINDING,
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
    parser.add_argument(
        "--adapter-path", default=None,
        help="Path to a LoRA/QLoRA adapter (from src/vlm/finetune_qwen.py) to load on top of the "
             "frozen qwen2-vl backbone for the agentic/hallucination/text-metrics eval. "
             "Use with --finetune-regime lora/qlora. Omit (with --finetune-regime frozen) to "
             "reproduce the paper's fully-frozen setup exactly (the default).",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to a full-fine-tuned standalone checkpoint (from src/vlm/finetune_qwen.py "
             "--regime full) to load INSTEAD OF the paper's base checkpoint. "
             "Use with --finetune-regime full.",
    )
    parser.add_argument(
        "--finetune-regime", choices=["frozen", "lora", "qlora", "full"], default="frozen",
        help="Label for which regime is being evaluated -- logged into config_snapshot so "
             "scripts/compile_finetune_comparison.py can group runs.",
    )
    parser.add_argument(
        "--agentic-offset", type=int, default=0,
        help="Row offset (into config.CHEXPERT_CSV_PATH) for the agentic/hallucination/text-metrics "
             "eval set. src/vlm/finetune_qwen.py always trains on rows starting at 0, so evaluating a "
             "LoRA/QLoRA/full-fine-tuned model with the default offset=0 would score it on studies it "
             "was also trained on, inflating every downstream quality metric. Set this to at least "
             "N_FINETUNE_STUDIES (plus a safety margin) when --finetune-regime is lora/qlora/full. "
             "Ablation is unaffected (never calls the VLM) and always uses offset 0.",
    )
    args = parser.parse_args()

    if not args.synthetic and not args.model:
        parser.error("--model is required unless --synthetic is set")
    if args.finetune_regime in ("lora", "qlora") and not args.adapter_path:
        parser.error("--finetune-regime lora/qlora requires --adapter-path")
    if args.finetune_regime == "full" and not args.model_path:
        parser.error("--finetune-regime full requires --model-path (the saved full-FT checkpoint dir)")
    if args.finetune_regime == "frozen" and (args.adapter_path or args.model_path):
        parser.error("--adapter-path/--model-path require --finetune-regime lora/qlora/full, not frozen")
    if args.adapter_path and args.model_path:
        parser.error("--adapter-path and --model-path are mutually exclusive (adapter-on-frozen-base vs. standalone full-FT checkpoint)")

    model_family = args.model or "qwen2-vl"

    if args.finetune_regime in ("lora", "qlora", "full") and args.agentic_offset == 0 and not args.synthetic:
        print(
            "  WARNING: --finetune-regime is a fine-tuned regime but --agentic-offset is 0 -- the "
            "agentic/hallucination/text-metrics eval will score studies that may also have been used "
            "for training, inflating these metrics. Pass --agentic-offset (see --help) unless this is "
            "intentional (e.g., checking training-set performance).\n"
        )

    print(f"=== CheXpert evaluation: model={model_family}, regime={args.finetune_regime}, synthetic={args.synthetic} ===\n")

    if args.synthetic:
        import numpy as np
        rng = np.random.default_rng(config.RANDOM_SEED)
        n = max(args.n_ablation, args.n_agentic + args.agentic_offset)
        studies = [
            {"study_id": str(i), "label": int(rng.integers(0, 2)), "report_text": "No Finding." if i % 2 == 0 else "Cardiomegaly present."}
            for i in range(n)
        ]
        ablation_studies = studies[: args.n_ablation]
        agentic_studies = studies[args.agentic_offset: args.agentic_offset + args.n_agentic]
    else:
        # Ablation never touches the VLM (pure sklearn over extracted features), so there's no
        # train/test leakage concern for it -- it always reads from row 0 regardless of
        # --agentic-offset. Only the agentic/hallucination/text-metrics eval (which actually
        # generates from the model) needs to be held out from whatever finetune_qwen.py trained on.
        ablation_studies = _load_real_studies(n=args.n_ablation, uncertain_policy=args.uncertain_policy)
        agentic_studies = _load_real_studies(
            n=args.n_agentic, uncertain_policy=args.uncertain_policy, offset=args.agentic_offset,
        )

    print(f"\n--- Ablation study ({len(ablation_studies)} studies) ---")
    run_ablation_on_chexpert(ablation_studies, model_family, args.synthetic)

    print(f"\n--- Agentic reasoning / hallucination / text metrics ({len(agentic_studies)} studies) ---")
    run_agentic_on_chexpert(
        agentic_studies, model_family, args.synthetic,
        adapter_path=args.adapter_path, finetune_regime=args.finetune_regime, model_path=args.model_path,
    )

    print(f"\nAll results appended to {config.RESULTS_LOG_PATH}")


if __name__ == "__main__":
    main()
