"""
Main entry point. Run this after you've set up your environment (see README).

Three modes:
    python run_pipeline.py demo            -- runs Stage 1+2 (feature extraction
                                               + feature card) on ONE synthetic
                                               study, no GPU/dataset needed.
                                               Good for confirming the pipeline
                                               logic before you have real data.

    python run_pipeline.py single --image path/to/xray.png --report path/to/report.txt
                                            -- full Stage 1-3 on one real study:
                                               extracts all features, builds the
                                               feature card, and calls the VLM
                                               in both baseline and context-driven
                                               modes so you can compare them side
                                               by side. REQUIRES GPU + Qwen2-VL.

    python run_pipeline.py ablation        -- runs the Table 1-style AUC ablation
                                               over your loaded dataset. Needs
                                               real feature vectors + labels; edit
                                               the TODO section below once your
                                               data loading is wired up. Pass
                                               --synthetic to instead run it on
                                               synthetic data end-to-end (no GPU
                                               or dataset needed) and log the
                                               result to the append-only
                                               results/experiment_log.json.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "features"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "vlm"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "ablation"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "evaluation"))

import config
from radiomics import extract_radiomics
from xai_gradcam import derive_spatial_statistics, extract_xai_features
from vocabulary import extract_vocabulary_features, load_vocabulary
from feature_card import build_feature_card, feature_card_to_prompt_text


def run_demo():
    """Stage 1 + 2 only, on synthetic data -- no GPU or dataset required."""
    import numpy as np

    print("Running Stage 1+2 demo on synthetic data (no GPU/dataset needed)...\n")

    rng = np.random.default_rng(0)
    synthetic_xray = rng.normal(120, 30, size=(256, 256)).clip(0, 255)
    synthetic_gradcam_heatmap = rng.uniform(0.2, 0.9, size=(64, 64))
    sample_report = (
        "No focal consolidation, pleural effusion, or pneumothorax. "
        "Cardiac silhouette within normal limits."
    )

    print("Step 1: extracting radiomics features (F_rad)...")
    rad = extract_radiomics(synthetic_xray)
    print(f"  -> {rad}\n")

    print("Step 2: summarizing (synthetic) Grad-CAM map into XAI stats (F_xai)...")
    print("  (in a real run, this heatmap comes from GradCAM() in src/vlm/../xai_gradcam.py")
    print("   applied to a pretrained chest X-ray classifier)")
    xai = derive_spatial_statistics(synthetic_gradcam_heatmap)
    print(f"  -> {xai}\n")

    print("Step 3: matching vocabulary terms in the report (F_voc)...")
    voc = extract_vocabulary_features(sample_report)
    print(f"  -> matched {voc['num_matched_terms']} terms: {voc['matched_terms']}\n")

    print("Step 4: serializing into the feature card (F_tool = Serialize(F_rad, F_xai, F_voc))...")
    card = build_feature_card(rad, xai, voc)
    print(feature_card_to_prompt_text(card))

    print(
        "\nThis feature card is what gets inserted into the VLM prompt in "
        "context-driven mode. Next step: run `single` mode with a real image "
        "+ report + GPU to see the actual before/after VLM outputs."
    )


def run_single(image_path: str, report_path: str, question: str):
    """
    Full Stage 1-3 on one real study: extract features, build feature card,
    then call the VLM in BOTH baseline and context-driven modes so you can
    directly compare the "before" and "after" behavior your professor wants.

    REQUIRES: torch, transformers, qwen-vl-utils, a GPU, and internet access
    to download Qwen2-VL-2B-Instruct weights the first time.
    """
    from PIL import Image
    import numpy as np
    from reasoning import QwenVLReasoner

    with open(report_path, "r") as f:
        report_text = f.read()

    image = np.array(Image.open(image_path).convert("L"))

    print("Extracting radiomics, XAI (Grad-CAM), and vocabulary features...")
    rad = extract_radiomics(image)
    voc = extract_vocabulary_features(report_text)
    xai = extract_xai_features(image)

    card = build_feature_card(rad, xai, voc)
    card_json = feature_card_to_prompt_text(card)

    print("\nLoading Qwen2-VL (this downloads weights on first run)...")
    reasoner = QwenVLReasoner()

    print("\n=== BASELINE (before): Y(0) = f_theta(I, R) ===")
    baseline_output = reasoner.baseline_reasoning(image_path, question, report_text)
    print(baseline_output)

    print("\n=== CONTEXT-DRIVEN (after): Y = f_theta(I, R, F_tool) ===")
    context_output = reasoner.context_aligned_reasoning(image_path, question, report_text, card_json)
    print(context_output)


def run_ablation(synthetic: bool = False):
    """
    Table 1-style AUC ablation. This needs real feature vectors and labels --
    fill in the TODO below once your dataset loading (src/data/dataset.py) and
    text-embedding step are wired up.

    Pass synthetic=True to instead run the same synthetic self-test that
    lives in src/ablation/ablation_study.py's __main__ block, but end-to-end
    through this entry point AND logged to the append-only results log --
    useful for verifying the whole pipeline + persistence layer works before
    you have a GPU or real dataset.
    """
    if not synthetic:
        print(
            "TODO: load your dataset via src/data/dataset.py, extract radiomics + "
            "XAI + text-embedding feature vectors for every study, then call:\n\n"
            "    from src.ablation.ablation_study import run_ablation, print_ablation_table\n"
            "    results = run_ablation(radiomics_features, xai_features, text_embeddings, labels)\n"
            "    print_ablation_table(results)\n\n"
            "See src/ablation/ablation_study.py's __main__ block for a working "
            "synthetic-data example you can adapt. Or pass --synthetic to run "
            "that example here and log it to the results database."
        )
        return

    import numpy as np
    from ablation_study import run_ablation as _run_ablation, print_ablation_table
    from results_store import append_run, new_run_record

    rng = np.random.default_rng(config.RANDOM_SEED)
    n = 300
    labels = rng.integers(0, 2, size=n)
    text_embeddings = rng.normal(0, 1, size=(n, 16)) + labels[:, None] * 2.0
    radiomics_features = rng.normal(0, 1, size=(n, 8)) + labels[:, None] * 0.3
    xai_features = rng.normal(0, 1, size=(n, 4)) + labels[:, None] * 0.2

    results = _run_ablation(
        radiomics_features, xai_features, text_embeddings, labels,
        random_state=config.RANDOM_SEED,
    )
    print_ablation_table(results)

    run_record = new_run_record(
        mode="ablation",
        config_snapshot={"seed": config.RANDOM_SEED, "n_studies": n, "dataset": "synthetic"},
        results={"ablation": results},
    )
    append_run(config.RESULTS_LOG_PATH, run_record)
    print(f"\nLogged to {config.RESULTS_LOG_PATH} (run_id={run_record['run_id']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-aligned medical VLM pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("demo", help="Run Stage 1+2 on synthetic data (no GPU needed)")

    single_parser = subparsers.add_parser("single", help="Run full pipeline on one real study (needs GPU)")
    single_parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    single_parser.add_argument("--report", required=True, help="Path to text file with the radiology report")
    single_parser.add_argument(
        "--question", default="Is there evidence of active cardiopulmonary abnormality?"
    )

    ablation_parser = subparsers.add_parser("ablation", help="Run Table 1-style AUC ablation study")
    ablation_parser.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data end-to-end and log the result (no GPU/dataset needed)",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "single":
        run_single(args.image, args.report, args.question)
    elif args.mode == "ablation":
        run_ablation(synthetic=args.synthetic)
