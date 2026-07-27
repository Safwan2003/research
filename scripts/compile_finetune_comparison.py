"""
Compile a single, professor-ready comparison table across the frozen /
LoRA / QLoRA fine-tuning regimes from results/experiment_log.json.

For each mode (ablation, hallucination, agentic_reasoning, text_metrics),
picks the LATEST real (non-synthetic) CheXpert run per finetune_regime and
renders one combined Markdown report -- Tables 1-4 style, side by side
across regimes, exactly as reported in that append-only log so nothing is
hand-edited or re-derived.

Usage (after running run_chexpert_eval.py for each regime you care about):
    python3 scripts/compile_finetune_comparison.py
    python3 scripts/compile_finetune_comparison.py --out results/my_report.md
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

REGIME_ORDER = ["frozen", "lora", "qlora", "full"]
REGIME_LABEL = {
    "frozen": "Frozen (paper's f_theta, no adapter)",
    "lora": "LoRA fine-tuned",
    "qlora": "QLoRA fine-tuned",
    "full": "Full fine-tuned (all params)",
}


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _latest_real_run_per_regime(runs: list, mode: str) -> dict:
    """Latest non-synthetic CheXpert run for `mode`, keyed by finetune_regime."""
    latest = {}
    for run in runs:
        if run.get("mode") != mode:
            continue
        cfg = run.get("config_snapshot", {})
        if cfg.get("synthetic") or cfg.get("dataset") != "chexpert":
            continue
        regime = cfg.get("finetune_regime", "frozen")
        if regime not in latest or run["timestamp_utc"] > latest[regime]["timestamp_utc"]:
            latest[regime] = run
    return latest


def build_report(log_path) -> str:
    with open(log_path) as f:
        log = json.load(f)
    runs = log.get("runs", [])

    ablation = _latest_real_run_per_regime(runs, "ablation")
    hallucination = _latest_real_run_per_regime(runs, "hallucination")
    agentic = _latest_real_run_per_regime(runs, "agentic_reasoning")
    text_metrics = _latest_real_run_per_regime(runs, "text_metrics")

    present_regimes = [r for r in REGIME_ORDER if r in hallucination or r in ablation]

    lines = []
    lines.append("# Context-Aligned Medical VLM -- Fine-Tuning Regime Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Dataset: CheXpert. AUC / fine-tuning target finding: **{config.CHEXPERT_TARGET_FINDING}**")
    lines.append("")
    lines.append(
        "> **Scope note:** `research_miss_sumra.pdf` reports results for a fully **frozen** "
        "VLM only (Eq. 3: *\"f_theta denotes the pretrained Qwen2-VL model with frozen "
        "parameters\"*) -- the paper's whole contribution is that reliability improves "
        "*without* training. The LoRA/QLoRA rows below go beyond the paper, added to check "
        "whether lightweight fine-tuning further improves the exact same metrics the paper "
        "defines. Every row here comes from an actual logged run in "
        "`results/experiment_log.json` (run id + timestamp shown per table) -- nothing is "
        "hand-edited."
    )
    lines.append("")

    if not present_regimes:
        lines.append("**No real (non-synthetic) CheXpert runs found in the results log yet.**")
        lines.append("Run `run_chexpert_eval.py` (optionally with `--adapter-path`) first.")
        return "\n".join(lines)

    # --- 1. Ablation (AUC) ---
    lines.append("## 1. Multimodal Ablation (AUC, Table 1-style)")
    lines.append("")
    run = ablation.get("frozen")
    if run:
        cfg, res = run["config_snapshot"], run["results"]["ablation"]
        lines.append(
            f"n_studies={cfg.get('n_studies')}, target_finding={cfg.get('target_finding', '?')} "
            f"(run `{run['run_id'][:8]}`, {run['timestamp_utc']})"
        )
        lines.append("")
        lines.append("| Feature Configuration | AUC |")
        lines.append("|---|---|")
        for name, auc in res.items():
            lines.append(f"| {name} | {_fmt(auc)} |")
        lines.append("")
        lines.append(
            "_Ablation is a pure scikit-learn logistic regression over extracted features and "
            "never calls the VLM -- it doesn't vary across frozen/LoRA/QLoRA. Only the tables "
            "below (which do call the VLM) differ by regime._"
        )
    else:
        lines.append("_No real ablation run found yet._")
    lines.append("")

    # --- 2. Hallucination ---
    lines.append("## 2. Hallucination Rate (Table 2-style, lower is better)")
    lines.append("")
    lines.append("| Regime | Baseline Y(0) image+text | Context-driven Y(2) full F_tool | Change | Run |")
    lines.append("|---|---|---|---|---|")
    for regime in present_regimes:
        run = hallucination.get(regime)
        if not run:
            lines.append(f"| {REGIME_LABEL[regime]} | - | - | - | - |")
            continue
        res = run["results"]["hallucination"]
        y0 = res.get("Image + Text (baseline Y0)")
        y2 = res.get("Image + Text + Radiomics + XAI (Y2)")
        delta = (y2 - y0) if isinstance(y0, (int, float)) and isinstance(y2, (int, float)) else None
        trend = "" if delta is None else (" (improved)" if delta < 0 else " (worse)" if delta > 0 else " (flat)")
        lines.append(
            f"| {REGIME_LABEL[regime]} | {_fmt(y0)} | {_fmt(y2)} | {_fmt(delta)}{trend} | "
            f"`{run['run_id'][:8]}` |"
        )
    lines.append("")

    # --- 3. Agentic reasoning ---
    lines.append("## 3. Agentic Reasoning (Table 3-style)")
    lines.append("")
    lines.append(
        "| Regime | Step | Uncertainty mean | Uncertainty std | Evidence length (words) | "
        "Limitations % | Safety notes % |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for regime in present_regimes:
        run = agentic.get(regime)
        if not run:
            continue
        res = run["results"]["agentic_reasoning"]
        step1 = (res.get("step0_vs_step1") or {}).get("step1")
        step2 = res.get("step2")
        for step_name, step in (("Step 1 (+radiomics)", step1), ("Step 2 (+F_tool, final)", step2)):
            if not step:
                continue
            lines.append(
                f"| {REGIME_LABEL[regime]} | {step_name} | {_fmt(step.get('uncertainty_mean'))} | "
                f"{_fmt(step.get('uncertainty_std'))} | {_fmt(step.get('evidence_length_mean'), 1)} | "
                f"{_fmt((step.get('presence_limitations') or 0) * 100, 0)} | "
                f"{_fmt((step.get('presence_safety_notes') or 0) * 100, 0)} |"
            )
    lines.append("")

    # --- 4. Text quality + Responsible AI ---
    lines.append("## 4. Text Quality + Responsible-AI Indicators (Table 4-style)")
    lines.append("")
    lines.append(
        "| Regime | ROUGE-1 | ROUGE-2 | ROUGE-L | SacreBLEU | BERTScore-F1 | PHI % | Unsafe % | "
        "Uncertainty-marker rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for regime in present_regimes:
        run = text_metrics.get(regime)
        if not run:
            continue
        res = run["results"]["text_metrics"]
        lines.append(
            f"| {REGIME_LABEL[regime]} | {_fmt(res.get('rouge1'))} | {_fmt(res.get('rouge2'))} | "
            f"{_fmt(res.get('rougeL'))} | {_fmt(res.get('sacrebleu'), 1)} | "
            f"{_fmt(res.get('bertscore_f1'))} | {_fmt(res.get('phi_pct'), 1)} | "
            f"{_fmt(res.get('unsafe_pct'), 1)} | {_fmt(res.get('unc_rate'))} |"
        )
    lines.append("")

    lines.append("## Paper's own reference numbers (OpenI dataset, frozen model -- not directly comparable to CheXpert above)")
    lines.append("")
    lines.append("- AUC: 0.918 (text only) -> 0.925 (radiomics + XAI + text)")
    lines.append("- Hallucination rate: 1.14 (image only) -> 0.25 (image + text) -> 0.28 (full context)")
    lines.append("- Evidence length: ~19.4 words -> ~15.3 words")
    lines.append("- Uncertainty: stays ~0.68-0.70 (frozen model; the paper never fine-tunes)")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-path", default=str(config.RESULTS_LOG_PATH))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "finetune_comparison_report.md"))
    args = parser.parse_args()

    report = build_report(args.log_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
