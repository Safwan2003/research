# Project: Context-Aligned Medical VLM Reasoning

## What this project is

This implements the framework from the paper *"Towards Responsible
Multimodal Medical Reasoning via Context-Aligned Vision-Language Models"*
(Khan, Chhabriya, Zafar, Arif, Muneer, Zafar, Raza, Qureshi — arXiv:2604.08815).
A copy of the paper's key equations/sections is summarized below so you don't
need to re-derive anything.

**The task**: this is a professor's PhD research codebase. The original
baseline only did `Y(0) = VLM(image, report)` — free-form answers, no
structured output, no auxiliary evidence. The goal is to make the model
**context-driven**: `Y = VLM(image, report, F_tool)`, where `F_tool` is a
"feature card" combining three independent evidence signals (radiomics, XAI,
vocabulary), and the VLM must output a structured, hallucination-resistant
response. This should improve results the way the paper reports: much lower
hallucination rate, more concise/calibrated answers, small AUC gain.

## Repo layout

```
config.py                      # edit dataset paths / model name / seed here
run_pipeline.py                 # main entry point: demo | single | ablation
vocabulary_list.txt              # curated radiology terms for F_voc
src/features/radiomics.py        # F_rad: intensity + GLCM + LBP texture stats
src/features/xai_gradcam.py      # F_xai: Grad-CAM computation + spatial stats
src/features/vocabulary.py       # F_voc: vocabulary term matching in reports
src/features/feature_card.py     # F_tool = Serialize(F_rad, F_xai, F_voc)
src/vlm/prompts.py                # baseline vs context-driven prompt templates
src/vlm/reasoning.py              # QwenVLReasoner: actual Qwen2-VL calls
src/ablation/ablation_study.py    # Table 1: AUC across modality combinations
src/evaluation/hallucination.py   # Table 2: hallucination rate (HR) metric
src/evaluation/text_metrics.py    # Table 4: ROUGE / SacreBLEU / BERTScore
src/data/dataset.py                # OpenI / CheXpert loading (Study dataclass)
```

## Current status (read this before assuming anything is broken)

Everything below was written to match the paper's equations exactly and
already **syntax-checked and unit-tested with synthetic data** in a sandbox
with no GPU/internet. Don't rewrite these from scratch — extend them.

| Module | Status |
|---|---|
| `radiomics.py` | Done, tested, works standalone |
| `xai_gradcam.py` | `derive_spatial_statistics()` tested standalone. The `GradCAM` class (hook-based, PyTorch) is written correctly but **not yet run** — it needs a real pretrained chest-X-ray classifier plugged in (see "Known gaps" below) |
| `vocabulary.py` | Done, tested, works standalone |
| `feature_card.py` | Done, tested, works standalone |
| `prompts.py` | Done, tested (string output verified) |
| `reasoning.py` | Written correctly against the Qwen2-VL `transformers` API, but **never executed** — needs GPU + internet to download `Qwen/Qwen2-VL-2B-Instruct` |
| `ablation_study.py` | Done, tested with synthetic feature vectors (logistic regression + AUC via scikit-learn) |
| `hallucination.py` | Done, tested |
| `text_metrics.py` | ROUGE + SacreBLEU tested. BERTScore written but not run (needs to download a BERT model) |
| `dataset.py` | OpenI XML parsing tested against one synthetic sample file. CheXpert loader written, not tested (no CheXpert access yet) |

`python run_pipeline.py demo` runs the full Stage 1+2 pipeline (radiomics +
XAI stats + vocabulary + feature card) end to end on synthetic data with zero
setup — use this to confirm your environment is sane before touching real
data or a GPU.

## Known gaps / your immediate priorities

1. **Plug in a real pretrained chest X-ray classifier for Grad-CAM.**
   ImageNet-pretrained DenseNet-121 will NOT give medically meaningful
   attention maps. Use `torchxrayvision` (pretrained on CheXpert/NIH/MIMIC)
   or a DenseNet-121 the lab already fine-tuned. Wire this into
   `src/features/xai_gradcam.py`'s `GradCAM` usage inside `run_pipeline.py`'s
   `run_single()`.
2. **Get real OpenI data** (public, no registration: https://openi.nlm.nih.gov/faq)
   and point `config.py`'s `OPENI_REPORTS_DIR` / `OPENI_IMAGES_DIR` at it.
   Verify `_parse_openi_xml()` in `src/data/dataset.py` against a few real
   XML files — OpenI XML schema has some version variance, so double check
   the `AbstractText`/`parentImage` tag names match what you actually have.
3. **Run Stage 3 for real** in Colab (T4 GPU is enough for the 2B model):
   install `requirements.txt`'s GPU section, then
   `python run_pipeline.py single --image <path> --report <path>` to see
   baseline vs context-driven output side by side.
4. **Wire up the ablation study with real features**: extract radiomics +
   XAI + text-embeddings (e.g. `sentence-transformers`) for ~1,000 OpenI
   studies, then call `run_ablation()` in `src/ablation/ablation_study.py`.
   Compare against paper's Table 1 (expect: text-only ~0.92 AUC,
   radiomics/XAI-only ~0.52-0.56, full combo ~0.925).
5. **Reproduce Table 2/3** (hallucination rate + stepwise agentic reasoning)
   on the 50-study subset using `QwenVLReasoner.stepwise_agentic_reasoning()`
   and `src/evaluation/hallucination.py`.
6. **Cross-dataset check on CheXpert** (Section 5.4) — expect the opposite
   pattern from OpenI (radiomics ~0.71 AUC beats text ~0.50, since CheXpert
   reports are short/label-centric). Don't "fix" this if you see it — it's
   the paper's own reported finding, not a bug.

## Conventions to follow when extending this code

- Keep heavy imports (`torch`, `transformers`) **local to functions/methods**,
  not at module top-level, so the rest of the codebase stays importable and
  testable without a GPU. This is why `reasoning.py` and `xai_gradcam.py`'s
  `GradCAM` class do `import torch` inside `__init__`/methods.
- Every new feature-extraction function should have a `if __name__ ==
  "__main__":` self-test block using synthetic data, mirroring the existing
  modules — this lets you sanity-check logic without real data/GPU.
- Match the paper's terminology exactly in code (`F_rad`, `F_xai`, `F_voc`,
  `F_tool`, `Y(0)`/`Y(1)`/`Y(2)`) so it's traceable back to the equations.
- Structured VLM output must always be valid JSON with exactly these keys:
  `impression`, `evidence`, `uncertainty` (float 0-1), `limitations`,
  `safety_note`. See `RESPONSE_SCHEMA_INSTRUCTIONS` in `src/vlm/prompts.py`.

## Expected results to sanity-check against (from the paper)

- AUC: 0.918 (text only) → 0.925 (radiomics + XAI + text) — small gain
- Hallucination rate: 1.14 (image only) → 0.25 (image + text) → ~0.28 (full)
- Evidence length: ~19.4 words → ~15.3 words (more concise with context)
- Uncertainty: stays ~0.68-0.70 (doesn't spike toward overconfidence)

If your numbers are wildly different from these shapes (not exact values —
your dataset subset/seed will differ), something in the pipeline is likely
off; if they're in the same ballpark and direction, you're on track.
