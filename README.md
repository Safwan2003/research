# Context-Aligned Medical VLM Reasoning

This is a working implementation of the framework from *"Towards Responsible
Multimodal Medical Reasoning via Context-Aligned Vision-Language Models"*
(Khan, Chhabriya, Zafar et al.). It takes the baseline setup your professor
already had — a VLM that just looks at an image + report and answers freely —
and adds the **context-driven** layer described in the paper on top of it, so
the model has to justify its answer using multiple independent evidence
sources instead of just guessing from one modality.

## The core idea in one sentence

Instead of `Y = VLM(image, report)`, we compute three extra "evidence
signals" from the image and report, bundle them into a small JSON **feature
card**, and force the VLM to reconcile all of them before answering in a
strict structured format (impression / evidence / uncertainty / limitations
/ safety note) — this is `Y = VLM(image, report, feature_card)`.

## Project structure

```
context_aligned_medical_vlm/
├── config.py                     # edit paths/settings here
├── requirements.txt
├── run_pipeline.py                # main entry point (see "How to run" below)
├── vocabulary_list.txt            # curated radiology terms for F_voc
├── src/
│   ├── features/
│   │   ├── radiomics.py           # F_rad: intensity + GLCM + LBP texture stats
│   │   ├── xai_gradcam.py         # F_xai: Grad-CAM + spatial statistics
│   │   ├── vocabulary.py          # F_voc: vocabulary matching in report text
│   │   └── feature_card.py        # F_tool = Serialize(F_rad, F_xai, F_voc)
│   ├── vlm/
│   │   ├── prompts.py             # baseline vs context-driven prompt templates
│   │   └── reasoning.py           # actual Qwen2-VL calls (needs GPU)
│   ├── ablation/
│   │   └── ablation_study.py      # Table 1: AUC across modality combinations
│   ├── evaluation/
│   │   ├── hallucination.py       # Table 2: hallucination rate (HR)
│   │   └── text_metrics.py        # Table 4: ROUGE / SacreBLEU / BERTScore
│   └── data/
│       └── dataset.py             # OpenI / CheXpert loading skeleton
└── tests/
```

## What runs where

I've already tested everything that doesn't need a GPU or internet access,
right in the sandbox this was built in. Here's the honest breakdown:

| Component | Tested here? | Needs |
|---|---|---|
| Radiomics (`radiomics.py`) | ✅ Yes, works | numpy, scikit-image only |
| XAI statistics function (`xai_gradcam.py`) | ✅ Yes, works | numpy only |
| Grad-CAM computation itself | ⬜ Not run here | torch + a pretrained CXR classifier + GPU |
| Vocabulary matching (`vocabulary.py`) | ✅ Yes, works | none |
| Feature card serialization | ✅ Yes, works | none |
| Prompt templates (`prompts.py`) | ✅ Yes, works | none |
| Ablation study / AUC (`ablation_study.py`) | ✅ Yes, works | scikit-learn |
| Hallucination rate (`hallucination.py`) | ✅ Yes, works | none |
| ROUGE / SacreBLEU | ✅ Yes, works | rouge-score, sacrebleu |
| BERTScore | ⬜ Not run here | internet (downloads a BERT model) |
| Actual Qwen2-VL calls (`reasoning.py`) | ⬜ Not run here | torch, transformers, GPU, internet |
| OpenI XML parsing | ✅ Yes, works (on a synthetic sample file) | your actual OpenI data |

Everything marked ⬜ is written and syntax-checked, following the exact
equations/sections of the paper, but needs a GPU + real data + internet to
actually execute — that's Google Colab (free T4 GPU) or your lab's server,
not this sandbox.

## How to run

### 1. Quick sanity check (no setup needed beyond `pip install numpy scikit-image scikit-learn`)

```bash
pip install numpy scikit-image scikit-learn
python run_pipeline.py demo
```

This runs Stage 1 (radiomics + XAI-stats + vocabulary) and Stage 2 (feature
card serialization) on synthetic data, so you can see the whole
feature-extraction pipeline work end to end before you touch real data or a
GPU.

### 2. Full pipeline on one real study (needs GPU — do this in Colab)

```bash
pip install -r requirements.txt
python run_pipeline.py single --image path/to/xray.png --report path/to/report.txt
```

This will print the **baseline** output (old behavior: image + report only,
free text) directly next to the **context-driven** output (new behavior:
image + report + feature card, structured JSON) — this side-by-side
comparison is exactly what your professor wants to see improve.

Before this fully works you need to plug in a real pretrained chest X-ray
classifier for Grad-CAM (see the docstring in `src/features/xai_gradcam.py`
— `torchxrayvision` is the easiest option, or use a DenseNet-121 your lab
already fine-tuned on OpenI/CheXpert).

### 3. Ablation study (Table 1 reproduction)

```bash
python run_pipeline.py ablation
```

This is currently a TODO stub pointing you to `src/ablation/ablation_study.py`,
which has a fully working synthetic-data example. Once you have:
- radiomics feature vectors per study (from `extract_radiomics`)
- XAI feature vectors per study (from `derive_spatial_statistics`)
- text embeddings per study (e.g. `sentence-transformers` on the report text)
- ground-truth labels (e.g. normal/abnormal from the report or CheXpert labels)

...call `run_ablation(radiomics_features, xai_features, text_embeddings, labels)`
directly to get the Table 1 AUC numbers.

### 4. Getting the datasets

- **OpenI**: public, no registration needed. https://openi.nlm.nih.gov/faq
  Has paired frontal/lateral X-rays + XML radiology reports — exactly what
  `src/data/dataset.py`'s `load_openi_dataset()` expects.
- **CheXpert**: requires a Stanford data use agreement. Ask your professor
  if the lab already has a local copy (most labs working on this do).

## What "context-driven" actually changes (so you know what to look for)

Once you're running real experiments, compare baseline vs context-driven on:
1. **Hallucination rate** (`src/evaluation/hallucination.py`) — should drop a
   lot (paper: 1.14 → 0.25 hallucinated keywords per study).
2. **Evidence length** — context-driven outputs should be shorter and more
   concise (paper: ~19 words → ~15 words).
3. **Uncertainty calibration** — should stay roughly stable, not spike toward
   overconfidence just because there's more context (paper: 0.70 → 0.68).
4. **AUC** — expect only a small bump (paper: 0.918 → 0.925). The real win is
   reliability/safety, not raw accuracy — don't be surprised or worried if
   your AUC gain is modest; that matches the paper's own finding.

## A note on the CheXpert cross-dataset result

The paper found that on CheXpert (shorter, more label-centric reports),
radiomics actually outperforms text (0.71 vs 0.50 AUC) — the opposite of
OpenI. If your professor asks you to also validate cross-dataset, don't be
alarmed if your OpenI-tuned intuitions flip on CheXpert — that's a genuine,
reported finding about how report richness changes which modality carries
more signal, not a bug in your code.

# research
