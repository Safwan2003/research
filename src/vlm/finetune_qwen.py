"""
LoRA / QLoRA / full supervised fine-tuning of Qwen2-VL-2B-Instruct on CheXpert.

IMPORTANT -- this is an extension BEYOND the paper, not a reproduction of it.
The paper's f_theta (Eq. 3) is explicitly frozen: "we demonstrate that
reliability can be improved without modifying model architecture or training
objectives, but by changing the decision protocol." Nothing here is required
to match the paper's own Tables 1-4 -- those come from run_chexpert_eval.py
with no --adapter-path, which is still the fully-frozen setup.

What this script does instead: teach the SAME frozen backbone (via a small
LoRA or QLoRA adapter, base weights untouched) to produce better-grounded
structured JSON output (Eq. 2 schema) when given the same F_tool-conditioned
prompt the frozen model already sees (build_context_aligned_prompt). The
training target for each study is a deterministic "silver" structured label
built from the study's ground-truth CheXpert finding column and its matched
vocabulary terms -- see build_target_completion(). This is supervised
fine-tuning towards the SAME metrics run_chexpert_eval.py measures
(hallucination rate, ROUGE/BERTScore vs. the reference report, calibrated
uncertainty), so results are directly comparable to the frozen baseline.

Usage (run inside WSL, GPU required):
    python3 src/vlm/finetune_qwen.py --regime lora  --n-studies 300 --epochs 1
    python3 src/vlm/finetune_qwen.py --regime qlora --n-studies 300 --epochs 1
    python3 src/vlm/finetune_qwen.py --regime full  --n-studies 300 --epochs 1

Bare `python3 finetune_qwen.py` (no args) runs only the synthetic,
torch-free self-test of build_target_completion(), per this project's
convention of every feature module being sanity-checkable without a GPU.
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _sub in ("", "src/data", "src/features", "src/vlm", "src/ablation", "src/evaluation"):
    sys.path.insert(0, str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT))


QUESTION = "Is there evidence of active cardiopulmonary abnormality?"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def build_target_completion(label: int, is_uncertain_finding: bool, target_finding: str, matched_terms: list) -> dict:
    """
    Deterministic "silver" structured-output target (Eq. 2 schema) for one
    CheXpert study, used as the SFT label. Unlike dataset.py's pseudo-report
    (an INPUT feature that must NOT leak the label), this is the TRAINING
    TARGET the model is supposed to learn to produce -- stating the correct
    finding here is the entire point, not a bug.

    Args:
        label: 1 (finding present) or 0 (absent), from dataset.py's
            _compute_chexpert_label for `target_finding`.
        is_uncertain_finding: whether the raw CheXpert column for
            `target_finding` was -1 ("uncertain") before the label policy
            collapsed it to 0/1 -- used to shape the uncertainty target so
            fine-tuning doesn't teach the model to be falsely confident on
            genuinely ambiguous cases.
        target_finding: the pathology name, e.g. "Cardiomegaly".
        matched_terms: F_voc matched vocabulary terms for this study's
            (leakage-free) pseudo-report, used to make "evidence" concrete
            rather than a generic template string.

    Returns:
        dict matching RESPONSE_SCHEMA_INSTRUCTIONS's schema exactly.
    """
    if label == 1:
        impression = f"Findings compatible with {target_finding}."
        finding_clause = f"radiographic features consistent with {target_finding.lower()}"
    else:
        impression = f"No definite radiographic evidence of {target_finding}."
        finding_clause = "no focal consolidation or silhouette abnormality identified"

    evidence_parts = []
    if matched_terms:
        evidence_parts.append(", ".join(matched_terms[:4]))
    evidence_parts.append(finding_clause)
    evidence = "; ".join(evidence_parts)

    if is_uncertain_finding:
        uncertainty = 0.55
    elif label == 1:
        uncertainty = 0.30
    else:
        uncertainty = 0.20

    return {
        "impression": impression,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "limitations": "Single frontal radiograph without prior comparison or clinical correlation.",
        "safety_note": "For research use only; not a substitute for expert radiologist interpretation.",
    }


def _extract_feature_card_json(image, report_text: str, cam) -> tuple:
    """Returns (feature_card_json, matched_vocab_terms) for one study's image + report."""
    from radiomics import extract_radiomics
    from vocabulary import extract_vocabulary_features
    from xai_gradcam import compute_xai_stats
    from feature_card import build_feature_card, feature_card_to_prompt_text

    radiomics = extract_radiomics(image)
    vocab = extract_vocabulary_features(report_text)
    xai = compute_xai_stats(image, cam)

    card = build_feature_card(radiomics, xai, vocab)
    return feature_card_to_prompt_text(card), vocab.get("matched_terms", [])


def _build_training_example(processor, image_path: str, prompt_text: str, target_json: dict, device: str):
    """
    Tokenizes one (image, prompt, target) triple into model inputs with
    `labels` masked so loss is only computed on the assistant's completion
    tokens (the target JSON), not the user prompt or image tokens.
    """
    target_text = json.dumps(target_json)
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text_rendered = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

    full_inputs = processor(text=[full_text], images=[image_path], padding=False, return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text_rendered], images=[image_path], padding=False, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels

    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full_inputs.items()}


def _build_lora_model(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def run_finetune(
    regime: str,
    n_studies: int = 300,
    output_dir: str = None,
    model_name: str = None,
    epochs: int = 1,
    lr: float = 1e-4,
    grad_accum_steps: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    log_every: int = 20,
):
    """
    Fine-tune Qwen2-VL-2B on `n_studies` CheXpert studies under one of three
    regimes:
      - "lora":  small adapter matrices trainable, base weights frozen+fp16.
      - "qlora": same, but base weights loaded 4-bit (NF4) to cut VRAM.
      - "full":  EVERY parameter trainable (no adapter) -- this is the
                 regime with no counterpart in the paper at all (the paper's
                 f_theta is always frozen; LoRA/QLoRA at least still leave
                 the base weights untouched). Full fine-tuning of even a 2B
                 VLM's optimizer state (Adam) plus gradients roughly
                 quadruples the ~4-5GB fp16 weight footprint, which is tight
                 on a 16GB GPU -- gradient checkpointing and an 8-bit Adam
                 optimizer (bitsandbytes) are both used here specifically to
                 make that fit. If it still OOMs, lower --n-studies (shorter
                 run, same peak memory) won't help; lower --grad-accum-steps
                 won't help either -- the fix is a smaller batch of image
                 tokens per step, which isn't currently exposed here, or
                 fall back to QLoRA.

    For "lora"/"qlora", `output_dir` gets a small peft adapter (a few MB) --
    reload via QwenVLReasoner(adapter_path=output_dir). For "full",
    `output_dir` gets a complete standalone model+processor checkpoint (a
    few GB) -- reload via QwenVLReasoner(model_name=output_dir) (no
    adapter_path).
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    import config
    from dataset import load_chexpert_dataset
    from prompts import build_context_aligned_prompt

    if regime not in ("lora", "qlora", "full"):
        raise ValueError(f"regime must be 'lora', 'qlora', or 'full', got {regime!r}")

    model_name = model_name or config.VLM_MODEL_NAME
    output_dir = output_dir or str(_PROJECT_ROOT / "models" / f"{regime}_chexpert_2b")
    device = config.VLM_DEVICE

    print(f"=== Fine-tuning {model_name} ({regime.upper()}) on {n_studies} CheXpert studies ===")

    studies = load_chexpert_dataset(
        csv_path=str(config.CHEXPERT_CSV_PATH),
        images_root=str(config.CHEXPERT_IMAGES_ROOT),
        limit=n_studies,
        target_finding=config.CHEXPERT_TARGET_FINDING,
    )
    print(f"Loaded {len(studies)} studies (target_finding={config.CHEXPERT_TARGET_FINDING!r}).")

    if regime == "qlora":
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, quantization_config=quant_config, device_map=device
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )

    if regime == "full":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required alongside gradient checkpointing
        for p in model.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning: {n_trainable:,} / {n_total:,} parameters trainable (100%).")
    else:
        model = _build_lora_model(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    model.train()

    processor = AutoProcessor.from_pretrained(model_name)

    if regime == "full":
        # 8-bit Adam (bitsandbytes) keeps optimizer state at ~1/4 the memory
        # of standard fp32 Adam state -- the difference between fitting on a
        # 16GB GPU and not, when every one of the model's ~2B params is
        # trainable (vs. LoRA/QLoRA's optimizer state being negligible
        # because only a few million adapter params are trainable).
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr)
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    from xai_gradcam import load_xai_backend
    cam = load_xai_backend(device=config.VLM_DEVICE)

    step = 0
    optimizer.zero_grad()
    for epoch in range(epochs):
        for study in studies:
            image = np.array(Image.open(study.frontal_image_path).convert("L"))
            card_json, matched_terms = _extract_feature_card_json(image, study.report_text, cam)
            prompt_text = build_context_aligned_prompt(QUESTION, study.report_text, card_json)

            raw_val = study.metadata.get("raw_row", {}).get(config.CHEXPERT_TARGET_FINDING)
            is_uncertain = raw_val in (-1, -1.0)
            target = build_target_completion(study.label, is_uncertain, config.CHEXPERT_TARGET_FINDING, matched_terms)

            example = _build_training_example(processor, study.frontal_image_path, prompt_text, target, device)

            outputs = model(**example)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            step += 1

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % log_every == 0:
                print(f"  epoch {epoch + 1}/{epochs} step {step}/{len(studies) * epochs}: loss={outputs.loss.item():.4f}")

    if step % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    if regime == "full":
        # Unlike the adapter regimes, "full" has no separate frozen base to
        # reload alongside it -- save the processor too so output_dir is a
        # complete, standalone checkpoint QwenVLReasoner(model_name=output_dir)
        # can load directly.
        processor.save_pretrained(output_dir)
        print(f"Saved full fine-tuned model + processor to {output_dir}")
    else:
        print(f"Saved {regime} adapter to {output_dir}")
    return output_dir


def _self_test_target_completion():
    """Torch-free self-test: build_target_completion() needs no GPU/model weights."""
    positive = build_target_completion(1, False, "Cardiomegaly", ["Edema", "Pleural Effusion"])
    negative = build_target_completion(0, False, "Cardiomegaly", [])
    uncertain = build_target_completion(0, True, "Cardiomegaly", [])

    assert "Cardiomegaly" in positive["impression"]
    assert 0.0 <= positive["uncertainty"] <= 1.0
    assert positive["uncertainty"] > negative["uncertainty"], "positive finding should not be MORE certain than negative"
    assert uncertain["uncertainty"] > negative["uncertainty"], "flagged-uncertain case should carry higher uncertainty"
    for target in (positive, negative, uncertain):
        assert target["limitations"] and target["safety_note"]
        json.dumps(target)  # must be JSON-serializable

    print("OK: build_target_completion() self-test passed.")
    print(f"  positive: {positive}")
    print(f"  negative: {negative}")
    print(f"  uncertain: {uncertain}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_test_target_completion()
        print(
            "\nNo arguments given -- ran the torch-free self-test only.\n"
            "To actually fine-tune (needs GPU + downloaded CheXpert data):\n\n"
            "    python3 src/vlm/finetune_qwen.py --regime lora  --n-studies 300 --epochs 1\n"
            "    python3 src/vlm/finetune_qwen.py --regime qlora --n-studies 300 --epochs 1\n"
            "    python3 src/vlm/finetune_qwen.py --regime full  --n-studies 300 --epochs 1\n"
        )
    else:
        import argparse

        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--regime", choices=["lora", "qlora", "full"], required=True)
        parser.add_argument("--n-studies", type=int, default=300)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--grad-accum-steps", type=int, default=8)
        parser.add_argument("--lora-r", type=int, default=16)
        parser.add_argument("--output-dir", default=None)
        args = parser.parse_args()

        run_finetune(
            regime=args.regime, n_studies=args.n_studies, epochs=args.epochs, lr=args.lr,
            grad_accum_steps=args.grad_accum_steps, lora_r=args.lora_r, output_dir=args.output_dir,
        )
