"""
Stage 3: Tool-Augmented Agentic Reasoning -- the actual VLM calls.

THIS MODULE REQUIRES: a GPU (or at least a machine with enough RAM), the
`transformers` + `torch` + `qwen-vl-utils` packages, and an internet
connection to download the Qwen2-VL model weights the first time you run
it. It will NOT run inside a sandboxed environment with no GPU/internet --
run this in Google Colab (with a T4/A100 GPU runtime) or your lab server.

This is the module that literally implements the difference between:
  - the OLD baseline your professor already had: Y(0) = f_theta(I, R)
  - the NEW context-driven version you are adding: Y = f_theta(I, R, F_tool)

Both call the SAME frozen VLM -- nothing about the model's weights changes.
Only the prompt (and therefore the model's available context) changes. This
is exactly the paper's key claim in the abstract: "these results... preserve
the underlying model architecture."
"""

import json

from prompts import build_baseline_prompt, build_context_aligned_prompt, build_stepwise_prompts


class QwenVLReasoner:
    """
    Thin wrapper around Qwen2-VL-2B/7B-Instruct for structured medical
    reasoning, following Section 4.3 of the paper.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cpu"):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.torch = torch
        self.device = device

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256) -> str:
        """Run one forward generation pass given an image path + prompt text."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"[Warning] Generation on {self.device} failed: {e}. Falling back to CPU execution...")
            self.device = "cpu"
            self.model = self.model.to("cpu").to(self.torch.float32)
            inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return output_text

    def baseline_reasoning(self, image_path: str, question: str, report_text: str) -> str:
        """Y(0) = f_theta(I, R) -- the OLD baseline, free-form text output."""
        prompt = build_baseline_prompt(question, report_text)
        return self._generate(image_path, prompt)

    def context_aligned_reasoning(
        self, image_path: str, question: str, report_text: str, feature_card_json: str
    ) -> dict:
        """
        Y = f_theta(I, R, F_tool) -- the NEW context-driven version.
        Returns a parsed dict (impression/evidence/uncertainty/limitations/safety_note).
        Falls back to returning the raw text if the model didn't produce valid JSON
        (useful for debugging early in development -- tighten this once your
        prompt/parsing is reliable).
        """
        prompt = build_context_aligned_prompt(question, report_text, feature_card_json)
        raw_output = self._generate(image_path, prompt)
        return self._safe_parse_json(raw_output)

    def stepwise_agentic_reasoning(
        self, image_path: str, question: str, report_text: str,
        radiomics_json: str, feature_card_json: str
    ) -> dict:
        """
        Runs all three steps from Eqs. (10)-(12): Y(0), Y(1), Y(2).
        Returns dict with keys "step0", "step1", "step2" holding each output
        (step0 is free text; step1/step2 are parsed structured dicts).
        """
        prompts = build_stepwise_prompts(question, report_text, radiomics_json, feature_card_json)

        step0_out = self._generate(image_path, prompts["step0"])
        step1_out = self._safe_parse_json(self._generate(image_path, prompts["step1"]))
        step2_out = self._safe_parse_json(self._generate(image_path, prompts["step2"]))

        return {"step0": step0_out, "step1": step1_out, "step2": step2_out}

    @staticmethod
    def _safe_parse_json(text: str):
        """Best-effort JSON parsing since VLMs occasionally wrap JSON in extra text."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"raw_output": text, "parse_error": True}


if __name__ == "__main__":
    print(
        "This module defines QwenVLReasoner but does not run standalone here --\n"
        "it requires torch/transformers + a GPU + downloaded model weights.\n"
        "Run it from run_pipeline.py in your Colab/GPU environment, e.g.:\n\n"
        "    from src.vlm.reasoning import QwenVLReasoner\n"
        "    reasoner = QwenVLReasoner()\n"
        "    result = reasoner.context_aligned_reasoning(\n"
        "        image_path='sample_xray.png',\n"
        "        question='Is there evidence of active cardiopulmonary abnormality?',\n"
        "        report_text='No focal consolidation, pleural effusion, or pneumothorax.',\n"
        "        feature_card_json=feature_card_json,\n"
        "    )\n"
    )
