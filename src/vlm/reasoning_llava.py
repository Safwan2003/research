"""
Stage 3: Tool-Augmented Agentic Reasoning -- LLaVA-1.5-7B backbone.

Mirrors src/vlm/reasoning.py's QwenVLReasoner exactly (same public method
names/signatures) so it's a drop-in replacement wherever a reasoner is used
-- this is what lets run_chexpert_eval.py pick either backbone via a single
--model flag. Reuses the SAME prompt builders and JSON-parsing logic as the
Qwen reasoner rather than duplicating them.

THIS MODULE REQUIRES: a GPU, `transformers` + `torch`, and internet to
download `llava-hf/llava-1.5-7b-hf` weights the first time. It will NOT run
inside a sandboxed environment with no GPU/internet -- run it on the lab
GPU machine, same as reasoning.py.
"""

from prompts import build_baseline_prompt, build_context_aligned_prompt, build_stepwise_prompts
from reasoning import QwenVLReasoner  # reuse _safe_parse_json, not duplicated


class LlavaVLReasoner:
    """
    Thin wrapper around LLaVA-1.5-7B for structured medical reasoning,
    following Section 4.3 of the paper (LLaVA-1.5-7B is the second model
    evaluated in Table 4 alongside Qwen2-VL-7B).
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        # Imports are local so everything ELSE in this project can be
        # imported/tested without requiring torch/transformers installed.
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self.torch = torch
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256) -> str:
        """
        Run one forward generation pass given an image path + prompt text.

        NOTE: LlavaProcessor's chat-template / image-input call signature can
        differ across transformers versions -- if apply_chat_template or the
        processor(...) call below doesn't match your installed version
        (requirements.txt pins transformers>=4.45.0), check
        `AutoProcessor.from_pretrained(model_name)`'s actual API rather than
        assuming it's identical to Qwen2-VL's (reasoning.py's _generate).
        """
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
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
        """
        prompt = build_context_aligned_prompt(question, report_text, feature_card_json)
        raw_output = self._generate(image_path, prompt)
        return QwenVLReasoner._safe_parse_json(raw_output)

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
        step1_out = QwenVLReasoner._safe_parse_json(self._generate(image_path, prompts["step1"]))
        step2_out = QwenVLReasoner._safe_parse_json(self._generate(image_path, prompts["step2"]))

        return {"step0": step0_out, "step1": step1_out, "step2": step2_out}


if __name__ == "__main__":
    print(
        "This module defines LlavaVLReasoner but does not run standalone here --\n"
        "it requires torch/transformers + a GPU + downloaded model weights.\n"
        "Run it from run_chexpert_eval.py --model llava-1.5-7b in your GPU\n"
        "environment, e.g.:\n\n"
        "    from src.vlm.reasoning_llava import LlavaVLReasoner\n"
        "    reasoner = LlavaVLReasoner()\n"
        "    result = reasoner.context_aligned_reasoning(\n"
        "        image_path='sample_xray.png',\n"
        "        question='Is there evidence of active cardiopulmonary abnormality?',\n"
        "        report_text='No focal consolidation, pleural effusion, or pneumothorax.',\n"
        "        feature_card_json=feature_card_json,\n"
        "    )\n"
    )
