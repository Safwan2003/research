"""
Stage 3: Tool-Augmented Agentic Reasoning -- LLaVA-1.5-7B VLM wrapper.

THIS MODULE REQUIRES: a GPU (or at least a machine with enough VRAM), the
`transformers` + `torch` + `pillow` packages, and an internet connection to
download the LLaVA model weights (llava-hf/llava-1.5-7b-hf) the first time you run it.

Mirrors the interface of src/vlm/reasoning.py's QwenVLReasoner for LLaVA-1.5-7B.
"""

import json
from PIL import Image

try:
    from prompts import build_baseline_prompt, build_context_aligned_prompt, build_stepwise_prompts
except ImportError:
    from src.vlm.prompts import build_baseline_prompt, build_context_aligned_prompt, build_stepwise_prompts



class LlavaVLReasoner:
    """
    Thin wrapper around LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) for structured medical
    reasoning, matching the public interface of QwenVLReasoner.
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        # Imports are local so that everything ELSE in this project can be
        # imported/tested without requiring torch/transformers to be installed.
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self.torch = torch
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _generate(self, image_path: str, prompt_text: str, max_new_tokens: int = 256) -> str:
        """Run one forward generation pass given an image path + prompt text."""
        raw_image = Image.open(image_path).convert("RGB")

        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return output_text.strip()

    def baseline_reasoning(self, image_path: str, question: str, report_text: str) -> str:
        """Y(0) = f_theta(I, R) -- free-form text output using LLaVA-1.5-7B."""
        prompt = build_baseline_prompt(question, report_text)
        return self._generate(image_path, prompt)

    def context_aligned_reasoning(
        self, image_path: str, question: str, report_text: str, feature_card_json: str
    ) -> dict:
        """
        Y = f_theta(I, R, F_tool) -- context-driven structured output.
        Returns a parsed dict (impression/evidence/uncertainty/limitations/safety_note).
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
        Returns dict with keys "step0", "step1", "step2".
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
        "This module defines LlavaVLReasoner but does not run standalone here --\n"
        "it requires torch/transformers + a GPU + downloaded model weights.\n"
        "Run it from your Colab/GPU environment, e.g.:\n\n"
        "    from src.vlm.reasoning_llava import LlavaVLReasoner\n"
        "    reasoner = LlavaVLReasoner()\n"
        "    result = reasoner.context_aligned_reasoning(\n"
        "        image_path='sample_xray.png',\n"
        "        question='Is there evidence of active cardiopulmonary abnormality?',\n"
        "        report_text='No focal consolidation, pleural effusion, or pneumothorax.',\n"
        "        feature_card_json=feature_card_json,\n"
        "    )\n"
    )
