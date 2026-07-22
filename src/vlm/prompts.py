"""
Prompt construction for Stage 3 (Tool-Augmented Agentic Reasoning).

This module builds the prompt text sent to the VLM. It has two modes,
matching exactly what your professor described:

  BASELINE (what existed before):
      Y(0) = f_theta(I, R)
      -> the model only sees the image and the raw report text, and answers
         however it likes, in free-form text.

  CONTEXT-DRIVEN (what you are adding):
      Y = f_theta(I, R, F_tool)
      -> the model additionally sees the serialized feature card (radiomics +
         XAI + vocabulary), and is FORCED to answer in a strict JSON schema
         containing impression, evidence, uncertainty, limitations, and a
         safety note (Section 3.8, "Responsible AI Constraints").

The prompt-level JSON schema constraint is what actually changes the model's
behavior -- lower hallucination, more concise evidence, calibrated
uncertainty -- as reported in Tables 2 and 3 of the paper.
"""

RESPONSE_SCHEMA_INSTRUCTIONS = """
You must respond with ONLY a single valid JSON object, no other text, in
exactly this schema:

{
  "impression": "<one or two sentence diagnostic impression>",
  "evidence": "<concise supporting visual/textual evidence, ideally under 20 words>",
  "uncertainty": <a number between 0.0 and 1.0>,
  "limitations": "<explicit statement of what limits this assessment, e.g. single view, no prior comparison>",
  "safety_note": "<mandatory disclaimer, e.g. that this is not a substitute for expert radiologist review>"
}

Rules you must follow:
- Do NOT make definitive diagnostic claims. Use cautious, hedged language.
- The "uncertainty" field must reflect genuine calibration, not just a fixed default.
- Always include a non-empty "limitations" field.
- Always include a non-empty "safety_note" field.
"""


def build_baseline_prompt(question: str, report_text: str) -> str:
    """
    Y(0) = f_theta(I, R) -- the "before" version, with no context alignment
    and no structured output requirement. This is your professor's original
    baseline behavior.
    """
    return (
        f"You are a radiology assistant. Given the attached chest X-ray image "
        f"and the following report, answer the question in free text.\n\n"
        f"Report:\n{report_text}\n\n"
        f"Question: {question}\n"
    )


def build_context_aligned_prompt(question: str, report_text: str, feature_card_json: str) -> str:
    """
    Y = f_theta(I, R, F_tool) -- the "after" / context-driven version.

    In addition to the image and report, this prompt injects the serialized
    feature card (radiomics + XAI + vocabulary) as structured evidence the
    model must reconcile, and enforces the JSON output schema from
    Section 3.8 of the paper.
    """
    return (
        f"You are a radiology assistant performing careful, evidence-grounded "
        f"reasoning. You are given a chest X-ray image, a radiology report, "
        f"and an auxiliary evidence 'feature card' derived from independent "
        f"tools (image texture statistics, model attention/explainability "
        f"statistics, and matched clinical vocabulary terms).\n\n"
        f"Your conclusion must be consistent with ALL of these evidence "
        f"sources. If they disagree, state that explicitly in your "
        f"limitations rather than picking one and ignoring the rest.\n\n"
        f"Report:\n{report_text}\n\n"
        f"Feature card (auxiliary evidence):\n{feature_card_json}\n\n"
        f"Question: {question}\n\n"
        f"{RESPONSE_SCHEMA_INSTRUCTIONS}"
    )


def build_stepwise_prompts(question: str, report_text: str, radiomics_json: str, feature_card_json: str) -> dict:
    """
    Builds the three prompts used in the "Stepwise Agentic Reasoning" mode
    (Eqs. 10-12 / Table 3 of the paper), so you can watch how the answer
    changes as more context is added, one signal at a time.

    Returns:
        dict with keys "step0", "step1", "step2".
    """
    step0 = build_baseline_prompt(question, report_text)

    step1 = (
        f"You are a radiology assistant. Given the chest X-ray image, the "
        f"report, and the following image-derived radiomic statistics, "
        f"answer the question.\n\n"
        f"Report:\n{report_text}\n\n"
        f"Radiomic statistics:\n{radiomics_json}\n\n"
        f"Question: {question}\n\n"
        f"{RESPONSE_SCHEMA_INSTRUCTIONS}"
    )

    step2 = build_context_aligned_prompt(question, report_text, feature_card_json)

    return {"step0": step0, "step1": step1, "step2": step2}


if __name__ == "__main__":
    demo_report = "No focal consolidation, pleural effusion, or pneumothorax."
    demo_card = '{"radiomics": {"mean_intensity": 120.0}, "vocabulary": {"num_matched_terms": 3}}'

    print("=== BASELINE PROMPT (before) ===")
    print(build_baseline_prompt("Is there evidence of active cardiopulmonary abnormality?", demo_report))

    print("\n=== CONTEXT-ALIGNED PROMPT (after) ===")
    print(build_context_aligned_prompt(
        "Is there evidence of active cardiopulmonary abnormality?", demo_report, demo_card
    ))
