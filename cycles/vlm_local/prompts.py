"""Versioned prompts used by local morphology-first inference."""

from __future__ import annotations

import json

from cycles.vlm_local.schema import MorphologyObservation

PROMPT_VERSION = "morphology-first-v2"


MORPHOLOGY_PROMPT = """You are reviewing five views from one rodent vaginal cytology slide: a whole-field overview followed by four overlapping quadrants.
Describe only visible morphology. Do not assign an estrous stage and do not infer exact percentages.
Return one JSON object with these exact fields:
- cornified_squames, nucleated_epithelial, leukocytes: absent|rare|present|dominant
- nuclear_state: clear_nuclei|ghost_nuclei|anucleate|mixed|not_assessable
- arrangement: isolated|clusters|sheets|mixed|not_assessable
- artifacts: JSON array chosen from visible findings such as mucus, debris, blood, crystals, blur, uneven_stain
- qc_status: usable|low_cellularity|out_of_focus|obscured|ungradable
- qc_reasons: JSON array of short strings
- evidence: JSON array of concise image-grounded observations
Return JSON only."""


def stage_prompt(morphology: MorphologyObservation) -> str:
    payload = {
        "cornified_squames": morphology.cornified_squames.value,
        "nucleated_epithelial": morphology.nucleated_epithelial.value,
        "leukocytes": morphology.leukocytes.value,
        "nuclear_state": morphology.nuclear_state.value,
        "arrangement": morphology.arrangement.value,
        "artifacts": list(morphology.artifacts),
        "qc_status": morphology.qc_status.value,
        "qc_reasons": list(morphology.qc_reasons),
        "evidence": list(morphology.evidence),
    }
    return f"""Score the same slide against the four canonical stages: diestrus, proestrus, estrus, and metestrus.
Use the images and the stage-blind morphology below. Do not invent findings that are absent from the evidence.
Morphology: {json.dumps(payload, sort_keys=True)}
Return JSON only, with these exact fields:
- raw_scores: object with all four stage names as keys and unbounded numeric scores as values
- probabilities: object with all four stage names as keys, each at least 0, together summing to 1
- primary_stage: exactly one of diestrus|proestrus|estrus|metestrus, the stage with the highest probability
- secondary_stage: the stage with the second-highest probability, or null if no runner-up is distinguishable
- confidence_tier: low|medium|high
- rationale: one concise evidence-bound paragraph
Never answer "unknown", "unclear", or "indeterminate", and never return all-zero probabilities. Weak or
conflicting evidence is reported by choosing the best-supported stage and setting confidence_tier to low;
slide-level unreadability was already recorded by the preceding morphology pass."""


def repair_prompt(response: str, error: Exception) -> str:
    return f"""Repair the following response into valid JSON matching the previously requested schema.
Do not add new image findings. Return JSON only.
Validation error: {error}
Response: {response}"""
