"""Versioned prompts used by local morphology-first inference."""

from __future__ import annotations

import json

from cyclonaut.vlm_local.schema import MorphologyObservation

PROMPT_VERSION = "morphology-first-v3"


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
    return f"""Score the same slide against these explicit rodent vaginal cytology criteria:
- diestrus: leukocytes dominate; epithelial cells are sparse and cornified squames are few or absent
- proestrus: nucleated epithelial cells dominate; leukocytes are few or absent and cornification is low
- estrus: anucleate cornified squames dominate, often in sheets; leukocytes and nucleated epithelial cells are few or absent
- metestrus: a transitional mixture of cornified squames, nucleated epithelial cells, and leukocytes; cornification alone is not metestrus
Use the stage-blind morphology below as the primary evidence summary and the images only to check it.
Do not invent findings that are absent from the evidence.
Morphology: {json.dumps(payload, sort_keys=True)}
Return JSON only, with these exact fields:
- raw_scores: object with all four stage names as keys and finite relative evidence scores as values; these are not probabilities
- rationale: one concise evidence-bound paragraph
Do not calculate probabilities, choose a stage label, or report confidence; those fields are derived
deterministically from raw_scores. Slide-level unreadability was already recorded by the morphology pass."""


def repair_prompt(original_prompt: str, response: str, error: Exception) -> str:
    return f"""Repair the following response into valid JSON matching this original request:
{original_prompt}

Do not add new image findings. Return JSON only.
Validation error: {error}
Response: {response}"""
