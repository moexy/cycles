from __future__ import annotations

from cycles.core.types import EstrousStage
from cycles.vlm_local.probes import PROBE_DISCLAIMER, morphology_sensitivity_cases


def test_morphology_sensitivity_probe_is_explicitly_not_ground_truth() -> None:
    cases = morphology_sensitivity_cases()

    assert {case.expected_stage for case in cases} == set(EstrousStage.canonical_stages())
    assert len({case.case_id for case in cases}) == 4
    assert "not independently annotated ground truth" in PROBE_DISCLAIMER
    assert "not biological accuracy" in PROBE_DISCLAIMER


def test_morphology_sensitivity_cases_encode_distinct_textbook_patterns() -> None:
    by_stage = {case.expected_stage: case.morphology for case in morphology_sensitivity_cases()}

    assert by_stage[EstrousStage.ESTRUS].cornified_squames.value == "dominant"
    assert by_stage[EstrousStage.DIESTRUS].leukocytes.value == "dominant"
    assert by_stage[EstrousStage.PROESTRUS].nucleated_epithelial.value == "dominant"
    assert by_stage[EstrousStage.METESTRUS].arrangement.value == "mixed"
