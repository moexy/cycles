"""Controlled, non-ground-truth probes for the morphology-to-stage pass."""

from __future__ import annotations

from dataclasses import dataclass

from cyclonaut.core.types import EstrousStage
from cyclonaut.vlm_local.schema import (
    Abundance,
    Arrangement,
    MorphologyObservation,
    NuclearState,
    QCStatus,
)

PROBE_DISCLAIMER = (
    "Expected stages are textbook design expectations, not independently annotated ground truth. "
    "The fixed source image can conflict with the counterfactual morphology; exact matches measure "
    "prompt sensitivity, not biological accuracy."
)


@dataclass(frozen=True, slots=True)
class MorphologyProbeCase:
    case_id: str
    expected_stage: EstrousStage
    morphology: MorphologyObservation


def morphology_sensitivity_cases() -> tuple[MorphologyProbeCase, ...]:
    """Return four explicit counterfactuals while holding the image constant."""
    return (
        MorphologyProbeCase(
            "textbook_estrus",
            EstrousStage.ESTRUS,
            _morphology(
                "dominant",
                "absent",
                "absent",
                "anucleate",
                "sheets",
                "Dense sheets of anucleate cornified squames, with no leukocytes or nucleated epithelial cells.",
            ),
        ),
        MorphologyProbeCase(
            "textbook_diestrus",
            EstrousStage.DIESTRUS,
            _morphology(
                "absent",
                "rare",
                "dominant",
                "clear_nuclei",
                "isolated",
                "Overwhelmingly leukocytes, with rare nucleated epithelial cells and no cornified squames.",
            ),
        ),
        MorphologyProbeCase(
            "textbook_proestrus",
            EstrousStage.PROESTRUS,
            _morphology(
                "rare",
                "dominant",
                "absent",
                "clear_nuclei",
                "clusters",
                "Uniform clusters of round nucleated epithelial cells with clear nuclei and almost no leukocytes.",
            ),
        ),
        MorphologyProbeCase(
            "textbook_metestrus",
            EstrousStage.METESTRUS,
            _morphology(
                "present",
                "present",
                "present",
                "mixed",
                "mixed",
                "A mixture of cornified squames, nucleated epithelial cells, and abundant leukocytes.",
            ),
        ),
    )


def _morphology(
    cornified: str,
    nucleated: str,
    leukocytes: str,
    nuclear_state: str,
    arrangement: str,
    evidence: str,
) -> MorphologyObservation:
    return MorphologyObservation(
        cornified_squames=Abundance(cornified),
        nucleated_epithelial=Abundance(nucleated),
        leukocytes=Abundance(leukocytes),
        nuclear_state=NuclearState(nuclear_state),
        arrangement=Arrangement(arrangement),
        artifacts=(),
        qc_status=QCStatus.USABLE,
        qc_reasons=(),
        evidence=(evidence,),
    )
