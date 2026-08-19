"""Explainable estrous staging from slide-level cellular proportions."""

from __future__ import annotations

import numpy as np

from cycles.core.types import EstrousStage, SlideCellMetrics, StagingResult

STAGE_CENTROIDS: dict[EstrousStage, np.ndarray] = {
    EstrousStage.DIESTRUS: np.array([0.75, 0.12, 0.13], dtype=np.float64),
    EstrousStage.PROESTRUS: np.array([0.08, 0.72, 0.20], dtype=np.float64),
    EstrousStage.ESTRUS: np.array([0.03, 0.12, 0.85], dtype=np.float64),
    EstrousStage.METESTRUS: np.array([0.35, 0.15, 0.50], dtype=np.float64),
}


def classify_stage_calibrated_rules(
    metrics: SlideCellMetrics,
) -> tuple[EstrousStage, bool]:
    """Apply calibrated biological composition rules.

    The boolean reports low cellularity (35 or fewer valid cells). Fewer than
    five cells is a hard QC failure and is never assigned a biological stage.
    """
    count = metrics.valid_cell_count
    if count < 5:
        return EstrousStage.INSUFFICIENT_CELLS, True

    f_l = metrics.leukocyte_fraction
    f_ne = metrics.nucleated_epithelial_fraction
    f_ce = metrics.cornified_squamous_fraction
    low_cell_flag = count <= 35

    # Diestrus: leukocyte dominance. Keep the stronger mixed L/CE phenotype
    # available to the metestrus rule below rather than treating any L plurality
    # as diestrus.
    if f_l >= 0.55 or (f_l >= 0.45 and f_l > f_ne and f_ce < 0.25):
        return EstrousStage.DIESTRUS, low_cell_flag

    # Estrus: overwhelmingly cornified material with very few leukocytes.
    if f_ce >= 0.60 and f_l < 0.15:
        return EstrousStage.ESTRUS, low_cell_flag

    # Proestrus: intact nucleated epithelial cells predominate.
    if f_ne >= 0.40 and f_ne >= f_l and f_ne >= f_ce:
        return EstrousStage.PROESTRUS, low_cell_flag

    # Metestrus: the characteristic returning-leukocyte/cornified mixture.
    if f_l >= 0.15 and f_ce >= 0.25:
        return EstrousStage.METESTRUS, low_cell_flag

    # Less-pure preparations retain a biologically meaningful plurality rule.
    if f_ne >= f_l and f_ne >= f_ce:
        return EstrousStage.PROESTRUS, low_cell_flag
    if f_ce >= f_l:
        return EstrousStage.ESTRUS, low_cell_flag
    return EstrousStage.DIESTRUS, low_cell_flag


def compute_stage_probabilities(
    metrics: SlideCellMetrics,
    *,
    temperature: float = 0.20,
) -> dict[EstrousStage, float]:
    """Softmax negative Euclidean distances to canonical stage centroids."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    stages = tuple(STAGE_CENTROIDS)
    if metrics.valid_cell_count < 5:
        uniform = 1.0 / len(stages)
        return {stage: uniform for stage in stages}

    proportions = np.array(
        [
            metrics.leukocyte_fraction,
            metrics.nucleated_epithelial_fraction,
            metrics.cornified_squamous_fraction,
        ],
        dtype=np.float64,
    )
    logits = np.array(
        [-float(np.linalg.norm(proportions - STAGE_CENTROIDS[stage])) / temperature for stage in stages],
        dtype=np.float64,
    )
    logits -= float(logits.max())
    exponentials = np.exp(logits)
    probabilities = exponentials / float(exponentials.sum())
    return {stage: float(probability) for stage, probability in zip(stages, probabilities, strict=True)}


def _transition_target(
    stage: EstrousStage,
    probabilities: dict[EstrousStage, float],
    metrics: SlideCellMetrics,
) -> EstrousStage | None:
    f_l = metrics.leukocyte_fraction
    f_ne = metrics.nucleated_epithelial_fraction
    f_ce = metrics.cornified_squamous_fraction

    if f_l >= 0.25 and f_ne >= 0.25 and f_ce < 0.30:
        return EstrousStage.PROESTRUS if stage == EstrousStage.DIESTRUS else EstrousStage.DIESTRUS
    if f_ne >= 0.25 and f_ce >= 0.30 and f_l < 0.20:
        return EstrousStage.ESTRUS if stage == EstrousStage.PROESTRUS else EstrousStage.PROESTRUS
    if f_ce >= 0.40 and 0.15 <= f_l <= 0.40:
        return EstrousStage.METESTRUS if stage == EstrousStage.ESTRUS else EstrousStage.ESTRUS
    if f_l >= 0.45 and 0.15 <= f_ce <= 0.40:
        return EstrousStage.METESTRUS if stage == EstrousStage.DIESTRUS else EstrousStage.DIESTRUS

    ranked = sorted(probabilities, key=probabilities.__getitem__, reverse=True)
    first, second = ranked[:2]
    adjacent = {
        frozenset((EstrousStage.DIESTRUS, EstrousStage.PROESTRUS)),
        frozenset((EstrousStage.PROESTRUS, EstrousStage.ESTRUS)),
        frozenset((EstrousStage.ESTRUS, EstrousStage.METESTRUS)),
        frozenset((EstrousStage.METESTRUS, EstrousStage.DIESTRUS)),
    }
    if probabilities[first] - probabilities[second] < 0.12 and frozenset((first, second)) in adjacent:
        return second if stage == first else first
    return None


def determine_stage(metrics: SlideCellMetrics) -> StagingResult:
    """Combine calibrated rules, centroid support, transition QC, and rationale."""
    probabilities = compute_stage_probabilities(metrics)
    stage, low_cell_flag = classify_stage_calibrated_rules(metrics)
    if stage == EstrousStage.INSUFFICIENT_CELLS:
        return StagingResult(
            stage=stage,
            confidence=0.0,
            probabilities=probabilities,
            is_transition=False,
            transition_to=None,
            low_cell_flag=True,
            rationale=(
                f"Only {metrics.valid_cell_count} valid cells were detected; fewer than 5 "
                "fails the cellularity QC guardrail (INSUFFICIENT_CELLS)."
            ),
            metrics=metrics,
        )

    ranked = sorted(probabilities, key=probabilities.__getitem__, reverse=True)
    centroid_stage = ranked[0]
    margin = probabilities[ranked[0]] - probabilities[ranked[1]]
    confidence = 0.55 * probabilities[stage] + 0.45 * max(margin, 0.0)
    if low_cell_flag:
        confidence *= 0.80
    transition_to = _transition_target(stage, probabilities, metrics)

    low_note = " Low-cell-count caution applies." if low_cell_flag else ""
    agreement = (
        f"centroid model agrees ({probabilities[stage]:.1%})"
        if centroid_stage == stage
        else f"centroid model favors {centroid_stage.display_name} ({probabilities[centroid_stage]:.1%})"
    )
    transition_note = (
        f" Composition is compatible with transition toward {transition_to.display_name}."
        if transition_to is not None
        else ""
    )
    rationale = (
        f"{metrics.valid_cell_count} valid cells: leukocytes {metrics.leukocyte_count} "
        f"({metrics.leukocyte_fraction:.1%}), nucleated epithelial "
        f"{metrics.nucleated_epithelial_count} ({metrics.nucleated_epithelial_fraction:.1%}), "
        f"cornified squamous {metrics.cornified_squamous_count} "
        f"({metrics.cornified_squamous_fraction:.1%}). Calibrated rules indicate "
        f"{stage.display_name}; {agreement}.{transition_note}{low_note}"
    )
    return StagingResult(
        stage=stage,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        probabilities=probabilities,
        is_transition=transition_to is not None,
        transition_to=transition_to,
        low_cell_flag=low_cell_flag,
        rationale=rationale,
        metrics=metrics,
    )
