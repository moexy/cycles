from __future__ import annotations

from cycles.core.types import EstrousStage
from cycles.vlm_local.schema import (
    Abundance,
    Arrangement,
    ConfidenceTier,
    ImagePrediction,
    LocalVLMRecord,
    MorphologyObservation,
    NuclearState,
    QCStatus,
    SequencePrediction,
)
from cycles.vlm_local.temporal import TemporalReconciler


def _record(
    sample: str,
    day: float,
    primary: EstrousStage,
    secondary: EstrousStage,
    probabilities: dict[EstrousStage, float],
    *,
    subject: str | None = "m1",
) -> LocalVLMRecord:
    morphology = MorphologyObservation(
        Abundance.PRESENT,
        Abundance.PRESENT,
        Abundance.RARE,
        NuclearState.MIXED,
        Arrangement.MIXED,
        (),
        QCStatus.USABLE,
        (),
        ("mixed population",),
    )
    prediction = ImagePrediction(
        primary,
        secondary,
        {stage: value for stage, value in probabilities.items()},
        probabilities,
        ConfidenceTier.LOW,
        "ambiguous morphology",
    )
    return LocalVLMRecord(
        sample,
        f"/{sample}.png",
        "a" * 64,
        subject,
        day,
        morphology,
        prediction,
        SequencePrediction(primary, False, "image_only"),
        {"model_id": "test"},
    )


def _probs(**values: float) -> dict[EstrousStage, float]:
    return {
        EstrousStage.DIESTRUS: values.get("d", 0.01),
        EstrousStage.PROESTRUS: values.get("p", 0.01),
        EstrousStage.ESTRUS: values.get("e", 0.01),
        EstrousStage.METESTRUS: values.get("m", 0.01),
    }


def test_reconciler_can_tie_break_uncertain_adjacent_call() -> None:
    records = [
        _record("d1", 1, EstrousStage.PROESTRUS, EstrousStage.ESTRUS, _probs(p=0.90, e=0.07)),
        _record("d2", 2, EstrousStage.METESTRUS, EstrousStage.ESTRUS, _probs(m=0.46, e=0.44, p=0.05, d=0.05)),
        _record("d3", 3, EstrousStage.METESTRUS, EstrousStage.DIESTRUS, _probs(m=0.90, d=0.07)),
    ]

    reconciled = TemporalReconciler(margin_threshold=0.10).reconcile(records)

    assert reconciled[1].image_prediction.primary_stage is EstrousStage.METESTRUS
    assert reconciled[1].sequence_prediction.final_stage is EstrousStage.ESTRUS
    assert reconciled[1].sequence_prediction.adjusted is True
    assert reconciled[1].sequence_prediction.reason == "adjacent_sequence_tiebreak"


def test_reconciler_never_overrides_confident_call() -> None:
    records = [
        _record("d1", 1, EstrousStage.PROESTRUS, EstrousStage.ESTRUS, _probs(p=0.90, e=0.07)),
        _record("d2", 2, EstrousStage.METESTRUS, EstrousStage.ESTRUS, _probs(m=0.85, e=0.10, p=0.03, d=0.02)),
        _record("d3", 3, EstrousStage.METESTRUS, EstrousStage.DIESTRUS, _probs(m=0.90, d=0.07)),
    ]

    reconciled = TemporalReconciler(margin_threshold=0.10).reconcile(records)

    assert reconciled[1].sequence_prediction.final_stage is EstrousStage.METESTRUS
    assert reconciled[1].sequence_prediction.adjusted is False
    assert reconciled[1].sequence_prediction.reason == "confident_image_call"


def test_reconciler_never_switches_between_nonadjacent_stages() -> None:
    records = [
        _record("d1", 1, EstrousStage.PROESTRUS, EstrousStage.ESTRUS, _probs(p=0.90, e=0.07)),
        _record("d2", 2, EstrousStage.DIESTRUS, EstrousStage.ESTRUS, _probs(d=0.46, e=0.44, p=0.05, m=0.05)),
        _record("d3", 3, EstrousStage.METESTRUS, EstrousStage.DIESTRUS, _probs(m=0.90, d=0.07)),
    ]

    reconciled = TemporalReconciler(margin_threshold=0.10).reconcile(records)

    assert reconciled[1].sequence_prediction.final_stage is EstrousStage.DIESTRUS
    assert reconciled[1].sequence_prediction.reason == "nonadjacent_runner_up"


def test_reconciler_leaves_image_only_record_without_sequence_metadata() -> None:
    record = _record(
        "single",
        1,
        EstrousStage.METESTRUS,
        EstrousStage.ESTRUS,
        _probs(m=0.46, e=0.44, p=0.05, d=0.05),
        subject=None,
    )

    reconciled = TemporalReconciler().reconcile([record])

    assert reconciled[0].sequence_prediction.final_stage is EstrousStage.METESTRUS
    assert reconciled[0].sequence_prediction.reason == "no_sequence_metadata"


def test_reconciler_handles_day_gaps() -> None:
    # Gap of 2 days: Proestrus on Day 1 -> Metestrus expected by Day 3
    records = [
        _record("d1", 1, EstrousStage.PROESTRUS, EstrousStage.ESTRUS, _probs(p=0.90, e=0.07)),
        _record("d3", 3, EstrousStage.ESTRUS, EstrousStage.METESTRUS, _probs(e=0.46, m=0.44, p=0.05, d=0.05)),
        _record("d4", 4, EstrousStage.DIESTRUS, EstrousStage.METESTRUS, _probs(d=0.90, m=0.07)),
    ]

    reconciled = TemporalReconciler(margin_threshold=0.10).reconcile(records)

    assert reconciled[1].sequence_prediction.final_stage is EstrousStage.METESTRUS
    assert reconciled[1].sequence_prediction.adjusted is True


def test_reconciler_handles_same_day_replicates() -> None:
    # Multiple images on same day (Day 1): should reinforce stage consistency
    records = [
        _record("d1_spot1", 1, EstrousStage.ESTRUS, EstrousStage.PROESTRUS, _probs(e=0.90, p=0.07)),
        _record("d1_spot2", 1, EstrousStage.PROESTRUS, EstrousStage.ESTRUS, _probs(p=0.46, e=0.44, m=0.05, d=0.05)),
    ]

    reconciled = TemporalReconciler(margin_threshold=0.10).reconcile(records)

    assert reconciled[1].sequence_prediction.final_stage is EstrousStage.ESTRUS
    assert reconciled[1].sequence_prediction.adjusted is True

