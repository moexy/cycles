from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyclonaut.core.types import EstrousStage
from cyclonaut.vlm_local.annotations import AnnotationStore, record_hash
from cyclonaut.vlm_local.schema import (
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


def _record(sample_id: str = "sample-1") -> LocalVLMRecord:
    probabilities = {
        EstrousStage.DIESTRUS: 0.05,
        EstrousStage.PROESTRUS: 0.05,
        EstrousStage.ESTRUS: 0.85,
        EstrousStage.METESTRUS: 0.05,
    }
    return LocalVLMRecord(
        sample_id,
        f"/{sample_id}.png",
        "b" * 64,
        "mouse-1",
        3,
        MorphologyObservation(
            Abundance.DOMINANT,
            Abundance.RARE,
            Abundance.ABSENT,
            NuclearState.ANUCLEATE,
            Arrangement.SHEETS,
            (),
            QCStatus.USABLE,
            (),
            ("cornified sheets",),
        ),
        ImagePrediction(
            EstrousStage.ESTRUS,
            EstrousStage.METESTRUS,
            probabilities,
            probabilities,
            ConfidenceTier.HIGH,
            "cornified sheets",
        ),
        SequencePrediction(EstrousStage.ESTRUS, False, "image_only"),
        {"model_id": "test/model"},
    )


def test_annotation_store_appends_versioned_events_without_rewriting(tmp_path: Path) -> None:
    record = _record()
    store = AnnotationStore(tmp_path / "reviews.jsonl")

    first = store.append(record, reviewer_id="reviewer-a", action="accept")
    second = store.append(
        record,
        reviewer_id="reviewer-a",
        action="correct",
        corrections={"primary_stage": "metestrus"},
        note="Leukocytes visible in two quadrants",
    )

    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert first.event_id != second.event_id
    assert first.record_hash == second.record_hash == record_hash(record)
    assert store.latest_by_sample()[record.sample_id].event_id == second.event_id
    assert json.loads(lines[0])["action"] == "accept"
    assert json.loads(lines[1])["corrections"]["primary_stage"] == "metestrus"


def test_frozen_export_uses_latest_review_and_preserves_model_record(tmp_path: Path) -> None:
    record = _record()
    store = AnnotationStore(tmp_path / "reviews.jsonl")
    store.append(record, reviewer_id="reviewer-a", action="accept")
    store.append(
        record,
        reviewer_id="reviewer-b",
        action="correct",
        corrections={"primary_stage": "metestrus", "confidence_tier": "medium"},
    )

    summary = store.export_teacher([record], tmp_path / "teacher-v1")

    payload = json.loads((tmp_path / "teacher-v1" / "dataset.jsonl").read_text())
    assert summary["exported_samples"] == 1
    assert payload["model_record"]["image_prediction"]["primary_stage"] == "estrus"
    assert payload["teacher_label"]["primary_stage"] == "metestrus"
    assert payload["teacher_label"]["secondary_stage"] is None
    assert payload["teacher_label"]["confidence_tier"] == "medium"
    assert len((tmp_path / "teacher-v1" / "manifest.sha256").read_text().split()[0]) == 64


def test_frozen_export_skips_deferred_samples_and_refuses_overwrite(tmp_path: Path) -> None:
    record = _record()
    store = AnnotationStore(tmp_path / "reviews.jsonl")
    store.append(record, reviewer_id="reviewer-a", action="defer")

    summary = store.export_teacher([record], tmp_path / "teacher-v1")

    assert summary["exported_samples"] == 0
    with pytest.raises(FileExistsError, match="already exists"):
        store.export_teacher([record], tmp_path / "teacher-v1")


def test_annotation_store_rejects_unversioned_or_unknown_corrections(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "reviews.jsonl")
    with pytest.raises(ValueError, match="unsupported correction field"):
        store.append(
            _record(),
            reviewer_id="reviewer-a",
            action="correct",
            corrections={"model_id": "mutated"},
        )

    with pytest.raises(ValueError, match="primary_stage"):
        store.append(
            _record(),
            reviewer_id="reviewer-a",
            action="correct",
            corrections={"primary_stage": "not-a-stage"},
        )


def test_annotation_store_rejects_corrections_on_non_correction_actions(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "reviews.jsonl")

    with pytest.raises(ValueError, match="only valid for correct"):
        store.append(
            _record(),
            reviewer_id="reviewer-a",
            action="accept",
            corrections={"primary_stage": "metestrus"},
        )


def test_annotation_store_revalidates_events_read_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    event = AnnotationStore(path).append(
        _record(), reviewer_id="reviewer-a", action="accept"
    ).to_dict()
    event["action"] = "silently_mutated"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid annotation event"):
        AnnotationStore(path).events()
