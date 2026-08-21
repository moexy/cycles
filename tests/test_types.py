from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from cyclonaut.core.types import (
    BatchClassificationResult,
    CellProfile,
    CellType,
    CheckpointMetadata,
    ClassificationResult,
    CycleFitResult,
    EstrousStage,
    SlideCellMetrics,
    StagingResult,
)


def test_estrous_stage_labels_and_canonical_order() -> None:
    assert EstrousStage.INSUFFICIENT_CELLS.display_name == "Insufficient Cells"
    assert [stage.value for stage in EstrousStage.canonical_stages()] == [
        "diestrus",
        "proestrus",
        "estrus",
        "metestrus",
    ], "Canonical stages should exclude the QC-only insufficient-cells value"
    assert [stage.abbreviation for stage in EstrousStage] == ["D", "P", "E", "M", "IC"]


def test_cell_type_labels_and_cellular_types() -> None:
    assert CellType.NUCLEATED_EPITHELIAL.display_name == "Nucleated Epithelial"
    assert CellType.cellular_types() == [
        CellType.LEUKOCYTE,
        CellType.NUCLEATED_EPITHELIAL,
        CellType.CORNIFIED_SQUAMOUS,
    ], "Debris is not a biological cell type"


def _profile(cell_type: CellType = CellType.LEUKOCYTE) -> CellProfile:
    return CellProfile(
        bbox=(1, 2, 8, 10),
        centroid=(4.5, 6.0),
        area=42.0,
        perimeter=24.0,
        circularity=0.92,
        aspect_ratio=1.2,
        mean_intensity=110.0,
        std_intensity=12.0,
        predicted_type=cell_type,
        confidence=0.88,
    )


def _metrics(profile: CellProfile) -> SlideCellMetrics:
    return SlideCellMetrics(
        total_cells_detected=2,
        valid_cell_count=1,
        leukocyte_count=1,
        nucleated_epithelial_count=0,
        cornified_squamous_count=0,
        debris_count=1,
        leukocyte_fraction=1.0,
        nucleated_epithelial_fraction=0.0,
        cornified_squamous_fraction=0.0,
        mean_cell_area=42.0,
        cell_profiles=[profile],
    )


def test_result_dataclasses_preserve_nested_domain_values(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.png"
    probabilities = {stage: 0.25 for stage in EstrousStage.canonical_stages()}
    classification = ClassificationResult(
        image_path=image_path,
        predicted_stage=EstrousStage.DIESTRUS,
        confidence=0.25,
        probabilities=probabilities,
        confidence_index=0.0,
        is_transition=True,
        transition_to=EstrousStage.PROESTRUS,
        raw_logits=[0.0] * 4,
    )
    batch = BatchClassificationResult(
        results=[classification],
        failed_images=[(tmp_path / "bad.png", "ValueError: corrupt")],
        total_processed=2,
        duration_seconds=0.5,
    )
    profile = _profile()
    metrics = _metrics(profile)
    staging = StagingResult(
        stage=EstrousStage.DIESTRUS,
        confidence=0.9,
        probabilities=probabilities,
        is_transition=False,
        transition_to=None,
        low_cell_flag=True,
        rationale="Leukocytes dominate.",
        metrics=metrics,
    )

    assert batch.results[0].image_path == image_path
    assert batch.failed_images[0][1].startswith("ValueError")
    assert staging.metrics.cell_profiles[0] is profile
    assert asdict(profile)["predicted_type"] is CellType.LEUKOCYTE


def test_checkpoint_metadata_has_independent_metrics_dicts() -> None:
    first = CheckpointMetadata("resnet50", ["a"], 224, "2026-01-01T00:00:00Z")
    second = CheckpointMetadata("resnet50", ["a"], 224, "2026-01-01T00:00:00Z")
    first.metrics["loss"] = 1.0

    assert second.metrics == {}, "default_factory must not share metrics between checkpoints"
    assert first.epoch == 0 and first.val_acc == pytest.approx(0.0)


def test_cycle_fit_result_retains_longitudinal_summary() -> None:
    result = CycleFitResult(
        mouse_id="M17",
        timestamps=["2026-01-01", "2026-01-02"],
        stages=[EstrousStage.PROESTRUS, EstrousStage.ESTRUS],
        regularity_score=1.0,
        estimated_cycle_length_days=4.2,
        is_pseudopregnant=False,
        consecutive_diestrus_days=0,
        anomalies=[(1, "unexpected")],
    )

    assert result.mouse_id == "M17"
    assert result.stages[-1] is EstrousStage.ESTRUS
    assert result.anomalies == [(1, "unexpected")]
