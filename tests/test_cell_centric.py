from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.draw import disk

from cycles.core.types import CellProfile, CellType, EstrousStage, SlideCellMetrics
from cycles.stages.cell_centric.classifier import CellClassifier
from cycles.stages.cell_centric.detector import CellDetector
from cycles.stages.cell_centric.pipeline import CellCentricPipeline
from cycles.stages.cell_centric.staging import classify_stage_calibrated_rules, determine_stage


def _profile(
    *,
    area: float,
    circularity: float,
    aspect_ratio: float,
    mean: float,
    std: float = 10.0,
    predicted_type: CellType = CellType.DEBRIS,
    confidence: float = 0.0,
    bbox: tuple[int, int, int, int] = (5, 5, 20, 20),
) -> CellProfile:
    return CellProfile(
        bbox=bbox,
        centroid=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        area=area,
        perimeter=50.0,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        mean_intensity=mean,
        std_intensity=std,
        predicted_type=predicted_type,
        confidence=confidence,
    )


def _metrics(leukocytes: int, nucleated: int, cornified: int, debris: int = 0) -> SlideCellMetrics:
    valid = leukocytes + nucleated + cornified
    divisor = max(valid, 1)
    return SlideCellMetrics(
        total_cells_detected=valid + debris,
        valid_cell_count=valid,
        leukocyte_count=leukocytes,
        nucleated_epithelial_count=nucleated,
        cornified_squamous_count=cornified,
        debris_count=debris,
        leukocyte_fraction=leukocytes / divisor,
        nucleated_epithelial_fraction=nucleated / divisor,
        cornified_squamous_fraction=cornified / divisor,
        mean_cell_area=100.0 if valid else 0.0,
        cell_profiles=[],
    )


def test_cell_detector_morphometry_watershed_finds_separate_dark_cells() -> None:
    image = np.full((128, 128, 3), 245, dtype=np.uint8)
    for center in ((32, 32), (32, 92), (92, 45), (88, 95)):
        rows, cols = disk(center, 9, shape=image.shape[:2])
        image[rows, cols] = (55, 45, 70)
    detector = CellDetector(mode="morphometry", min_area=30)

    profiles = detector.detect(image)

    assert len(profiles) >= 4, "Watershed should keep four isolated, internal cell candidates"
    assert detector.last_labels is not None and int(detector.last_labels.max()) == len(profiles)
    assert len(detector.last_nuclear_ratios) == len(profiles)
    assert all(profile.area >= 30 for profile in profiles)


def test_cell_detector_falls_back_when_yolo_weights_are_missing(tmp_path: Path) -> None:
    detector = CellDetector(mode="yolo", yolo_weights_path=tmp_path / "missing.pt")
    blank = np.full((32, 32, 3), 255, dtype=np.uint8)

    profiles = detector.detect(blank)

    assert detector.mode == "morphometry"
    assert detector.fallback_reason == "YOLO weights were not found"
    assert profiles == [], "Fallback morphometry should safely accept a blank slide"


def test_cell_detector_falls_back_after_yolo_inference_failure(mocker) -> None:
    detector = CellDetector(mode="morphometry")
    detector.mode = "yolo"
    detector._yolo_model = mocker.Mock()
    mocker.patch.object(detector, "_detect_yolo", side_effect=RuntimeError("backend failed"))
    fallback = mocker.patch.object(detector, "_detect_morphometry", return_value=[])

    assert detector.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []
    assert detector.mode == "morphometry" and "backend failed" in (detector.fallback_reason or "")
    fallback.assert_called_once()


def test_cell_classifier_types_profiles_and_aggregates_metrics() -> None:
    profiles = [
        _profile(area=180, circularity=0.85, aspect_ratio=1.1, mean=90),
        _profile(area=800, circularity=0.75, aspect_ratio=1.3, mean=150, std=35),
        _profile(area=2200, circularity=0.45, aspect_ratio=2.0, mean=190, std=8),
        _profile(area=8, circularity=0.05, aspect_ratio=5.0, mean=100),
    ]
    classifier = CellClassifier()

    classifier.classify(profiles, nuclear_to_cytoplasmic_ratios=[0.5, 0.2, 0.01, 0.0])
    metrics = classifier.aggregate(profiles)

    assert [profile.predicted_type for profile in profiles] == [
        CellType.LEUKOCYTE,
        CellType.NUCLEATED_EPITHELIAL,
        CellType.CORNIFIED_SQUAMOUS,
        CellType.DEBRIS,
    ]
    assert metrics.total_cells_detected == 4 and metrics.valid_cell_count == 3
    assert metrics.debris_count == 1
    assert metrics.leukocyte_fraction == pytest.approx(1 / 3)
    assert metrics.nucleated_epithelial_fraction == pytest.approx(1 / 3)
    assert metrics.cornified_squamous_fraction == pytest.approx(1 / 3)
    assert metrics.mean_cell_area == pytest.approx((180 + 800 + 2200) / 3)


def test_cell_classifier_preserves_confident_yolo_typing() -> None:
    profile = _profile(
        area=2000,
        circularity=0.55,
        aspect_ratio=1.4,
        mean=160,
        predicted_type=CellType.CORNIFIED_SQUAMOUS,
        confidence=0.95,
    )
    cell_type, confidence = CellClassifier().classify_cell(profile, 0.0)
    assert cell_type is CellType.CORNIFIED_SQUAMOUS and confidence == pytest.approx(0.95)

    # Guardrail: Small dense objects predicted as Cornified by YOLO are corrected to Leukocyte
    dense_profile = _profile(
        area=100,
        circularity=0.9,
        aspect_ratio=1.0,
        mean=80,
        predicted_type=CellType.CORNIFIED_SQUAMOUS,
        confidence=0.85,
    )
    guardrail_type, _ = CellClassifier().classify_cell(dense_profile, 0.9)
    assert guardrail_type is CellType.LEUKOCYTE

@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (_metrics(8, 1, 1), EstrousStage.DIESTRUS),
        (_metrics(1, 8, 1), EstrousStage.PROESTRUS),
        (_metrics(0, 1, 9), EstrousStage.ESTRUS),
        (_metrics(4, 1, 5), EstrousStage.METESTRUS),
    ],
)
def test_calibrated_stage_rules_cover_canonical_compositions(
    metrics: SlideCellMetrics,
    expected: EstrousStage,
) -> None:
    stage, low_cell = classify_stage_calibrated_rules(metrics)
    assert stage is expected
    assert low_cell is True, "Ten valid cells should retain the low-cell caution"


def test_determine_stage_enforces_insufficient_cells_qc_guardrail() -> None:
    result = determine_stage(_metrics(2, 1, 1))

    assert result.stage is EstrousStage.INSUFFICIENT_CELLS
    assert result.confidence == 0.0 and result.low_cell_flag
    assert result.transition_to is None and not result.is_transition
    assert "INSUFFICIENT_CELLS" in result.rationale


def test_cell_centric_pipeline_generates_overlay_and_csv(tmp_path: Path, mocker) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (64, 64), (235, 225, 240)).save(image_path)
    pipeline = CellCentricPipeline(detector_mode="morphometry")
    labels = np.zeros((64, 64), dtype=np.int32)
    profiles: list[CellProfile] = []
    for index in range(6):
        row = 5 + (index // 3) * 25
        col = 5 + (index % 3) * 18
        bbox = (row, col, row + 10, col + 10)
        labels[row : row + 10, col : col + 10] = index + 1
        profiles.append(
            _profile(area=100, circularity=0.9, aspect_ratio=1.0, mean=80, bbox=bbox)
        )
    pipeline.detector.last_rgb = np.asarray(Image.open(image_path)).copy()
    pipeline.detector.last_labels = labels
    pipeline.detector.last_nuclear_ratios = [0.5] * len(profiles)
    mocker.patch.object(pipeline.detector, "detect", return_value=profiles)
    overlay = tmp_path / "outputs" / "overlay.png"

    result = pipeline.process_image(image_path, overlay)
    csv_path = pipeline.export_results_csv([result], tmp_path / "outputs" / "results.csv")

    assert result.stage is EstrousStage.DIESTRUS
    assert overlay.is_file() and Image.open(overlay).size == (64, 64)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["image_path"] == str(image_path)
    assert rows[0]["stage"] == "diestrus" and rows[0]["valid_cell_count"] == "6"
