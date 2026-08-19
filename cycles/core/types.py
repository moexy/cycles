"""Shared domain types for estrous staging and cyclicity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class EstrousStage(StrEnum):
    """Canonical estrous stages plus the low-cell quality-control outcome."""

    DIESTRUS = "diestrus"
    PROESTRUS = "proestrus"
    ESTRUS = "estrus"
    METESTRUS = "metestrus"
    INSUFFICIENT_CELLS = "insufficient_cells"

    @property
    def display_name(self) -> str:
        """Return a human-readable stage label."""
        return self.value.replace("_", " ").title()

    @classmethod
    def canonical_stages(cls) -> list[EstrousStage]:
        """Return the four biological stages in canonical cycle order."""
        return [cls.DIESTRUS, cls.PROESTRUS, cls.ESTRUS, cls.METESTRUS]

    @property
    def abbreviation(self) -> str:
        """Return a compact, unambiguous stage label."""
        return {
            EstrousStage.DIESTRUS: "D",
            EstrousStage.PROESTRUS: "P",
            EstrousStage.ESTRUS: "E",
            EstrousStage.METESTRUS: "M",
            EstrousStage.INSUFFICIENT_CELLS: "IC",
        }[self]


class CellType(StrEnum):
    """Cytological cell classes used by cell-centric staging."""

    LEUKOCYTE = "leukocyte"
    NUCLEATED_EPITHELIAL = "nucleated_epithelial"
    CORNIFIED_SQUAMOUS = "cornified_squamous"
    DEBRIS = "debris"

    @property
    def display_name(self) -> str:
        """Return a human-readable cell-type label."""
        return self.value.replace("_", " ").title()

    @classmethod
    def cellular_types(cls) -> list[CellType]:
        """Return biological cell types, excluding non-cellular debris."""
        return [cls.LEUKOCYTE, cls.NUCLEATED_EPITHELIAL, cls.CORNIFIED_SQUAMOUS]


@dataclass(slots=True)
class ClassificationResult:
    """Prediction and confidence information for one image."""

    image_path: Path
    predicted_stage: EstrousStage
    confidence: float
    probabilities: dict[EstrousStage, float]
    confidence_index: float
    is_transition: bool
    transition_to: EstrousStage | None
    raw_logits: list[float] | None = None


@dataclass(slots=True)
class BatchClassificationResult:
    """Results and isolated failures from a batch classification run."""

    results: list[ClassificationResult]
    failed_images: list[tuple[Path, str]]
    total_processed: int
    duration_seconds: float


@dataclass(slots=True)
class CellProfile:
    """Morphometric, intensity, and predicted-type profile for one cell."""

    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    mean_intensity: float
    std_intensity: float
    predicted_type: CellType
    confidence: float = 0.0


@dataclass(slots=True)
class SlideCellMetrics:
    """Cell counts, composition fractions, and profiles for one slide image."""

    total_cells_detected: int
    valid_cell_count: int
    leukocyte_count: int
    nucleated_epithelial_count: int
    cornified_squamous_count: int
    debris_count: int
    leukocyte_fraction: float
    nucleated_epithelial_fraction: float
    cornified_squamous_fraction: float
    mean_cell_area: float
    cell_profiles: list[CellProfile]


@dataclass(slots=True)
class StagingResult:
    """Explainable cell-centric stage assessment for one slide image."""

    stage: EstrousStage
    confidence: float
    probabilities: dict[EstrousStage, float]
    is_transition: bool
    transition_to: EstrousStage | None
    low_cell_flag: bool
    rationale: str
    metrics: SlideCellMetrics


@dataclass(slots=True)
class CheckpointMetadata:
    """Portable metadata stored alongside learned model parameters."""

    architecture: str
    classes: list[str]
    img_size: int
    created_at: str
    epoch: int = 0
    val_acc: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CycleFitResult:
    """Longitudinal cycle-fit summary for one animal."""

    mouse_id: str
    timestamps: list[str]
    stages: list[EstrousStage]
    regularity_score: float
    estimated_cycle_length_days: float
    is_pseudopregnant: bool
    consecutive_diestrus_days: int
    anomalies: list[tuple[int, str]]
