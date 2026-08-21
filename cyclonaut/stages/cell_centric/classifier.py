"""Morphometric cell typing and slide-level composition aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cyclonaut.core.types import CellProfile, CellType, SlideCellMetrics


class CellClassifier:
    """Classify cell profiles with interpretable cytomorphology rules."""

    def __init__(self, *, trust_yolo_confidence: float = 0.25) -> None:
        if not 0.0 <= trust_yolo_confidence <= 1.0:
            raise ValueError("trust_yolo_confidence must be between 0 and 1")
        self.trust_yolo_confidence = float(trust_yolo_confidence)

    def classify_cell(
        self,
        profile: CellProfile,
        nuclear_to_cytoplasmic_ratio: float = 0.0,
    ) -> tuple[CellType, float]:
        """Return a cell type and confidence from morphology and intensity spread."""
        area = profile.area
        circularity = profile.circularity
        aspect_ratio = profile.aspect_ratio
        nuclear_ratio = float(np.clip(nuclear_to_cytoplasmic_ratio, 0.0, 1.0))
        intensity_cv = profile.std_intensity / max(profile.mean_intensity, 1.0)
        initial_type = profile.predicted_type

        if area < 15.0 or aspect_ratio > 5.0 or circularity < 0.08:
            return CellType.DEBRIS, 0.90

        # 1. Biological verification for upstream (YOLO/detector) predictions
        if initial_type != CellType.DEBRIS and profile.confidence >= self.trust_yolo_confidence:
            # Guardrail: Small dense objects cannot be cornified squamous cells
            if initial_type == CellType.CORNIFIED_SQUAMOUS and area <= 450.0:
                if nuclear_ratio >= 0.20 or profile.mean_intensity <= 130.0:
                    return CellType.LEUKOCYTE, 0.88
                return CellType.DEBRIS, 0.70

            # Guardrail: Intermediate cells with prominent nucleus are nucleated epithelial
            if initial_type == CellType.CORNIFIED_SQUAMOUS and 200.0 <= area <= 3000.0 and nuclear_ratio >= 0.08:
                return CellType.NUCLEATED_EPITHELIAL, 0.85

            # Guardrail: Huge cells cannot be single leukocytes
            if initial_type == CellType.LEUKOCYTE and area >= 800.0:
                if nuclear_ratio < 0.06:
                    return CellType.CORNIFIED_SQUAMOUS, 0.85
                return CellType.NUCLEATED_EPITHELIAL, 0.80

            # Guardrail: Large anucleate sheets cannot be nucleated epithelial
            if initial_type == CellType.NUCLEATED_EPITHELIAL and area >= 1800.0 and nuclear_ratio < 0.04:
                return CellType.CORNIFIED_SQUAMOUS, 0.88

            return initial_type, profile.confidence

        # 2. Pure morphometry decision rules
        # Leukocyte: Small round/dense cells
        if area <= 450.0 and circularity >= 0.45 and aspect_ratio <= 2.2:
            nuclear_evidence = nuclear_ratio >= 0.18 or profile.mean_intensity <= 130.0
            if nuclear_evidence:
                confidence = 0.72 + 0.18 * circularity
                return CellType.LEUKOCYTE, min(confidence, 0.98)

        # Cornified Squamous: Large, flat, anucleate sheets
        if area >= 1000.0:
            anucleate = nuclear_ratio < 0.08
            irregular = circularity < 0.70 or aspect_ratio > 1.40
            if anucleate or irregular or area >= 2500.0:
                confidence = 0.70 + min(area / 10000.0, 0.15) + (0.08 if anucleate else 0.0)
                return CellType.CORNIFIED_SQUAMOUS, min(confidence, 0.98)

        # Nucleated Epithelial: Intermediate cells with nucleus
        if 200.0 <= area <= 3500.0:
            nuclear_evidence = nuclear_ratio >= 0.06 or intensity_cv >= 0.15
            if nuclear_evidence and circularity >= 0.25 and aspect_ratio <= 3.2:
                confidence = 0.65 + min(nuclear_ratio, 0.35) * 0.50 + 0.10 * circularity
                return CellType.NUCLEATED_EPITHELIAL, min(confidence, 0.95)

        if area >= 800.0 and nuclear_ratio < 0.06:
            return CellType.CORNIFIED_SQUAMOUS, 0.60
        if area <= 400.0 and circularity >= 0.40:
            return CellType.LEUKOCYTE, 0.58
        if aspect_ratio <= 3.0 and circularity >= 0.20:
            return CellType.NUCLEATED_EPITHELIAL, 0.55
        return CellType.DEBRIS, 0.65
    def classify(
        self,
        profiles: list[CellProfile],
        *,
        nuclear_to_cytoplasmic_ratios: Sequence[float] | None = None,
    ) -> list[CellProfile]:
        """Classify profiles in place and return them for convenient chaining."""
        ratios = () if nuclear_to_cytoplasmic_ratios is None else nuclear_to_cytoplasmic_ratios
        for index, profile in enumerate(profiles):
            ratio = float(ratios[index]) if index < len(ratios) else 0.0
            predicted_type, confidence = self.classify_cell(profile, ratio)
            profile.predicted_type = predicted_type
            profile.confidence = float(confidence)
        return profiles

    def aggregate(self, profiles: list[CellProfile]) -> SlideCellMetrics:
        """Aggregate classified cells into counts, proportions, and mean area."""
        leukocytes = sum(p.predicted_type == CellType.LEUKOCYTE for p in profiles)
        nucleated = sum(p.predicted_type == CellType.NUCLEATED_EPITHELIAL for p in profiles)
        cornified = sum(p.predicted_type == CellType.CORNIFIED_SQUAMOUS for p in profiles)
        debris = sum(p.predicted_type == CellType.DEBRIS for p in profiles)
        valid_count = int(leukocytes + nucleated + cornified)

        if valid_count:
            inverse_count = 1.0 / valid_count
            leukocyte_fraction = float(leukocytes * inverse_count)
            nucleated_fraction = float(nucleated * inverse_count)
            cornified_fraction = float(cornified * inverse_count)
            mean_area = float(
                np.mean([p.area for p in profiles if p.predicted_type != CellType.DEBRIS])
            )
        else:
            leukocyte_fraction = nucleated_fraction = cornified_fraction = mean_area = 0.0

        return SlideCellMetrics(
            total_cells_detected=len(profiles),
            valid_cell_count=valid_count,
            leukocyte_count=int(leukocytes),
            nucleated_epithelial_count=int(nucleated),
            cornified_squamous_count=int(cornified),
            debris_count=int(debris),
            leukocyte_fraction=leukocyte_fraction,
            nucleated_epithelial_fraction=nucleated_fraction,
            cornified_squamous_fraction=cornified_fraction,
            mean_cell_area=mean_area,
            cell_profiles=profiles,
        )
