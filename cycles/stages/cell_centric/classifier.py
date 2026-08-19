"""Morphometric cell typing and slide-level composition aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cycles.core.types import CellProfile, CellType, SlideCellMetrics


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
        """Return a cell type and confidence from morphology and intensity spread.

        ``nuclear_to_cytoplasmic_ratio`` is supplied by the detector because the
        compact shared ``CellProfile`` stores summary intensity statistics only.
        """
        if (
            profile.predicted_type != CellType.DEBRIS
            and profile.confidence >= self.trust_yolo_confidence
        ):
            return profile.predicted_type, profile.confidence

        area = profile.area
        circularity = profile.circularity
        aspect_ratio = profile.aspect_ratio
        nuclear_ratio = float(np.clip(nuclear_to_cytoplasmic_ratio, 0.0, 1.0))
        intensity_cv = profile.std_intensity / max(profile.mean_intensity, 1.0)

        if area < 15.0 or aspect_ratio > 4.5 or circularity < 0.10:
            return CellType.DEBRIS, 0.90

        # Small round, optically dense cells are predominantly leukocytes. A
        # uniformly dark candidate is retained because a leukocyte can be almost
        # entirely occupied by its nucleus.
        if area <= 450.0 and circularity >= 0.55 and aspect_ratio <= 1.9:
            nuclear_evidence = nuclear_ratio >= 0.22 or profile.mean_intensity <= 125.0
            confidence = 0.68 + 0.18 * circularity + (0.10 if nuclear_evidence else 0.0)
            return CellType.LEUKOCYTE, min(confidence, 0.98)

        # Large, irregular, low-variance and anucleate sheets are cornified.
        if area >= 1500.0:
            anucleate = nuclear_ratio < 0.10
            irregular = circularity < 0.68 or aspect_ratio > 1.55
            homogeneous = intensity_cv < 0.22
            if anucleate or irregular or (area >= 2800.0 and homogeneous):
                confidence = 0.66 + min(area / 12000.0, 0.14)
                confidence += 0.10 * (1.0 - circularity) + (0.07 if anucleate else 0.0)
                return CellType.CORNIFIED_SQUAMOUS, min(confidence, 0.98)

        # Intermediate round/oval cells with an appreciable dark nuclear region
        # are nucleated epithelial cells.
        if 250.0 <= area <= 3500.0:
            nuclear_evidence = nuclear_ratio >= 0.07 or intensity_cv >= 0.16
            if nuclear_evidence and circularity >= 0.30 and aspect_ratio <= 3.0:
                confidence = 0.62 + min(nuclear_ratio, 0.35) * 0.55 + 0.10 * circularity
                return CellType.NUCLEATED_EPITHELIAL, min(confidence, 0.95)

        if area >= 1200.0:
            return CellType.CORNIFIED_SQUAMOUS, 0.58
        if area <= 500.0 and circularity >= 0.42:
            return CellType.LEUKOCYTE, 0.56
        if aspect_ratio <= 3.0 and circularity >= 0.25:
            return CellType.NUCLEATED_EPITHELIAL, 0.54
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
