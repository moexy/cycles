"""End-to-end cell-centric estrous staging and result export."""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

from cyclonaut.core.types import CellType, EstrousStage, StagingResult
from cyclonaut.stages.cell_centric.classifier import CellClassifier
from cyclonaut.stages.cell_centric.detector import CellDetector, DetectorMode
from cyclonaut.stages.cell_centric.staging import determine_stage

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CELL_TYPE_COLORS: dict[CellType, tuple[int, int, int]] = {
    CellType.LEUKOCYTE: (36, 120, 255),
    CellType.NUCLEATED_EPITHELIAL: (34, 180, 85),
    CellType.CORNIFIED_SQUAMOUS: (245, 95, 35),
    CellType.DEBRIS: (145, 145, 145),
}


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


class CellCentricPipeline:
    """Detect, type, aggregate, and stage cells in vaginal cytology images."""

    def __init__(
        self,
        detector_mode: DetectorMode = "auto",
        yolo_weights_path: Path | str | None = None,
        *,
        device: str | None = None,
        is_phase_contrast: bool = False,
        confidence_threshold: float = 0.15,
    ) -> None:
        self.detector = CellDetector(
            mode=detector_mode,
            yolo_weights_path=yolo_weights_path,
            device=device,
            confidence_threshold=confidence_threshold,
            is_phase_contrast=is_phase_contrast,
        )
        self.classifier = CellClassifier()
        self.processing_errors: list[tuple[Path, str]] = []
        self._result_paths: dict[int, Path] = {}

    def process_image(
        self,
        image_path: Path | str,
        save_overlay_path: Path | str | None = None,
    ) -> StagingResult:
        """Process one image and optionally save a self-contained annotated overlay."""
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        profiles = self.detector.detect(path)
        self.classifier.classify(
            profiles,
            nuclear_to_cytoplasmic_ratios=self.detector.last_nuclear_ratios,
        )
        result = determine_stage(self.classifier.aggregate(profiles))
        self._result_paths[id(result)] = path

        if save_overlay_path is not None:
            overlay_path = Path(save_overlay_path).expanduser()
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            self._render_overlay(result).save(overlay_path)
        return result

    def process_folder(
        self,
        folder_path: Path | str,
        save_overlays_dir: Path | str | None = None,
        recursive: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[StagingResult]:
        """Process a folder with per-image error isolation.

        Failures are recorded in ``processing_errors`` and do not prevent later
        images from being staged.
        """
        folder = Path(folder_path).expanduser()
        if not folder.is_dir():
            raise NotADirectoryError(f"Image folder not found: {folder}")
        iterator = folder.rglob("*") if recursive else folder.iterdir()
        image_paths = sorted(
            (
                path
                for path in iterator
                if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=_natural_key,
        )
        overlay_dir = Path(save_overlays_dir).expanduser() if save_overlays_dir else None
        if overlay_dir is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)

        self.processing_errors = []
        self._result_paths.clear()
        results: list[StagingResult] = []
        total = len(image_paths)
        for index, path in enumerate(image_paths, start=1):
            if progress_callback is not None:
                progress_callback(index - 1, total, f"Processing {path.name}")
            overlay_path = overlay_dir / f"{path.stem}_cells.png" if overlay_dir else None
            try:
                results.append(self.process_image(path, overlay_path))
            except Exception as exc:  # A corrupt slide must not abort the batch.
                self.processing_errors.append((path, f"{type(exc).__name__}: {exc}"))
        if progress_callback is not None:
            progress_callback(total, total, "Cell-centric processing complete")
        return results

    def _render_overlay(self, result: StagingResult) -> Image.Image:
        rgb = self.detector.last_rgb
        if rgb is None:
            raise RuntimeError("No detector image is available for overlay rendering")
        canvas = rgb.copy()
        profiles = result.metrics.cell_profiles
        labels = self.detector.last_labels

        if labels is not None and labels.shape == canvas.shape[:2] and profiles:
            color_lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
            for label, profile in enumerate(profiles, start=1):
                if label < len(color_lut):
                    color_lut[label] = CELL_TYPE_COLORS[profile.predicted_type]
            cell_pixels = labels > 0
            colors = color_lut[labels]
            canvas[cell_pixels] = (
                0.72 * canvas[cell_pixels] + 0.28 * colors[cell_pixels]
            ).astype(np.uint8)
            boundary = find_boundaries(labels, mode="inner")
            canvas[boundary] = colors[boundary]

        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image, "RGBA")
        line_width = max(2, round(min(image.size) / 500))
        font = ImageFont.load_default()
        for profile in profiles:
            y1, x1, y2, x2 = profile.bbox
            color = (*CELL_TYPE_COLORS[profile.predicted_type], 255)
            draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=color, width=line_width)

        self._draw_summary(draw, result, font)
        return image

    @staticmethod
    def _draw_summary(
        draw: ImageDraw.ImageDraw,
        result: StagingResult,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> None:
        metrics = result.metrics
        transition = (
            f" -> {result.transition_to.display_name}" if result.transition_to is not None else ""
        )
        lines = [
            f"Stage: {result.stage.display_name}{transition} ({result.confidence:.0%})",
            f"Valid: {metrics.valid_cell_count}   Debris: {metrics.debris_count}",
            f"L {metrics.leukocyte_count}   NE {metrics.nucleated_epithelial_count}   CE {metrics.cornified_squamous_count}",
        ]
        legend = (
            (CellType.LEUKOCYTE, "Leukocyte"),
            (CellType.NUCLEATED_EPITHELIAL, "Nucleated epithelial"),
            (CellType.CORNIFIED_SQUAMOUS, "Cornified squamous"),
            (CellType.DEBRIS, "Debris"),
        )
        sample_box = draw.textbbox((0, 0), max(lines + [name for _, name in legend], key=len), font=font)
        row_height = max(15, sample_box[3] - sample_box[1] + 5)
        panel_width = max(260, sample_box[2] - sample_box[0] + 45)
        panel_height = 10 + row_height * (len(lines) + len(legend))
        draw.rounded_rectangle((8, 8, 8 + panel_width, 8 + panel_height), radius=6, fill=(0, 0, 0, 185))
        y = 13
        for line in lines:
            draw.text((15, y), line, fill=(255, 255, 255, 255), font=font)
            y += row_height
        for cell_type, name in legend:
            color = CELL_TYPE_COLORS[cell_type]
            draw.rectangle((15, y + 2, 25, y + 12), fill=(*color, 255))
            draw.text((31, y), name, fill=(255, 255, 255, 255), font=font)
            y += row_height

    def export_results_csv(
        self,
        results: list[StagingResult],
        output_path: Path | str,
    ) -> Path:
        """Export staging decisions, composition, probabilities, and rationale."""
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        probability_columns = tuple(EstrousStage.canonical_stages())
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "image_path", "stage", "confidence", "is_transition", "transition_to",
                    "low_cell_flag", "total_cells_detected", "valid_cell_count", "leukocyte_count",
                    "nucleated_epithelial_count", "cornified_squamous_count", "debris_count",
                    "leukocyte_fraction", "nucleated_epithelial_fraction", "cornified_squamous_fraction",
                    "mean_cell_area", *[f"probability_{stage.value}" for stage in probability_columns],
                    "rationale",
                ]
            )
            for result in results:
                metrics = result.metrics
                source = self._result_paths.get(id(result))
                writer.writerow(
                    [
                        str(source) if source is not None else "",
                        result.stage.value,
                        f"{result.confidence:.6f}",
                        result.is_transition,
                        result.transition_to.value if result.transition_to else "",
                        result.low_cell_flag,
                        metrics.total_cells_detected,
                        metrics.valid_cell_count,
                        metrics.leukocyte_count,
                        metrics.nucleated_epithelial_count,
                        metrics.cornified_squamous_count,
                        metrics.debris_count,
                        f"{metrics.leukocyte_fraction:.6f}",
                        f"{metrics.nucleated_epithelial_fraction:.6f}",
                        f"{metrics.cornified_squamous_fraction:.6f}",
                        f"{metrics.mean_cell_area:.3f}",
                        *[f"{result.probabilities.get(stage, 0.0):.6f}" for stage in probability_columns],
                        result.rationale,
                    ]
                )
        return destination
