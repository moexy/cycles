"""Cell detection with optional YOLOv8 inference and watershed fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from cycles.core.types import CellProfile, CellType

LOGGER = logging.getLogger(__name__)
DetectorMode = Literal["auto", "yolo", "morphometry"]


def _default_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def _candidate_yolo_weights() -> tuple[Path, ...]:
    project_root = Path(__file__).resolve().parents[3]
    configured = os.environ.get("CYCLES_YOLO_WEIGHTS")
    candidates = [
        project_root / "ODES_Object_Detection_For_Estrous_Staging" / "ODES" / "finalweight.pt",
        project_root.parent / "EstrousNet" / "ODES_Object_Detection_For_Estrous_Staging" / "ODES" / "finalweight.pt",
        project_root.parent / "ODES_Object_Detection_For_Estrous_Staging" / "ODES" / "finalweight.pt",
        Path.cwd() / "finalweight.pt",
    ]
    if configured:
        candidates.insert(0, Path(configured).expanduser())
    return tuple(candidates)


def _load_rgb(image_or_path: Path | str | np.ndarray) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        with Image.open(Path(image_or_path)) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    array = np.asarray(image_or_path)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected an HxW or HxWx3/4 image, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    gray = np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _nuclear_ratio(intensities: np.ndarray) -> float:
    """Estimate the dark nuclear fraction within a candidate region."""
    values = np.asarray(intensities, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size < 15 or float(np.ptp(values)) < 6.0:
        return 0.0
    try:
        threshold = float(filters.threshold_otsu(values))
    except ValueError:
        return 0.0
    dark = values <= threshold
    if not np.any(dark) or not np.any(~dark):
        return 0.0
    contrast = float(values[~dark].mean() - values[dark].mean())
    return float(np.count_nonzero(dark) / values.size) if contrast >= 6.0 else 0.0


class CellDetector:
    """Detect cells with YOLOv8 or morphology-driven watershed.

    Missing weights, model-load failures, and per-image YOLO inference failures
    all switch the detector to morphometry rather than aborting processing.
    """

    def __init__(
        self,
        mode: DetectorMode = "auto",
        yolo_weights_path: Path | str | None = None,
        *,
        device: str | None = None,
        confidence_threshold: float = 0.15,
        min_area: float = 15.0,
        max_area: float | None = None,
        is_phase_contrast: bool = False,
    ) -> None:
        if mode not in {"auto", "yolo", "morphometry"}:
            raise ValueError("mode must be one of: auto, yolo, morphometry")
        if min_area < 15.0:
            raise ValueError("min_area must be at least 15 pixels")
        self.requested_mode: DetectorMode = mode
        self.mode: DetectorMode = "morphometry"
        self.device = device or _default_device()
        self.confidence_threshold = float(confidence_threshold)
        self.min_area = float(min_area)
        self.max_area = float(max_area) if max_area is not None else None
        self.is_phase_contrast = is_phase_contrast
        self.yolo_weights_path = self._resolve_weights(yolo_weights_path)
        self.fallback_reason: str | None = None
        self.last_labels: np.ndarray | None = None
        self.last_rgb: np.ndarray | None = None
        self.last_nuclear_ratios: list[float] = []
        self._yolo_model: Any | None = None
        if mode in {"auto", "yolo"}:
            self._try_load_yolo()

    @staticmethod
    def _resolve_weights(weights_path: Path | str | None) -> Path | None:
        if weights_path is not None:
            path = Path(weights_path).expanduser()
            return path if path.is_file() else None
        return next((path for path in _candidate_yolo_weights() if path.is_file()), None)

    def _try_load_yolo(self) -> None:
        if self.yolo_weights_path is None:
            self.fallback_reason = "YOLO weights were not found"
            return
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(self.yolo_weights_path))
            self.mode = "yolo"
        except Exception as exc:  # Ultralytics raises backend-specific exceptions.
            self._yolo_model = None
            self.fallback_reason = f"YOLO model could not be loaded: {exc}"
            LOGGER.warning("%s; using morphometry", self.fallback_reason)

    def detect(self, image_or_path: Path | str | np.ndarray) -> list[CellProfile]:
        """Detect cells, falling back per image when necessary."""
        rgb = _load_rgb(image_or_path)
        self.last_rgb = rgb
        self.last_labels = None
        self.last_nuclear_ratios = []
        if self.mode == "yolo" and self._yolo_model is not None:
            try:
                return self._detect_yolo(rgb)
            except Exception as exc:  # One bad accelerator/model must not lose the slide.
                self.mode = "morphometry"
                self.fallback_reason = f"YOLO inference failed: {exc}"
                LOGGER.warning("%s; using morphometry", self.fallback_reason)
        return self._detect_morphometry(rgb)

    def _detect_yolo(self, rgb: np.ndarray) -> list[CellProfile]:
        assert self._yolo_model is not None
        results = self._yolo_model.predict(source=rgb, conf=self.confidence_threshold, device=self.device, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        height, width = rgb.shape[:2]
        names = getattr(results[0], "names", None) or getattr(self._yolo_model, "names", {})
        gray = _to_grayscale(rgb)
        profiles: list[CellProfile] = []
        arrays = (boxes.xyxy.detach().cpu().numpy(), boxes.cls.detach().cpu().numpy(), boxes.conf.detach().cpu().numpy())
        for bounds, class_id, confidence in zip(*arrays, strict=False):
            x1f, y1f, x2f, y2f = (float(value) for value in bounds)
            x1, y1 = max(0, int(np.floor(x1f))), max(0, int(np.floor(y1f)))
            x2, y2 = min(width, int(np.ceil(x2f))), min(height, int(np.ceil(y2f)))
            box_width, box_height = x2 - x1, y2 - y1
            area = float(box_width * box_height)
            if area < self.min_area or x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            class_name = str(names.get(int(class_id), "debris")).lower()
            if "leuko" in class_name or "white" in class_name:
                cell_type = CellType.LEUKOCYTE
            elif "corn" in class_name or "squam" in class_name:
                cell_type = CellType.CORNIFIED_SQUAMOUS
            elif "nucle" in class_name or "epithel" in class_name:
                cell_type = CellType.NUCLEATED_EPITHELIAL
            else:
                cell_type = CellType.DEBRIS
            crop = gray[y1:y2, x1:x2]
            perimeter = float(2 * (box_width + box_height))
            profiles.append(CellProfile(
                bbox=(y1, x1, y2, x2), centroid=((y1 + y2) / 2.0, (x1 + x2) / 2.0),
                area=area, perimeter=perimeter,
                circularity=float(np.clip(4.0 * np.pi * area / max(perimeter**2, 1.0), 0.0, 1.0)),
                aspect_ratio=float(max(box_width, box_height) / max(min(box_width, box_height), 1)),
                mean_intensity=float(crop.mean()) if crop.size else 0.0,
                std_intensity=float(crop.std()) if crop.size else 0.0,
                predicted_type=cell_type, confidence=float(confidence),
            ))
            self.last_nuclear_ratios.append(_nuclear_ratio(crop))
        return profiles

    def _detect_morphometry(self, rgb: np.ndarray) -> list[CellProfile]:
        gray = _to_grayscale(rgb)
        gray_float = gray.astype(np.float32) / 255.0
        height, width = gray.shape
        if min(height, width) < 8 or float(np.ptp(gray_float)) < 1.0 / 255.0:
            self.last_labels = np.zeros_like(gray, dtype=np.int32)
            return []
        if self.is_phase_contrast:
            enhanced = filters.sobel(gray_float)
            binary = enhanced > float(filters.threshold_otsu(enhanced))
        else:
            sigma = float(np.clip(min(height, width) / 80.0, 3.0, 35.0))
            background = ndi.gaussian_filter(gray_float, sigma=sigma)
            dark_contrast = np.clip(background - gray_float, 0.0, None)
            positive = dark_contrast[dark_contrast > 0]
            if positive.size < int(self.min_area):
                self.last_labels = np.zeros_like(gray, dtype=np.int32)
                return []
            global_mask = dark_contrast > max(float(filters.threshold_otsu(positive)), 2.0 / 255.0)
            block_size = min(101, min(height, width) - (1 - min(height, width) % 2))
            if block_size >= 9:
                local_threshold = filters.threshold_local(gray_float, block_size=block_size, method="gaussian", offset=0.015)
                binary = global_mask | (gray_float < local_threshold)
            else:
                binary = global_mask
        binary = morphology.opening(binary, morphology.disk(1))
        binary = morphology.closing(binary, morphology.disk(2))
        binary = ndi.binary_fill_holes(binary)
        components, _ = ndi.label(binary)
        component_sizes = np.bincount(components.ravel())
        binary = (components > 0) & (
            component_sizes[components] >= int(np.ceil(self.min_area))
        )
        binary = segmentation.clear_border(binary, buffer_size=1)
        if not np.any(binary):
            self.last_labels = np.zeros_like(gray, dtype=np.int32)
            return []
        distance = ndi.distance_transform_edt(binary)
        coordinates = feature.peak_local_max(distance, labels=binary, min_distance=max(2, int(np.sqrt(self.min_area) / 2.0)), threshold_abs=1.5, exclude_border=False)
        markers = np.zeros_like(gray, dtype=np.int32)
        if coordinates.size:
            markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        else:
            markers, _ = ndi.label(binary)
        labels = segmentation.watershed(-distance, markers, mask=binary, compactness=0.001)
        keep = np.zeros(int(labels.max()) + 1, dtype=bool)
        for region in measure.regionprops(labels):
            if region.area < self.min_area or (self.max_area is not None and region.area > self.max_area):
                continue
            min_row, min_col, max_row, max_col = region.bbox
            if min_row > 0 and min_col > 0 and max_row < height and max_col < width:
                keep[region.label] = True
        labels = np.where(keep[labels], labels, 0)
        labels, _, _ = segmentation.relabel_sequential(labels)
        self.last_labels = labels.astype(np.int32, copy=False)
        profiles: list[CellProfile] = []
        for region in measure.regionprops(self.last_labels, intensity_image=gray):
            min_row, min_col, max_row, max_col = region.bbox
            box_height, box_width = max_row - min_row, max_col - min_col
            perimeter, area = float(region.perimeter), float(region.area)
            intensity_values = gray[region.slice][region.image]
            profiles.append(CellProfile(
                bbox=(int(min_row), int(min_col), int(max_row), int(max_col)),
                centroid=(float(region.centroid[0]), float(region.centroid[1])), area=area,
                perimeter=perimeter,
                circularity=float(np.clip(4.0 * np.pi * area / max(perimeter**2, 1.0), 0.0, 1.0)),
                aspect_ratio=float(max(box_height, box_width) / max(min(box_height, box_width), 1)),
                mean_intensity=float(np.mean(intensity_values)), std_intensity=float(np.std(intensity_values)),
                predicted_type=CellType.DEBRIS,
            ))
            self.last_nuclear_ratios.append(_nuclear_ratio(intensity_values))
        return profiles
