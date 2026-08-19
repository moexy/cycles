"""Cell detection with optional YOLOv8 inference and multi-scale cytomorphology fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

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
    if rgb.ndim == 2:
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _nuclear_ratio(crop: np.ndarray) -> float:
    """Estimate the dark nuclear fraction within a candidate region."""
    if crop.size < 15:
        return 0.0
    crop_f = crop.astype(np.float32)
    min_v, max_v = float(crop_f.min()), float(crop_f.max())
    if max_v - min_v < 6.0:
        return 0.0
    thresh = float(np.percentile(crop_f, 35))
    dark = crop_f <= thresh
    contrast = float(crop_f[~dark].mean() - crop_f[dark].mean()) if np.any(~dark) else 0.0
    return float(np.count_nonzero(dark) / crop_f.size) if contrast >= 6.0 else 0.0


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
        height, width = rgb.shape[:2]
        gray = _to_grayscale(rgb)
        names = getattr(self._yolo_model, "names", {})

        # High-resolution sliding window tiling if image exceeds 1024px
        if max(height, width) > 1024:
            raw_boxes, raw_classes, raw_scores = self._tiled_yolo_predict(rgb)
        else:
            results = self._yolo_model.predict(
                source=rgb,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return []
            boxes = results[0].boxes
            names = getattr(results[0], "names", None) or names
            raw_boxes = boxes.xyxy.detach().cpu().numpy()
            raw_classes = boxes.cls.detach().cpu().numpy().astype(int)
            raw_scores = boxes.conf.detach().cpu().numpy()

        if len(raw_boxes) == 0:
            return self._detect_morphometry(rgb)

        # Estimate smooth background for local contrast evaluation
        bg_ksize = min(51, min(height, width) // 4 * 2 + 1)
        bg_est = cv2.medianBlur(gray, bg_ksize)
        contrast = cv2.subtract(bg_est, gray)

        profiles: list[CellProfile] = []
        self.last_nuclear_ratios = []
        labels = np.zeros((height, width), dtype=np.int32)

        for idx, (bounds, class_id, confidence) in enumerate(
            zip(raw_boxes, raw_classes, raw_scores, strict=False), start=1
        ):
            x1f, y1f, x2f, y2f = (float(val) for val in bounds)
            x1, y1 = max(0, int(np.floor(x1f))), max(0, int(np.floor(y1f)))
            x2, y2 = min(width, int(np.ceil(x2f))), min(height, int(np.ceil(y2f)))
            box_w, box_h = x2 - x1, y2 - y1
            area = float(box_w * box_h)
            if area < self.min_area or x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            crop = gray[y1:y2, x1:x2]
            crop_contrast = contrast[y1:y2, x1:x2]
            nc_ratio = _nuclear_ratio(crop)

            class_name = str(names.get(int(class_id), "debris")).lower()
            if "leuko" in class_name or "white" in class_name:
                initial_type = CellType.LEUKOCYTE
            elif "corn" in class_name or "squam" in class_name:
                initial_type = CellType.CORNIFIED_SQUAMOUS
            elif "nucle" in class_name or "epithel" in class_name:
                initial_type = CellType.NUCLEATED_EPITHELIAL
            else:
                initial_type = CellType.DEBRIS

            max_contrast = float(crop_contrast.max()) if crop_contrast.size else 0.0

            # Cytomorphological verification on YOLO candidate
            if initial_type == CellType.CORNIFIED_SQUAMOUS and area <= 450.0:
                if nc_ratio >= 0.20 or max_contrast >= 22.0:
                    final_type = CellType.LEUKOCYTE
                else:
                    final_type = CellType.DEBRIS
            elif initial_type == CellType.CORNIFIED_SQUAMOUS and 200.0 <= area <= 3000.0 and nc_ratio >= 0.08:
                final_type = CellType.NUCLEATED_EPITHELIAL
            elif initial_type == CellType.NUCLEATED_EPITHELIAL and area <= 300.0 and nc_ratio >= 0.40:
                final_type = CellType.LEUKOCYTE
            elif initial_type == CellType.NUCLEATED_EPITHELIAL and area >= 1500.0 and nc_ratio < 0.04:
                final_type = CellType.CORNIFIED_SQUAMOUS
            elif initial_type == CellType.LEUKOCYTE and area >= 800.0:
                final_type = CellType.CORNIFIED_SQUAMOUS if nc_ratio < 0.06 else CellType.NUCLEATED_EPITHELIAL
            else:
                final_type = initial_type

            perimeter = float(2 * (box_w + box_h))
            circularity = float(np.clip(4.0 * np.pi * area / max(perimeter**2, 1.0), 0.0, 1.0))
            aspect_ratio = float(max(box_w, box_h) / max(min(box_w, box_h), 1))

            profiles.append(
                CellProfile(
                    bbox=(y1, x1, y2, x2),
                    centroid=((y1 + y2) / 2.0, (x1 + x2) / 2.0),
                    area=area,
                    perimeter=perimeter,
                    circularity=circularity,
                    aspect_ratio=aspect_ratio,
                    mean_intensity=float(crop.mean()) if crop.size else 0.0,
                    std_intensity=float(crop.std()) if crop.size else 0.0,
                    predicted_type=final_type,
                    confidence=float(confidence),
                )
            )
            self.last_nuclear_ratios.append(nc_ratio)
            labels[y1:y2, x1:x2] = idx

        # Always merge high-contrast morphological leukocytes that YOLO missed
        morph_profiles = self._detect_morphometry(rgb)
        morph_leukos = [p for p in morph_profiles if p.predicted_type == CellType.LEUKOCYTE]
        if morph_leukos:
            # Exclude morphological leukocytes that heavily overlap with existing leukocyte detections
            existing_leuko_boxes = [p.bbox for p in profiles if p.predicted_type == CellType.LEUKOCYTE]
            for ml in morph_leukos:
                my1, mx1, my2, mx2 = ml.bbox
                overlap = False
                for ey1, ex1, ey2, ex2 in existing_leuko_boxes:
                    inter_h = max(0, min(my2, ey2) - max(my1, ey1))
                    inter_w = max(0, min(mx2, ex2) - max(mx1, ex1))
                    if inter_h * inter_w > 0:
                        overlap = True
                        break
                if not overlap:
                    profiles.append(ml)
                    self.last_nuclear_ratios.append(0.50)

        self.last_labels = labels
        return profiles

    def _tiled_yolo_predict(
        self,
        rgb: np.ndarray,
        tile_size: int = 640,
        overlap: int = 128,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run sliding-window tiled YOLO inference with coordinate mapping and NMS."""
        assert self._yolo_model is not None
        height, width = rgb.shape[:2]
        stride = tile_size - overlap
        all_boxes: list[np.ndarray] = []
        all_classes: list[int] = []
        all_scores: list[float] = []

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                crop = rgb[y_start:y_end, x_start:x_end]
                results = self._yolo_model.predict(
                    source=crop,
                    conf=self.confidence_threshold,
                    imgsz=tile_size,
                    device=self.device,
                    verbose=False,
                )
                if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                    continue
                b = results[0].boxes.xyxy.detach().cpu().numpy()
                c = results[0].boxes.cls.detach().cpu().numpy().astype(int)
                s = results[0].boxes.conf.detach().cpu().numpy()

                b[:, [0, 2]] += x_start
                b[:, [1, 3]] += y_start

                all_boxes.extend(b)
                all_classes.extend(c)
                all_scores.extend(s)

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,))

        boxes_arr = np.array(all_boxes, dtype=np.float32)
        classes_arr = np.array(all_classes, dtype=int)
        scores_arr = np.array(all_scores, dtype=np.float32)

        # Class-aware NMS
        keep_indices: list[int] = []
        unique_classes = np.unique(classes_arr)
        for cls_id in unique_classes:
            cls_mask = np.where(classes_arr == cls_id)[0]
            cls_boxes = boxes_arr[cls_mask]
            cls_scores = scores_arr[cls_mask]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            while order.size > 0:
                i = order[0]
                keep_indices.append(int(cls_mask[i]))
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
                inds = np.where(iou <= 0.45)[0]
                order = order[inds + 1]

        keep_indices.sort()
        return boxes_arr[keep_indices], classes_arr[keep_indices], scores_arr[keep_indices]

    def _detect_morphometry(self, rgb: np.ndarray) -> list[CellProfile]:
        """Multi-scale cytomorphology detection of sheets, nucleated cells, and leukocytes."""
        gray = _to_grayscale(rgb)
        height, width = gray.shape
        if min(height, width) < 8:
            self.last_labels = np.zeros_like(gray, dtype=np.int32)
            return []

        # 1. Fast Background Estimation & Contrast
        bg_ksize = min(51, min(height, width) // 4 * 2 + 1)
        bg = cv2.medianBlur(gray, bg_ksize)
        contrast = cv2.subtract(bg, gray)

        # 2. Cornified Squamous Sheets & Large Cell Bodies via Edge Closing
        edges = cv2.Canny(gray, 18, 55)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cornified_mask = np.zeros((height, width), dtype=np.uint8)
        total_cornified_area = 0.0

        profiles: list[CellProfile] = []
        self.last_nuclear_ratios = []
        labels = np.zeros((height, width), dtype=np.int32)
        curr_label = 1

        # Extract Cornified Sheets
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area >= 800.0:
                total_cornified_area += area
                cv2.drawContours(cornified_mask, [cnt], -1, 255, -1)
                eff_cells = max(1, int(np.round(area / 1500.0)))
                x, y, w, h = cv2.boundingRect(cnt)
                for _ in range(eff_cells):
                    profiles.append(
                        CellProfile(
                            bbox=(int(y), int(x), int(y + h), int(x + w)),
                            centroid=(float(y + h / 2.0), float(x + w / 2.0)),
                            area=area / eff_cells,
                            perimeter=float(cv2.arcLength(cnt, True)) / eff_cells,
                            circularity=0.35,
                            aspect_ratio=float(max(w, h) / max(min(w, h), 1)),
                            mean_intensity=float(gray[y:y + h, x:x + w].mean()),
                            std_intensity=float(gray[y:y + h, x:x + w].std()),
                            predicted_type=CellType.CORNIFIED_SQUAMOUS,
                            confidence=0.92,
                        )
                    )
                    self.last_nuclear_ratios.append(0.0)

        sheet_cov_pct = total_cornified_area / (height * width) * 100.0

        # 3. Solid Dense Nuclear / Leukocyte Detection
        _, dark_blobs = cv2.threshold(contrast, 24, 255, cv2.THRESH_BINARY)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_blobs = cv2.morphologyEx(dark_blobs, cv2.MORPH_OPEN, kernel3)

        num_blobs, blob_labels, blob_stats, blob_centroids = cv2.connectedComponentsWithStats(
            dark_blobs, connectivity=8
        )

        for i in range(1, num_blobs):
            area = float(blob_stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > 3500.0:
                continue
            x = int(blob_stats[i, cv2.CC_STAT_LEFT])
            y = int(blob_stats[i, cv2.CC_STAT_TOP])
            w = int(blob_stats[i, cv2.CC_STAT_WIDTH])
            h = int(blob_stats[i, cv2.CC_STAT_HEIGHT])
            aspect_ratio = float(max(w, h) / max(min(w, h), 1))
            if aspect_ratio > 3.0:
                continue
            solidity = area / max(w * h, 1)
            if solidity < 0.35:
                continue

            cx, cy = blob_centroids[i]
            is_inside_sheet = cornified_mask[int(cy), int(cx)] > 0

            # If inside massive cornified sheet on high-coverage Estrus slides, ignore intracellular speckles
            if is_inside_sheet and sheet_cov_pct >= 60.0:
                continue

            crop = gray[y:y + h, x:x + w]
            nc_ratio = _nuclear_ratio(crop)
            perimeter = float(2 * (w + h))
            circularity = float(np.clip(4.0 * np.pi * area / max(perimeter**2, 1.0), 0.0, 1.0))

            if area <= 300.0 and (solidity >= 0.40 or circularity >= 0.35):
                cell_type = CellType.LEUKOCYTE
                conf = 0.88
            elif area >= 150.0 and nc_ratio >= 0.06:
                cell_type = CellType.NUCLEATED_EPITHELIAL
                conf = 0.82
            else:
                cell_type = CellType.DEBRIS
                conf = 0.50

            profiles.append(
                CellProfile(
                    bbox=(y, x, y + h, x + w),
                    centroid=(float(cy), float(cx)),
                    area=area,
                    perimeter=perimeter,
                    circularity=circularity,
                    aspect_ratio=aspect_ratio,
                    mean_intensity=float(crop.mean()) if crop.size else 0.0,
                    std_intensity=float(crop.std()) if crop.size else 0.0,
                    predicted_type=cell_type,
                    confidence=conf,
                )
            )
            self.last_nuclear_ratios.append(nc_ratio)
            labels[y:y + h, x:x + w] = curr_label
            curr_label += 1

        self.last_labels = labels
        return profiles
