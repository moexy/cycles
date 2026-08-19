"""Patch extraction and foreground filtering for high-resolution cytology images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu


@dataclass(frozen=True, slots=True)
class PatchInfo:
    """Location and foreground content of a patch in the source image."""

    x: int
    y: int
    width: int
    height: int
    tissue_ratio: float


class PatchExtractor:
    """Tile an image and discard patches containing mostly glass background.

    Returned patch images always have ``patch_size`` square dimensions. Metadata
    dimensions describe the portion that came from the source image, which can
    be smaller when an input dimension is shorter than ``patch_size``.
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 256,
        min_tissue_ratio: float = 0.15,
        max_patches: int | None = 256,
        saturation_threshold: float = 0.08,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if not 0.0 <= min_tissue_ratio <= 1.0:
            raise ValueError("min_tissue_ratio must be between 0 and 1")
        if max_patches is not None and max_patches <= 0:
            raise ValueError("max_patches must be positive or None")
        if not 0.0 <= saturation_threshold <= 1.0:
            raise ValueError("saturation_threshold must be between 0 and 1")

        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_ratio = min_tissue_ratio
        self.max_patches = max_patches
        self.saturation_threshold = saturation_threshold

    def __call__(
        self, image_or_path: Image.Image | np.ndarray | Path | str
    ) -> tuple[list[Image.Image], list[PatchInfo]]:
        return self.extract(image_or_path)

    def extract(
        self,
        image_or_path: Image.Image | np.ndarray | Path | str,
    ) -> tuple[list[Image.Image], list[PatchInfo]]:
        """Return retained RGB patches and matching spatial metadata.

        Otsu intensity segmentation is combined with colour saturation so both
        dark nuclei and lightly stained cells count as tissue. Completely blank
        images legitimately return an empty bag; callers decide how to report
        that condition.
        """
        rgb = self._as_rgb_array(image_or_path)
        height, width = rgb.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("image has no pixels")

        tissue_mask = self._tissue_mask(rgb)
        candidates: list[tuple[float, int, Image.Image, PatchInfo]] = []
        sequence = 0
        for y in self._axis_starts(height):
            for x in self._axis_starts(width):
                valid_height = min(self.patch_size, height - y)
                valid_width = min(self.patch_size, width - x)
                region = rgb[y : y + valid_height, x : x + valid_width]
                region_mask = tissue_mask[y : y + valid_height, x : x + valid_width]
                tissue_ratio = float(region_mask.mean())
                if tissue_ratio < self.min_tissue_ratio:
                    sequence += 1
                    continue

                patch_array = self._pad_patch(region)
                info = PatchInfo(
                    x=x,
                    y=y,
                    width=valid_width,
                    height=valid_height,
                    tissue_ratio=tissue_ratio,
                )
                candidates.append(
                    (tissue_ratio, sequence, Image.fromarray(patch_array, mode="RGB"), info)
                )
                sequence += 1

        if self.max_patches is not None and len(candidates) > self.max_patches:
            # Stable ranking keeps deterministic spatial order for equal-density patches.
            candidates = sorted(candidates, key=lambda item: (-item[0], item[1]))[
                : self.max_patches
            ]
            candidates.sort(key=lambda item: item[1])

        patches = [item[2] for item in candidates]
        infos = [item[3] for item in candidates]
        return patches, infos

    def _axis_starts(self, extent: int) -> Sequence[int]:
        if extent <= self.patch_size:
            return (0,)
        last = extent - self.patch_size
        starts = list(range(0, last + 1, self.stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _pad_patch(self, patch: np.ndarray) -> np.ndarray:
        pad_height = self.patch_size - patch.shape[0]
        pad_width = self.patch_size - patch.shape[1]
        if pad_height == 0 and pad_width == 0:
            return np.ascontiguousarray(patch)

        # NumPy reflection requires at least two samples on every padded axis.
        mode = "reflect" if patch.shape[0] > 1 and patch.shape[1] > 1 else "edge"
        return np.pad(
            patch,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode=mode,
        )

    def _tissue_mask(self, rgb: np.ndarray) -> np.ndarray:
        rgb_float = rgb.astype(np.float32) / 255.0
        maximum = rgb_float.max(axis=2)
        minimum = rgb_float.min(axis=2)
        saturation = np.divide(
            maximum - minimum,
            maximum,
            out=np.zeros_like(maximum),
            where=maximum > 1e-6,
        )
        gray = (
            0.2126 * rgb_float[..., 0]
            + 0.7152 * rgb_float[..., 1]
            + 0.0722 * rgb_float[..., 2]
        )
        try:
            otsu_threshold = float(threshold_otsu(gray))
        except ValueError:
            otsu_threshold = float(gray.mean())

        # An Otsu split is unreliable on nearly uniform glass; neither a tiny
        # intensity fluctuation nor neutral white pixels should become tissue.
        contrast = float(np.percentile(gray, 95)) - gray
        dark_tissue = (gray < otsu_threshold) & (contrast > 0.04)
        stained_tissue = saturation >= self.saturation_threshold
        return dark_tissue | stained_tissue

    @staticmethod
    def _as_rgb_array(
        image_or_path: Image.Image | np.ndarray | Path | str,
    ) -> np.ndarray:
        if isinstance(image_or_path, (str, Path)):
            with Image.open(image_or_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                return np.asarray(image, dtype=np.uint8).copy()
        if isinstance(image_or_path, Image.Image):
            return np.asarray(ImageOps.exif_transpose(image_or_path).convert("RGB"), dtype=np.uint8)

        array = np.asarray(image_or_path)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim != 3:
            raise ValueError(f"expected a 2D or 3D image array, got shape {array.shape}")
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] >= 3:
            array = array[..., :3]
        else:
            raise ValueError(f"unsupported channel count: {array.shape[2]}")

        if np.issubdtype(array.dtype, np.floating):
            finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            if finite.size and float(finite.max()) <= 1.0:
                finite = finite * 255.0
            array = np.clip(finite, 0.0, 255.0)
        elif array.dtype != np.uint8:
            type_max = float(np.iinfo(array.dtype).max)
            array = array.astype(np.float32) * (255.0 / type_max)
        return np.ascontiguousarray(array.astype(np.uint8))
