"""Deterministic multi-view preparation without lossy intermediate files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

VIEW_PACK_VERSION = "overview-quadrants-v1"


@dataclass(frozen=True, slots=True)
class ImageView:
    label: str
    image: Image.Image


def build_view_pack(
    image_path: Path | str,
    *,
    overview_max_edge: int = 1536,
    overlap_fraction: float = 0.10,
) -> list[ImageView]:
    """Return one overview and four overlapping full-resolution quadrants."""
    if not 0 <= overlap_fraction < 0.5:
        raise ValueError("overlap_fraction must be in [0, 0.5)")
    with Image.open(image_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
        source.load()

    overview = source.copy()
    overview.thumbnail((overview_max_edge, overview_max_edge), Image.Resampling.LANCZOS)
    width, height = source.size
    midpoint_x, midpoint_y = width // 2, height // 2
    overlap_x = round(width * overlap_fraction / 2)
    overlap_y = round(height * overlap_fraction / 2)
    boxes = (
        ("top_left", (0, 0, min(width, midpoint_x + overlap_x), min(height, midpoint_y + overlap_y))),
        ("top_right", (max(0, midpoint_x - overlap_x), 0, width, min(height, midpoint_y + overlap_y))),
        ("bottom_left", (0, max(0, midpoint_y - overlap_y), min(width, midpoint_x + overlap_x), height)),
        ("bottom_right", (max(0, midpoint_x - overlap_x), max(0, midpoint_y - overlap_y), width, height)),
    )
    return [ImageView("overview", overview), *[ImageView(label, source.crop(box)) for label, box in boxes]]
