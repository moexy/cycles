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
    quadrant_max_edge: int | None = None,
) -> list[ImageView]:
    """Return one overview and four overlapping quadrants.

    Quadrants are full resolution unless ``quadrant_max_edge`` caps their longest
    edge. Vision prefill scales with pixel count, so the cap is the main lever on
    latency; whether the discarded detail carries stage-relevant signal is an
    empirical question and the default keeps the original behavior.
    """
    if not 0 <= overlap_fraction < 0.5:
        raise ValueError("overlap_fraction must be in [0, 0.5)")
    if quadrant_max_edge is not None and quadrant_max_edge <= 0:
        raise ValueError("quadrant_max_edge must be positive")
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
    quadrants = []
    for label, box in boxes:
        crop = source.crop(box)
        if quadrant_max_edge is not None:
            crop.thumbnail((quadrant_max_edge, quadrant_max_edge), Image.Resampling.LANCZOS)
        quadrants.append(ImageView(label, crop))
    return [ImageView("overview", overview), *quadrants]
