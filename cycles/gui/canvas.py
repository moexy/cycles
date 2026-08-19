"""Interactive Matplotlib canvases embedded in the PySide6 application."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image
from PySide6.QtWidgets import QSizePolicy, QWidget

PHASE_ORDER = ("diestrus", "proestrus", "estrus", "metestrus")
CELL_BOX_COLORS: dict[str, str] = {
    "leukocyte": "#38bdf8",
    "nucleated_epithelial": "#22c55e",
    "cornified_squamous": "#ef4444",
    "debris": "#94a3b8",
}
PHASE_COLORS: dict[str, str] = {
    "diestrus": "#3b82f6",
    "proestrus": "#22c55e",
    "estrus": "#ef4444",
    "metestrus": "#f97316",
}


def _stage_key(stage: object) -> str:
    value = getattr(stage, "value", stage)
    return str(value).strip().lower().replace(" ", "_")


class ImageOverlayCanvas(FigureCanvasQTAgg):
    """Image viewer supporting bounding-box and attention-map overlays."""

    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 7.0,
        height: float = 6.0,
    ) -> None:
        self.figure = Figure(figsize=(width, height), dpi=100, layout="constrained")
        self.figure.patch.set_facecolor("#0f172a")
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_shape: tuple[int, int] | None = None
        self.mpl_connect("scroll_event", self._on_scroll)
        self.clear()

    def clear(self, message: str = "Select an image to inspect") -> None:
        """Clear the viewer and render a neutral status message."""
        self.axes.clear()
        self.axes.set_facecolor("#111827")
        self.axes.text(
            0.5,
            0.5,
            message,
            transform=self.axes.transAxes,
            ha="center",
            va="center",
            color="#94a3b8",
            fontsize=11,
        )
        self.axes.set_axis_off()
        self._image_shape = None
        self.draw_idle()

    def show_image(
        self,
        image: Path | str | Image.Image | np.ndarray,
        *,
        bounding_boxes: Iterable[object] | None = None,
        attention_heatmap: np.ndarray | None = None,
        title: str | None = None,
    ) -> None:
        """Render an RGB/grayscale image and optional explainability overlays.

        Boxes may be mappings or objects with ``bbox`` and optional ``label`` /
        ``cell_type`` / ``confidence`` attributes. Coordinates are interpreted as
        ``(y1, x1, y2, x2)``, matching the cell-centric pipeline.
        """
        pixels = self._load_pixels(image)
        self.axes.clear()
        self.axes.set_facecolor("#111827")
        self.axes.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None)
        self._image_shape = (int(pixels.shape[0]), int(pixels.shape[1]))

        if attention_heatmap is not None:
            heatmap = np.asarray(attention_heatmap, dtype=np.float32)
            if heatmap.ndim == 3:
                heatmap = heatmap.mean(axis=-1)
            self.axes.imshow(
                heatmap,
                cmap="inferno",
                alpha=0.42,
                interpolation="bilinear",
                extent=(0, pixels.shape[1], pixels.shape[0], 0),
            )

        for box in bounding_boxes or ():
            parsed = self._parse_box(box)
            if parsed is None:
                continue
            y1, x1, y2, x2, label, color = parsed
            self.axes.add_patch(
                Rectangle(
                    (x1, y1),
                    max(0.0, x2 - x1),
                    max(0.0, y2 - y1),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                )
            )
            if label:
                self.axes.text(
                    x1,
                    max(0.0, y1 - 3),
                    label,
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": color, "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
                )

        if title:
            self.axes.set_title(title, color="#f8fafc", fontsize=11, pad=8)
        self.axes.set_axis_off()
        self.draw_idle()

    @staticmethod
    def _load_pixels(image: Path | str | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            pixels = image
        elif isinstance(image, Image.Image):
            pixels = np.asarray(image.convert("RGB"))
        else:
            with Image.open(Path(image)) as source:
                pixels = np.asarray(source.convert("RGB"))
        if pixels.ndim not in (2, 3):
            raise ValueError(f"expected a 2D or 3D image array, got shape {pixels.shape}")
        return pixels

    @staticmethod
    def _parse_box(box: object) -> tuple[float, float, float, float, str, str] | None:
        if isinstance(box, dict):
            coordinates = box.get("bbox", box.get("box"))
            label_value = box.get("label", box.get("cell_type", box.get("predicted_type", "")))
            confidence = box.get("confidence")
            label_key = _stage_key(label_value)
            color = str(box.get("color", CELL_BOX_COLORS.get(label_key, "#38bdf8")))
        else:
            coordinates = getattr(box, "bbox", box if isinstance(box, Sequence) else None)
            label_value = getattr(
                box,
                "label",
                getattr(box, "cell_type", getattr(box, "predicted_type", "")),
            )
            confidence = getattr(box, "confidence", None)
            label_key = _stage_key(label_value)
            color = str(getattr(box, "color", CELL_BOX_COLORS.get(label_key, "#38bdf8")))
        if coordinates is None or len(coordinates) != 4:
            return None
        y1, x1, y2, x2 = (float(value) for value in coordinates)
        label = str(getattr(label_value, "value", label_value)).replace("_", " ").title()
        if confidence is not None:
            label = f"{label} {float(confidence):.0%}".strip()
        return y1, x1, y2, x2, label, color

    def _on_scroll(self, event: Any) -> None:
        if self._image_shape is None or event.xdata is None or event.ydata is None:
            return
        x_limits = self.axes.get_xlim()
        y_limits = self.axes.get_ylim()
        scale = 0.8 if event.button == "up" else 1.25
        width = (x_limits[1] - x_limits[0]) * scale
        height = (y_limits[1] - y_limits[0]) * scale
        relative_x = (event.xdata - x_limits[0]) / (x_limits[1] - x_limits[0])
        relative_y = (event.ydata - y_limits[0]) / (y_limits[1] - y_limits[0])
        self.axes.set_xlim(event.xdata - width * relative_x, event.xdata + width * (1 - relative_x))
        self.axes.set_ylim(event.ydata - height * relative_y, event.ydata + height * (1 - relative_y))
        self.draw_idle()


class CycleTimelineCanvas(FigureCanvasQTAgg):
    """Longitudinal Day-vs-Phase plot with canonical phase color bands."""

    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 5.0,
        height: float = 3.0,
    ) -> None:
        self.figure = Figure(figsize=(width, height), dpi=100, layout="constrained")
        self.figure.patch.set_facecolor("#0f172a")
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_timeline([], [])

    def update_timeline(
        self,
        days: Sequence[float | int | str],
        stages: Sequence[object],
        *,
        current_index: int | None = None,
    ) -> None:
        """Render phase observations against elapsed day or sample labels."""
        self.axes.clear()
        self.axes.set_facecolor("#111827")
        for index, phase in enumerate(PHASE_ORDER):
            self.axes.axhspan(index - 0.45, index + 0.45, color=PHASE_COLORS[phase], alpha=0.22)

        if stages:
            if len(days) != len(stages):
                raise ValueError("days and stages must have equal length")
            values = [PHASE_ORDER.index(_stage_key(stage)) if _stage_key(stage) in PHASE_ORDER else np.nan for stage in stages]
            x_values: Sequence[float | int] = list(range(1, len(days) + 1))
            numeric_days = all(isinstance(day, (int, float)) for day in days)
            if numeric_days:
                x_values = [float(day) for day in days]  # type: ignore[arg-type]
            self.axes.plot(x_values, values, color="#e2e8f0", marker="o", linewidth=2, zorder=3)
            if current_index is not None and 0 <= current_index < len(values):
                self.axes.scatter(
                    [x_values[current_index]],
                    [values[current_index]],
                    marker="*",
                    s=140,
                    color="#facc15",
                    edgecolors="white",
                    zorder=4,
                )
            if not numeric_days:
                self.axes.set_xticks(list(x_values), [str(day) for day in days], rotation=35, ha="right")
        else:
            self.axes.text(
                0.5,
                0.5,
                "Cycle timeline appears after staging",
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                color="#94a3b8",
            )

        self.axes.set_yticks(range(4), [phase.title() for phase in PHASE_ORDER])
        self.axes.set_ylim(-0.5, 3.5)
        self.axes.set_xlabel("Day", color="#e2e8f0")
        self.axes.set_ylabel("Phase", color="#e2e8f0")
        self.axes.set_title("Longitudinal Estrous Cycle", color="#f8fafc", fontsize=10)
        self.axes.tick_params(colors="#cbd5e1", labelsize=8)
        self.axes.grid(axis="x", color="#64748b", alpha=0.25, linestyle="--")
        for spine in self.axes.spines.values():
            spine.set_color("#475569")
        self.draw_idle()

    def update_cycle(
        self,
        stages: Sequence[object],
        current_index: int | None = None,
    ) -> None:
        """Convenience wrapper using one sample per day."""
        self.update_timeline(list(range(1, len(stages) + 1)), stages, current_index=current_index)
