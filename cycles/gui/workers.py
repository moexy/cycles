"""Background Qt workers used by the desktop interface."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any

from PySide6.QtCore import QThread, Signal

from cycles.core.types import BatchClassificationResult

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class ClassificationWorker(QThread):
    """Run a staging pipeline without blocking the GUI event loop.

    Cancellation is cooperative: every mode checks the request between images.
    The worker always emits the completed or partial batch so controls can return
    to their idle state cleanly.
    """

    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        classifier: object,
        folder: Path | str,
        *,
        mode: str = "cnn",
        recursive: bool = False,
        artifact_dir: Path | str | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.classifier = classifier
        self.folder = Path(folder)
        self.mode = mode.strip().lower()
        self.recursive = recursive
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._cancel_event = Event()

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def cancel(self) -> None:
        """Cooperatively request cancellation at the next safe boundary."""
        self._cancel_event.set()
        self.requestInterruption()

    def run(self) -> None:
        """Execute the configured engine and emit its result or a readable error."""
        try:
            result = self._run_pipeline()
            # Partial results are deliberately emitted after cancellation so the
            # GUI can retain completed work and leave its running state cleanly.
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _run_pipeline(self) -> Any:
        if self.mode == "cnn":
            return self.classifier.classify_folder(  # type: ignore[attr-defined, no-any-return]
                self.folder,
                recursive=self.recursive,
                progress_callback=self._emit_progress,
                cancel_flag=self._should_cancel,
            )
        if self.mode in {"cell", "cell-centric", "cell_centric"}:
            return self._run_cell_pipeline()
        if self.mode in {"mil", "attention-mil", "attention_mil"}:
            return self._run_mil_pipeline()
        raise ValueError(f"unsupported classification mode: {self.mode}")

    def _images(self) -> list[Path]:
        iterator = self.folder.rglob("*") if self.recursive else self.folder.iterdir()
        return sorted(
            (
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(self.folder).as_posix().casefold(),
        )

    def _run_cell_pipeline(self) -> list[object]:
        images = self._images()
        results: list[object] = []
        failures: list[tuple[Path, str]] = []
        self._emit_progress(0, len(images), f"Found {len(images)} image(s)")
        for current, image_path in enumerate(images, start=1):
            if self.is_cancelled:
                break
            relative = image_path.relative_to(self.folder)
            overlay_path = (
                self.artifact_dir / relative.parent / f"{image_path.stem}_cells.png"
                if self.artifact_dir is not None
                else None
            )
            try:
                results.append(self.classifier.process_image(image_path, overlay_path))  # type: ignore[attr-defined]
                message = f"Profiled {image_path.name}"
            except Exception as exc:
                failures.append((image_path, f"{type(exc).__name__}: {exc}"))
                message = f"Failed {image_path.name}"
            self._emit_progress(current, len(images), message)
        self.classifier.processing_errors = failures
        return results

    def _run_mil_pipeline(self) -> BatchClassificationResult:
        images = self._images()
        results = []
        failures: list[tuple[Path, str]] = []
        started = time.perf_counter()
        self._emit_progress(0, len(images), f"Found {len(images)} image(s)")
        for current, image_path in enumerate(images, start=1):
            if self.is_cancelled:
                break
            relative = image_path.relative_to(self.folder)
            heatmap_path = (
                self.artifact_dir / relative.parent / f"{image_path.stem}_attention.png"
                if self.artifact_dir is not None
                else None
            )
            try:
                results.append(self.classifier.process_image(image_path, heatmap_path))  # type: ignore[attr-defined]
                message = f"Encoded {image_path.name}"
            except Exception as exc:
                failures.append((image_path, f"{type(exc).__name__}: {exc}"))
                message = f"Failed {image_path.name}"
            self._emit_progress(current, len(images), message)
        return BatchClassificationResult(
            results=results,
            failed_images=failures,
            total_processed=len(results) + len(failures),
            duration_seconds=time.perf_counter() - started,
        )

    def _should_cancel(self) -> bool:
        return self.is_cancelled

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        if not self.is_cancelled:
            self.progress.emit(current, total, message)
