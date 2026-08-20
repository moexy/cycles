"""Main PySide6 window for staging and longitudinal cycle review."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QStyle,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from cycles.core.types import EstrousStage
from cycles.gui.canvas import PHASE_COLORS, CycleTimelineCanvas, ImageOverlayCanvas
from cycles.gui.vlm_review import VLMReviewWorkspace
from cycles.gui.workers import ClassificationWorker

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _stage_value(stage: object | None) -> str:
    if stage is None:
        return "unavailable"
    return str(getattr(stage, "value", stage)).lower().replace(" ", "_")


def _humanise(value: object | None) -> str:
    return _stage_value(value).replace("_", " ").title()


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()  # type: ignore[no-any-return, union-attr]
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class MainWindow(QMainWindow):
    """Three-pane workspace for CNN, cell-centric, and Attention-MIL staging."""

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        classifier: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("cycles — Rodent Estrous Staging")
        self.resize(1480, 900)
        self.setMinimumSize(1120, 720)

        self.selected_folder: Path | None = None
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self._injected_classifier = classifier
        self._engine: object | None = classifier
        self._worker: ClassificationWorker | None = None
        self._batch_result: object | None = None
        self._results: list[object] = []
        self._image_paths: list[Path] = []
        self._artifact_temp: TemporaryDirectory[str] | None = None

        self._build_toolbar()
        self._build_workspace()
        self._apply_theme()
        self._set_running(False)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Staging Controls", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        style = self.style()

        self.select_folder_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), "Select Folder", self
        )
        self.select_folder_action.triggered.connect(self.select_folder)
        toolbar.addAction(self.select_folder_action)

        self.select_checkpoint_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "Select Checkpoint",
            self,
        )
        self.select_checkpoint_action.triggered.connect(self.select_checkpoint)
        toolbar.addAction(self.select_checkpoint_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("CNN", "cnn")
        self.mode_combo.addItem("Cell-Centric", "cell-centric")
        self.mode_combo.addItem("Attention-MIL", "mil")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_combo)
        toolbar.addSeparator()

        self.run_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Run Staging", self
        )
        self.run_action.triggered.connect(self.run_staging)
        toolbar.addAction(self.run_action)

        self.cancel_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Cancel", self
        )
        self.cancel_action.triggered.connect(self.cancel_staging)
        toolbar.addAction(self.cancel_action)

        self.export_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Export Results", self
        )
        self.export_action.triggered.connect(self.export_results)
        toolbar.addAction(self.export_action)
        toolbar.addSeparator()

        self.toggle_labels_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView),
            "Hide Labels",
            self,
        )
        self.toggle_labels_action.setCheckable(True)
        self.toggle_labels_action.setChecked(True)
        self.toggle_labels_action.setShortcut("L")
        self.toggle_labels_action.setToolTip("Toggle Cell Labels & Overlays Visibility (L)")
        self.toggle_labels_action.triggered.connect(self._on_toggle_labels)
        toolbar.addAction(self.toggle_labels_action)
    def _build_workspace(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_file_sidebar())
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_details_sidebar())
        splitter.setSizes([270, 760, 390])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.workspace_tabs = QTabWidget(self)
        self.workspace_tabs.addTab(splitter, "Staging")
        self.vlm_review_workspace = VLMReviewWorkspace(parent=self.workspace_tabs)
        self.workspace_tabs.addTab(self.vlm_review_workspace, "VLM Review")
        self.setCentralWidget(self.workspace_tabs)

        status = QStatusBar(self)
        self.status_label = QLabel("Select an image folder to begin")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(330)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        status.addWidget(self.status_label, 1)
        status.addPermanentWidget(self.progress_bar)
        self.setStatusBar(status)

    def _build_file_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sidebar")
        layout = QVBoxLayout(frame)
        title = QLabel("Images")
        title.setObjectName("panelTitle")
        self.folder_summary = QLabel("No folder selected")
        self.folder_summary.setWordWrap(True)
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._display_index)
        layout.addWidget(title)
        layout.addWidget(self.folder_summary)
        layout.addWidget(self.file_list, 1)
        return frame

    def _build_image_panel(self) -> QWidget:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        self.stage_banner = QLabel("No stage prediction")
        self.stage_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_banner.setObjectName("stageBanner")
        header_row = QHBoxLayout()
        self.image_caption = QLabel("Image and explainability overlay")
        self.labels_checkbox = QCheckBox("Show Overlay Labels (L)")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.toggled.connect(self._on_toggle_labels)
        header_row.addWidget(self.image_caption, 1)
        header_row.addWidget(self.labels_checkbox)

        self.image_canvas = ImageOverlayCanvas(frame)
        layout.addWidget(self.stage_banner)
        layout.addLayout(header_row)
        layout.addWidget(self.image_canvas, 1)
        return frame

    def _build_details_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(355)
        content = QWidget()
        layout = QVBoxLayout(content)

        probabilities = QGroupBox("Detailed Stage Probabilities")
        probabilities_layout = QVBoxLayout(probabilities)
        self.probability_bars: dict[EstrousStage, QProgressBar] = {}
        for stage in EstrousStage.canonical_stages():
            row = QHBoxLayout()
            label = QLabel(stage.display_name)
            label.setMinimumWidth(78)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFormat("0.0%")
            row.addWidget(label)
            row.addWidget(bar, 1)
            probabilities_layout.addLayout(row)
            self.probability_bars[stage] = bar

        confidence_group = QGroupBox("Assessment")
        confidence_layout = QVBoxLayout(confidence_group)
        self.confidence_label = QLabel("Confidence Index: —")
        self.transition_label = QLabel("Transition Warning: None")
        self.transition_label.setWordWrap(True)
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.transition_label)

        cells_group = QGroupBox("Cell Counts & Proportions")
        cells_layout = QVBoxLayout(cells_group)
        self.cell_count_label = QLabel("No cell-centric metrics")
        self.cell_count_label.setWordWrap(True)
        self.cell_proportion_label = QLabel("")
        self.cell_proportion_label.setWordWrap(True)
        cells_layout.addWidget(self.cell_count_label)
        cells_layout.addWidget(self.cell_proportion_label)

        timeline_group = QGroupBox("Cyclicity Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        self.timeline_canvas = CycleTimelineCanvas(timeline_group, width=4.0, height=3.2)
        timeline_layout.addWidget(self.timeline_canvas)

        layout.addWidget(probabilities)
        layout.addWidget(confidence_group)
        layout.addWidget(cells_group)
        layout.addWidget(timeline_group, 1)
        scroll.setWidget(content)
        return scroll

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1220; color: #e2e8f0; }
            QToolBar { background: #111827; border-bottom: 1px solid #334155; spacing: 5px; padding: 5px; }
            QToolButton { background: #1e3a5f; border-radius: 5px; padding: 6px; }
            QToolButton:hover { background: #2563eb; }
            QToolButton:disabled { color: #64748b; background: #1e293b; }
            QComboBox { background: #1e293b; border: 1px solid #475569; border-radius: 4px; padding: 5px; }
            QFrame#sidebar { background: #0f172a; border-right: 1px solid #334155; }
            QLabel#panelTitle { color: #7dd3fc; font-size: 16px; font-weight: 700; }
            QLabel#stageBanner { background: #334155; border-radius: 6px; padding: 10px; font-size: 20px; font-weight: 700; }
            QListWidget { background: #111827; border: 1px solid #334155; border-radius: 5px; }
            QListWidget::item { padding: 7px; border-bottom: 1px solid #1e293b; }
            QListWidget::item:selected { background: #1d4ed8; }
            QGroupBox { border: 1px solid #334155; border-radius: 6px; margin-top: 10px; padding: 9px; font-weight: 700; color: #7dd3fc; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QProgressBar { border: 1px solid #475569; border-radius: 4px; background: #1e293b; text-align: center; color: white; }
            QProgressBar::chunk { background: #0ea5e9; }
            QStatusBar { background: #111827; border-top: 1px solid #334155; }
            QScrollArea { border: none; }
            """
        )

    @Slot()
    def select_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Cytology Image Folder")
        if selected:
            self.set_folder(Path(selected))

    def set_folder(self, folder: Path | str) -> None:
        """Load supported images from a folder into the file-status sidebar."""
        path = Path(folder).expanduser()
        if not path.is_dir():
            raise ValueError(f"image folder does not exist: {path}")
        images = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS),
            key=lambda item: item.name.casefold(),
        )
        if not images:
            QMessageBox.warning(self, "No Images", f"No supported images were found in:\n{path}")
            return
        self.selected_folder = path
        self._image_paths = images
        self._results = []
        self._batch_result = None
        self.file_list.clear()
        file_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        for image_path in images:
            item = QListWidgetItem(file_icon, f"{image_path.name}  [Pending]")
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self.file_list.addItem(item)
        self.folder_summary.setText(f"{path.name}\n{len(images)} image(s)")
        self.status_label.setText(f"Ready to stage {len(images)} images")
        self.export_action.setEnabled(False)
        self.run_action.setEnabled(True)
        self.file_list.setCurrentRow(0)

    @Slot()
    def select_checkpoint(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            str(self.checkpoint.parent if self.checkpoint else Path.cwd()),
            "PyTorch Checkpoints (*.pt *.pth *.ckpt);;All Files (*)",
        )
        if selected:
            self.checkpoint = Path(selected)
            self._engine = self._injected_classifier
            self.status_label.setText(f"Checkpoint: {self.checkpoint.name}")

    @Slot()
    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        self.select_checkpoint_action.setEnabled(mode in {"cnn", "mil"})
        self._engine = self._injected_classifier if mode == "cnn" else None
        self.status_label.setText(f"Mode selected: {self.mode_combo.currentText()}")

    def _create_engine(self) -> object:
        mode = str(self.mode_combo.currentData())
        if mode == "cnn":
            if self._engine is not None:
                return self._engine
            if self.checkpoint is None:
                raise ValueError("select a CNN checkpoint before running staging")
            from cycles.stages.cnn import CNNClassifierService

            return CNNClassifierService.from_checkpoint(self.checkpoint, device=None)
        if mode == "cell-centric":
            from cycles.stages.cell_centric import CellCentricPipeline

            return CellCentricPipeline(detector_mode="auto")
        if mode == "mil":
            from cycles.stages.mil import AttentionMILPipeline

            return AttentionMILPipeline(weights_path=self.checkpoint, device=None)
        raise ValueError(f"unknown staging mode: {mode}")

    @Slot()
    def run_staging(self) -> None:
        if self.selected_folder is None:
            QMessageBox.information(self, "Select Folder", "Select an image folder first.")
            return
        try:
            self._engine = self._create_engine()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Start", str(exc))
            return

        mode = str(self.mode_combo.currentData())
        self._results = []
        self._batch_result = None
        self._mark_all_pending()
        if self._artifact_temp is not None:
            self._artifact_temp.cleanup()
            self._artifact_temp = None
        artifact_dir: Path | None = None
        if mode in {"cell-centric", "mil"}:
            self._artifact_temp = TemporaryDirectory(prefix="cycles-gui-")
            artifact_dir = Path(self._artifact_temp.name)
        self._worker = ClassificationWorker(
            self._engine,
            self.selected_folder,
            mode=mode,
            recursive=False,
            artifact_dir=artifact_dir,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._set_running(True)
        self.status_label.setText(f"Running {self.mode_combo.currentText()} staging…")
        self._worker.start()

    @Slot()
    def cancel_staging(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.cancel_action.setEnabled(False)
            self.status_label.setText("Cancellation requested; finishing the current image…")

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(min(current, max(total, 1)))
        self.status_label.setText(message)
        if current > 0 and current - 1 < self.file_list.count():
            self._set_item_status(current - 1, "Complete", success=True)

    @Slot(object)
    def _on_finished(self, result: object) -> None:
        was_cancelled = self._worker is not None and self._worker.is_cancelled
        self._batch_result = result
        values = getattr(result, "results", result)
        self._results = list(values) if values is not None else []
        self._set_running(False)
        self.export_action.setEnabled(bool(self._results))
        prefix = "Staging cancelled" if was_cancelled else "Staging complete"
        self.status_label.setText(f"{prefix}: {len(self._results)} result(s)")
        self._apply_result_statuses()
        if self.file_list.count():
            self._display_index(max(self.file_list.currentRow(), 0))

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText(f"Staging failed: {message}")
        QMessageBox.critical(self, "Staging Error", message)

    def _set_running(self, running: bool) -> None:
        has_folder = self.selected_folder is not None
        self.select_folder_action.setEnabled(not running)
        self.select_checkpoint_action.setEnabled(not running and self.mode_combo.currentData() in {"cnn", "mil"})
        self.mode_combo.setEnabled(not running)
        self.run_action.setEnabled(not running and has_folder)
        self.cancel_action.setEnabled(running)
        self.export_action.setEnabled(not running and bool(self._results))
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, 0)

    def _mark_all_pending(self) -> None:
        for index, path in enumerate(self._image_paths):
            item = self.file_list.item(index)
            item.setText(f"{path.name}  [Pending]")
            item.setForeground(QColor("#e2e8f0"))
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))

    def _set_item_status(self, index: int, status: str, *, success: bool) -> None:
        if not 0 <= index < self.file_list.count():
            return
        item = self.file_list.item(index)
        path = item.data(Qt.ItemDataRole.UserRole)
        item.setText(f"{Path(path).name}  [{status}]")
        icon = QStyle.StandardPixmap.SP_DialogApplyButton if success else QStyle.StandardPixmap.SP_MessageBoxWarning
        item.setIcon(self.style().standardIcon(icon))
        item.setForeground(QColor("#86efac" if success else "#fca5a5"))

    def _failed_paths(self) -> set[Path]:
        failures = getattr(self._batch_result, "failed_images", None)
        if failures is None and self._engine is not None:
            failures = getattr(self._engine, "processing_errors", ())
        return {Path(path).resolve() for path, _message in failures or ()}

    def _apply_result_statuses(self) -> None:
        failed_paths = self._failed_paths()
        result_paths = {
            Path(path).resolve()
            for result in self._results
            if (path := getattr(result, "image_path", None)) is not None
        }
        if result_paths or failed_paths:
            for index, path in enumerate(self._image_paths):
                resolved = path.resolve()
                success = resolved in result_paths or (
                    not result_paths and resolved not in failed_paths and index < len(self._results) + len(failed_paths)
                )
                self._set_item_status(index, "Complete" if success else "Failed", success=success)
        else:
            for index in range(len(self._image_paths)):
                success = index < len(self._results)
                self._set_item_status(index, "Complete" if success else "Failed", success=success)

    def _result_at(self, index: int) -> object | None:
        if not 0 <= index < len(self._image_paths):
            return None
        target = self._image_paths[index].resolve()
        for result in self._results:
            image_path = getattr(result, "image_path", None)
            if image_path is not None and Path(image_path).resolve() == target:
                return result
        if target in self._failed_paths():
            return None
        successful_index = sum(
            1
            for path in self._image_paths[:index]
            if path.resolve() not in self._failed_paths()
        )
        return self._results[successful_index] if successful_index < len(self._results) else None

    @Slot(int)
    def _display_index(self, index: int) -> None:
        if not 0 <= index < len(self._image_paths):
            return
        path = self._image_paths[index]
        result = self._result_at(index)
        boxes: list[object] = []
        heatmap = None
        display_path = path
        if result is not None:
            metrics = getattr(result, "metrics", None)
            boxes = list(getattr(metrics, "cell_profiles", [])) if metrics is not None else []
            heatmap = getattr(result, "attention_heatmap", getattr(result, "heatmap", None))
            if self.mode_combo.currentData() == "mil" and self._artifact_temp is not None:
                rendered_heatmap = Path(self._artifact_temp.name) / f"{path.stem}_attention.png"
                if rendered_heatmap.is_file():
                    display_path = rendered_heatmap
        try:
            self.image_canvas.show_image(
                display_path,
                bounding_boxes=boxes,
                attention_heatmap=heatmap,
                title=path.name,
            )
        except Exception as exc:
            self.image_canvas.clear(f"Unable to display image: {exc}")
        self.image_caption.setText(f"{index + 1} / {len(self._image_paths)} — {path.name}")
        self._update_details(result, index)

    def _update_details(self, result: object | None, current_index: int) -> None:
        if result is None:
            self.stage_banner.setText("Pending")
            self.stage_banner.setStyleSheet("")
            probabilities: dict[object, float] = {}
        else:
            stage = getattr(result, "predicted_stage", getattr(result, "stage", None))
            stage_key = _stage_value(stage)
            self.stage_banner.setText(_humanise(stage))
            color = PHASE_COLORS.get(stage_key, "#64748b")
            self.stage_banner.setStyleSheet(
                f"background: {color}; color: white; border-radius: 6px; padding: 10px; font-size: 20px; font-weight: 700;"
            )
            probabilities = getattr(result, "probabilities", getattr(result, "stage_probabilities", {}))

        for stage, bar in self.probability_bars.items():
            probability = float(probabilities.get(stage, probabilities.get(stage.value, 0.0)))
            bar.setValue(round(probability * 1000))
            bar.setFormat(f"{probability:.1%}")

        if result is None:
            self.confidence_label.setText("Confidence Index: —")
            self.transition_label.setText("Transition Warning: None")
            self.cell_count_label.setText("No cell-centric metrics")
            self.cell_proportion_label.clear()
        else:
            confidence_index = getattr(result, "confidence_index", None)
            confidence = getattr(result, "confidence", None)
            if confidence_index is None:
                ranked_probabilities = sorted(
                    (float(value) for value in probabilities.values()),
                    reverse=True,
                )
                if len(ranked_probabilities) >= 2:
                    denominator = ranked_probabilities[0] + ranked_probabilities[1]
                    confidence_index = (
                        (ranked_probabilities[0] - ranked_probabilities[1]) / denominator
                        if denominator > 0
                        else 0.0
                    )
                else:
                    confidence_index = 0.0
            self.confidence_label.setText(
                f"Confidence Index: {float(confidence_index):.3f}   |   Confidence: {float(confidence or 0.0):.1%}"
            )
            is_transition = bool(getattr(result, "is_transition", False))
            transition_to = getattr(result, "transition_to", getattr(result, "transition_label", None))
            self.transition_label.setText(
                f"Transition Warning: {_humanise(transition_to)}" if is_transition else "Transition Warning: None"
            )
            self.transition_label.setStyleSheet("color: #fbbf24;" if is_transition else "color: #86efac;")
            self._update_cell_metrics(getattr(result, "metrics", None))

        stages = [
            getattr(item, "predicted_stage", getattr(item, "stage", None))
            for item in self._results
        ]
        stages = [stage for stage in stages if stage is not None]
        self.timeline_canvas.update_cycle(stages, current_index=min(current_index, len(stages) - 1) if stages else None)

    def _update_cell_metrics(self, metrics: object | None) -> None:
        if metrics is None:
            self.cell_count_label.setText("No cell-centric metrics")
            self.cell_proportion_label.clear()
            return
        total = int(getattr(metrics, "valid_cell_count", getattr(metrics, "total_cells", 0)))
        leukocytes = int(getattr(metrics, "leukocyte_count", 0))
        nucleated = int(getattr(metrics, "nucleated_epithelial_count", 0))
        cornified = int(getattr(metrics, "cornified_squamous_count", 0))
        self.cell_count_label.setText(
            f"Valid cells: {total}\nLeukocytes: {leukocytes}\nNucleated epithelial: {nucleated}\nCornified squamous: {cornified}"
        )
        self.cell_proportion_label.setText(
            "Proportions: "
            f"L {float(getattr(metrics, 'leukocyte_fraction', 0.0)):.1%}  ·  "
            f"N {float(getattr(metrics, 'nucleated_epithelial_fraction', 0.0)):.1%}  ·  "
            f"C {float(getattr(metrics, 'cornified_squamous_fraction', 0.0)):.1%}"
        )

    @Slot()
    def export_results(self) -> None:
        if not self._results or self._engine is None:
            return
        selected, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Staging Results",
            str((self.selected_folder or Path.cwd()) / "cycles_results.csv"),
            "CSV Files (*.csv);;JSON Files (*.json)",
        )
        if not selected:
            return
        output = Path(selected)
        try:
            if output.suffix.lower() == ".json" or "JSON" in selected_filter:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(self._results, default=_json_default, indent=2), encoding="utf-8")
            else:
                payload = self._results if self.mode_combo.currentData() == "cell-centric" else self._batch_result
                self._engine.export_results_csv(payload, output)  # type: ignore[attr-defined]
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        self.status_label.setText(f"Exported results to {output}")
        QMessageBox.information(self, "Export Complete", f"Results written to:\n{output}")

    def _on_toggle_labels(self, checked: bool) -> None:
        """Toggle explainability bounding boxes and text badges on canvas."""
        self.image_canvas.set_show_labels(checked)
        self.toggle_labels_action.setChecked(checked)
        self.toggle_labels_action.setText("Hide Labels" if checked else "Show Labels")
        if hasattr(self, "labels_checkbox") and self.labels_checkbox.isChecked() != checked:
            self.labels_checkbox.setChecked(checked)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        if self._artifact_temp is not None:
            self._artifact_temp.cleanup()
            self._artifact_temp = None
        super().closeEvent(event)
