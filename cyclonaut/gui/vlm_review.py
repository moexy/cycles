"""Evidence-first review workspace for local morphology VLM predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QImage, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cyclonaut.core.cycle import STAGE_CYCLE_ORDER
from cyclonaut.core.types import EstrousStage
from cyclonaut.vlm_local.annotations import AnnotationStore
from cyclonaut.vlm_local.schema import (
    Abundance,
    Arrangement,
    ConfidenceTier,
    LocalVLMRecord,
    NuclearState,
    QCStatus,
)
from cyclonaut.vlm_local.views import build_view_pack


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._has_image = False

    def set_pil_image(self, image: Image.Image) -> None:
        rgb = image.convert("RGB")
        width, height = rgb.size
        qimage = QImage(
            rgb.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888
        ).copy()
        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, width, height)
        self._has_image = True
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if not self._has_image:
            return super().wheelEvent(event)
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._has_image and self.transform().m11() == 1.0:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class VLMReviewWorkspace(QWidget):
    """Review predictions without mutating model records or adapting live."""

    def __init__(
        self,
        *,
        reviewer_id: str = "local-reviewer",
        blinded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.reviewer_id = reviewer_id
        self.records: list[LocalVLMRecord] = []
        self.annotation_store: AnnotationStore | None = None
        # Blinded by default: an annotator who forgets the toggle is still protected.
        self._blinded = bool(blinded)
        self._display_order: list[int] = []
        self._display_number: dict[int, int] = {}
        self._build_ui()
        self._build_shortcuts()
        self._apply_blinding()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.load_button = QPushButton("Load Predictions")
        self.load_button.clicked.connect(self._choose_results)
        self.export_button = QPushButton("Export Frozen Teacher Data")
        self.export_button.clicked.connect(self._choose_export)
        self.export_button.setEnabled(False)
        self.summary_label = QLabel("No v3 predictions loaded")
        self.blind_checkbox = QCheckBox("Blinded")
        self.blind_checkbox.setChecked(self._blinded)
        self.blind_checkbox.toggled.connect(self._on_blind_toggled)
        self.blind_status_label = QLabel()
        self.blind_status_label.setObjectName("blindStatus")
        toolbar.addWidget(self.load_button)
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.summary_label, 1)
        toolbar.addWidget(self.blind_status_label)
        toolbar.addWidget(self.blind_checkbox)
        outer.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_queue())
        splitter.addWidget(self._build_evidence_panel())
        splitter.addWidget(self._build_annotation_panel())
        splitter.setSizes([260, 760, 430])
        outer.addWidget(splitter, 1)

    def _build_queue(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        title = QLabel("Review Queue")
        title.setObjectName("panelTitle")
        self.queue_filter = QComboBox()
        for label, value in (
            ("All", "all"),
            ("Pending", "pending"),
            ("Accepted", "accept"),
            ("Corrected", "correct"),
            ("Ungradable", "ungradable"),
            ("Deferred", "defer"),
        ):
            self.queue_filter.addItem(label, value)
        self.queue_filter.currentIndexChanged.connect(self._refresh_queue)
        self.queue_list = QListWidget()
        self.queue_list.currentRowChanged.connect(self._display_current)
        layout.addWidget(title)
        layout.addWidget(self.queue_filter)
        layout.addWidget(self.queue_list, 1)
        return panel

    def _build_evidence_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        self.image_heading = QLabel("Whole-field overview")
        self.image_heading.setObjectName("panelTitle")
        self.overview_view = ZoomableImageView()
        tile_row = QHBoxLayout()
        self.tile_labels: list[QLabel] = []
        for _ in range(4):
            label = QLabel("Tile")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(120, 90)
            label.setStyleSheet("border: 1px solid #334155;")
            self.tile_labels.append(label)
            tile_row.addWidget(label, 1)
        layout.addWidget(self.image_heading)
        layout.addWidget(self.overview_view, 1)
        layout.addLayout(tile_row)
        return panel

    def _build_annotation_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        morphology_group = QGroupBox("Editable morphology and QC")
        form = QFormLayout(morphology_group)
        self.qc_combo = _enum_combo(QCStatus)
        self.cornified_combo = _enum_combo(Abundance)
        self.nucleated_combo = _enum_combo(Abundance)
        self.leukocyte_combo = _enum_combo(Abundance)
        self.nuclear_combo = _enum_combo(NuclearState)
        self.arrangement_combo = _enum_combo(Arrangement)
        form.addRow("QC", self.qc_combo)
        form.addRow("Cornified squames", self.cornified_combo)
        form.addRow("Nucleated epithelium", self.nucleated_combo)
        form.addRow("Leukocytes", self.leukocyte_combo)
        form.addRow("Nuclear state", self.nuclear_combo)
        form.addRow("Arrangement", self.arrangement_combo)

        prediction_group = QGroupBox("Stage assessment")
        prediction_form = QFormLayout(prediction_group)
        self.prediction_form = prediction_form
        self.primary_stage_combo = _stage_combo(optional=False)
        self.secondary_stage_combo = _stage_combo(optional=True)
        self.confidence_combo = _enum_combo(ConfidenceTier)
        self.image_only_label = QLabel("—")
        self.final_stage_label = QLabel("—")
        prediction_form.addRow("Image-only call", self.image_only_label)
        prediction_form.addRow("Final call", self.final_stage_label)
        prediction_form.addRow("Reviewed primary", self.primary_stage_combo)
        prediction_form.addRow("Reviewed secondary", self.secondary_stage_combo)
        prediction_form.addRow("Confidence", self.confidence_combo)

        self.transition_panel = QGroupBox("Adjacent-stage transition adjudication")
        transition_layout = QVBoxLayout(self.transition_panel)
        summary_row = QHBoxLayout()
        self.transition_primary = QLabel("Primary: —")
        self.transition_secondary = QLabel("Secondary: —")
        summary_row.addWidget(self.transition_primary)
        summary_row.addWidget(self.transition_secondary)
        transition_layout.addLayout(summary_row)

        self.transition_neighbors_container = QWidget()
        neighbors_layout = QHBoxLayout(self.transition_neighbors_container)
        neighbors_layout.setContentsMargins(0, 0, 0, 0)

        prev_box = QVBoxLayout()
        self.neighbor_prev_title = QLabel("Previous Observation")
        self.neighbor_prev_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.neighbor_prev_image = QLabel()
        self.neighbor_prev_image.setFixedSize(140, 100)
        self.neighbor_prev_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.neighbor_prev_image.setStyleSheet("border: 1px solid #d1d5db; background: #f3f4f6;")
        self.neighbor_prev_text = QLabel("—")
        self.neighbor_prev_text.setWordWrap(True)
        prev_box.addWidget(self.neighbor_prev_title)
        prev_box.addWidget(self.neighbor_prev_image)
        prev_box.addWidget(self.neighbor_prev_text)

        next_box = QVBoxLayout()
        self.neighbor_next_title = QLabel("Next Observation")
        self.neighbor_next_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.neighbor_next_image = QLabel()
        self.neighbor_next_image.setFixedSize(140, 100)
        self.neighbor_next_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.neighbor_next_image.setStyleSheet("border: 1px solid #d1d5db; background: #f3f4f6;")
        self.neighbor_next_text = QLabel("—")
        self.neighbor_next_text.setWordWrap(True)
        next_box.addWidget(self.neighbor_next_title)
        next_box.addWidget(self.neighbor_next_image)
        next_box.addWidget(self.neighbor_next_text)

        neighbors_layout.addLayout(prev_box)
        neighbors_layout.addLayout(next_box)
        transition_layout.addWidget(self.transition_neighbors_container)
        self.transition_neighbors_container.hide()
        self.transition_panel.hide()

        evidence_group = QGroupBox("Evidence")
        evidence_layout = QVBoxLayout(evidence_group)
        self.rationale_label = QLabel("—")
        self.rationale_label.setWordWrap(True)
        self.evidence_edit = QPlainTextEdit()
        self.evidence_edit.setPlaceholderText("One image-grounded observation per line")
        self.note_edit = QPlainTextEdit()
        self.note_edit.setMaximumHeight(70)
        self.note_edit.setPlaceholderText("Reviewer note")
        evidence_layout.addWidget(self.rationale_label)
        evidence_layout.addWidget(self.evidence_edit)
        evidence_layout.addWidget(self.note_edit)

        actions = QHBoxLayout()
        self.accept_button = QPushButton("Accept [A]")
        self.correct_button = QPushButton("Correct [C]")
        self.ungradable_button = QPushButton("Ungradable [U]")
        self.defer_button = QPushButton("Defer [D]")
        for button, action in (
            (self.accept_button, "accept"),
            (self.correct_button, "correct"),
            (self.ungradable_button, "ungradable"),
            (self.defer_button, "defer"),
        ):
            button.clicked.connect(lambda _checked=False, value=action: self.review(value))
            actions.addWidget(button)

        layout.addWidget(morphology_group)
        layout.addWidget(prediction_group)
        layout.addWidget(self.transition_panel)
        layout.addWidget(evidence_group, 1)
        layout.addLayout(actions)
        scroll.setWidget(content)
        return scroll

    def _build_shortcuts(self) -> None:
        for text, shortcut, callback in (
            ("Accept review", "A", lambda: self.review("accept")),
            ("Correct review", "C", lambda: self.review("correct")),
            ("Mark ungradable", "U", lambda: self.review("ungradable")),
            ("Defer review", "D", lambda: self.review("defer")),
            ("Previous sample", "Left", self.previous),
            ("Next sample", "Right", self.next),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            self.addAction(action)

    def set_records(
        self,
        records: list[LocalVLMRecord],
        annotation_store: AnnotationStore,
    ) -> None:
        self.records = list(records)
        self.annotation_store = annotation_store
        self._recompute_display_order()
        self.export_button.setEnabled(bool(records))
        self.summary_label.setText(
            f"{len(records)} prediction(s) · reviews: {annotation_store.path.name}"
        )
        self._refresh_queue()

    def load_results(
        self,
        results_path: Path | str,
        annotation_path: Path | str | None = None,
    ) -> None:
        path = Path(results_path)
        records = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(LocalVLMRecord.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid v3 prediction at {path}:{line_number}") from exc
        review_path = Path(annotation_path) if annotation_path else path.with_suffix(".reviews.jsonl")
        self.set_records(records, AnnotationStore(review_path))

    @Slot()
    def _choose_results(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Open v3 Predictions", str(Path.cwd()), "JSON Lines (*.jsonl)"
        )
        if not selected:
            return
        try:
            self.load_results(selected)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Load Predictions", str(exc))

    @Slot()
    def _refresh_queue(self) -> None:
        if self.annotation_store is None:
            self.queue_list.clear()
            return
        latest = self.annotation_store.latest_by_sample()
        selected_filter = self.queue_filter.currentData()
        current_sample = self.current_record().sample_id if self.current_record() else None
        self.queue_list.clear()
        selected_row = 0
        for record_index in self._display_order:
            record = self.records[record_index]
            event = latest.get(record.sample_id)
            status = event.action if event else "pending"
            if selected_filter != "all" and status != selected_filter:
                continue
            item = QListWidgetItem(self._queue_label(record_index, record, status))
            item.setData(Qt.ItemDataRole.UserRole, record_index)
            self.queue_list.addItem(item)
            if record.sample_id == current_sample:
                selected_row = self.queue_list.count() - 1
        if self.queue_list.count():
            self.queue_list.setCurrentRow(selected_row)

    def _current_record_index(self) -> int | None:
        item = self.queue_list.currentItem()
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        return int(index) if index is not None else None

    def current_record(self) -> LocalVLMRecord | None:
        index = self._current_record_index()
        return self.records[index] if index is not None else None

    @property
    def blinded(self) -> bool:
        return self._blinded

    def set_blinded(self, blinded: bool) -> None:
        """Hide or reveal subject identity, day, and the sequence-derived call."""
        blinded = bool(blinded)
        if blinded == self._blinded:
            return
        self._blinded = blinded
        if self.blind_checkbox.isChecked() != blinded:
            self.blind_checkbox.setChecked(blinded)
        self._recompute_display_order()
        self._apply_blinding()
        self._refresh_queue()
        self._display_current(self.queue_list.currentRow())

    @Slot(bool)
    def _on_blind_toggled(self, checked: bool) -> None:
        self.set_blinded(checked)

    def _recompute_display_order(self) -> None:
        # Blinded review must not present subjects in day order, so sort by image
        # content hash: deterministic across runs and independent of acquisition order.
        indices = list(range(len(self.records)))
        if self._blinded:
            indices.sort(key=lambda index: (self.records[index].image_sha256, self.records[index].sample_id))
        self._display_order = indices
        self._display_number = {index: position for position, index in enumerate(indices, start=1)}

    def _apply_blinding(self) -> None:
        # The final call is derived from day ordering; showing it would reintroduce sequence.
        self.prediction_form.setRowVisible(self.final_stage_label, not self._blinded)
        if self._blinded:
            self.neighbor_prev_image.clear()
            self.neighbor_next_image.clear()
            self.neighbor_prev_text.clear()
            self.neighbor_next_text.clear()
            self.transition_neighbors_container.hide()
        self.blind_status_label.setText(
            "Blinded — subject, day and sequence call hidden"
            if self._blinded
            else "IDENTIFIED — subject and day are visible"
        )

    def _queue_label(self, record_index: int, record: LocalVLMRecord, status: str) -> str:
        if self._blinded:
            return f"Sample {self._display_number[record_index]:03d}  [{_status_label(status)}]"
        day = record.day if record.day is not None else "—"
        return f"{record.sample_id} · day {day}  [{_status_label(status)}]"

    def _heading_text(self, record_index: int, record: LocalVLMRecord) -> str:
        if self._blinded:
            return f"Sample {self._display_number[record_index]:03d} — whole field"
        return f"{record.sample_id} — whole field"

    @Slot(int)
    def _display_current(self, _row: int) -> None:
        record_index = self._current_record_index()
        if record_index is None:
            return
        record = self.records[record_index]
        views = build_view_pack(record.image_path)
        self.overview_view.set_pil_image(views[0].image)
        self.image_heading.setText(self._heading_text(record_index, record))
        for label, view in zip(self.tile_labels, views[1:], strict=True):
            pixmap = _pixmap(view.image).scaled(
                180,
                130,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(pixmap)
            label.setToolTip(view.label.replace("_", " ").title())
        morphology = record.morphology
        _set_combo_data(self.qc_combo, morphology.qc_status.value)
        _set_combo_data(self.cornified_combo, morphology.cornified_squames.value)
        _set_combo_data(self.nucleated_combo, morphology.nucleated_epithelial.value)
        _set_combo_data(self.leukocyte_combo, morphology.leukocytes.value)
        _set_combo_data(self.nuclear_combo, morphology.nuclear_state.value)
        _set_combo_data(self.arrangement_combo, morphology.arrangement.value)
        prediction = record.image_prediction
        _set_combo_data(
            self.primary_stage_combo,
            prediction.primary_stage.value if prediction.primary_stage else None,
        )
        _set_combo_data(
            self.secondary_stage_combo,
            prediction.secondary_stage.value if prediction.secondary_stage else None,
        )
        _set_combo_data(self.confidence_combo, prediction.confidence_tier.value)
        self.image_only_label.setText(
            prediction.primary_stage.display_name if prediction.primary_stage else "Ungradable"
        )
        final = record.sequence_prediction.final_stage
        self.final_stage_label.setText(final.display_name if final else "Ungradable")
        self.rationale_label.setText(prediction.rationale)
        self.evidence_edit.setPlainText("\n".join(morphology.evidence))
        self.note_edit.clear()
        ambiguous = _is_ambiguous_adjacent(record)
        self.transition_panel.setVisible(ambiguous)
        if ambiguous:
            self.transition_primary.setText(
                f"Primary: {prediction.primary_stage.display_name if prediction.primary_stage else '—'}"
            )
            self.transition_secondary.setText(
                f"Secondary: {prediction.secondary_stage.display_name if prediction.secondary_stage else '—'}"
            )
            if not self._blinded and record.subject_id is not None and record.day is not None:
                same_subject = [
                    r
                    for r in self.records
                    if r.subject_id == record.subject_id
                    and r.day is not None
                    and r.sample_id != record.sample_id
                ]
                prev_records = [r for r in same_subject if float(r.day or 0) < float(record.day or 0)]
                next_records = [r for r in same_subject if float(r.day or 0) > float(record.day or 0)]
                prev_record = max(prev_records, key=lambda r: float(r.day or 0)) if prev_records else None
                next_record = min(next_records, key=lambda r: float(r.day or 0)) if next_records else None

                has_neighbor = False
                if prev_record is not None:
                    has_neighbor = True
                    try:
                        p_views = build_view_pack(prev_record.image_path)
                        p_pix = _pixmap(p_views[0].image).scaled(
                            140,
                            100,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        self.neighbor_prev_image.setPixmap(p_pix)
                    except Exception:
                        self.neighbor_prev_image.setText("No Preview")
                    stage_name = (
                        prev_record.image_prediction.primary_stage.display_name
                        if prev_record.image_prediction.primary_stage
                        else "Ungradable"
                    )
                    self.neighbor_prev_text.setText(f"Day {prev_record.day}: {stage_name}")
                else:
                    self.neighbor_prev_image.clear()
                    self.neighbor_prev_image.setText("None")
                    self.neighbor_prev_text.setText("—")

                if next_record is not None:
                    has_neighbor = True
                    try:
                        n_views = build_view_pack(next_record.image_path)
                        n_pix = _pixmap(n_views[0].image).scaled(
                            140,
                            100,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        self.neighbor_next_image.setPixmap(n_pix)
                    except Exception:
                        self.neighbor_next_image.setText("No Preview")
                    stage_name = (
                        next_record.image_prediction.primary_stage.display_name
                        if next_record.image_prediction.primary_stage
                        else "Ungradable"
                    )
                    self.neighbor_next_text.setText(f"Day {next_record.day}: {stage_name}")
                else:
                    self.neighbor_next_image.clear()
                    self.neighbor_next_image.setText("None")
                    self.neighbor_next_text.setText("—")

                self.transition_neighbors_container.setVisible(has_neighbor)
            else:
                self.neighbor_prev_image.clear()
                self.neighbor_next_image.clear()
                self.neighbor_prev_text.clear()
                self.neighbor_next_text.clear()
                self.transition_neighbors_container.hide()
        else:
            self.neighbor_prev_image.clear()
            self.neighbor_next_image.clear()
            self.neighbor_prev_text.clear()
            self.neighbor_next_text.clear()
            self.transition_neighbors_container.hide()

    def review(self, action: str) -> None:
        record = self.current_record()
        if record is None or self.annotation_store is None:
            return
        corrections = self._corrections(record) if action == "correct" else {}
        try:
            self.annotation_store.append(
                record,
                reviewer_id=self.reviewer_id,
                action=action,
                corrections=corrections,
                note=self.note_edit.toPlainText(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Review Not Saved", str(exc))
            return
        self._refresh_queue()

    def _corrections(self, record: LocalVLMRecord) -> dict[str, Any]:
        morphology = record.morphology
        prediction = record.image_prediction
        current: dict[str, Any] = {
            "primary_stage": prediction.primary_stage.value if prediction.primary_stage else None,
            "secondary_stage": prediction.secondary_stage.value if prediction.secondary_stage else None,
            "confidence_tier": prediction.confidence_tier.value,
            "qc_status": morphology.qc_status.value,
            "cornified_squames": morphology.cornified_squames.value,
            "nucleated_epithelial": morphology.nucleated_epithelial.value,
            "leukocytes": morphology.leukocytes.value,
            "nuclear_state": morphology.nuclear_state.value,
            "arrangement": morphology.arrangement.value,
            "evidence": list(morphology.evidence),
        }
        edited: dict[str, Any] = {
            "primary_stage": self.primary_stage_combo.currentData(),
            "secondary_stage": self.secondary_stage_combo.currentData(),
            "confidence_tier": self.confidence_combo.currentData(),
            "qc_status": self.qc_combo.currentData(),
            "cornified_squames": self.cornified_combo.currentData(),
            "nucleated_epithelial": self.nucleated_combo.currentData(),
            "leukocytes": self.leukocyte_combo.currentData(),
            "nuclear_state": self.nuclear_combo.currentData(),
            "arrangement": self.arrangement_combo.currentData(),
            "evidence": [line.strip() for line in self.evidence_edit.toPlainText().splitlines() if line.strip()],
        }
        return {key: value for key, value in edited.items() if value != current[key]}

    def previous(self) -> None:
        if self.queue_list.currentRow() > 0:
            self.queue_list.setCurrentRow(self.queue_list.currentRow() - 1)

    def next(self) -> None:
        if self.queue_list.currentRow() + 1 < self.queue_list.count():
            self.queue_list.setCurrentRow(self.queue_list.currentRow() + 1)

    def export_teacher(self, output_dir: Path | str) -> dict[str, Any]:
        if self.annotation_store is None:
            raise ValueError("no annotation log is loaded")
        return self.annotation_store.export_teacher(self.records, output_dir)

    @Slot()
    def _choose_export(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose Parent Folder for Frozen Teacher Export"
        )
        if not selected:
            return
        destination = Path(selected) / "cycles-teacher-export"
        try:
            summary = self.export_teacher(destination)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Frozen Export Complete",
            f"Exported {summary['exported_samples']} sample(s) to:\n{destination}",
        )


def _enum_combo(enum_type: Any) -> QComboBox:
    combo = QComboBox()
    for value in enum_type:
        combo.addItem(value.value.replace("_", " ").title(), value.value)
    return combo


def _stage_combo(*, optional: bool) -> QComboBox:
    combo = QComboBox()
    if optional:
        combo.addItem("None", None)
    for stage in EstrousStage.canonical_stages():
        combo.addItem(stage.display_name, stage.value)
    return combo


def _set_combo_data(combo: QComboBox, value: Any) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _pixmap(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    width, height = rgb.size
    qimage = QImage(
        rgb.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(qimage)


def _is_ambiguous_adjacent(record: LocalVLMRecord, margin_threshold: float = 0.15) -> bool:
    prediction = record.image_prediction
    primary, secondary = prediction.primary_stage, prediction.secondary_stage
    if primary is None or secondary is None:
        return False
    probabilities = sorted(prediction.probabilities.values(), reverse=True)
    if len(probabilities) < 2 or probabilities[0] - probabilities[1] > margin_threshold:
        return False
    first, second = STAGE_CYCLE_ORDER.index(primary), STAGE_CYCLE_ORDER.index(secondary)
    return (first - second) % len(STAGE_CYCLE_ORDER) in {1, len(STAGE_CYCLE_ORDER) - 1}


def _status_label(status: str) -> str:
    return {
        "accept": "Accepted",
        "correct": "Corrected",
        "ungradable": "Ungradable",
        "defer": "Deferred",
        "pending": "Pending",
    }[status]
