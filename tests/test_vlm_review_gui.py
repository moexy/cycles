from __future__ import annotations

import os
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget
from pytestqt.qtbot import QtBot

from cycles.core.types import EstrousStage
from cycles.gui.vlm_review import VLMReviewWorkspace
from cycles.vlm_local.annotations import AnnotationStore
from cycles.vlm_local.schema import (
    Abundance,
    Arrangement,
    ConfidenceTier,
    ImagePrediction,
    LocalVLMRecord,
    MorphologyObservation,
    NuclearState,
    QCStatus,
    SequencePrediction,
)


def _record(
    image_path: Path,
    *,
    primary: EstrousStage = EstrousStage.METESTRUS,
    secondary: EstrousStage = EstrousStage.ESTRUS,
    primary_probability: float = 0.46,
    secondary_probability: float = 0.44,
    sample_id: str = "s1",
    subject_id: str = "mouse-1",
    day: float = 3,
    image_sha256: str = "c" * 64,
) -> LocalVLMRecord:
    remaining = (1.0 - primary_probability - secondary_probability) / 2
    probabilities = {stage: remaining for stage in EstrousStage.canonical_stages()}
    probabilities[primary] = primary_probability
    probabilities[secondary] = secondary_probability
    return LocalVLMRecord(
        sample_id,
        str(image_path),
        image_sha256,
        subject_id,
        day,
        MorphologyObservation(
            Abundance.PRESENT,
            Abundance.RARE,
            Abundance.PRESENT,
            NuclearState.MIXED,
            Arrangement.SHEETS,
            ("mucus",),
            QCStatus.USABLE,
            (),
            ("Cornified sheets and leukocytes are visible.",),
        ),
        ImagePrediction(
            primary,
            secondary,
            probabilities,
            probabilities,
            ConfidenceTier.LOW,
            "Mixed cornified cells and leukocytes.",
        ),
        SequencePrediction(primary, False, "image_only"),
        {"model_id": "test/model"},
    )


def test_review_workspace_shows_evidence_views_and_ambiguous_transition(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records([_record(image_path)], AnnotationStore(tmp_path / "reviews.jsonl"))
    workspace.show()
    qapp.processEvents()

    assert workspace.queue_list.count() == 1
    assert workspace.overview_view.scene() is not None
    assert len(workspace.tile_labels) == 4
    assert all(label.pixmap() is not None for label in workspace.tile_labels)
    assert workspace.transition_panel.isHidden() is False
    assert workspace.primary_stage_combo.currentData() == "metestrus"
    assert "Cornified sheets" in workspace.evidence_edit.toPlainText()


def test_review_actions_are_append_only_and_correction_keeps_model_prediction(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    record = _record(image_path)
    store = AnnotationStore(tmp_path / "reviews.jsonl")
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records([record], store)

    qtbot.mouseClick(workspace.accept_button, Qt.MouseButton.LeftButton)
    workspace.primary_stage_combo.setCurrentText("Estrus")
    qtbot.mouseClick(workspace.correct_button, Qt.MouseButton.LeftButton)

    events = store.events()
    assert [event.action for event in events] == ["accept", "correct"]
    assert events[-1].corrections["primary_stage"] == "estrus"
    assert record.image_prediction.primary_stage is EstrousStage.METESTRUS
    assert "Corrected" in workspace.queue_list.item(0).text()


def test_confident_call_does_not_open_transition_adjudication(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records(
        [
            _record(
                image_path,
                primary=EstrousStage.ESTRUS,
                secondary=EstrousStage.METESTRUS,
                primary_probability=0.90,
                secondary_probability=0.05,
            )
        ],
        AnnotationStore(tmp_path / "reviews.jsonl"),
    )

    assert workspace.transition_panel.isHidden() is True


def test_review_workspace_exports_frozen_teacher_dataset(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records([_record(image_path)], AnnotationStore(tmp_path / "reviews.jsonl"))
    workspace.review("accept")

    summary = workspace.export_teacher(tmp_path / "teacher-v1")

    assert summary["exported_samples"] == 1
    assert (tmp_path / "teacher-v1" / "dataset.jsonl").is_file()


def _widget_text(workspace: VLMReviewWorkspace) -> str:
    """Every string the blinded reviewer can actually see."""
    fragments: list[str] = []
    for widget in workspace.findChildren(QWidget):
        for accessor in ("text", "toPlainText", "currentText", "title", "placeholderText"):
            method = getattr(widget, accessor, None)
            if callable(method):
                value = method()
                if isinstance(value, str):
                    fragments.append(value)
        tooltip = widget.toolTip()
        if tooltip:
            fragments.append(tooltip)
    for row in range(workspace.queue_list.count()):
        fragments.append(workspace.queue_list.item(row).text())
    return "\n".join(fragments)


def test_blinded_by_default_hides_sample_id_day_and_sequence_call(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records([_record(image_path)], AnnotationStore(tmp_path / "reviews.jsonl"))
    workspace.show()
    qapp.processEvents()

    assert workspace.blinded is True
    assert workspace.queue_list.item(0).text() == "Sample 001  [Pending]"
    assert workspace.image_heading.text() == "Sample 001 — whole field"
    # the final call is derived from day ordering, so it stays hidden while blinded
    assert workspace.final_stage_label.isVisible() is False


def test_unblinding_reveals_sample_id_and_day(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records([_record(image_path)], AnnotationStore(tmp_path / "reviews.jsonl"))
    workspace.show()

    workspace.set_blinded(False)
    qapp.processEvents()

    assert workspace.blind_checkbox.isChecked() is False
    assert "s1" in workspace.queue_list.item(0).text()
    assert "day 3" in workspace.queue_list.item(0).text()
    assert workspace.image_heading.text() == "s1 — whole field"
    assert workspace.final_stage_label.isVisible() is True


def test_blinded_queue_order_is_content_derived_not_acquisition_order(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    records = [
        _record(image_path, sample_id="day-1", day=1, image_sha256="b" * 64),
        _record(image_path, sample_id="day-2", day=2, image_sha256="a" * 64),
    ]
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records(records, AnnotationStore(tmp_path / "reviews.jsonl"))

    # sorted by image hash, so the later day is presented first
    assert [workspace.queue_list.item(row).text() for row in range(2)] == [
        "Sample 001  [Pending]",
        "Sample 002  [Pending]",
    ]
    assert workspace.queue_list.item(0).data(Qt.ItemDataRole.UserRole) == 1

    workspace.set_blinded(False)
    assert [workspace.queue_list.item(row).data(Qt.ItemDataRole.UserRole) for row in range(2)] == [0, 1]


def test_blinded_workspace_leaks_no_identifier_into_any_widget_text(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_path = tmp_path / "zzsecretslide.png"
    Image.new("RGB", (100, 80), (20, 80, 160)).save(image_path)
    workspace = VLMReviewWorkspace(reviewer_id="reviewer-a")
    qtbot.addWidget(workspace)
    workspace.set_records(
        [
            _record(
                image_path,
                sample_id="sample-zzz-secret",
                subject_id="mouse-qqq-secret",
                day=7,
            )
        ],
        AnnotationStore(tmp_path / "reviews.jsonl"),
    )
    workspace.show()
    qapp.processEvents()

    visible = _widget_text(workspace)
    for identifier in ("sample-zzz-secret", "mouse-qqq-secret", "zzsecretslide", "day 7"):
        assert identifier not in visible, f"blinded review leaked {identifier!r}"

    workspace.set_blinded(False)
    qapp.processEvents()
    assert "sample-zzz-secret" in _widget_text(workspace)
