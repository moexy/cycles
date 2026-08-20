from __future__ import annotations

import os
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
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
) -> LocalVLMRecord:
    remaining = (1.0 - primary_probability - secondary_probability) / 2
    probabilities = {stage: remaining for stage in EstrousStage.canonical_stages()}
    probabilities[primary] = primary_probability
    probabilities[secondary] = secondary_probability
    return LocalVLMRecord(
        "s1",
        str(image_path),
        "c" * 64,
        "mouse-1",
        3,
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
