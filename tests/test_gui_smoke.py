from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QToolBar
from pytestqt.qtbot import QtBot

from cyclonaut.gui.canvas import CycleTimelineCanvas, ImageOverlayCanvas
from cyclonaut.gui.main_window import MainWindow
from cyclonaut.gui.workers import ClassificationWorker


def _save_image(path: Path) -> Path:
    Image.new("RGB", (24, 18), (30, 90, 140)).save(path)
    return path


def test_main_window_initialization_modes_and_toolbar(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    assert QApplication.instance() is qapp
    classifier = object()
    window = MainWindow(classifier=classifier)
    qtbot.addWidget(window)
    window.show()
    qapp.processEvents()

    assert window.windowTitle() == "cycles — Rodent Estrous Staging"
    assert window.centralWidget() is not None
    assert [window.mode_combo.itemData(index) for index in range(window.mode_combo.count())] == [
        "cnn",
        "cell-centric",
        "mil",
    ]
    toolbar = window.findChild(QToolBar, "")
    assert toolbar is not None
    action_texts = {action.text() for action in toolbar.actions() if action.text()}
    assert {
        "Select Folder",
        "Select Checkpoint",
        "Run Staging",
        "Cancel",
        "Export Results",
    } <= action_texts
    assert window.cancel_action.isEnabled() is False
    assert window.run_action.isEnabled() is False
    assert window.export_action.isEnabled() is False

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _save_image(image_dir / "slide.png")
    window.set_folder(image_dir)
    assert window.file_list.count() == 1
    assert window.run_action.isEnabled() is True

    window.mode_combo.setCurrentIndex(1)
    assert window.mode_combo.currentData() == "cell-centric"
    assert window.select_checkpoint_action.isEnabled() is False
    window.mode_combo.setCurrentIndex(2)
    assert window.mode_combo.currentData() == "mil"
    assert window.select_checkpoint_action.isEnabled() is True
    window.mode_combo.setCurrentIndex(0)
    assert window._create_engine() is classifier


class _ImmediateClassifier:
    def classify_folder(
        self,
        folder: Path,
        *,
        recursive: bool,
        progress_callback: object,
        cancel_flag: object,
    ) -> SimpleNamespace:
        assert folder.is_dir()
        assert recursive is True
        assert not cancel_flag()  # type: ignore[operator]
        progress_callback(0, 2, "Found 2 images")  # type: ignore[operator]
        progress_callback(1, 2, "Processed first.png")  # type: ignore[operator]
        return SimpleNamespace(results=["first"], failed_images=[])


def test_classification_worker_emits_progress_and_finished(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    worker = ClassificationWorker(_ImmediateClassifier(), image_dir, mode="cnn", recursive=True)
    progress: list[tuple[int, int, str]] = []
    finished: list[object] = []
    errors: list[str] = []
    worker.progress.connect(lambda current, total, message: progress.append((current, total, message)))
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    with qtbot.waitSignal(worker.finished, timeout=3000):
        worker.start()
    assert worker.wait(3000)
    qapp.processEvents()

    assert errors == []
    assert progress == [(0, 2, "Found 2 images"), (1, 2, "Processed first.png")]
    assert len(finished) == 1
    assert finished[0].results == ["first"]


class _BlockingClassifier:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.cancel_observed = False

    def classify_folder(
        self,
        folder: Path,
        *,
        recursive: bool,
        progress_callback: object,
        cancel_flag: object,
    ) -> SimpleNamespace:
        progress_callback(1, 2, "Processed first.png")  # type: ignore[operator]
        if not self.release.wait(timeout=3):
            raise TimeoutError("test did not release classifier")
        self.cancel_observed = bool(cancel_flag())  # type: ignore[operator]
        if not self.cancel_observed:
            progress_callback(2, 2, "Processed second.png")  # type: ignore[operator]
        return SimpleNamespace(results=["first"], cancelled=self.cancel_observed)


def test_classification_worker_cancel_returns_partial_result(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    classifier = _BlockingClassifier()
    worker = ClassificationWorker(classifier, image_dir, mode="cnn")
    progress: list[tuple[int, int, str]] = []
    finished: list[object] = []
    worker.progress.connect(lambda current, total, message: progress.append((current, total, message)))
    worker.finished.connect(finished.append)

    worker.start()
    qtbot.waitUntil(lambda: progress == [(1, 2, "Processed first.png")], timeout=3000)
    worker.cancel()
    assert worker.is_cancelled is True
    with qtbot.waitSignal(worker.finished, timeout=3000):
        classifier.release.set()
    assert worker.wait(3000)
    qapp.processEvents()

    assert classifier.cancel_observed is True
    assert len(finished) == 1
    assert finished[0].cancelled is True
    assert finished[0].results == ["first"]
    assert progress == [(1, 2, "Processed first.png")]


def test_image_overlay_canvas_renders_image_heatmap_and_boxes(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    canvas = ImageOverlayCanvas()
    qtbot.addWidget(canvas)
    pixels = np.zeros((32, 40, 3), dtype=np.uint8)
    pixels[..., 1] = 160
    heatmap = np.linspace(0, 1, 32 * 40, dtype=np.float32).reshape(32, 40)

    canvas.show_image(
        pixels,
        attention_heatmap=heatmap,
        bounding_boxes=[
            {
                "bbox": (3, 5, 20, 25),
                "cell_type": "leukocyte",
                "confidence": 0.85,
            }
        ],
        title="Synthetic overlay",
    )
    canvas.draw()
    qapp.processEvents()

    assert canvas._image_shape == (32, 40)
    assert len(canvas.axes.images) == 2
    assert len(canvas.axes.patches) == 1
    assert canvas.axes.get_title() == "Synthetic overlay"
    assert any("Leukocyte 85%" in text.get_text() for text in canvas.axes.texts)
    output = tmp_path / "overlay.png"
    canvas.figure.savefig(output)
    assert output.read_bytes().startswith(b"\x89PNG")


def test_cycle_timeline_canvas_renders_timeline_and_current_sample(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    canvas = CycleTimelineCanvas()
    qtbot.addWidget(canvas)

    canvas.update_timeline(
        [1, 2, 3, 4],
        ["diestrus", "proestrus", "estrus", "metestrus"],
        current_index=2,
    )
    canvas.draw()
    qapp.processEvents()

    assert len(canvas.axes.lines) == 1
    assert list(canvas.axes.lines[0].get_xdata()) == [1.0, 2.0, 3.0, 4.0]
    assert list(canvas.axes.lines[0].get_ydata()) == [0, 1, 2, 3]
    assert len(canvas.axes.collections) == 1
    assert [label.get_text() for label in canvas.axes.get_yticklabels()] == [
        "Diestrus",
        "Proestrus",
        "Estrus",
        "Metestrus",
    ]
    output = tmp_path / "timeline.png"
    canvas.figure.savefig(output)
    assert output.read_bytes().startswith(b"\x89PNG")


def test_cycle_timeline_rejects_mismatched_inputs(qtbot: QtBot) -> None:
    canvas = CycleTimelineCanvas()
    qtbot.addWidget(canvas)

    with pytest.raises(ValueError, match="equal length"):
        canvas.update_timeline([1, 2], ["estrus"])
