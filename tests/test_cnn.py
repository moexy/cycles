from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import cyclonaut.stages.cnn as cnn_module
from cyclonaut.core.types import EstrousStage
from cyclonaut.stages.cnn import CNNClassifierService, CNNTrainerService, CNNTrainingConfig


class ConstantLogitModel(nn.Module):
    def __init__(self, logits: tuple[float, float, float, float] = (0.0, 3.0, 1.0, -1.0)) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.expand(inputs.shape[0], -1)


class TinyImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(inputs).flatten(1))


def _save_image(path: Path, colour: tuple[int, int, int] = (130, 40, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), colour).save(path)


def test_classify_image_returns_probabilities_logits_and_transition_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.png"
    _save_image(image_path)
    service = CNNClassifierService(ConstantLogitModel(), torch.device("cpu"), img_size=16)

    result = service.classify_image(image_path)

    assert result.predicted_stage is EstrousStage.PROESTRUS
    assert result.image_path == image_path
    assert result.raw_logits == pytest.approx([0.0, 3.0, 1.0, -1.0])
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert result.confidence_index > 0.5 and not result.is_transition


def test_classify_folder_isolates_corrupt_images_and_reports_progress(tmp_path: Path) -> None:
    _save_image(tmp_path / "good.png")
    (tmp_path / "bad.png").write_bytes(b"corrupt")
    progress: list[tuple[int, int, str]] = []
    service = CNNClassifierService(ConstantLogitModel(), torch.device("cpu"), img_size=8)

    batch = service.classify_folder(tmp_path, progress_callback=lambda *args: progress.append(args))

    assert batch.total_processed == 2
    assert len(batch.results) == 1 and batch.results[0].image_path.name == "good.png"
    assert len(batch.failed_images) == 1 and batch.failed_images[0][0].name == "bad.png"
    assert "ValueError" in batch.failed_images[0][1]
    assert progress[0][:2] == (0, 2) and progress[-1][0] == 2


def test_classify_folder_honors_cancellation_before_next_image(tmp_path: Path) -> None:
    _save_image(tmp_path / "one.png")
    _save_image(tmp_path / "two.png")
    cancelled = threading.Event()
    cancelled.set()
    service = CNNClassifierService(ConstantLogitModel(), torch.device("cpu"), img_size=8)

    batch = service.classify_folder(tmp_path, cancel_flag=cancelled)

    assert batch.total_processed == 0
    assert batch.results == [] and batch.failed_images == []


def test_csv_and_json_exports_include_successes_and_failures(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    _save_image(good)
    bad.write_bytes(b"bad")
    service = CNNClassifierService(ConstantLogitModel(), torch.device("cpu"), img_size=8)
    batch = service.classify_folder(tmp_path)

    csv_path = service.export_results_csv(batch, tmp_path / "exports" / "results.csv")
    json_path = service.export_results_json(batch, tmp_path / "exports" / "results.json")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert {Path(row["image_path"]).name for row in rows} == {"good.png", "bad.png"}
    assert any(row["error"] for row in rows), "Failed images need an exported error"
    assert payload["successful"] == 1 and payload["failed"] == 1
    assert payload["results"][0]["predicted_stage"] == "proestrus"


def _make_stage_dataset(root: Path) -> None:
    colours = ((40, 40, 40), (200, 60, 60), (220, 220, 120), (80, 80, 200))
    for stage, colour in zip(EstrousStage.canonical_stages(), colours, strict=True):
        _save_image(root / stage.value / "sample.png", colour)


def test_cnn_trainer_runs_one_basic_training_and_validation_step(tmp_path: Path, mocker) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_stage_dataset(train_dir)
    _make_stage_dataset(val_dir)
    model = TinyImageClassifier()
    initial_weight = model.fc.weight.detach().clone()
    mocker.patch.object(cnn_module, "build_model", return_value=model)
    config = CNNTrainingConfig(
        architecture="resnet50",
        img_size=8,
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        pretrained=False,
        output_path=tmp_path / "best.pt",
    )

    result = CNNTrainerService(torch.device("cpu")).train(train_dir, val_dir, config)

    assert len(result.history) == 1 and result.best_epoch == 1
    assert result.checkpoint_path.is_file(), "The best validation epoch should be checkpointed"
    assert not result.cancelled and not result.stopped_early
    assert not torch.equal(initial_weight, model.fc.weight), "A training step should update model weights"
    assert 0.0 <= result.history[0].train_accuracy <= 1.0


def test_cnn_trainer_cancellation_writes_a_valid_checkpoint(tmp_path: Path, mocker) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_stage_dataset(train_dir)
    _make_stage_dataset(val_dir)
    mocker.patch.object(cnn_module, "build_model", return_value=TinyImageClassifier())
    cancelled = threading.Event()
    cancelled.set()
    config = CNNTrainingConfig(
        img_size=8,
        epochs=2,
        batch_size=4,
        pretrained=False,
        output_path=tmp_path / "cancelled.pt",
    )

    result = CNNTrainerService(torch.device("cpu")).train(
        train_dir,
        val_dir,
        config,
        cancel_flag=cancelled,
    )

    assert result.cancelled and result.history == []
    assert result.checkpoint_path.is_file(), "Cancellation should still preserve resumable weights"
