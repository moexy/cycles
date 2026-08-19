"""CNN inference and fine-tuning services for estrous stage assessment."""

from __future__ import annotations

import csv
import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cycles.core.cycle import compute_confidence_index
from cycles.core.models import build_model, load_checkpoint, save_checkpoint
from cycles.core.preprocessing import (
    discover_images,
    get_inference_transforms,
    get_train_transforms,
    load_image,
)
from cycles.core.types import (
    BatchClassificationResult,
    CheckpointMetadata,
    ClassificationResult,
    EstrousStage,
)

_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
CancelFlag = threading.Event | Callable[[], bool]


class OptimizerName(StrEnum):
    """Optimizers supported by :class:`CNNTrainerService`."""

    RMSPROP = "rmsprop"
    ADAMW = "adamw"


@dataclass(slots=True)
class CNNTrainingConfig:
    """Configuration for transfer learning on stage-labelled image folders."""

    architecture: str = "resnet50"
    img_size: int = 224
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 1e-4
    optimizer: OptimizerName | str = OptimizerName.ADAMW
    weight_decay: float = 1e-4
    rmsprop_alpha: float = 0.99
    lr_factor: float = 0.5
    lr_patience: int = 2
    early_stopping_patience: int = 5
    min_delta: float = 0.0
    num_workers: int = 0
    pretrained: bool = True
    output_path: Path | str = Path("runs/cnn_best.pt")

    def __post_init__(self) -> None:
        """Validate hyperparameters before allocating a model or data loaders."""
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 < self.rmsprop_alpha <= 1.0:
            raise ValueError("rmsprop_alpha must be in (0, 1]")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("lr_factor must be in (0, 1)")
        if self.lr_patience < 0:
            raise ValueError("lr_patience cannot be negative")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        try:
            self.optimizer = OptimizerName(str(self.optimizer).lower())
        except ValueError as error:
            choices = ", ".join(item.value for item in OptimizerName)
            raise ValueError(f"optimizer must be one of: {choices}") from error


@dataclass(slots=True, frozen=True)
class CNNEpochMetrics:
    """Training and validation measurements for one completed epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float


@dataclass(slots=True)
class CNNTrainingResult:
    """Outcome of a CNN fine-tuning run."""

    model: nn.Module
    history: list[CNNEpochMetrics]
    checkpoint_path: Path
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    stopped_early: bool
    cancelled: bool

    def __iter__(self) -> Iterator[object]:
        """Allow legacy ``model, history, path = result`` unpacking."""
        yield self.model
        yield self.history
        yield self.checkpoint_path


class _StageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset with an explicit biological class order instead of alphabetic order."""

    def __init__(self, root: Path | str, transform: Callable[[Any], torch.Tensor]) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {self.root}")

        directories = {
            child.name.casefold(): child
            for child in self.root.iterdir()
            if child.is_dir()
        }
        missing: list[str] = []
        for class_index, stage in enumerate(EstrousStage.canonical_stages()):
            class_dir = directories.get(stage.value.casefold())
            if class_dir is None:
                missing.append(stage.value)
                continue
            class_images = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in _IMAGE_EXTENSIONS
            )
            if not class_images:
                missing.append(stage.value)
                continue
            self.samples.extend((path, class_index) for path in class_images)

        if missing:
            expected = ", ".join(stage.value for stage in EstrousStage.canonical_stages())
            absent = ", ".join(missing)
            raise ValueError(
                f"Dataset '{self.root}' is missing images for: {absent}. "
                f"Expected populated class folders: {expected}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        image = load_image(image_path)
        return self.transform(image), target


def _cancel_requested(cancel_flag: CancelFlag | None) -> bool:
    if cancel_flag is None:
        return False
    if isinstance(cancel_flag, threading.Event):
        return cancel_flag.is_set()
    return bool(cancel_flag())


def _logits_from_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Model returned unsupported output type: {type(output).__name__}")


class CNNClassifierService:
    """Classify individual images or folders with a four-class CNN."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        architecture: str = "resnet50",
        img_size: int = 224,
    ) -> None:
        if img_size <= 0:
            raise ValueError("img_size must be positive")
        self.device = device
        self.architecture = architecture
        self.img_size = img_size
        self.model = model.to(device)
        self.model.eval()
        self.transform = get_inference_transforms(img_size=img_size)
        self._stages = tuple(EstrousStage.canonical_stages())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device | None = None,
    ) -> CNNClassifierService:
        """Restore a classifier from a self-describing checkpoint."""
        model, metadata = load_checkpoint(checkpoint_path, device=device)
        resolved_device = device or next(model.parameters(), torch.empty(0)).device
        if not isinstance(resolved_device, torch.device):
            resolved_device = torch.device(resolved_device)
        service = cls(
            model=model,
            device=resolved_device,
            architecture=metadata.architecture,
            img_size=metadata.img_size,
        )

        stages: list[EstrousStage] = []
        for class_name in metadata.classes:
            try:
                stages.append(EstrousStage(class_name.casefold()))
            except ValueError as error:
                raise ValueError(
                    f"Checkpoint contains unsupported class label: {class_name!r}"
                ) from error
        canonical = set(EstrousStage.canonical_stages())
        if len(stages) != len(canonical) or set(stages) != canonical:
            raise ValueError(
                "Checkpoint classes must contain exactly diestrus, proestrus, "
                "estrus, and metestrus"
            )
        service._stages = tuple(stages)
        return service

    def classify_image(self, image_path: Path | str) -> ClassificationResult:
        """Load and classify one image."""
        path = Path(image_path)
        image = load_image(path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = _logits_from_output(self.model(tensor))
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError(
                    "CNN must return one two-dimensional logit row per input image; "
                    f"received shape {tuple(logits.shape)}"
                )
            if logits.shape[1] != len(self._stages):
                raise ValueError(
                    f"CNN returned {logits.shape[1]} classes, expected {len(self._stages)}"
                )
            probabilities_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
            raw_logits = logits[0].detach().cpu().tolist()

        probabilities = {
            stage: float(probabilities_tensor[index].item())
            for index, stage in enumerate(self._stages)
        }
        predicted_stage = max(probabilities, key=probabilities.__getitem__)
        confidence = probabilities[predicted_stage]
        confidence_index, is_transition, transition_to = compute_confidence_index(probabilities)
        return ClassificationResult(
            image_path=path,
            predicted_stage=predicted_stage,
            confidence=confidence,
            probabilities=probabilities,
            confidence_index=confidence_index,
            is_transition=is_transition,
            transition_to=transition_to,
            raw_logits=[float(value) for value in raw_logits],
        )

    def classify_folder(
        self,
        folder_path: Path | str,
        recursive: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_flag: CancelFlag | None = None,
    ) -> BatchClassificationResult:
        """Classify all supported images while isolating failures per image."""
        started_at = time.perf_counter()
        images = discover_images(folder_path, recursive=recursive)
        total = len(images)
        results: list[ClassificationResult] = []
        failed_images: list[tuple[Path, str]] = []
        processed = 0

        if progress_callback is not None:
            progress_callback(0, total, f"Found {total} image(s)")

        for image_path in images:
            if _cancel_requested(cancel_flag):
                if progress_callback is not None:
                    progress_callback(processed, total, "Classification cancelled")
                break
            try:
                results.append(self.classify_image(image_path))
                message = f"Classified {image_path.name}"
            except Exception as error:
                failed_images.append((image_path, f"{type(error).__name__}: {error}"))
                message = f"Failed {image_path.name}"
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total, message)

        return BatchClassificationResult(
            results=results,
            failed_images=failed_images,
            total_processed=processed,
            duration_seconds=time.perf_counter() - started_at,
        )

    @staticmethod
    def export_results_csv(
        result: BatchClassificationResult,
        output_path: Path | str,
    ) -> Path:
        """Export successful classifications and isolated failures as UTF-8 CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stage_columns = list(EstrousStage.canonical_stages())
        fieldnames = [
            "image_path",
            "predicted_stage",
            "confidence",
            "confidence_index",
            "is_transition",
            "transition_to",
            *(f"probability_{stage.value}" for stage in stage_columns),
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for item in result.results:
                row: dict[str, object] = {
                    "image_path": str(item.image_path),
                    "predicted_stage": item.predicted_stage.value,
                    "confidence": item.confidence,
                    "confidence_index": item.confidence_index,
                    "is_transition": item.is_transition,
                    "transition_to": item.transition_to.value if item.transition_to else "",
                    "error": "",
                }
                row.update(
                    {
                        f"probability_{stage.value}": item.probabilities.get(stage, 0.0)
                        for stage in stage_columns
                    }
                )
                writer.writerow(row)
            for failed_path, error in result.failed_images:
                writer.writerow({"image_path": str(failed_path), "error": error})
        return path

    @staticmethod
    def export_results_json(
        result: BatchClassificationResult,
        output_path: Path | str,
    ) -> Path:
        """Export a complete batch result as UTF-8 JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_processed": result.total_processed,
            "successful": len(result.results),
            "failed": len(result.failed_images),
            "duration_seconds": result.duration_seconds,
            "results": [
                {
                    "image_path": str(item.image_path),
                    "predicted_stage": item.predicted_stage.value,
                    "confidence": item.confidence,
                    "confidence_index": item.confidence_index,
                    "is_transition": item.is_transition,
                    "transition_to": item.transition_to.value if item.transition_to else None,
                    "probabilities": {
                        stage.value: probability
                        for stage, probability in item.probabilities.items()
                    },
                    "raw_logits": item.raw_logits,
                }
                for item in result.results
            ],
            "failed_images": [
                {"image_path": str(image_path), "error": error}
                for image_path, error in result.failed_images
            ],
        }
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return path


class CNNTrainerService:
    """Fine-tune a torchvision CNN on four custom stage folders."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _default_device()

    def train(
        self,
        train_dir: Path | str,
        val_dir: Path | str,
        config: CNNTrainingConfig | None = None,
        progress_callback: Callable[[CNNEpochMetrics], None] | None = None,
        cancel_flag: CancelFlag | None = None,
    ) -> CNNTrainingResult:
        """Train, validate, early-stop, and export the best checkpoint."""
        cfg = config or CNNTrainingConfig()
        train_dataset = _StageFolderDataset(
            train_dir,
            transform=get_train_transforms(img_size=cfg.img_size, augment=True),
        )
        val_dataset = _StageFolderDataset(
            val_dir,
            transform=get_inference_transforms(img_size=cfg.img_size),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        model = build_model(
            architecture=cfg.architecture,
            num_classes=len(EstrousStage.canonical_stages()),
            pretrained=cfg.pretrained,
        ).to(self.device)
        optimizer = self._build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
        )
        criterion = nn.CrossEntropyLoss()
        checkpoint_path = Path(cfg.output_path)
        history: list[CNNEpochMetrics] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_early = False
        cancelled = False

        for epoch in range(1, cfg.epochs + 1):
            if _cancel_requested(cancel_flag):
                cancelled = True
                break

            train_loss, train_accuracy, cancelled = self._train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cancel_flag,
            )
            if cancelled:
                break
            val_loss, val_accuracy = self._validate_epoch(model, val_loader, criterion)
            scheduler.step(val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            metrics = CNNEpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=current_lr,
            )
            history.append(metrics)

            if val_loss < best_val_loss - cfg.min_delta:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                self._save_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    cfg,
                    epoch=best_epoch,
                    val_loss=best_val_loss,
                    val_accuracy=best_val_accuracy,
                )
            else:
                epochs_without_improvement += 1

            if progress_callback is not None:
                progress_callback(metrics)
            if epochs_without_improvement >= cfg.early_stopping_patience:
                stopped_early = True
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            self._save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                cfg,
                epoch=0,
                val_loss=0.0,
                val_accuracy=0.0,
            )
            best_val_loss = 0.0

        model.eval()
        return CNNTrainingResult(
            model=model,
            history=history,
            checkpoint_path=checkpoint_path,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            stopped_early=stopped_early,
            cancelled=cancelled,
        )

    def _build_optimizer(
        self,
        model: nn.Module,
        config: CNNTrainingConfig,
    ) -> torch.optim.Optimizer:
        if config.optimizer is OptimizerName.RMSPROP:
            return torch.optim.RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                alpha=config.rmsprop_alpha,
                weight_decay=config.weight_decay,
            )
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        cancel_flag: CancelFlag | None,
    ) -> tuple[float, float, bool]:
        model.train()
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        for inputs, targets in loader:
            if _cancel_requested(cancel_flag):
                return (
                    loss_sum / sample_count if sample_count else 0.0,
                    correct / sample_count if sample_count else 0.0,
                    True,
                )
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            targets = targets.to(self.device, non_blocking=self.device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = _logits_from_output(model(inputs))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.shape[0]
            loss_sum += float(loss.detach().item()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            sample_count += batch_size
        if sample_count == 0:
            raise ValueError("Training dataset produced no samples")
        return loss_sum / sample_count, correct / sample_count, False

    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
    ) -> tuple[float, float]:
        model.eval()
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
                targets = targets.to(self.device, non_blocking=self.device.type == "cuda")
                logits = _logits_from_output(model(inputs))
                loss = criterion(logits, targets)
                batch_size = targets.shape[0]
                loss_sum += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                sample_count += batch_size
        if sample_count == 0:
            raise ValueError("Validation dataset produced no samples")
        return loss_sum / sample_count, correct / sample_count

    @staticmethod
    def _save_training_checkpoint(
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: CNNTrainingConfig,
        *,
        epoch: int,
        val_loss: float,
        val_accuracy: float,
    ) -> None:
        metadata = CheckpointMetadata(
            architecture=config.architecture,
            classes=[stage.value for stage in EstrousStage.canonical_stages()],
            img_size=config.img_size,
            created_at=datetime.now(UTC).isoformat(),
            epoch=epoch,
            val_acc=val_accuracy,
            metrics={"val_loss": val_loss, "val_accuracy": val_accuracy},
        )
        save_checkpoint(checkpoint_path, model, metadata, optimizer=optimizer)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
