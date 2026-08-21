"""Torchvision backbone construction and self-describing checkpoint utilities."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torchvision import models

from cyclonaut.core.types import CheckpointMetadata

_CANONICAL_CLASSES: Final[list[str]] = ["diestrus", "proestrus", "estrus", "metestrus"]
BACKBONE_INPUT_SIZES: Final[dict[str, int]] = {
    "resnet50": 224,
    "inception_v3": 299,
    "vgg19": 224,
    "mobilenet_v2": 224,
    "convnext_tiny": 224,
}

ModelBuilder = Callable[[int, bool], nn.Module]


def _build_with_weights(
    constructor: Callable[..., nn.Module],
    weights: object | None,
    pretrained: bool,
    **kwargs: object,
) -> nn.Module:
    """Instantiate a model, falling back to random weights if a download fails."""
    try:
        return constructor(weights=weights if pretrained else None, **kwargs)
    except (OSError, RuntimeError) as error:
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained weights ({error}); using random initialization instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return constructor(weights=None, **kwargs)


def _build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if hasattr(models.ResNet50_Weights, "IMAGENET1K_V2") else models.ResNet50_Weights.DEFAULT
    model = _build_with_weights(
        models.resnet50,
        weights,
        pretrained,
    )
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise RuntimeError("Unexpected torchvision resnet50 classification head")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_inception_v3(num_classes: int, pretrained: bool) -> nn.Module:
    model = _build_with_weights(
        models.inception_v3,
        models.Inception_V3_Weights.DEFAULT,
        pretrained,
        aux_logits=True,
    )
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise RuntimeError("Unexpected torchvision inception_v3 classification head")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    aux_logits = getattr(model, "AuxLogits", None)
    if aux_logits is not None:
        if not hasattr(aux_logits, "fc") or not isinstance(aux_logits.fc, nn.Linear):
            raise RuntimeError("Unexpected torchvision inception_v3 auxiliary head")
        aux_logits.fc = nn.Linear(aux_logits.fc.in_features, num_classes)
    return model


def _replace_sequential_head(
    model: nn.Module,
    index: int,
    num_classes: int,
    architecture: str,
) -> nn.Module:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[index], nn.Linear):
        raise RuntimeError(f"Unexpected torchvision {architecture} classification head")
    classifier[index] = nn.Linear(classifier[index].in_features, num_classes)
    return model


def _build_vgg19(num_classes: int, pretrained: bool) -> nn.Module:
    model = _build_with_weights(models.vgg19, models.VGG19_Weights.DEFAULT, pretrained)
    return _replace_sequential_head(model, 6, num_classes, "vgg19")


def _build_mobilenet_v2(num_classes: int, pretrained: bool) -> nn.Module:
    model = _build_with_weights(
        models.mobilenet_v2,
        models.MobileNet_V2_Weights.DEFAULT,
        pretrained,
    )
    return _replace_sequential_head(model, 1, num_classes, "mobilenet_v2")


def _build_convnext_tiny(num_classes: int, pretrained: bool) -> nn.Module:
    model = _build_with_weights(
        models.convnext_tiny,
        models.ConvNeXt_Tiny_Weights.DEFAULT,
        pretrained,
    )
    return _replace_sequential_head(model, 2, num_classes, "convnext_tiny")


BACKBONE_REGISTRY: Final[dict[str, ModelBuilder]] = {
    "resnet50": _build_resnet50,
    "inception_v3": _build_inception_v3,
    "vgg19": _build_vgg19,
    "mobilenet_v2": _build_mobilenet_v2,
    "convnext_tiny": _build_convnext_tiny,
}
SUPPORTED_ARCHITECTURES: Final[tuple[str, ...]] = tuple(BACKBONE_REGISTRY)


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except (AttributeError, RuntimeError):
        return False


def _device_is_available(device: torch.device) -> bool:
    if device.type == "cpu":
        return True
    if device.type == "mps":
        return _mps_is_available()
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        return device.index is None or device.index < torch.cuda.device_count()
    return False


def get_device(preferred: str | None = None) -> torch.device:
    """Select an available MPS, CUDA, or CPU device.

    An available explicit preference is honored. If the requested accelerator is
    unavailable, selection falls back through MPS, CUDA, and finally CPU. CUDA
    indices such as ``"cuda:1"`` are supported. Unknown device types are rejected
    rather than silently changing the caller's intent.

    Args:
        preferred: Preferred device string, ``"auto"``, or ``None``.

    Raises:
        ValueError: If *preferred* is not a valid MPS, CUDA, or CPU device.
    """
    requested: torch.device | None = None
    if preferred is not None and preferred.strip().lower() != "auto":
        try:
            requested = torch.device(preferred.strip().lower())
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid preferred device '{preferred}'") from error
        if requested.type not in {"mps", "cuda", "cpu"}:
            raise ValueError(
                f"Unsupported device type '{requested.type}'. Supported: mps, cuda, cpu"
            )
        if _device_is_available(requested):
            return requested

    for candidate in (torch.device("mps"), torch.device("cuda"), torch.device("cpu")):
        if requested is not None and candidate.type == requested.type:
            continue
        if _device_is_available(candidate):
            return candidate
    return torch.device("cpu")


def _normalize_architecture(architecture: str) -> str:
    normalized = architecture.strip().lower().replace("-", "_")
    if normalized not in BACKBONE_REGISTRY:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {supported}")
    return normalized


def freeze_layers(
    model: nn.Module,
    trainable_prefixes: Sequence[str] = ("layer4", "fc", "classifier"),
) -> None:
    """Freeze all model parameters except those starting with trainable_prefixes."""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)


def build_model(
    architecture: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    trainable_prefixes: Sequence[str] = ("layer4", "fc", "classifier"),
) -> nn.Module:
    """Build a torchvision classifier with a task-specific output head."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")
    normalized = _normalize_architecture(architecture)
    model = BACKBONE_REGISTRY[normalized](num_classes, pretrained)
    if freeze_backbone:
        freeze_layers(model, trainable_prefixes)
    model.architecture = normalized  # type: ignore[attr-defined]
    model.num_classes = num_classes  # type: ignore[attr-defined]
    return model

def _load_payload(path: Path, map_location: torch.device | str) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Compatibility with torch releases predating the weights_only keyword.
        return torch.load(path, map_location=map_location)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise RuntimeError(f"Could not load checkpoint '{path}': {error}") from error


def _looks_like_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(isinstance(item, (torch.Tensor, nn.Parameter)) for item in value.values())
    )


def _normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    for prefix in ("module.", "_orig_mod.", "model."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {key.removeprefix(prefix): value for key, value in normalized.items()}
    return normalized


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    candidate: object = payload
    if isinstance(payload, Mapping):
        if "state_dict" in payload:
            candidate = payload["state_dict"]
        elif "model_state_dict" in payload:
            candidate = payload["model_state_dict"]
    if not _looks_like_state_dict(candidate):
        raise ValueError("Checkpoint does not contain a valid model state_dict")
    return _normalize_state_dict(candidate)  # type: ignore[arg-type]


def _infer_architecture(state_dict: Mapping[str, torch.Tensor]) -> str:
    keys = state_dict.keys()
    if any(key.startswith(("Mixed_", "AuxLogits.")) for key in keys):
        return "inception_v3"
    if "classifier.6.weight" in state_dict:
        return "vgg19"
    if "classifier.1.weight" in state_dict and any(key.startswith("features.18.") for key in keys):
        return "mobilenet_v2"
    if "classifier.2.weight" in state_dict and any(key.startswith("features.7.") for key in keys):
        return "convnext_tiny"
    if "fc.weight" in state_dict and any(key.startswith("layer4.") for key in keys):
        return "resnet50"
    raise ValueError(
        "Could not infer architecture from raw state_dict; save a self-describing checkpoint instead"
    )


def _infer_num_classes(state_dict: Mapping[str, torch.Tensor], architecture: str) -> int:
    head_key = {
        "resnet50": "fc.weight",
        "inception_v3": "fc.weight",
        "vgg19": "classifier.6.weight",
        "mobilenet_v2": "classifier.1.weight",
        "convnext_tiny": "classifier.2.weight",
    }[architecture]
    weight = state_dict.get(head_key)
    if weight is None or weight.ndim < 1 or int(weight.shape[0]) <= 0:
        raise ValueError(f"Could not infer class count from checkpoint key '{head_key}'")
    return int(weight.shape[0])


def _checkpoint_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _metadata_from_mapping(value: Mapping[object, object], path: Path) -> CheckpointMetadata:
    try:
        architecture = _normalize_architecture(str(value["architecture"]))
        raw_classes = value["classes"]
        if isinstance(raw_classes, (str, bytes)) or not isinstance(raw_classes, (list, tuple)):
            raise TypeError("classes must be a list of labels")
        classes = [str(label) for label in raw_classes]
        if not classes or any(not label for label in classes):
            raise ValueError("classes must contain at least one non-empty label")
        img_size = int(value.get("img_size", BACKBONE_INPUT_SIZES[architecture]))
        if img_size <= 0:
            raise ValueError("img_size must be positive")
        created_at = str(value.get("created_at", _checkpoint_timestamp(path)))
        epoch = int(value.get("epoch", 0))
        val_acc = float(value.get("val_acc", 0.0))
        raw_metrics = value.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        metrics = {str(name): float(metric) for name, metric in raw_metrics.items()}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint metadata in '{path}': {error}") from error
    return CheckpointMetadata(
        architecture=architecture,
        classes=classes,
        img_size=img_size,
        created_at=created_at,
        epoch=epoch,
        val_acc=val_acc,
        metrics=metrics,
    )


def _extract_metadata(payload: object, path: Path) -> CheckpointMetadata | None:
    if not isinstance(payload, Mapping):
        return None
    metadata = payload.get("metadata")
    if isinstance(metadata, CheckpointMetadata):
        return metadata
    if isinstance(metadata, Mapping) and "architecture" in metadata and "classes" in metadata:
        return _metadata_from_mapping(metadata, path)
    # Gracefully read early checkpoints that stored descriptive fields at top level.
    if "architecture" in payload and "classes" in payload:
        return _metadata_from_mapping(payload, path)
    return None


def _raw_checkpoint_metadata(
    path: Path,
    architecture: str,
    num_classes: int,
) -> CheckpointMetadata:
    classes = (
        list(_CANONICAL_CLASSES)
        if num_classes == len(_CANONICAL_CLASSES)
        else [f"class_{index}" for index in range(num_classes)]
    )
    return CheckpointMetadata(
        architecture=architecture,
        classes=classes,
        img_size=BACKBONE_INPUT_SIZES[architecture],
        created_at=_checkpoint_timestamp(path),
    )


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save model parameters and portable metadata in one checkpoint.

    The checkpoint contains ``state_dict`` and ``metadata`` entries plus an
    ``optimizer_state_dict`` entry when an optimizer is supplied. Metadata is
    serialized as primitive values so it remains loadable with PyTorch's safe
    ``weights_only`` loader.

    Returns:
        The checkpoint destination as a :class:`Path`.
    """
    destination = Path(path).expanduser()
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Checkpoint destination is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    architecture = _normalize_architecture(metadata.architecture)
    state_dict = _normalize_state_dict(model.state_dict())
    expected_classes = _infer_num_classes(state_dict, architecture)
    if expected_classes != len(metadata.classes):
        raise ValueError(
            "Checkpoint metadata class count does not match model output: "
            f"{len(metadata.classes)} labels for {expected_classes} outputs"
        )
    metadata_payload = asdict(metadata)
    metadata_payload["architecture"] = architecture
    checkpoint: dict[str, object] = {
        "state_dict": state_dict,
        "metadata": metadata_payload,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    try:
        torch.save(checkpoint, destination)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Could not save checkpoint '{destination}': {error}") from error
    return destination


def load_checkpoint(
    path: Path | str,
    device: torch.device | None = None,
) -> tuple[nn.Module, CheckpointMetadata]:
    """Load a classifier and its metadata from a checkpoint.

    Self-describing checkpoints are reconstructed directly. For a raw state_dict,
    the supported torchvision architecture and output width are inferred from its
    parameter names, and conservative synthetic metadata is returned.
    """
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise IsADirectoryError(f"Checkpoint path is not a file: {checkpoint_path}")

    target_device = get_device(str(device)) if device is not None else get_device()
    payload = _load_payload(checkpoint_path, "cpu")
    state_dict = _extract_state_dict(payload)
    metadata = _extract_metadata(payload, checkpoint_path)
    if metadata is None:
        architecture = _infer_architecture(state_dict)
        num_classes = _infer_num_classes(state_dict, architecture)
        metadata = _raw_checkpoint_metadata(checkpoint_path, architecture, num_classes)
    else:
        architecture = _normalize_architecture(metadata.architecture)
        num_classes = len(metadata.classes)

    model = build_model(architecture=architecture, num_classes=num_classes, pretrained=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint parameters are incompatible with {architecture}: {error}"
        ) from error
    model.to(target_device)
    model.eval()
    return model, metadata


def load_checkpoint_metadata(path: Path | str) -> CheckpointMetadata | None:
    """Read metadata without constructing a model; return ``None`` for raw weights."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise IsADirectoryError(f"Checkpoint path is not a file: {checkpoint_path}")
    payload = _load_payload(checkpoint_path, "cpu")
    return _extract_metadata(payload, checkpoint_path)
