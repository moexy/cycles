from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cyclonaut.core.models as model_utils
from cyclonaut.core.types import CheckpointMetadata


class TinyResNet(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(nn.Linear(3, 3))
        self.fc = nn.Linear(3, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer4(inputs))


def test_get_device_honors_available_preference_and_falls_back(mocker) -> None:
    mocker.patch.object(model_utils, "_mps_is_available", return_value=False)
    mocker.patch.object(torch.cuda, "is_available", return_value=False)

    assert model_utils.get_device("cpu") == torch.device("cpu")
    assert model_utils.get_device("cuda") == torch.device("cpu"), "Unavailable CUDA must fall back"
    assert model_utils.get_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="Unsupported device"):
        model_utils.get_device("xpu")
    with pytest.raises(ValueError, match="Invalid preferred device"):
        model_utils.get_device("not a device")


def _fake_resnet() -> nn.Module:
    return SimpleNamespace(fc=nn.Linear(8, 1000))  # type: ignore[return-value]


def _fake_inception() -> nn.Module:
    return SimpleNamespace(
        fc=nn.Linear(8, 1000),
        AuxLogits=SimpleNamespace(fc=nn.Linear(6, 1000)),
    )  # type: ignore[return-value]


def _fake_sequential(index: int) -> nn.Module:
    layers: list[nn.Module] = [nn.Identity() for _ in range(index + 1)]
    layers[index] = nn.Linear(8, 1000)
    return SimpleNamespace(classifier=nn.Sequential(*layers))  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("architecture", "constructor_name", "factory", "head"),
    [
        ("resnet50", "resnet50", _fake_resnet, "fc"),
        ("inception_v3", "inception_v3", _fake_inception, "fc"),
        ("vgg19", "vgg19", lambda: _fake_sequential(6), "classifier.6"),
        ("mobilenet_v2", "mobilenet_v2", lambda: _fake_sequential(1), "classifier.1"),
        ("convnext_tiny", "convnext_tiny", lambda: _fake_sequential(2), "classifier.2"),
    ],
)
def test_build_model_replaces_each_supported_backbone_head(
    architecture: str,
    constructor_name: str,
    factory,
    head: str,
    mocker,
) -> None:
    constructor = mocker.patch.object(
        model_utils.models,
        constructor_name,
        side_effect=lambda **_kwargs: factory(),
    )

    built = model_utils.build_model(architecture, num_classes=3, pretrained=False)
    output_head = built
    for part in head.split("."):
        output_head = output_head[int(part)] if part.isdigit() else getattr(output_head, part)

    assert isinstance(output_head, nn.Linear) and output_head.out_features == 3
    assert built.architecture == architecture and built.num_classes == 3
    assert constructor.call_args.kwargs["weights"] is None
    if architecture == "inception_v3":
        assert built.AuxLogits.fc.out_features == 3, "Inception auxiliary head must match class count"


def test_build_model_normalizes_name_and_rejects_invalid_counts(mocker) -> None:
    builder = mocker.patch.dict(
        model_utils.BACKBONE_REGISTRY,
        {"resnet50": lambda num_classes, pretrained: TinyResNet(num_classes)},
    )
    del builder
    assert model_utils.build_model(" ResNet50 ", 2, False).num_classes == 2
    with pytest.raises(ValueError, match="positive integer"):
        model_utils.build_model(num_classes=0)
    with pytest.raises(ValueError, match="Unsupported architecture"):
        model_utils.build_model("unknown", pretrained=False)


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        architecture="resnet50",
        classes=["diestrus", "proestrus", "estrus", "metestrus"],
        img_size=32,
        created_at="2026-08-19T00:00:00+00:00",
        epoch=3,
        val_acc=0.875,
        metrics={"val_loss": 0.25},
    )


def test_save_and_load_checkpoint_round_trip_with_metadata(tmp_path: Path, mocker) -> None:
    original = TinyResNet()
    optimizer = torch.optim.AdamW(original.parameters(), lr=0.01)
    checkpoint = tmp_path / "nested" / "model.pt"

    assert model_utils.save_checkpoint(checkpoint, original, _metadata(), optimizer) == checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" in payload
    mocker.patch.object(model_utils, "build_model", side_effect=lambda **kwargs: TinyResNet(kwargs["num_classes"]))
    restored, metadata = model_utils.load_checkpoint(checkpoint, device=torch.device("cpu"))

    assert metadata == _metadata()
    assert not restored.training
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), f"Parameter {name} changed on round trip"
    assert model_utils.load_checkpoint_metadata(checkpoint) == metadata


def test_load_raw_state_dict_infers_resnet_and_synthetic_metadata(tmp_path: Path, mocker) -> None:
    raw_path = tmp_path / "raw.pt"
    raw_model = TinyResNet(num_classes=2)
    torch.save(raw_model.state_dict(), raw_path)
    mocker.patch.object(model_utils, "build_model", side_effect=lambda **kwargs: TinyResNet(kwargs["num_classes"]))

    restored, metadata = model_utils.load_checkpoint(raw_path, device=torch.device("cpu"))

    assert isinstance(restored, TinyResNet)
    assert metadata.architecture == "resnet50"
    assert metadata.classes == ["class_0", "class_1"]
    assert model_utils.load_checkpoint_metadata(raw_path) is None


def test_checkpoint_errors_are_contextual_and_missing_weights_are_rejected(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="Could not load checkpoint"):
        model_utils.load_checkpoint(corrupt, device=torch.device("cpu"))

    missing_weights = tmp_path / "missing_weights.pt"
    torch.save({"metadata": {"architecture": "resnet50", "classes": ["a"]}}, missing_weights)
    with pytest.raises(ValueError, match="valid model state_dict"):
        model_utils.load_checkpoint(missing_weights, device=torch.device("cpu"))

    with pytest.raises(FileNotFoundError):
        model_utils.load_checkpoint(tmp_path / "absent.pt")


def test_save_checkpoint_rejects_class_count_mismatch(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata.classes = ["only-one"]
    with pytest.raises(ValueError, match="class count"):
        model_utils.save_checkpoint(tmp_path / "bad.pt", TinyResNet(), metadata)

def test_freeze_layers_freezes_backbone_and_keeps_trainable_heads() -> None:
    model = TinyResNet(num_classes=4)
    model.layer1 = nn.Linear(3, 3)
    model_utils.freeze_layers(model, trainable_prefixes=("layer4", "fc"))

    assert not model.layer1.weight.requires_grad
    assert model.layer4[0].weight.requires_grad
    assert model.fc.weight.requires_grad

    built = model_utils.build_model("resnet50", num_classes=4, pretrained=False, freeze_backbone=True)
    assert not built.conv1.weight.requires_grad
    assert built.layer4[0].conv1.weight.requires_grad
    assert built.fc.weight.requires_grad
