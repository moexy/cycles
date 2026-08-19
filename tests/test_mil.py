from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import cycles.stages.mil.encoder as encoder_module
from cycles.core.types import EstrousStage
from cycles.stages.mil.encoder import PatchEncoder
from cycles.stages.mil.model import GatedAttentionMIL
from cycles.stages.mil.patching import PatchExtractor, PatchInfo
from cycles.stages.mil.pipeline import AttentionMILPipeline


def test_patch_extractor_tiles_image_and_preserves_spatial_metadata() -> None:
    stained = np.zeros((32, 32, 3), dtype=np.uint8)
    stained[..., 0] = 220
    stained[..., 1] = 30
    stained[..., 2] = 100
    extractor = PatchExtractor(patch_size=16, stride=16, min_tissue_ratio=0.5, max_patches=None)

    patches, info = extractor.extract(stained)

    assert len(patches) == 4 and len(info) == 4
    assert [(item.x, item.y) for item in info] == [(0, 0), (16, 0), (0, 16), (16, 16)]
    assert all(patch.size == (16, 16) for patch in patches)
    assert all(item.tissue_ratio == pytest.approx(1.0) for item in info)


def test_patch_extractor_filters_blank_background_and_caps_dense_patches() -> None:
    blank = np.full((24, 24, 3), 255, dtype=np.uint8)
    assert PatchExtractor(patch_size=12, min_tissue_ratio=0.1).extract(blank) == ([], [])

    tissue = np.full((24, 24, 3), (180, 10, 80), dtype=np.uint8)
    patches, info = PatchExtractor(
        patch_size=12,
        stride=12,
        min_tissue_ratio=0.1,
        max_patches=2,
    ).extract(tissue)
    assert len(patches) == len(info) == 2, "max_patches should cap even fully stained images"


def test_patch_extractor_reflection_pads_images_smaller_than_patch() -> None:
    source = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[20, 30, 40], [50, 60, 70], [80, 90, 100]],
        ],
        dtype=np.uint8,
    )
    patches, info = PatchExtractor(patch_size=5, min_tissue_ratio=0.0).extract(source)
    padded = np.asarray(patches[0])

    assert len(patches) == 1 and patches[0].size == (5, 5)
    assert info[0] == PatchInfo(x=0, y=0, width=3, height=2, tissue_ratio=info[0].tissue_ratio)
    np.testing.assert_array_equal(padded[:2, :3], source)
    np.testing.assert_array_equal(
        padded[2, :3], source[0], err_msg="The short vertical axis should use reflection padding"
    )


class TinyConvNext(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 768, kernel_size=1), nn.ReLU())


def test_patch_encoder_batches_mixed_patch_inputs_and_normalizes_embeddings(mocker) -> None:
    mocker.patch.object(
        encoder_module.models,
        "convnext_tiny",
        side_effect=lambda **_kwargs: TinyConvNext(),
    )
    encoder = PatchEncoder(
        embedding_dim=6,
        device="cpu",
        pretrained=False,
        image_size=8,
    )
    patches = [
        Image.new("RGB", (7, 9), (200, 30, 80)),
        np.full((8, 8, 3), 0.5, dtype=np.float32),
        torch.full((3, 8, 8), 0.25),
    ]

    embeddings = encoder(patches, batch_size=2)

    assert embeddings.shape == (3, 6)
    torch.testing.assert_close(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3))
    assert encoder([], batch_size=1).shape == (0, 6)
    with pytest.raises(ValueError, match="positive"):
        encoder(patches, batch_size=0)


def test_gated_attention_mil_normalizes_attention_for_single_and_batched_bags() -> None:
    model = GatedAttentionMIL(dim=8, attention_dim=4, num_classes=4)
    single_logits, single_probabilities, single_attention = model(torch.randn(5, 8))
    batch_logits, batch_probabilities, batch_attention = model(torch.randn(2, 5, 8))

    assert single_logits.shape == single_probabilities.shape == (4,)
    assert single_attention.shape == (5,) and single_attention.sum().item() == pytest.approx(1.0)
    assert batch_logits.shape == batch_probabilities.shape == (2, 4)
    torch.testing.assert_close(batch_attention.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(batch_probabilities.sum(dim=1), torch.ones(2))
    with pytest.raises(ValueError, match="at least one"):
        model(torch.empty(0, 8))
    with pytest.raises(ValueError, match="expected embedding dimension"):
        model(torch.randn(2, 7))


class StubEncoder(nn.Module):
    embedding_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")

    def forward(self, patches, batch_size: int = 32) -> torch.Tensor:
        count = len(patches)
        values = torch.arange(1, count * self.embedding_dim + 1, dtype=torch.float32)
        return values.reshape(count, self.embedding_dim)


def _deterministic_mil_model() -> GatedAttentionMIL:
    model = GatedAttentionMIL(dim=4, attention_dim=3, num_classes=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 3.0, 1.0, -1.0]))
    return model


def _save_tissue_image(path: Path) -> None:
    image = np.full((32, 32, 3), (210, 20, 100), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_attention_mil_pipeline_generates_heatmap_and_classification(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.png"
    heatmap_path = tmp_path / "outputs" / "attention.png"
    _save_tissue_image(image_path)
    pipeline = AttentionMILPipeline(
        device="cpu",
        model=_deterministic_mil_model(),
        encoder=StubEncoder(),
        patch_size=16,
        stride=16,
        min_tissue_ratio=0.1,
        batch_size=2,
    )

    result = pipeline.process_image(image_path, heatmap_path)

    assert result.predicted_stage is EstrousStage.PROESTRUS
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.raw_logits == pytest.approx([0.0, 3.0, 1.0, -1.0])
    assert heatmap_path.is_file() and Image.open(heatmap_path).size == (32, 32)


def test_attention_heatmap_validates_metadata_and_blends_covered_pixels() -> None:
    pipeline = AttentionMILPipeline(
        device="cpu",
        model=_deterministic_mil_model(),
        encoder=StubEncoder(),
        pretrained_encoder=False,
    )
    source = np.full((10, 10, 3), 100, dtype=np.uint8)
    info = [PatchInfo(0, 0, 5, 5, 1.0), PatchInfo(5, 5, 5, 5, 1.0)]
    heatmap = pipeline.render_attention_heatmap(source, info, [0.2, 0.8])

    assert heatmap.shape == source.shape and heatmap.dtype == np.uint8
    assert not np.array_equal(heatmap[:5, :5], source[:5, :5])
    np.testing.assert_array_equal(heatmap[:5, 5:], source[:5, 5:])
    with pytest.raises(ValueError, match="equal lengths"):
        pipeline.render_attention_heatmap(source, info, [0.2])


def test_attention_mil_folder_isolates_corrupt_file_and_exports_csv(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    corrupt = tmp_path / "corrupt.png"
    _save_tissue_image(good)
    corrupt.write_bytes(b"not an image")
    pipeline = AttentionMILPipeline(
        device="cpu",
        model=_deterministic_mil_model(),
        encoder=StubEncoder(),
        patch_size=16,
        stride=16,
        min_tissue_ratio=0.1,
    )
    csv_path = tmp_path / "exports" / "results.csv"

    batch = pipeline.process_folder(
        tmp_path,
        output_csv=csv_path,
        save_heatmaps_dir=tmp_path / "heatmaps",
    )

    assert batch.total_processed == 2
    assert len(batch.results) == 1 and len(batch.failed_images) == 1
    assert batch.failed_images[0][0] == corrupt
    assert (tmp_path / "heatmaps" / "good_attention.png").is_file()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["predicted_stage"] == "proestrus"
