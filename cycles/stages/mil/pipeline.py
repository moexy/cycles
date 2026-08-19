"""End-to-end attention-MIL inference and spatial explanation export."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps

from cycles.core.types import BatchClassificationResult, ClassificationResult, EstrousStage
from cycles.stages.mil.encoder import PatchEncoder, resolve_device
from cycles.stages.mil.model import GatedAttentionMIL
from cycles.stages.mil.patching import PatchExtractor, PatchInfo

_SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


class AttentionMILPipeline:
    """Run patch extraction, feature encoding, and gated-attention staging."""

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: torch.device | str | None = None,
        *,
        model: GatedAttentionMIL | None = None,
        encoder: PatchEncoder | None = None,
        patch_size: int = 256,
        stride: int = 256,
        min_tissue_ratio: float = 0.15,
        max_patches: int | None = 256,
        batch_size: int = 32,
        heatmap_alpha: float = 0.5,
        colormap: str = "inferno",
        pretrained_encoder: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= heatmap_alpha <= 1.0:
            raise ValueError("heatmap_alpha must be between 0 and 1")
        if colormap not in matplotlib.colormaps:
            raise ValueError(f"unknown matplotlib colormap: {colormap!r}")

        self.device = resolve_device(device)
        self.extractor = PatchExtractor(
            patch_size=patch_size,
            stride=stride,
            min_tissue_ratio=min_tissue_ratio,
            max_patches=max_patches,
        )
        self.batch_size = batch_size
        self.heatmap_alpha = heatmap_alpha
        self.colormap = colormap
        self.class_order = EstrousStage.canonical_stages()

        checkpoint = self._read_checkpoint(weights_path) if weights_path is not None else None
        model_config = self._model_config(checkpoint)
        self.encoder = encoder or PatchEncoder(
            embedding_dim=int(model_config.get("dim", 512)),
            device=self.device,
            pretrained=pretrained_encoder,
        )
        self.encoder.device = self.device
        self.encoder.to(self.device).eval()
        encoder_dim = int(getattr(self.encoder, "embedding_dim", model_config.get("dim", 512)))
        self.model = model or GatedAttentionMIL(
            dim=encoder_dim,
            attention_dim=int(model_config.get("attention_dim", 128)),
            num_classes=int(model_config.get("num_classes", 4)),
        )
        if self.model.dim != encoder_dim:
            raise ValueError(
                f"encoder outputs {encoder_dim} features but MIL model expects {self.model.dim}"
            )
        if self.model.num_classes != len(self.class_order):
            raise ValueError(
                f"MIL model has {self.model.num_classes} classes; expected {len(self.class_order)}"
            )

        self.model.to(self.device).eval()
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    @torch.inference_mode()
    def process_image(
        self,
        image_path: Path | str,
        save_heatmap_path: Path | str | None = None,
    ) -> ClassificationResult:
        """Classify one slide and optionally save its blended attention map."""
        path = Path(image_path)
        with Image.open(path) as opened:
            rgb_image = ImageOps.exif_transpose(opened).convert("RGB")
            raw_rgb = np.asarray(rgb_image, dtype=np.uint8).copy()

        patches, patch_info = self.extractor.extract(raw_rgb)
        if not patches:
            raise ValueError(
                "no patches met the tissue threshold; the image appears blank or contains too little tissue"
            )

        embeddings = self.encoder(patches, batch_size=self.batch_size)
        if embeddings.shape != (len(patches), self.model.dim):
            raise RuntimeError(
                "patch encoder returned shape "
                f"{tuple(embeddings.shape)}, expected ({len(patches)}, {self.model.dim})"
            )
        logits, probability_tensor, attention_tensor = self.model(embeddings)
        logits_cpu = logits.detach().to("cpu", dtype=torch.float32)
        probabilities_cpu = probability_tensor.detach().to("cpu", dtype=torch.float32)
        attention = attention_tensor.detach().to("cpu", dtype=torch.float32).numpy()

        probabilities = {
            stage: float(probabilities_cpu[index])
            for index, stage in enumerate(self.class_order)
        }
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        predicted_stage, confidence = ranked[0]
        second_probability = ranked[1][1]
        denominator = confidence + second_probability
        confidence_index = (
            (confidence - second_probability) / denominator if denominator > 0.0 else 0.0
        )

        if save_heatmap_path is not None:
            heatmap = self.render_attention_heatmap(raw_rgb, patch_info, attention)
            output_path = Path(save_heatmap_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(heatmap, mode="RGB").save(output_path)

        return ClassificationResult(
            image_path=path,
            predicted_stage=predicted_stage,
            confidence=confidence,
            probabilities=probabilities,
            confidence_index=float(confidence_index),
            is_transition=False,
            transition_to=None,
            raw_logits=logits_cpu.tolist(),
        )

    def process_folder(
        self,
        folder: Path | str,
        output_csv: Path | str | None = None,
        save_heatmaps_dir: Path | str | None = None,
        recursive: bool = False,
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> BatchClassificationResult:
        """Classify supported images while isolating failures per image."""
        start_time = time.perf_counter()
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise NotADirectoryError(folder_path)

        iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()
        image_paths = sorted(
            (
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path.relative_to(folder_path)).casefold(),
        )
        results: list[ClassificationResult] = []
        failures: list[tuple[Path, str]] = []
        heatmap_dir = Path(save_heatmaps_dir) if save_heatmaps_dir is not None else None
        if heatmap_dir is not None:
            heatmap_dir.mkdir(parents=True, exist_ok=True)

        total = len(image_paths)
        for index, path in enumerate(image_paths, start=1):
            relative_path = path.relative_to(folder_path)
            heatmap_path = (
                heatmap_dir / relative_path.parent / f"{path.stem}_attention.png"
                if heatmap_dir is not None
                else None
            )
            try:
                results.append(self.process_image(path, heatmap_path))
            except Exception as error:
                failures.append((path, f"{type(error).__name__}: {error}"))
            if progress_callback is not None:
                progress_callback(index, total, path)

        batch_result = BatchClassificationResult(
            results=results,
            failed_images=failures,
            total_processed=len(image_paths),
            duration_seconds=time.perf_counter() - start_time,
        )
        if output_csv is not None:
            self.export_results_csv(batch_result, output_csv)
        return batch_result

    def render_attention_heatmap(
        self,
        raw_rgb: np.ndarray,
        patch_info: Sequence[PatchInfo],
        attention: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Alpha blend max-normalized patch attention over the source image."""
        image = np.asarray(raw_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("raw_rgb must have shape (height, width, 3)")
        attention_array = np.asarray(attention, dtype=np.float32).reshape(-1)
        if len(patch_info) != len(attention_array):
            raise ValueError("patch metadata and attention must have equal lengths")
        if not patch_info:
            return image.copy()

        height, width = image.shape[:2]
        accumulated = np.zeros((height, width), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.float32)
        for info, score in zip(patch_info, attention_array, strict=True):
            x_stop = min(width, info.x + info.width)
            y_stop = min(height, info.y + info.height)
            if info.x < 0 or info.y < 0 or x_stop <= info.x or y_stop <= info.y:
                continue
            accumulated[info.y:y_stop, info.x:x_stop] += max(0.0, float(score))
            coverage[info.y:y_stop, info.x:x_stop] += 1.0

        covered = coverage > 0.0
        attention_map = np.divide(
            accumulated,
            coverage,
            out=np.zeros_like(accumulated),
            where=covered,
        )
        maximum = float(attention_map.max())
        if maximum > 0.0:
            attention_map /= maximum

        colour_map = matplotlib.colormaps[self.colormap]
        coloured = (colour_map(attention_map)[..., :3] * 255.0).astype(np.uint8)
        blended = image.copy()
        alpha = self.heatmap_alpha
        blended[covered] = (
            (1.0 - alpha) * image[covered].astype(np.float32)
            + alpha * coloured[covered].astype(np.float32)
        ).astype(np.uint8)
        return blended

    @staticmethod
    def export_results_csv(
        batch_result: BatchClassificationResult,
        output_csv: Path | str,
    ) -> Path:
        """Export successful slide predictions and per-class probabilities."""
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stages = EstrousStage.canonical_stages()
        headers = [
            "image_path",
            "predicted_stage",
            "confidence",
            "confidence_index",
            *[f"probability_{stage.value}" for stage in stages],
        ]
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for result in batch_result.results:
                writer.writerow(
                    [
                        str(result.image_path),
                        result.predicted_stage.value,
                        f"{result.confidence:.8f}",
                        f"{result.confidence_index:.8f}",
                        *[f"{result.probabilities.get(stage, 0.0):.8f}" for stage in stages],
                    ]
                )
        return output_path

    @staticmethod
    def _read_checkpoint(weights_path: Path | str) -> Mapping[str, Any]:
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("MIL checkpoint must be a self-describing mapping")
        if "metadata" not in checkpoint:
            raise ValueError("MIL checkpoint is missing required metadata")
        if "model_state_dict" not in checkpoint and "state_dict" not in checkpoint:
            raise ValueError("MIL checkpoint is missing model_state_dict")
        return checkpoint

    @staticmethod
    def _model_config(checkpoint: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if checkpoint is None:
            return {}
        config = checkpoint.get("model_config", {})
        if not isinstance(config, Mapping):
            raise ValueError("checkpoint model_config must be a mapping")
        return config

    def _load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint metadata must be a mapping")
        architecture = metadata.get("architecture")
        if architecture not in (None, "gated_attention_mil", "GatedAttentionMIL"):
            raise ValueError(f"unsupported MIL checkpoint architecture: {architecture!r}")
        class_names = metadata.get("classes")
        if class_names is not None:
            expected = [stage.value for stage in self.class_order]
            if list(class_names) != expected:
                raise ValueError(
                    f"checkpoint classes {list(class_names)!r} do not match expected {expected!r}"
                )

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint model state must be a mapping")
        self.model.load_state_dict(state_dict, strict=True)
        encoder_state = checkpoint.get("encoder_state_dict")
        if encoder_state is not None:
            if not isinstance(encoder_state, Mapping):
                raise ValueError("checkpoint encoder state must be a mapping")
            self.encoder.load_state_dict(encoder_state, strict=True)
        self.model.to(self.device).eval()
        self.encoder.to(self.device).eval()
