"""Batched feature encoding for cytology image patches."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from cyclonaut.core.models import get_device

PatchLike = Image.Image | np.ndarray | torch.Tensor


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve a PyTorch device using the project's shared fallback policy."""
    return get_device(str(device) if device is not None else None)


class PatchEncoder(nn.Module):
    """ConvNeXt-Tiny patch encoder producing fixed 512-dimensional features."""

    def __init__(
        self,
        embedding_dim: int = 512,
        device: torch.device | str | None = None,
        *,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.embedding_dim = embedding_dim
        self.device = resolve_device(device)
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            base = models.convnext_tiny(weights=weights)
        except Exception as error:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load pretrained ConvNeXt-Tiny weights ({error}); "
                "using randomly initialized backbone weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            base = models.convnext_tiny(weights=None)

        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(768, embedding_dim)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.to(self.device)
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        patches: Sequence[PatchLike] | torch.Tensor,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode patches into an ``(N, embedding_dim)`` feature matrix."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        prepared = self._prepare_input(patches)
        if not prepared:
            return torch.empty((0, self.embedding_dim), device=self.device)

        outputs: list[torch.Tensor] = []
        for start in range(0, len(prepared), batch_size):
            batch = torch.stack(prepared[start : start + batch_size]).to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            features = self.backbone(batch)
            pooled = self.pool(features).flatten(1)
            embeddings = self.projection(pooled)
            outputs.append(nn.functional.normalize(embeddings, p=2, dim=1))
        return torch.cat(outputs, dim=0)

    def _prepare_input(
        self,
        patches: Sequence[PatchLike] | torch.Tensor,
    ) -> list[torch.Tensor]:
        if isinstance(patches, torch.Tensor):
            if patches.ndim == 3:
                patches = patches.unsqueeze(0)
            if patches.ndim != 4:
                raise ValueError(
                    "tensor patches must have shape (N, C, H, W) or (C, H, W)"
                )
            return [self._prepare_patch(patch) for patch in patches]
        return [self._prepare_patch(patch) for patch in patches]

    def _prepare_patch(self, patch: PatchLike) -> torch.Tensor:
        if isinstance(patch, Image.Image):
            image = patch.convert("RGB")
        else:
            array = patch.detach().cpu().numpy() if isinstance(patch, torch.Tensor) else np.asarray(patch)
            if array.ndim != 3:
                raise ValueError(f"patch must be a 3D image, got shape {array.shape}")
            if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            elif array.shape[-1] >= 3:
                array = array[..., :3]
            else:
                raise ValueError(f"unsupported patch channel count: {array.shape[-1]}")
            if np.issubdtype(array.dtype, np.floating):
                array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
                if array.size and float(array.max()) <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0.0, 255.0)
            elif array.dtype != np.uint8:
                array = array.astype(np.float32) * (255.0 / float(np.iinfo(array.dtype).max))
            image = Image.fromarray(np.ascontiguousarray(array.astype(np.uint8)), mode="RGB")
        return self.transform(image)
