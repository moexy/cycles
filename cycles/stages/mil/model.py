"""Gated-attention multiple-instance learning for slide-level staging."""

from __future__ import annotations

import torch
from torch import nn


class GatedAttentionMIL(nn.Module):
    """Aggregate a variable-size bag of patch embeddings with gated attention."""

    def __init__(
        self,
        dim: int = 512,
        attention_dim: int = 128,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        self.dim = dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.attention_v = nn.Linear(dim, attention_dim)
        self.attention_u = nn.Linear(dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return class logits, class probabilities, and patch attention.

        ``embeddings`` may describe one bag as ``(N, D)`` or a same-sized
        batch of bags as ``(B, N, D)``. Attention is normalized independently
        across the patch dimension and therefore sums to one per bag.
        """
        if embeddings.ndim not in (2, 3):
            raise ValueError(
                f"embeddings must have shape (N, D) or (B, N, D), got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[-1] != self.dim:
            raise ValueError(
                f"expected embedding dimension {self.dim}, got {embeddings.shape[-1]}"
            )
        patch_dimension = 0 if embeddings.ndim == 2 else 1
        if embeddings.shape[patch_dimension] == 0:
            raise ValueError("MIL requires at least one patch embedding")

        gated = torch.tanh(self.attention_v(embeddings)) * torch.sigmoid(
            self.attention_u(embeddings)
        )
        attention_logits = self.attention_w(gated).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=patch_dimension)
        slide_embedding = torch.sum(
            attention.unsqueeze(-1) * embeddings,
            dim=patch_dimension,
        )
        logits = self.classifier(slide_embedding)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities, attention
