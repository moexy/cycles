"""Attention-based multiple-instance learning for whole-slide staging."""

from cycles.stages.mil.encoder import PatchEncoder
from cycles.stages.mil.model import GatedAttentionMIL
from cycles.stages.mil.patching import PatchExtractor
from cycles.stages.mil.pipeline import AttentionMILPipeline

__all__ = [
    "AttentionMILPipeline",
    "GatedAttentionMIL",
    "PatchEncoder",
    "PatchExtractor",
]
