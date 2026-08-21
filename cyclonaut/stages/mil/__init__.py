"""Attention-based multiple-instance learning for whole-slide staging."""

from cyclonaut.stages.mil.encoder import PatchEncoder
from cyclonaut.stages.mil.model import GatedAttentionMIL
from cyclonaut.stages.mil.patching import PatchExtractor
from cyclonaut.stages.mil.pipeline import AttentionMILPipeline

__all__ = [
    "AttentionMILPipeline",
    "GatedAttentionMIL",
    "PatchEncoder",
    "PatchExtractor",
]
