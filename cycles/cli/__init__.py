"""Command-line interface for :mod:`cycles`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["build_parser", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("cycles.cli.main"), name)
    raise AttributeError(name)
