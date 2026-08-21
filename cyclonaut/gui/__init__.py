"""PySide6 desktop interface for cycles."""

from __future__ import annotations

__all__ = ["MainWindow"]


def __getattr__(name: str) -> object:
    if name == "MainWindow":
        from cyclonaut.gui.main_window import MainWindow

        return MainWindow
    raise AttributeError(name)
