"""Application launcher for the cycles PySide6 desktop interface."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _display_is_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("QT_QPA_PLATFORM") in {"offscreen", "minimal", "vnc"}
        )
    return True


def main(
    checkpoint: Path | str | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """Launch the Qt application, returning a nonzero status if no display exists."""
    if not _display_is_available():
        print(
            "cycles: GUI unavailable because no display server is configured; "
            "use a CLI subcommand or set QT_QPA_PLATFORM=offscreen for automated smoke checks.",
            file=sys.stderr,
        )
        return 2

    try:
        from PySide6.QtWidgets import QApplication

        from cyclonaut.gui.main_window import MainWindow
    except (ImportError, RuntimeError) as exc:
        print(f"cycles: unable to load the PySide6 GUI: {exc}", file=sys.stderr)
        return 2

    application = QApplication.instance()
    owns_application = application is None
    try:
        if application is None:
            application = QApplication(list(argv) if argv is not None else sys.argv)
        window = MainWindow(checkpoint=checkpoint)
        window.show()
        # Keep the Python wrapper alive for applications embedded in another Qt host.
        application._cycles_main_window = window
        return int(application.exec()) if owns_application else 0
    except (RuntimeError, OSError) as exc:
        print(f"cycles: unable to start the GUI display: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
