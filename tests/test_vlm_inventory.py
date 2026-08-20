from __future__ import annotations

from pathlib import Path

from PIL import Image

from cycles.vlm_local.inventory import build_blind_inventory


def _image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path)


def test_blind_inventory_freezes_content_without_exposing_paths(tmp_path: Path) -> None:
    _image(tmp_path / "mouse-7" / "day-1.png", "red")
    _image(tmp_path / "revealing-stage-name.webp", "blue")

    inventory = build_blind_inventory(tmp_path)
    encoded = str(inventory)

    assert inventory["image_count"] == 2
    assert len(inventory["content_inventory_sha256"]) == 64
    assert len(inventory["images"]) == 2
    assert all(set(row) == {"bytes", "image_sha256"} for row in inventory["images"])
    assert "mouse-7" not in encoded
    assert "revealing-stage-name" not in encoded
    assert str(tmp_path) not in encoded


def test_blind_inventory_is_stable_when_files_are_renamed(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    _image(first, "red")
    before = build_blind_inventory(tmp_path)
    first.rename(tmp_path / "renamed.png")

    after = build_blind_inventory(tmp_path)

    assert after == before
