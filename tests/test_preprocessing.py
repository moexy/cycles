from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cycles.core.preprocessing import (
    discover_images,
    get_inference_transforms,
    get_train_transforms,
    load_image,
    normalize_luminance,
    save_preprocessed_image,
)


def _gradient_rgb(height: int = 32, width: int = 40) -> np.ndarray:
    values = np.linspace(20, 230, height * width, dtype=np.uint8).reshape(height, width)
    return np.stack((values, np.roll(values, 3, axis=1), np.flipud(values)), axis=-1)


def test_discover_images_filters_extensions_sorts_and_honors_recursion(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    for path in (tmp_path / "B.JPG", tmp_path / "a.png", nested / "c.TIFF"):
        Image.new("RGB", (4, 4), "white").save(path)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    assert [path.name for path in discover_images(tmp_path, recursive=False)] == ["a.png", "B.JPG"]
    assert [path.name for path in discover_images(tmp_path)] == ["a.png", "B.JPG", "c.TIFF"]
    (tmp_path / "empty").mkdir()
    assert discover_images(tmp_path / "empty") == []


def test_discover_images_rejects_missing_or_file_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "slide.png"
    Image.new("RGB", (2, 2)).save(file_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_images(tmp_path / "missing")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        discover_images(file_path)


def test_load_image_detaches_data_and_converts_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (7, 5), 127).save(path)

    image = load_image(path)
    path.unlink()

    assert image.mode == "RGB"
    assert image.size == (7, 5)
    assert image.getpixel((0, 0)) == (127, 127, 127), "loaded pixels must outlive the file handle"


def test_load_image_reports_corrupt_and_invalid_paths(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a PNG")

    with pytest.raises(ValueError, match="Could not decode image"):
        load_image(corrupt)
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")
    with pytest.raises(IsADirectoryError):
        load_image(tmp_path)


@pytest.mark.parametrize("method", ["clahe", "percentile", "minmax"])
def test_normalize_luminance_methods_return_rgb_uint8_with_full_shape(method: str) -> None:
    source = _gradient_rgb()
    normalized = normalize_luminance(source, method)

    assert normalized.shape == source.shape
    assert normalized.dtype == np.uint8
    assert np.ptp(normalized) > 0, f"{method} should retain nonconstant image information"


def test_normalize_luminance_handles_constant_float_and_aliases() -> None:
    constant = np.full((5, 6), 0.5, dtype=np.float32)
    normalized = normalize_luminance(constant, "min-max normalization")

    assert normalized.shape == (5, 6, 3)
    assert np.all(normalized == 128), "constant images should not be spuriously equalized"
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_luminance(constant, "unknown")
    with pytest.raises(ValueError, match="at least one pixel"):
        normalize_luminance(np.empty((0, 0), dtype=np.uint8))


def test_inference_transforms_are_deterministic_and_optionally_normalized() -> None:
    image = Image.fromarray(_gradient_rgb(17, 19), mode="RGB")
    normalized = get_inference_transforms(img_size=24)(image)
    unnormalized = get_inference_transforms(img_size=24, normalize=False)(np.asarray(image))

    assert normalized.shape == (3, 24, 24)
    assert torch.equal(normalized, get_inference_transforms(img_size=24)(image))
    assert float(unnormalized.min()) >= 0.0 and float(unnormalized.max()) <= 1.0
    with pytest.raises(ValueError, match="positive integer"):
        get_inference_transforms(img_size=0)


def test_train_transforms_support_augmented_and_deterministic_paths() -> None:
    image = Image.fromarray(_gradient_rgb(20, 28), mode="RGB")
    deterministic = get_train_transforms(img_size=16, augment=False)
    augmented = get_train_transforms(img_size=16, augment=True)

    fixed = deterministic(image)
    random_view = augmented(image)
    assert fixed.shape == random_view.shape == (3, 16, 16)
    assert torch.isfinite(fixed).all() and torch.isfinite(random_view).all()
    with pytest.raises(ValueError, match="positive integer"):
        get_train_transforms(img_size=True)


def test_save_preprocessed_image_creates_parents_and_round_trips_rgb(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "prepared.png"
    returned = save_preprocessed_image(_gradient_rgb(8, 9), destination)

    assert returned == destination and destination.is_file()
    reloaded = load_image(destination)
    assert reloaded.mode == "RGB" and reloaded.size == (9, 8)
    with pytest.raises(ValueError, match="file extension"):
        save_preprocessed_image(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "extensionless")
