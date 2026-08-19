"""Image discovery, loading, normalization, and transform utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import color, exposure
from torchvision import transforms

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def discover_images(folder: Path | str, recursive: bool = True) -> list[Path]:
    """Return supported image files below *folder* in deterministic order.

    Args:
        folder: Directory to search.
        recursive: Search all descendants when true, or only the directory itself.

    Raises:
        FileNotFoundError: If *folder* does not exist.
        NotADirectoryError: If *folder* is not a directory.
    """
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    images = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(
        images,
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        ),
    )


def load_image(path: Path | str) -> Image.Image:
    """Load an image as a detached RGB :class:`PIL.Image.Image`.

    The returned image owns its pixel data and therefore remains usable after the
    underlying file handle is closed.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If Pillow cannot decode the file as an image.
    """
    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise IsADirectoryError(f"Image path is not a file: {image_path}")

    try:
        with Image.open(image_path) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not decode image '{image_path}': {error}") from error


def _as_rgb_uint8(img: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a Pillow image or numeric array to an RGB uint8 array."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    array = np.asarray(img)
    if array.ndim == 2:
        array = np.repeat(array[..., np.newaxis], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(
            "Image array must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)"
        )

    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"Image array must contain numeric values, got {array.dtype}")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    converted = np.asarray(array, dtype=np.float32)
    converted = np.nan_to_num(converted, nan=0.0, posinf=255.0, neginf=0.0)
    if converted.size and float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0:
        converted = converted * 255.0
    return np.clip(np.rint(converted), 0.0, 255.0).astype(np.uint8)


def normalize_luminance(
    img: Image.Image | np.ndarray,
    method: str = "clahe",
) -> np.ndarray:
    """Normalize image luminance while retaining RGB chromatic information.

    Supported methods are ``"clahe"`` (contrast-limited adaptive histogram
    equalization), ``"percentile"`` (2nd-to-98th percentile contrast stretch),
    and ``"minmax"``. Common hyphenated and descriptive aliases are accepted.
    The result is always an ``(H, W, 3)`` uint8 RGB array.

    Args:
        img: Pillow image or NumPy image array.
        method: Luminance normalization method.

    Raises:
        ValueError: If the image shape is invalid or the method is unsupported.
        TypeError: If an array contains non-numeric data.
    """
    aliases = {
        "clahe": "clahe",
        "percentile": "percentile",
        "percentile_contrast": "percentile",
        "percentile_contrast_stretching": "percentile",
        "contrast_stretch": "percentile",
        "contrast_stretching": "percentile",
        "minmax": "minmax",
        "min_max": "minmax",
        "min_max_normalization": "minmax",
    }
    normalized_name = method.strip().lower().replace("-", "_").replace(" ", "_")
    selected = aliases.get(normalized_name)
    if selected is None:
        supported = "clahe, percentile, minmax"
        raise ValueError(f"Unsupported luminance normalization method '{method}'. Supported: {supported}")

    rgb = _as_rgb_uint8(img)
    if rgb.size == 0:
        raise ValueError("Image must contain at least one pixel")

    rgb_float = rgb.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb_float)
    luminance = lab[..., 0] / 100.0
    luminance_range = float(np.ptp(luminance))

    # Equalizing a constant plane is undefined and can turn a valid blank image
    # fully black or white depending on the library version.
    if luminance_range <= np.finfo(np.float32).eps:
        return rgb.copy()

    if selected == "clahe":
        adjusted = exposure.equalize_adapthist(luminance, clip_limit=0.01)
    elif selected == "percentile":
        low, high = np.percentile(luminance, (2.0, 98.0))
        if float(high - low) <= np.finfo(np.float32).eps:
            return rgb.copy()
        adjusted = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    else:
        low = float(luminance.min())
        adjusted = (luminance - low) / luminance_range

    lab[..., 0] = np.asarray(adjusted, dtype=np.float32) * 100.0
    with np.errstate(invalid="ignore"):
        normalized_rgb = color.lab2rgb(lab)
    return np.clip(np.rint(normalized_rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def save_preprocessed_image(
    img: Image.Image | np.ndarray,
    dest_path: Path | str,
) -> Path:
    """Save an image as RGB, creating destination directories as needed.

    Returns:
        The destination path supplied by the caller as a :class:`Path`.

    Raises:
        ValueError: If the destination has no file extension or the image is invalid.
        OSError: If Pillow cannot encode or write the destination.
    """
    destination = Path(dest_path).expanduser()
    if not destination.suffix:
        raise ValueError(f"Destination must include an image file extension: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        Image.fromarray(_as_rgb_uint8(img), mode="RGB").save(destination)
    except OSError as error:
        raise OSError(f"Could not save preprocessed image '{destination}': {error}") from error
    return destination


def _to_rgb_pil(img: Image.Image | np.ndarray) -> Image.Image:
    """Make transform pipelines accept both Pillow images and NumPy arrays."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(_as_rgb_uint8(img), mode="RGB")


def _validate_img_size(img_size: int) -> None:
    if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size <= 0:
        raise ValueError(f"img_size must be a positive integer, got {img_size!r}")


def get_inference_transforms(
    img_size: int = 224,
    normalize: bool = True,
) -> transforms.Compose:
    """Build the deterministic preprocessing pipeline used for inference."""
    _validate_img_size(img_size)
    operations: list[object] = [
        transforms.Lambda(_to_rgb_pil),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
    ]
    if normalize:
        operations.append(transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD))
    return transforms.Compose(operations)


def get_train_transforms(
    img_size: int = 224,
    augment: bool = True,
    *,
    aggressive_stain: bool = True,
) -> transforms.Compose:
    """Build a training transform pipeline for orientation-free cytology images."""
    _validate_img_size(img_size)
    operations: list[object] = [transforms.Lambda(_to_rgb_pil)]
    if augment:
        if aggressive_stain:
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        (img_size, img_size),
                        scale=(0.7, 1.0),
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
                ]
            )
        else:
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        (img_size, img_size),
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                ]
            )
    else:
        operations.append(transforms.Resize((img_size, img_size), antialias=True))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)
