"""Apply white balance normalization to all WebP images and save to a new output folder."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path("/Volumes/SSD/Imaging/Cycles/test")
DST_DIR = Path("/Volumes/SSD/Imaging/Cycles/test_whitebalanced")


def white_balance_image(arr: np.ndarray, target_wp: float = 255.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute slide background white point and scale channels to target_wp."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Estimate background from top 5% brightest pixels (excluding top 0.2% outliers/hot pixels)
    p95 = np.percentile(lum, 95)
    p998 = np.percentile(lum, 99.8)
    mask = (lum >= p95) & (lum <= p998)
    if not np.any(mask):
        mask = lum >= p95

    bg_color = np.median(arr[mask], axis=0)
    bg_color = np.clip(bg_color, 1.0, 255.0)

    scale = target_wp / bg_color
    balanced = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return balanced, bg_color


def process_image(src_file: Path) -> tuple[Path, Path, bool, str, float]:
    rel = src_file.relative_to(SRC_DIR)
    dst_file = DST_DIR / rel
    t0 = time.time()
    try:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_file) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            balanced_arr, _ = white_balance_image(arr, target_wp=255.0)
            balanced_img = Image.fromarray(balanced_arr, mode="RGB")
            balanced_img.save(dst_file, format="WEBP", quality=80)
        elapsed = time.time() - t0
        return (src_file, dst_file, True, "", elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        return (src_file, dst_file, False, str(e), elapsed)


def main() -> None:
    src_files = sorted(list(SRC_DIR.rglob("*.webp")))
    print(f"Found {len(src_files)} WebP images in {SRC_DIR}")
    print(f"Output directory: {DST_DIR}")

    workers = min(os.cpu_count() or 4, 16)
    print(f"Running white balance with {workers} parallel processes...")

    start_time = time.time()
    succeeded = 0
    failed: list[tuple[Path, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(process_image, p): p for p in src_files}
        for idx, future in enumerate(as_completed(future_map), 1):
            src_file, dst_file, ok, err, elapsed = future.result()
            if ok:
                succeeded += 1
            else:
                failed.append((src_file, err))
            if idx % 50 == 0 or idx == len(src_files):
                print(
                    f"Progress: {idx}/{len(src_files)} processed ({succeeded} ok, {len(failed)} failed) "
                    f"[{time.time() - start_time:.1f}s]"
                )

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f}s.")
    print(f"Successfully processed: {succeeded}/{len(src_files)}")
    if failed:
        print(f"Failures ({len(failed)}):")
        for src_file, err in failed:
            print(f"  {src_file}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
