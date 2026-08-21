"""Convert TIFF images from samples to WebP 80 preserving directory structure."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

SRC_DIR = Path("/Volumes/SSD/Imaging/Cycles/samples")
DST_DIR = Path("/Volumes/SSD/Imaging/Cycles/test")


def get_dest_path(src_file: Path, src_root: Path = SRC_DIR, dst_root: Path = DST_DIR) -> Path:
    rel = src_file.relative_to(src_root)
    name = src_file.name
    name_lower = name.lower()
    if name_lower.endswith(".ome.tif"):
        clean_name = name[:-8] + ".webp"
    elif name_lower.endswith(".ome.tiff"):
        clean_name = name[:-9] + ".webp"
    elif name_lower.endswith(".tiff"):
        clean_name = name[:-5] + ".webp"
    elif name_lower.endswith(".tif"):
        clean_name = name[:-4] + ".webp"
    else:
        clean_name = src_file.stem + ".webp"
    return dst_root / rel.parent / clean_name


def convert_image(src_file: Path) -> tuple[Path, Path, bool, str, float]:
    dst_file = get_dest_path(src_file)
    t0 = time.time()
    try:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_file) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            rgb.save(dst_file, format="WEBP", quality=80)
        elapsed = time.time() - t0
        return (src_file, dst_file, True, "", elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        return (src_file, dst_file, False, str(e), elapsed)


def main() -> None:
    tif_files = sorted(
        [
            p
            for p in SRC_DIR.rglob("*")
            if p.is_file() and (p.name.lower().endswith(".tif") or p.name.lower().endswith(".tiff"))
        ]
    )
    print(f"Discovered {len(tif_files)} TIFF images in {SRC_DIR}")
    print(f"Destination root: {DST_DIR}")

    workers = min(os.cpu_count() or 4, 16)
    print(f"Converting using {workers} parallel processes (WebP quality=80)...")

    start_time = time.time()
    succeeded = 0
    failed: list[tuple[Path, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(convert_image, p): p for p in tif_files}
        for idx, future in enumerate(as_completed(future_map), 1):
            src_file, dst_file, ok, err, elapsed = future.result()
            if ok:
                succeeded += 1
            else:
                failed.append((src_file, err))
            if idx % 50 == 0 or idx == len(tif_files):
                print(
                    f"Progress: {idx}/{len(tif_files)} processed ({succeeded} ok, {len(failed)} failed) "
                    f"[{time.time() - start_time:.1f}s]"
                )

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f}s.")
    print(f"Successfully converted: {succeeded}/{len(tif_files)}")
    if failed:
        print(f"Errors ({len(failed)}):")
        for src_file, err in failed:
            print(f"  {src_file}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
