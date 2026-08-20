"""Convert EstrousBank dataset to WebP 80 and package into WebDataset (.tar) shards for LLM/VLM ingest."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

STAGE_TO_CLASS_INDEX: dict[str, int] = {
    "diestrus": 0,
    "proestrus": 1,
    "estrus": 2,
    "metestrus": 3,
}

STAGE_DESCRIPTIONS: dict[str, str] = {
    "diestrus": "characterized predominantly by small, dense polymorphonuclear leukocytes with low estrogen levels",
    "proestrus": "characterized predominantly by round or oval nucleated epithelial cells in sheets with peak estrogen production",
    "estrus": "characterized almost exclusively by large, flat, anucleated cornified squamous epithelial cells during ovulation",
    "metestrus": "characterized by the co-occurrence of leukocytes and cornified squamous cells in the transitional phase",
}


@dataclass(slots=True)
class SampleMetadata:
    __key__: str
    stage: str
    class_index: int
    split: str
    source_lab: str
    magnification: str
    stain: str
    species: str
    strain: str
    group_id: str
    original_filename: str
    width: int
    height: int
    caption: str
    conversations: list[dict[str, str]]


def process_single_image(
    row: dict[str, str],
    sample_key: str,
    splits_dir: Path,
    webp_quality: int = 80,
) -> tuple[str, bytes, bytes, bytes, bytes] | None:
    """Process a single image: convert to WebP 80 and build WebDataset metadata entries."""
    filename = row["Filename"]
    stage = row["Stage"].strip().lower()
    split = row["Split"].strip().lower()

    image_path = splits_dir / split / stage / filename
    if not image_path.is_file():
        image_path = Path(row["OriginalPath"])
        if not image_path.is_file():
            return None

    try:
        with Image.open(image_path) as raw_img:
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            width, height = img.size
            buffer = io.BytesIO()
            img.save(buffer, format="WEBP", quality=webp_quality, method=0)
            webp_bytes = buffer.getvalue()
    except Exception as exc:
        print(f"Error processing {image_path}: {exc}")
        return None

    class_idx = STAGE_TO_CLASS_INDEX.get(stage, 0)
    stain = row.get("Stain", "Unknown Stain")
    mag = row.get("Magnification", "10X")
    species = row.get("Species", "Rodent")
    strain = row.get("Strain", "WT")
    lab = row.get("SourceLab", "EstrousBank")
    group_id = row.get("GroupID", "")

    stage_desc = STAGE_DESCRIPTIONS.get(stage, f"in the {stage} stage")
    caption = (
        f"A rodent vaginal cytology smear ({species}, {strain}) in the {stage} phase of the estrous cycle, "
        f"{stage_desc}, stained with {stain} at {mag} magnification."
    )

    conversations = [
        {
            "from": "human",
            "value": (
                "<image>\n"
                "Examine this rodent vaginal cytology smear. What stage of the estrous cycle is shown, "
                "and what are the key cytological characteristics?"
            ),
        },
        {
            "from": "gpt",
            "value": (
                f"This cytology smear is in the **{stage.upper()}** phase of the rodent estrous cycle.\n\n"
                f"- **Stage:** {stage.title()}\n"
                f"- **Staining & Protocol:** {stain} ({mag} magnification)\n"
                f"- **Animal:** {species} ({strain})\n"
                f"- **Cytological Hallmarks:** {stage_desc.capitalize()}."
            ),
        },
    ]

    meta = SampleMetadata(
        __key__=sample_key,
        stage=stage,
        class_index=class_idx,
        split=split,
        source_lab=lab,
        magnification=mag,
        stain=stain,
        species=species,
        strain=strain,
        group_id=group_id,
        original_filename=filename,
        width=width,
        height=height,
        caption=caption,
        conversations=conversations,
    )

    json_bytes = json.dumps(asdict(meta), indent=2).encode("utf-8")
    cls_bytes = str(class_idx).encode("utf-8")
    txt_bytes = caption.encode("utf-8")

    return sample_key, webp_bytes, json_bytes, cls_bytes, txt_bytes


def write_tar_shard(
    shard_path: Path,
    samples: list[tuple[str, bytes, bytes, bytes, bytes]],
) -> int:
    """Write a list of samples into a single WebDataset .tar shard."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tar:
        for key, webp_bytes, json_bytes, cls_bytes, txt_bytes in samples:
            # 1. Image
            ti_img = tarfile.TarInfo(name=f"{key}.webp")
            ti_img.size = len(webp_bytes)
            ti_img.mtime = int(time.time())
            tar.addfile(ti_img, io.BytesIO(webp_bytes))

            # 2. JSON Metadata & LLM conversation
            ti_json = tarfile.TarInfo(name=f"{key}.json")
            ti_json.size = len(json_bytes)
            ti_json.mtime = int(time.time())
            tar.addfile(ti_json, io.BytesIO(json_bytes))

            # 3. Class label
            ti_cls = tarfile.TarInfo(name=f"{key}.cls")
            ti_cls.size = len(cls_bytes)
            ti_cls.mtime = int(time.time())
            tar.addfile(ti_cls, io.BytesIO(cls_bytes))

            # 4. Text caption
            ti_txt = tarfile.TarInfo(name=f"{key}.txt")
            ti_txt.size = len(txt_bytes)
            ti_txt.mtime = int(time.time())
            tar.addfile(ti_txt, io.BytesIO(txt_bytes))

    return len(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert EstrousBank images to WebP 80 and WebDataset shards")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/Volumes/SSD/Bioinformatics/EstrousBank_Work/split_manifest.csv"),
        help="Path to split manifest CSV",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("/Volumes/SSD/Bioinformatics/EstrousBank_Work/splits"),
        help="Path to splits directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/SSD/Bioinformatics/shards"),
        help="Path to output WebDataset shards",
    )
    parser.add_argument("--max-shard-size", type=int, default=1000, help="Samples per shard")
    parser.add_argument("--quality", type=int, default=80, help="WebP compression quality (1-100)")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel worker processes")
    args = parser.parse_args()

    print("==================================================")
    print("EstrousBank -> WebP 80 WebDataset Sharder")
    print(f"Manifest:   {args.manifest}")
    print(f"Splits Dir: {args.splits_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Quality:    WebP {args.quality}")
    print(f"Shard Size: {args.max_shard_size} samples/shard")
    print(f"Workers:    {args.num_workers}")
    print("==================================================")

    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} total records from manifest.")

    # Group records by split
    by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        split = r["Split"].strip().lower()
        if split in by_split:
            by_split[split].append(r)
        else:
            by_split.setdefault(split, []).append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    manifest_summary: dict[str, Any] = {
        "dataset_name": "EstrousBank-Cytology",
        "format": "WebDataset (tar)",
        "image_format": f"WebP (quality={args.quality})",
        "total_samples": len(rows),
        "classes": STAGE_TO_CLASS_INDEX,
        "splits": {},
    }

    for split_name, split_rows in by_split.items():
        split_t0 = time.perf_counter()
        print(f"\nProcessing split: {split_name} ({len(split_rows)} samples)...")
        split_out_dir = args.output_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for idx, row in enumerate(split_rows):
                sample_key = f"estrousbank_{split_name}_{idx:06d}"
                tasks.append(
                    executor.submit(
                        process_single_image,
                        row,
                        sample_key,
                        args.splits_dir,
                        args.quality,
                    )
                )

            processed_samples: list[tuple[str, bytes, bytes, bytes, bytes]] = []
            for future in as_completed(tasks):
                res = future.result()
                if res is not None:
                    processed_samples.append(res)

        processed_samples.sort(key=lambda x: x[0])
        print(f"Converted {len(processed_samples)}/{len(split_rows)} images to WebP {args.quality} in {time.perf_counter() - split_t0:.1f}s.")

        # Write shards
        shard_idx = 0
        total_split_shards = []
        for i in range(0, len(processed_samples), args.max_shard_size):
            shard_chunk = processed_samples[i : i + args.max_shard_size]
            shard_filename = f"{split_name}-{shard_idx:06d}.tar"
            shard_path = split_out_dir / shard_filename
            write_tar_shard(shard_path, shard_chunk)
            shard_size_mb = shard_path.stat().st_size / (1024 * 1024)
            print(f"  Wrote {shard_filename}: {len(shard_chunk)} samples ({shard_size_mb:.2f} MB)")
            total_split_shards.append(
                {
                    "filename": shard_filename,
                    "relative_path": f"{split_name}/{shard_filename}",
                    "samples": len(shard_chunk),
                    "size_mb": round(shard_size_mb, 2),
                }
            )
            shard_idx += 1

        manifest_summary["splits"][split_name] = {
            "total_samples": len(processed_samples),
            "total_shards": len(total_split_shards),
            "shards": total_split_shards,
        }

    # Write top-level dataset_info.json and README in shards directory
    info_path = args.output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)

    readme_path = args.output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            f"""# EstrousBank WebDataset Shards (WebP 80)

This directory contains **{len(rows)}** rodent vaginal cytology images converted to **WebP (quality {args.quality})** and packaged into standard **WebDataset (`.tar`) shards** for multimodal LLM / VLM training, visual question answering, and high-throughput PyTorch DataLoader ingestion.

## Directory Structure
```
shards/
├── dataset_info.json
├── train/
│   ├── train-000000.tar
│   ├── train-000001.tar
│   └── ... ({len(manifest_summary['splits']['train']['shards'])} shards, {manifest_summary['splits']['train']['total_samples']} samples)
├── val/
│   ├── val-000000.tar
│   └── ... ({len(manifest_summary['splits']['val']['shards'])} shards, {manifest_summary['splits']['val']['total_samples']} samples)
└── test/
    ├── test-000000.tar
    └── ... ({len(manifest_summary['splits']['test']['shards'])} shards, {manifest_summary['splits']['test']['total_samples']} samples)
```

## Sample Structure inside Each Shard
Each sample shares a unique `__key__` with four aligned companion files:
- **`{sample_key}.webp`**: WebP encoded cytology image (quality 80)
- **`{sample_key}.json`**: Structured metadata dictionary including `stage`, `stain`, `magnification`, `species`, `strain`, `lab`, and LLaVA/ShareGPT-style `conversations`
- **`{sample_key}.cls`**: Integer class index (`0`: diestrus, `1`: proestrus, `2`: estrus, `3`: metestrus)
- **`{sample_key}.txt`**: Natural language descriptive caption

## Loading with PyTorch / WebDataset
```python
import webdataset as wds
from torchvision import transforms

dataset = (
    wds.WebDataset("/Volumes/SSD/Bioinformatics/shards/train/train-{{000000..000010}}.tar")
    .decode("pil")
    .to_tuple("webp", "json", "cls")
)
for image, metadata, label in dataset:
    print(image.size, metadata["stage"], label)
```
"""
        )

    print("\n==================================================")
    print(f"All Shards Successfully Created in {(time.perf_counter() - t_start):.1f}s!")
    print(f"Manifest written to: {info_path}")
    print(f"README written to:   {readme_path}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
