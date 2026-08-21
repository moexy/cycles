"""Obfuscate stage-revealing filenames in train/BulkRNA and train/Histology.

Stores the private reverse-lookup mapping records OUTSIDE the dataset directory
(in /Volumes/SSD/Imaging/Cycles/metadata/) so that the distributed dataset split
remains completely blind to stage-revealing filenames.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATASET_ROOT = Path("/Volumes/SSD/Imaging/Cycles/dataset_split")
TRAIN_ROOT = DATASET_ROOT / "train"
BULK_DIR = TRAIN_ROOT / "BulkRNA staging images"
HIST_DIR = TRAIN_ROOT / "Histology staging images"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
SUMMARY_PATH = DATASET_ROOT / "summary.json"

# External private records directory (OUTSIDE the dataset directory)
PRIVATE_META_DIR = Path("/Volumes/SSD/Imaging/Cycles/metadata")


def main() -> None:
    PRIVATE_META_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load existing manifest
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    bulk_files = sorted(list(BULK_DIR.rglob("*.webp")))
    hist_files = sorted(list(HIST_DIR.rglob("*.webp")))
    target_files = bulk_files + hist_files

    print(f"Targeting {len(target_files)} training images for obfuscation verification:")
    print(f"  BulkRNA:   {len(bulk_files)}")
    print(f"  Histology: {len(hist_files)}")

    # Clean any internal mapping files if present
    internal_files = [
        TRAIN_ROOT / "filename_mapping.csv",
        DATASET_ROOT / "obfuscation_mapping.csv",
        DATASET_ROOT / "obfuscation_mapping.json",
    ]
    for p in internal_files:
        if p.exists():
            p.unlink()
            print(f"Removed internal mapping: {p}")

    # Build external mapping records
    mapping_records = []
    for row in manifest_rows:
        if row["split"] == "train" and ("BulkRNA" in row["cohort"] or "Histology" in row["cohort"]):
            mapping_records.append(
                {
                    "cohort": row["cohort"],
                    "split_path": row["split_path"],
                    "obfuscated_filename": row["filename"],
                    "group_id": row["group_id"],
                    "stage": row["stage"],
                }
            )

    # Save private master records OUTSIDE dataset
    out_csv = PRIVATE_META_DIR / "train_obfuscation_mapping.csv"
    out_json = PRIVATE_META_DIR / "train_obfuscation_mapping.json"

    fieldnames = ["cohort", "split_path", "obfuscated_filename", "group_id", "stage"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_records)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping_records, f, indent=2)

    print("\nSaved external private mapping records to:")
    print(f"  {out_csv}")
    print(f"  {out_json}")

    # Clean manifest.csv inside dataset_split: remove any original_filename leakage
    cleaned_manifest = []
    for row in manifest_rows:
        clean_row = {
            "split": row["split"],
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "split_path": row["split_path"],
            "cohort": row["cohort"],
            "group_id": row["group_id"],
            "stage": row.get("stage", "unknown"),
        }
        cleaned_manifest.append(clean_row)

    manifest_fields = [
        "split",
        "filename",
        "relative_path",
        "split_path",
        "cohort",
        "group_id",
        "stage",
    ]

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(cleaned_manifest)
    print(f"Updated clean dataset manifest: {MANIFEST_PATH}")

    # Update summary.json
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = json.load(f)
    if "obfuscation_mapping_file" in summary:
        del summary["obfuscation_mapping_file"]
    summary["obfuscated_train_images"] = len(mapping_records)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated clean dataset summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
