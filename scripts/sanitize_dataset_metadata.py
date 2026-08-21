"""Sanitize dataset_split metadata to remove all references to stage.

Stores all ground-truth stage annotations and original filenames OUTSIDE the dataset
in /Volumes/SSD/Imaging/Cycles/metadata/.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

DATASET_ROOT = Path("/Volumes/SSD/Imaging/Cycles/dataset_split")
ORIGINAL_WB_ROOT = Path("/Volumes/SSD/Imaging/Cycles/test_whitebalanced")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
SUMMARY_PATH = DATASET_ROOT / "summary.json"

EXTERNAL_META_DIR = Path("/Volumes/SSD/Imaging/Cycles/metadata")


def main() -> None:
    EXTERNAL_META_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build lookup from original whitebalanced files
    orig_hash_to_path: dict[str, Path] = {}
    for p in ORIGINAL_WB_ROOT.rglob("*.webp"):
        rel = p.relative_to(ORIGINAL_WB_ROOT)
        h = hashlib.sha256(str(rel).encode()).hexdigest()[:8]
        orig_hash_to_path[f"sample_{h}.webp"] = rel

    # 2. Read existing manifest
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    full_ground_truth_rows = []
    blinded_manifest_rows = []
    train_obfuscation_rows = []

    for row in manifest_rows:
        fn = row["filename"]
        split = row["split"]
        stage = row.get("stage", "unknown")
        cohort = row["cohort"]
        gid = row["group_id"]
        split_path = row["split_path"]
        rel_path = row["relative_path"]

        is_obf = fn.startswith("sample_") and fn in orig_hash_to_path
        if is_obf:
            orig_rel = orig_hash_to_path[fn]
            orig_fn = orig_rel.name
        else:
            orig_rel = Path(rel_path)
            orig_fn = fn

        # External ground truth record
        gt_record = {
            "split": split,
            "filename": fn,
            "original_filename": orig_fn,
            "is_obfuscated": str(is_obf),
            "relative_path": rel_path,
            "split_path": split_path,
            "cohort": cohort,
            "group_id": gid,
            "stage": stage,
        }
        full_ground_truth_rows.append(gt_record)

        if is_obf:
            train_obfuscation_rows.append(
                {
                    "cohort": cohort,
                    "directory": str(Path(rel_path).parent),
                    "original_filename": orig_fn,
                    "obfuscated_filename": fn,
                    "original_relative_path": str(orig_rel),
                    "obfuscated_relative_path": rel_path,
                    "original_split_path": f"{split}/{orig_rel}",
                    "obfuscated_split_path": split_path,
                    "group_id": gid,
                    "stage": stage,
                }
            )

        # Blinded internal manifest (ZERO stage leakage)
        blinded_record = {
            "split": split,
            "filename": fn,
            "relative_path": rel_path,
            "split_path": split_path,
            "cohort": cohort,
            "group_id": gid,
        }
        blinded_manifest_rows.append(blinded_record)

    # 3. Write external ground truth records
    gt_csv = EXTERNAL_META_DIR / "ground_truth_metadata.csv"
    gt_json = EXTERNAL_META_DIR / "ground_truth_metadata.json"

    gt_fields = [
        "split",
        "filename",
        "original_filename",
        "is_obfuscated",
        "relative_path",
        "split_path",
        "cohort",
        "group_id",
        "stage",
    ]

    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gt_fields)
        writer.writeheader()
        writer.writerows(full_ground_truth_rows)

    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(full_ground_truth_rows, f, indent=2)

    # Write external train obfuscation mapping
    obf_csv = EXTERNAL_META_DIR / "train_obfuscation_mapping.csv"
    obf_json = EXTERNAL_META_DIR / "train_obfuscation_mapping.json"

    obf_fields = [
        "cohort",
        "directory",
        "original_filename",
        "obfuscated_filename",
        "original_relative_path",
        "obfuscated_relative_path",
        "original_split_path",
        "obfuscated_split_path",
        "group_id",
        "stage",
    ]

    with open(obf_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=obf_fields)
        writer.writeheader()
        writer.writerows(train_obfuscation_rows)

    with open(obf_json, "w", encoding="utf-8") as f:
        json.dump(train_obfuscation_rows, f, indent=2)

    print("Saved external ground truth metadata:")
    print(f"  {gt_csv} ({len(full_ground_truth_rows)} rows)")
    print(f"  {obf_csv} ({len(train_obfuscation_rows)} rows)")

    # 4. Write blinded manifest to dataset_split/manifest.csv
    blinded_fields = [
        "split",
        "filename",
        "relative_path",
        "split_path",
        "cohort",
        "group_id",
    ]

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=blinded_fields)
        writer.writeheader()
        writer.writerows(blinded_manifest_rows)
    print(f"\nUpdated blinded dataset manifest: {MANIFEST_PATH}")

    # 5. Overwrite dataset_split/summary.json (removing any stage info)
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = json.load(f)

    if "stage_breakdown" in summary:
        del summary["stage_breakdown"]
    if "obfuscation_mapping_file" in summary:
        del summary["obfuscation_mapping_file"]

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated blinded dataset summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
