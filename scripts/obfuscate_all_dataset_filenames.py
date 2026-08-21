"""Obfuscate all stage-revealing filenames and group_ids across the dataset.

Saves full reverse-lookup mappings, raw group IDs, and ground-truth records OUTSIDE
the dataset in /Volumes/SSD/Imaging/Cycles/metadata/.
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

    # 2. Load stage and raw group info from external ground truth if present
    existing_stages: dict[str, str] = {}
    existing_raw_groups: dict[str, str] = {}
    gt_existing_path = EXTERNAL_META_DIR / "ground_truth_metadata.csv"
    if gt_existing_path.is_file():
        with open(gt_existing_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_stages[row["original_filename"]] = row["stage"]
                existing_raw_groups[row["original_filename"]] = row.get(
                    "raw_group_id", row.get("group_id", "unknown")
                )

    all_dataset_files = sorted(list(DATASET_ROOT.rglob("*.webp")))
    # Filter out symlinked 'val' files to avoid duplicates
    all_dataset_files = [p for p in all_dataset_files if "val/" not in p.as_posix()]

    # Collect unique raw group IDs to generate deterministic opaque IDs
    raw_group_set: set[str] = set()
    file_records: list[dict[str, str]] = []

    for p in all_dataset_files:
        rel_from_root = p.relative_to(DATASET_ROOT)
        split = rel_from_root.parts[0]
        cohort = rel_from_root.parts[1]
        rel_path = Path(*rel_from_root.parts[1:])
        fn = p.name

        is_obf = fn.startswith("sample_") and fn in orig_hash_to_path
        if is_obf:
            orig_rel = orig_hash_to_path[fn]
            orig_fn = orig_rel.name
        else:
            orig_rel = rel_path
            orig_fn = fn

        stage = existing_stages.get(orig_fn, "unknown")
        raw_gid = existing_raw_groups.get(orig_fn, "unknown")
        if raw_gid == "unknown":
            raw_gid = (
                f"{cohort}_{p.parent.name}_{orig_fn.split()[0]}"
                if not cohort.startswith("batch_")
                else f"{cohort}_{p.parent.name}"
            )

        raw_group_set.add(raw_gid)

        file_records.append(
            {
                "split": split,
                "filename": fn,
                "original_filename": orig_fn,
                "is_obfuscated": str(is_obf),
                "relative_path": str(rel_path),
                "split_path": f"{split}/{rel_path}",
                "cohort": cohort,
                "raw_group_id": raw_gid,
                "original_relative_path": str(orig_rel),
                "stage": stage,
            }
        )

    # 3. Create deterministic opaque group IDs: group_0001, group_0002, ...
    sorted_raw_groups = sorted(list(raw_group_set))
    raw_to_opaque_group = {
        raw_g: f"group_{idx:04d}" for idx, raw_g in enumerate(sorted_raw_groups, 1)
    }

    full_ground_truth_rows = []
    blinded_manifest_rows = []
    obfuscation_mapping_rows = []

    for r in file_records:
        opaque_gid = raw_to_opaque_group[r["raw_group_id"]]
        is_obf = r["is_obfuscated"] == "True"

        # External ground truth row
        gt_row = {
            "split": r["split"],
            "filename": r["filename"],
            "original_filename": r["original_filename"],
            "is_obfuscated": r["is_obfuscated"],
            "relative_path": r["relative_path"],
            "split_path": r["split_path"],
            "cohort": r["cohort"],
            "group_id": opaque_gid,
            "raw_group_id": r["raw_group_id"],
            "stage": r["stage"],
        }
        full_ground_truth_rows.append(gt_row)

        if is_obf:
            obf_row = {
                "split": r["split"],
                "cohort": r["cohort"],
                "directory": str(Path(r["relative_path"]).parent),
                "original_filename": r["original_filename"],
                "obfuscated_filename": r["filename"],
                "original_relative_path": r["original_relative_path"],
                "obfuscated_relative_path": r["relative_path"],
                "original_split_path": f"{r['split']}/{r['original_relative_path']}",
                "obfuscated_split_path": r["split_path"],
                "group_id": opaque_gid,
                "raw_group_id": r["raw_group_id"],
                "stage": r["stage"],
            }
            obfuscation_mapping_rows.append(obf_row)

        # Blinded internal manifest (opaque group_id, NO stage, NO original_filename)
        blinded_row = {
            "split": r["split"],
            "filename": r["filename"],
            "relative_path": r["relative_path"],
            "split_path": r["split_path"],
            "cohort": r["cohort"],
            "group_id": opaque_gid,
        }
        blinded_manifest_rows.append(blinded_row)

    # 4. Write external ground truth records
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
        "raw_group_id",
        "stage",
    ]
    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gt_fields)
        writer.writeheader()
        writer.writerows(full_ground_truth_rows)
    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(full_ground_truth_rows, f, indent=2)

    obf_csv = EXTERNAL_META_DIR / "all_obfuscation_mapping.csv"
    obf_json = EXTERNAL_META_DIR / "all_obfuscation_mapping.json"
    obf_fields = [
        "split",
        "cohort",
        "directory",
        "original_filename",
        "obfuscated_filename",
        "original_relative_path",
        "obfuscated_relative_path",
        "original_split_path",
        "obfuscated_split_path",
        "group_id",
        "raw_group_id",
        "stage",
    ]
    with open(obf_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=obf_fields)
        writer.writeheader()
        writer.writerows(obfuscation_mapping_rows)
    with open(obf_json, "w", encoding="utf-8") as f:
        json.dump(obfuscation_mapping_rows, f, indent=2)

    print("Saved external ground truth metadata:")
    print(f"  {gt_csv} ({len(full_ground_truth_rows)} rows)")
    print(f"  {obf_csv} ({len(obfuscation_mapping_rows)} rows)")

    # 5. Write blinded manifest to dataset_split/manifest.csv
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

    # 6. Overwrite dataset_split/summary.json
    summary_data = {
        "total_images": len(full_ground_truth_rows),
        "total_unique_groups": len(sorted_raw_groups),
        "split_counts": {
            "train": len([r for r in full_ground_truth_rows if r["split"] == "train"]),
            "validate": len([r for r in full_ground_truth_rows if r["split"] == "validate"]),
            "test": len([r for r in full_ground_truth_rows if r["split"] == "test"]),
        },
        "total_obfuscated_images": len(obfuscation_mapping_rows),
        "obfuscated_by_split": {
            "train": len([r for r in obfuscation_mapping_rows if r["split"] == "train"]),
            "validate": len([r for r in obfuscation_mapping_rows if r["split"] == "validate"]),
            "test": len([r for r in obfuscation_mapping_rows if r["split"] == "test"]),
        },
        "cohort_counts": {
            "Histology staging images": len(
                [r for r in full_ground_truth_rows if "Histology" in r["cohort"]]
            ),
            "BulkRNA staging images": len(
                [r for r in full_ground_truth_rows if "BulkRNA" in r["cohort"]]
            ),
            "batch_1": len([r for r in full_ground_truth_rows if r["cohort"] == "batch_1"]),
            "batch_2": len([r for r in full_ground_truth_rows if r["cohort"] == "batch_2"]),
            "batch_3": len([r for r in full_ground_truth_rows if r["cohort"] == "batch_3"]),
            "batch_4": len([r for r in full_ground_truth_rows if r["cohort"] == "batch_4"]),
            "batch_5": len([r for r in full_ground_truth_rows if r["cohort"] == "batch_5"]),
        },
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Updated clean dataset summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
