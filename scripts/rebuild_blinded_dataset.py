"""Rebuild the completely blinded dataset with opaque group IDs and zero group leakage.

1. Partitions images at the canonical subject/animal entity level (zero leakage).
2. Obfuscates all BulkRNA and Histology filenames across train, validate, test to sample_<hash>.webp.
3. Anonymizes group_id in manifest.csv to opaque group_0001..group_0316.
4. Stores complete ground-truth metadata, original filenames, and raw group IDs OUTSIDE the dataset
   in /Volumes/SSD/Imaging/Cycles/metadata/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SRC_DIR = Path("/Volumes/SSD/Imaging/Cycles/test_whitebalanced")
SAMPLES_DIR = Path("/Volumes/SSD/Imaging/Cycles/samples")
DST_DIR = Path("/Volumes/SSD/Imaging/Cycles/dataset_split")
EXTERNAL_META_DIR = Path("/Volumes/SSD/Imaging/Cycles/metadata")
MANIFEST_PATH = DST_DIR / "manifest.csv"
SUMMARY_PATH = DST_DIR / "summary.json"


def load_batch_stage_map() -> dict[tuple[str, str], str]:
    batch_map: dict[tuple[str, str], str] = {}
    for csv_path in SAMPLES_DIR.glob("batch_*_results.csv"):
        batch_name = csv_path.stem.replace("_results", "")
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row["Filename"]
                base = fname.replace(".ome.tif", "").replace(".tif", "")
                stage = row["FinalStage"].lower()
                batch_map[(batch_name, base)] = stage
    batch_map[("batch_5", "mouse1D2a")] = "metestrus"
    batch_map[("batch_5", "mouse1D2b")] = "metestrus"
    return batch_map


def extract_canonical_record(
    p: Path, batch_map: dict[tuple[str, str], str]
) -> dict[str, str | Path]:
    rel = p.relative_to(SRC_DIR)
    top = rel.parts[0]
    stem = p.stem
    n = stem.lower()

    stage = "unknown"
    cohort = top

    if top.startswith("batch_"):
        mouse_name = rel.parts[1] if len(rel.parts) > 2 else stem.split("D")[0]
        group_id = f"{top}_{mouse_name}"
        st = batch_map.get((top, stem))
        if st:
            stage = st
        else:
            for (b, f), val in batch_map.items():
                if b == top and (f in stem or stem in f):
                    stage = val
                    break
    elif top.startswith("BulkRNA"):
        date_folder = rel.parts[1]
        tokens = stem.split()
        if tokens:
            last = tokens[-1].lower()
            if last == "e" or stem.endswith("e"):
                stage = "estrus"
            elif last == "p" or stem.endswith("p"):
                stage = "proestrus"
            elif last == "d" or stem.endswith("d"):
                stage = "diestrus"
            elif last == "m" or stem.endswith("m"):
                stage = "metestrus"
        animal_token = tokens[0] if tokens else "mouse"
        m = re.match(r"^([A-Za-z]+\d+)", animal_token)
        base_animal = m.group(1).upper() if m else animal_token.upper()
        group_id = f"bulk_{date_folder}_{base_animal}"
    elif top.startswith("Histology"):
        date_folder = rel.parts[1]
        if (
            "diest" in n
            or "die" in n
            or " di" in n
            or n.endswith(" d")
            or n.endswith("d")
            or " d " in n
        ):
            stage = "diestrus"
        elif (
            "proest" in n
            or "pro" in n
            or " pro" in n
            or n.endswith(" p")
            or n.endswith("p")
            or " p " in n
        ):
            stage = "proestrus"
        elif (
            "estrous" in n
            or "estrus" in n
            or " est" in n
            or n.endswith(" e")
            or n.endswith("e")
            or " e " in n
            or " es" in n
        ):
            stage = "estrus"
        elif (
            "metest" in n
            or "met" in n
            or " met" in n
            or n.endswith(" m")
            or n.endswith("m")
            or " m " in n
        ):
            stage = "metestrus"

        clean_stem = (
            stem.replace("SPOT", "")
            .replace("spot", "")
            .replace("SPOT2", "")
            .replace("SPOT3", "")
            .strip()
        )
        tokens = clean_stem.split()
        raw_code = tokens[0] if tokens else "hist"
        m = re.match(r"^([A-Za-z]+\d+)", raw_code)
        base_animal = m.group(1).upper() if m else re.sub(r"[^A-Za-z0-9_-]", "", raw_code).upper()
        group_id = f"hist_{date_folder}_{base_animal}"

    return {
        "src_path": p,
        "rel_path": rel,
        "cohort": cohort,
        "stem": stem,
        "raw_group_id": group_id,
        "stage": stage,
    }


def main() -> None:
    EXTERNAL_META_DIR.mkdir(parents=True, exist_ok=True)
    batch_map = load_batch_stage_map()
    all_files = sorted(list(SRC_DIR.rglob("*.webp")))
    records = [extract_canonical_record(p, batch_map) for p in all_files]

    # Group by canonical animal entity
    groups: dict[str, list[dict[str, str | Path]]] = defaultdict(list)
    for r in records:
        groups[str(r["raw_group_id"])].append(r)

    group_meta: list[dict[str, object]] = []
    for gid, recs in groups.items():
        cohort = str(recs[0]["cohort"]).split()[0]
        stages = [r["stage"] for r in recs]
        dominant_stage = Counter(stages).most_common(1)[0][0]
        group_meta.append(
            {
                "raw_gid": gid,
                "cohort": cohort,
                "dominant_stage": dominant_stage,
                "records": recs,
                "count": len(recs),
            }
        )

    # Stratify by (cohort, dominant_stage)
    strata: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for g in group_meta:
        strata[(str(g["cohort"]), str(g["dominant_stage"]))].append(g)

    # Multi-objective balancing
    random.seed(42)
    best_splits = None
    best_score = float("inf")

    target_train_pct = 0.70
    target_val_pct = 0.15
    target_test_pct = 0.15
    total_images = len(records)

    for _ in range(1000):
        train_g: list[dict[str, object]] = []
        val_g: list[dict[str, object]] = []
        test_g: list[dict[str, object]] = []
        train_cnt, val_cnt, test_cnt = 0, 0, 0

        for _, g_list in sorted(strata.items()):
            shuffled = list(g_list)
            random.shuffle(shuffled)
            for g in shuffled:
                c = int(g["count"])  # type: ignore[arg-type]
                d_t = target_train_pct * total_images - train_cnt
                d_v = target_val_pct * total_images - val_cnt
                d_te = target_test_pct * total_images - test_cnt

                deficits = [
                    (d_t / target_train_pct, "train"),
                    (d_v / target_val_pct, "val"),
                    (d_te / target_test_pct, "test"),
                ]
                deficits.sort(reverse=True)
                chosen = deficits[0][1]

                if chosen == "train":
                    train_g.append(g)
                    train_cnt += c
                elif chosen == "val":
                    val_g.append(g)
                    val_cnt += c
                else:
                    test_g.append(g)
                    test_cnt += c

        score = (
            abs(train_cnt / total_images - target_train_pct)
            + abs(val_cnt / total_images - target_val_pct)
            + abs(test_cnt / total_images - target_test_pct)
        )
        if score < best_score:
            best_score = score
            best_splits = (train_g, val_g, test_g, train_cnt, val_cnt, test_cnt)

    assert best_splits is not None
    train_g, val_g, test_g, train_cnt, val_cnt, test_cnt = best_splits

    # Create opaque group IDs sorted deterministically
    sorted_raw_groups = sorted(list(groups.keys()))
    raw_to_opaque_group = {
        raw_g: f"group_{idx:04d}" for idx, raw_g in enumerate(sorted_raw_groups, 1)
    }

    # Reset destination directory
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    splits_dict: dict[str, list[dict[str, object]]] = {
        "train": train_g,
        "validate": val_g,
        "test": test_g,
    }

    full_ground_truth_rows = []
    blinded_manifest_rows = []
    obfuscation_mapping_rows = []

    for split_name, group_list in splits_dict.items():
        split_dir = DST_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for g in group_list:
            raw_gid = str(g["raw_gid"])
            opaque_gid = raw_to_opaque_group[raw_gid]
            recs = g["records"]  # type: ignore[union-attr]

            for r in recs:
                src_path = Path(r["src_path"])  # type: ignore[arg-type]
                orig_rel = Path(r["rel_path"])  # type: ignore[arg-type]
                cohort = str(r["cohort"])
                stage = str(r["stage"])

                is_obf = cohort.startswith("BulkRNA") or cohort.startswith("Histology")
                if is_obf:
                    h = hashlib.sha256(str(orig_rel).encode()).hexdigest()[:8]
                    out_filename = f"sample_{h}.webp"
                else:
                    out_filename = src_path.name

                out_rel = orig_rel.parent / out_filename
                dst_file = split_dir / out_rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_file)

                split_path = f"{split_name}/{out_rel}"

                # Ground truth external record
                full_ground_truth_rows.append(
                    {
                        "split": split_name,
                        "filename": out_filename,
                        "original_filename": src_path.name,
                        "is_obfuscated": str(is_obf),
                        "relative_path": str(out_rel),
                        "split_path": split_path,
                        "cohort": cohort,
                        "group_id": opaque_gid,
                        "raw_group_id": raw_gid,
                        "stage": stage,
                    }
                )

                if is_obf:
                    obfuscation_mapping_rows.append(
                        {
                            "split": split_name,
                            "cohort": cohort,
                            "directory": str(out_rel.parent),
                            "original_filename": src_path.name,
                            "obfuscated_filename": out_filename,
                            "original_relative_path": str(orig_rel),
                            "obfuscated_relative_path": str(out_rel),
                            "original_split_path": f"{split_name}/{orig_rel}",
                            "obfuscated_split_path": split_path,
                            "group_id": opaque_gid,
                            "raw_group_id": raw_gid,
                            "stage": stage,
                        }
                    )

                # Blinded internal manifest
                blinded_manifest_rows.append(
                    {
                        "split": split_name,
                        "filename": out_filename,
                        "relative_path": str(out_rel),
                        "split_path": split_path,
                        "cohort": cohort,
                        "group_id": opaque_gid,
                    }
                )

    # Create 'val' symlink
    val_symlink = DST_DIR / "val"
    try:
        val_symlink.symlink_to("validate", target_is_directory=True)
    except OSError:
        pass

    # Save external metadata
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

    # Save blinded manifest
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
    print(f"Updated clean dataset manifest: {MANIFEST_PATH}")

    # Save summary.json
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
