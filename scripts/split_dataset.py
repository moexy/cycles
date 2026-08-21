"""Sensibly divide the whitebalanced dataset into train, validate, and test splits.

Performs strict group-level stratified partitioning to prevent data leakage:
all images belonging to the same mouse subject (e.g. longitudinal D1-D5 days)
or slide entity (multi-spot slide duplicates) are kept strictly together
within the same split partition.
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SRC_DIR = Path("/Volumes/SSD/Imaging/Cycles/test_whitebalanced")
SAMPLES_DIR = Path("/Volumes/SSD/Imaging/Cycles/samples")
DST_DIR = Path("/Volumes/SSD/Imaging/Cycles/dataset_split")


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


def extract_group_and_stage(
    p: Path, batch_map: dict[tuple[str, str], str]
) -> dict[str, str | Path]:
    rel = p.relative_to(SRC_DIR)
    top = rel.parts[0]
    stem = p.stem
    n = stem.lower()

    stage = "unknown"
    cohort = top

    if top.startswith("batch_"):
        # Longitudinal mouse tracking: group all days D1..D5 for that mouse
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
        animal_id = tokens[0] if tokens else "mouse"
        group_id = f"bulk_{date_folder}_{animal_id.upper()}"
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

        # Multi-spot duplicates and slide replicates for same animal stay together
        clean_stem = (
            stem.replace("SPOT", "")
            .replace("spot", "")
            .replace("SPOT2", "")
            .replace("SPOT3", "")
            .strip()
        )
        tokens = clean_stem.split()
        animal_code = tokens[0] if tokens else "hist"
        animal_code = re.sub(r"[^A-Za-z0-9_-]", "", animal_code)
        group_id = f"hist_{date_folder}_{animal_code.upper()}"

    return {
        "path": p,
        "rel": rel,
        "cohort": cohort,
        "stem": stem,
        "group_id": group_id,
        "stage": stage,
    }


def main() -> None:
    batch_map = load_batch_stage_map()
    all_files = sorted(list(SRC_DIR.rglob("*.webp")))
    records = [extract_group_and_stage(p, batch_map) for p in all_files]

    # Group records by indivisible group_id
    groups: dict[str, list[dict[str, str | Path]]] = defaultdict(list)
    for r in records:
        groups[str(r["group_id"])].append(r)

    group_meta: list[dict[str, object]] = []
    for gid, recs in groups.items():
        cohort = str(recs[0]["cohort"]).split()[0]
        stages = [r["stage"] for r in recs]
        dominant_stage = Counter(stages).most_common(1)[0][0]
        group_meta.append(
            {
                "gid": gid,
                "cohort": cohort,
                "dominant_stage": dominant_stage,
                "records": recs,
                "count": len(recs),
            }
        )

    # Stratify groups by (cohort, dominant_stage)
    strata: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for g in group_meta:
        strata[(str(g["cohort"]), str(g["dominant_stage"]))].append(g)

    # Multi-objective balancing to achieve ~70% / 15% / 15% with zero group leakage
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

    # Strict group exclusivity verification
    g_train = {str(g["gid"]) for g in train_g}
    g_val = {str(g["gid"]) for g in val_g}
    g_test = {str(g["gid"]) for g in test_g}

    assert len(g_train & g_val) == 0, "Leakage between train and validate!"
    assert len(g_train & g_test) == 0, "Leakage between train and test!"
    assert len(g_val & g_test) == 0, "Leakage between validate and test!"

    # Re-create destination directory
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    splits_dict: dict[str, list[dict[str, object]]] = {
        "train": train_g,
        "validate": val_g,
        "test": test_g,
    }

    manifest_rows = []
    split_record_counts: dict[str, int] = {}
    stage_breakdown: dict[str, dict[str, int]] = {}
    cohort_breakdown: dict[str, dict[str, int]] = {}

    for split_name, group_list in splits_dict.items():
        split_dir = DST_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        cur_records = [r for g in group_list for r in g["records"]]  # type: ignore[union-attr]
        split_record_counts[split_name] = len(cur_records)
        stage_breakdown[split_name] = dict(Counter(str(r["stage"]) for r in cur_records))
        cohort_breakdown[split_name] = dict(Counter(str(r["cohort"]) for r in cur_records))

        for r in cur_records:
            src_path = Path(r["path"])  # type: ignore[arg-type]
            rel_path = Path(r["rel"])  # type: ignore[arg-type]
            dst_file = split_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_file)

            manifest_rows.append(
                {
                    "split": split_name,
                    "filename": src_path.name,
                    "relative_path": str(rel_path),
                    "split_path": f"{split_name}/{rel_path}",
                    "cohort": str(r["cohort"]),
                    "group_id": str(r["group_id"]),
                    "stage": str(r["stage"]),
                }
            )

    # Create 'val' symlink to 'validate' for framework compatibility
    val_symlink = DST_DIR / "val"
    try:
        val_symlink.symlink_to("validate", target_is_directory=True)
    except OSError:
        pass

    # Save manifest.csv
    manifest_csv = DST_DIR / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "filename",
                "relative_path",
                "split_path",
                "cohort",
                "group_id",
                "stage",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save summary.json
    summary_data = {
        "total_images": total_images,
        "total_unique_groups": len(groups),
        "split_image_counts": split_record_counts,
        "split_group_counts": {k: len(v) for k, v in splits_dict.items()},
        "split_percentages": {
            k: f"{v / total_images * 100:.1f}%" for k, v in split_record_counts.items()
        },
        "stage_breakdown": stage_breakdown,
        "cohort_breakdown": cohort_breakdown,
        "group_leakage_verified": "0 overlap across all split pairs",
    }
    with open(DST_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("Zero-Leakage Group-Level Dataset Split Complete:")
    print(
        f"  Train:    {split_record_counts['train']} images ({len(train_g)} groups) [{split_record_counts['train'] / total_images * 100:.1f}%]"
    )
    print(
        f"  Validate: {split_record_counts['validate']} images ({len(val_g)} groups) [{split_record_counts['validate'] / total_images * 100:.1f}%]"
    )
    print(
        f"  Test:     {split_record_counts['test']} images ({len(test_g)} groups) [{split_record_counts['test'] / total_images * 100:.1f}%]"
    )
    print(f"  Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
