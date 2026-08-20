"""Prepare auditable single-image MLX-VLM supervised datasets."""

from __future__ import annotations

import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
CANONICAL_STAGES = {"diestrus", "proestrus", "estrus", "metestrus"}
_STAGE_KEYS = ("stage", "label", "estrous_stage", "class_name")


def prepare_sft_dataset(
    source: str,
    input_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Create MLX-VLM JSONL without carrying source-generated rationales."""
    source_path = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(f"SFT output already exists: {destination}")
    destination.mkdir(parents=True)
    counts: Counter[str] = Counter()
    try:
        if source == "estrousbank":
            _prepare_estrousbank(source_path, destination, counts)
        elif source == "blind-teacher":
            _prepare_teacher(source_path, destination, counts)
        else:
            raise ValueError("source must be 'estrousbank' or 'blind-teacher'")
    except Exception:
        shutil.rmtree(destination)
        raise
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "source": source,
        "source_path": str(source_path),
        "samples_by_split": dict(sorted(counts.items())),
        "format": "mlx-vlm-single-image-messages",
    }
    (destination / "metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _prepare_estrousbank(source: Path, destination: Path, counts: Counter[str]) -> None:
    shards = sorted(source.rglob("*.tar")) if source.is_dir() else [source]
    if not shards or not all(shard.is_file() for shard in shards):
        raise FileNotFoundError(f"no EstrousBank tar shards found at {source}")
    streams: dict[str, Any] = {}
    image_destinations: set[Path] = set()
    try:
        for shard in shards:
            split = _split_from_name(shard.name)
            with tarfile.open(shard) as archive:
                members: dict[str, dict[str, tarfile.TarInfo]] = {}
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower()
                    if suffix == ".json" or suffix in IMAGE_SUFFIXES:
                        sample_key = Path(member.name).with_suffix("").as_posix()
                        members.setdefault(sample_key, {})[suffix] = member
                for sample_key, sample_members in sorted(members.items()):
                    metadata_member = sample_members.get(".json")
                    image_member = next(
                        (sample_members[suffix] for suffix in IMAGE_SUFFIXES if suffix in sample_members),
                        None,
                    )
                    if metadata_member is None or image_member is None:
                        continue
                    metadata_stream = archive.extractfile(metadata_member)
                    image_stream = archive.extractfile(image_member)
                    if metadata_stream is None or image_stream is None:
                        continue
                    metadata = json.load(metadata_stream)
                    stage = _find_stage(metadata)
                    safe_id = _safe_id(f"{shard.stem}-{sample_key}")
                    image_relative = Path("images") / split / f"{safe_id}{Path(image_member.name).suffix.lower()}"
                    if image_relative in image_destinations:
                        raise ValueError(f"safe filename collision: {image_relative}")
                    image_destinations.add(image_relative)
                    image_path = destination / image_relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_stream.read())
                    row = _sft_row(
                        sample_id=safe_id,
                        image_path=image_relative,
                        prompt="Classify this rodent vaginal cytology image into one canonical estrous stage. Return JSON only.",
                        answer={"primary_stage": stage},
                        supervision="broad_stage",
                    )
                    stream = streams.setdefault(
                        split,
                        (destination / f"{split}.jsonl").open("a", encoding="utf-8"),
                    )
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()


def _prepare_teacher(source: Path, destination: Path, counts: Counter[str]) -> None:
    dataset_path = source / "dataset.jsonl" if source.is_dir() else source
    if not dataset_path.is_file():
        raise FileNotFoundError(f"teacher dataset not found: {dataset_path}")
    output = (destination / "train.jsonl").open("x", encoding="utf-8")
    image_destinations: set[Path] = set()
    try:
        with dataset_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                model_record = payload.get("model_record")
                teacher_label = payload.get("teacher_label")
                if not isinstance(model_record, dict) or not isinstance(teacher_label, dict):
                    raise ValueError(f"teacher row {line_number} lacks model_record or teacher_label")
                sample_id = str(payload.get("sample_id") or model_record.get("sample_id") or "").strip()
                if not sample_id:
                    raise ValueError(f"teacher row {line_number} lacks sample_id")
                source_image = Path(str(model_record.get("image_path", ""))).expanduser()
                if not source_image.is_file():
                    raise FileNotFoundError(f"teacher image not found: {source_image}")
                safe_id = _safe_id(sample_id)
                relative = Path("images") / "train" / f"{safe_id}{source_image.suffix.lower()}"
                if relative in image_destinations:
                    raise ValueError(f"safe filename collision: {relative}")
                image_destinations.add(relative)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, target)
                row = _sft_row(
                    sample_id=sample_id,
                    image_path=relative,
                    prompt="Describe visible cytology morphology, QC, uncertainty, and the most supported estrous stage. Return JSON only.",
                    answer=teacher_label,
                    supervision="teacher",
                )
                output.write(json.dumps(row, sort_keys=True) + "\n")
                counts["train"] += 1
    finally:
        output.close()


def _sft_row(
    *,
    sample_id: str,
    image_path: Path,
    prompt: str,
    answer: dict[str, Any],
    supervision: str,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "images": [image_path.as_posix()],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": json.dumps(answer, sort_keys=True)},
        ],
        "metadata": {"supervision": supervision},
    }


def _find_stage(value: Any) -> str:
    if isinstance(value, dict):
        for key in _STAGE_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip().lower() in CANONICAL_STAGES:
                return candidate.strip().lower()
        for nested in value.values():
            try:
                return _find_stage(nested)
            except ValueError:
                continue
    elif isinstance(value, list):
        for nested in value:
            try:
                return _find_stage(nested)
            except ValueError:
                continue
    raise ValueError("EstrousBank sample has no canonical stage label")


def _split_from_name(name: str) -> str:
    lowered = name.lower()
    if re.search(r"(^|[-_.])(val|valid|validation)([-_.]|$)", lowered):
        return "valid"
    if re.search(r"(^|[-_.])test([-_.]|$)", lowered):
        return "test"
    return "train"


def _safe_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not sanitized:
        raise ValueError("sample id cannot be converted to a safe filename")
    return sanitized
