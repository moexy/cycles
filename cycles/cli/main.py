"""Unified command-line entry point for all cycles analysis modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def _existing_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _progress(current: int, total: int, message: str) -> None:
    print(f"[{current}/{total}] {message}", file=sys.stderr)


def _device(value: str) -> str | None:
    return None if value == "auto" else value


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()  # type: ignore[no-any-return, union-attr]
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _build_cnn_service(model_spec: str, device: str | None) -> Any:
    from cycles.core.models import build_model, get_device
    from cycles.stages.cnn import CNNClassifierService

    resolved_device = get_device(device)
    checkpoint = Path(model_spec).expanduser()
    if checkpoint.is_file():
        return CNNClassifierService.from_checkpoint(checkpoint, device=resolved_device)
    if checkpoint.suffix:
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    model = build_model(model_spec, num_classes=4, pretrained=False)
    return CNNClassifierService(
        model=model,
        device=resolved_device,
        architecture=model_spec,
    )


def _cmd_classify(args: argparse.Namespace) -> int:
    service = _build_cnn_service(args.model, _device(args.device))
    result = service.classify_folder(
        args.folder,
        recursive=args.recursive,
        progress_callback=_progress,
        cancel_flag=lambda: False,
    )
    service.export_results_csv(result, args.output)
    print(f"Classified {len(result.results)} image(s); wrote {args.output}")
    return 0


def _cmd_cell_centric(args: argparse.Namespace) -> int:
    from cycles.stages.cell_centric import CellCentricPipeline

    pipeline = CellCentricPipeline(detector_mode=args.detector)
    results = pipeline.process_folder(
        args.folder,
        save_overlays_dir=args.save_overlays,
        recursive=False,
        progress_callback=_progress,
    )
    pipeline.export_results_csv(results, args.output)
    print(
        f"Processed {len(results)} image(s), {len(pipeline.processing_errors)} failed; "
        f"wrote {args.output}"
    )
    return 0


def _cmd_mil(args: argparse.Namespace) -> int:
    from cycles.stages.mil import AttentionMILPipeline

    pipeline = AttentionMILPipeline()
    result = pipeline.process_folder(
        args.folder,
        save_heatmaps_dir=args.save_heatmaps,
        recursive=False,
        progress_callback=lambda current, total, path: _progress(
            current,
            total,
            f"Processed {Path(path).name}",
        ),
    )
    pipeline.export_results_csv(result, args.output)
    print(
        f"Processed {len(result.results)} image(s), {len(result.failed_images)} failed; "
        f"wrote {args.output}"
    )
    return 0


def _normalise_column(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _first_value(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    normalised = {_normalise_column(key): value for key, value in row.items()}
    for name in names:
        value = normalised.get(_normalise_column(name))
        if value is not None and value.strip():
            return value.strip()
    return None


def _read_cycle_jsonl(path: Path, mouse_id: str | None = None) -> tuple[list[str], list[Any]]:
    from cycles.core.types import EstrousStage

    items: list[tuple[str, EstrousStage]] = []
    with path.open(encoding="utf-8") as stream:
        for _line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_mouse_id = payload.get("subject_id")
            if mouse_id is not None and row_mouse_id is not None and str(row_mouse_id) != str(mouse_id):
                continue
            seq = payload.get("sequence_prediction") or {}
            img = payload.get("image_prediction") or {}
            stage_val = seq.get("final_stage") or img.get("primary_stage")
            if not stage_val:
                continue
            try:
                stage = EstrousStage(stage_val)
            except ValueError:
                continue

            day = payload.get("day")
            if day is not None:
                timestamp = _coerce_timestamp(str(day))
            else:
                timestamp = _timestamp_from_filename(payload.get("image_path") or payload.get("sample_id"))
                if timestamp is None:
                    timestamp = (datetime(1970, 1, 1) + timedelta(days=len(items))).isoformat()
            items.append((timestamp, stage))

    if not items:
        raise ValueError(f"cycle input JSONL contains no valid observations: {path}")

    items.sort(key=lambda x: x[0])
    return [t for t, _ in items], [s for _, s in items]


def _read_cycle_observations(path: Path, mouse_id: str | None = None) -> tuple[list[str], list[Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_cycle_jsonl(path, mouse_id=mouse_id)
    return _read_cycle_csv(path, mouse_id=mouse_id)


def _read_cycle_csv(path: Path, mouse_id: str | None = None) -> tuple[list[str], list[Any]]:
    from cycles.core.types import EstrousStage

    timestamps: list[str] = []
    stages: list[EstrousStage] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("cycle input CSV has no header")
        for index, row in enumerate(reader):
            row_mouse_id = _first_value(row, ("mouse_id", "mouse", "animal_id", "animal"))
            if mouse_id is not None and row_mouse_id is not None and row_mouse_id != mouse_id:
                continue
            stage_text = _first_value(
                row,
                ("predicted_stage", "stage", "final_stage", "net_stage", "phase"),
            )
            if stage_text is None:
                raise ValueError(
                    "cycle input requires a stage column (PredictedStage, Stage, FinalStage, or Phase)"
                )
            try:
                stage = EstrousStage(stage_text.strip().lower().replace(" ", "_"))
            except ValueError as exc:
                raise ValueError(f"row {index + 2}: unknown estrous stage {stage_text!r}") from exc

            timestamp = _first_value(row, ("timestamp", "date", "datetime", "day"))
            if timestamp is None:
                image_name = _first_value(row, ("image_path", "filename", "file", "path"))
                timestamp = _timestamp_from_filename(image_name) if image_name else None
            if timestamp is None:
                timestamp = (datetime(1970, 1, 1) + timedelta(days=len(stages))).isoformat()
            else:
                timestamp = _coerce_timestamp(timestamp)
            timestamps.append(timestamp)
            stages.append(stage)

    if not stages:
        raise ValueError("cycle input CSV contains no observations")
    return timestamps, stages


def _coerce_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = None
    if parsed is not None:
        return parsed.isoformat()
    for pattern in ("%Y%m%d", "%y%m%d"):
        try:
            return datetime.strptime(value, pattern).isoformat()
        except ValueError:
            continue
    try:
        return (datetime(1970, 1, 1) + timedelta(days=float(value))).isoformat()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid timestamp or day value: {value!r}") from exc


def _timestamp_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    stem = Path(filename).stem
    digits = "".join(character for character in stem if character.isdigit())
    for size, pattern in ((8, "%Y%m%d"), (6, "%y%m%d")):
        if len(digits) >= size:
            try:
                return datetime.strptime(digits[:size], pattern).isoformat()
            except ValueError:
                continue
    return None


def _write_cycle_plot(plot_data: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    phase_names = ["Diestrus", "Proestrus", "Estrus", "Metestrus"]
    phase_colors = ["#3b82f6", "#22c55e", "#ef4444", "#f97316"]
    raw_days = plot_data.get("days") or plot_data.get("x")
    if raw_days:
        days = list(raw_days)
    else:
        timestamps = plot_data.get("timestamps", [])
        try:
            parsed = [datetime.fromisoformat(str(value)) for value in timestamps]
            days = [
                (value - parsed[0]).total_seconds() / 86_400.0
                for value in parsed
            ]
        except (TypeError, ValueError):
            days = list(range(len(timestamps)))
    values = plot_data.get("phase_values") or plot_data.get("stage_values") or plot_data.get("y")
    if values is None:
        raw_stages = plot_data.get("stages", [])
        phase_index = {name.lower(): idx for idx, name in enumerate(phase_names)}
        values = [phase_index.get(str(getattr(stage, "value", stage)).lower(), 0) for stage in raw_stages]
    values = [
        value - 1 if isinstance(value, (int, float)) and 1 <= value <= 4 else value
        for value in values
    ]
    if not days:
        days = list(range(len(values)))

    figure = Figure(figsize=(10, 4.5), dpi=140)
    axis = figure.subplots()
    for index, color in enumerate(phase_colors):
        axis.axhspan(index - 0.45, index + 0.45, color=color, alpha=0.15)
    axis.plot(days, values, color="#111827", marker="o", linewidth=2)
    axis.set_xlabel("Day")
    axis.set_ylabel("Phase")
    axis.set_yticks(range(4), phase_names)
    axis.set_ylim(-0.5, 3.5)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def _cmd_cycle_fit(args: argparse.Namespace) -> int:
    from cycles.core.cycle import fit_cyclicity, generate_cycle_plot_data

    timestamps, stages = _read_cycle_observations(args.input, mouse_id=args.mouse_id)
    fit = fit_cyclicity(timestamps, stages, mouse_id=args.mouse_id or "Mouse1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(
            json.dumps(fit, default=_json_default, indent=2),
            encoding="utf-8",
        )
    else:
        _write_cycle_plot(generate_cycle_plot_data(timestamps, stages), args.output)
    print(f"Wrote cycle fit to {args.output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from cycles.eval.benchmark import BenchmarkHarness

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name")
    harness = BenchmarkHarness()
    harness.run_benchmark(
        image_dir=args.image_dir,
        models=models,
        output_json=args.output,
        markdown_report=args.markdown_report,
        plot_dir=args.plot_dir,
    )
    print(f"Wrote benchmark report to {args.output}")
    return 0

def _cmd_train(args: argparse.Namespace) -> int:
    from cycles.core.models import get_device
    from cycles.stages.cnn import CNNTrainerService, CNNTrainingConfig

    device = get_device(args.device if args.device != "auto" else None)
    config = CNNTrainingConfig(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=not args.no_freeze,
        output_path=args.output,
    )
    trainer = CNNTrainerService(device=device)
    result = trainer.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        config=config,
        progress_callback=lambda m: print(
            f"Epoch {m.epoch}/{config.epochs}: Train Loss={m.train_loss:.4f} (Acc={m.train_accuracy*100:.1f}%), "
            f"Val Loss={m.val_loss:.4f} (Acc={m.val_accuracy*100:.1f}%) [LR={m.learning_rate:.6f}]"
        ),
    )
    print(f"Training complete! Best checkpoint saved to {result.checkpoint_path} (Val Acc={result.best_val_accuracy*100:.1f}%)")
    return 0


_VLM_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _build_local_vlm_pipeline(
    model: str,
    adapter: Path | None,
    model_revision: str,
    calibrator_path: Path | None,
    reuse_prompt_prefix: bool = True,
) -> Any:
    from cycles.vlm_local.backend import MLXVLMBackend
    from cycles.vlm_local.calibration import TemperatureCalibrator
    from cycles.vlm_local.pipeline import LocalVLMPipeline

    backend = MLXVLMBackend(
        model,
        adapter_path=adapter,
        model_revision=model_revision,
        reuse_prompt_prefix=reuse_prompt_prefix,
    )
    calibrator = TemperatureCalibrator.load(calibrator_path) if calibrator_path else None
    return LocalVLMPipeline(
        backend,
        software_lock_hash=_software_lock_hash(),
        calibrator=calibrator,
    )


def _software_lock_hash() -> str:
    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else "unlocked"


def _build_temporal_reconciler(margin_threshold: float, adjustment_threshold: float) -> Any:
    from cycles.vlm_local.temporal import TemporalReconciler

    return TemporalReconciler(
        margin_threshold=margin_threshold,
        adjustment_threshold=adjustment_threshold,
    )


def _vlm_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in _VLM_IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image format: {input_path}")
        return [input_path.resolve()]
    images = sorted(
        path.resolve()
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _VLM_IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no supported images found in {input_path}")
    return images


def _read_vlm_sequence_manifest(path: Path) -> dict[Path, dict[str, Any]]:
    required = ["sample_id", "image_path", "subject_id", "day"]
    rows: dict[Path, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != required:
            raise ValueError(
                "sequence manifest columns must be exactly: " + ",".join(required)
            )
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            subject_id = (row.get("subject_id") or "").strip()
            raw_image = (row.get("image_path") or "").strip()
            if not sample_id or not subject_id or not raw_image:
                raise ValueError(f"sequence manifest row {row_number} has empty required values")
            image_path = Path(raw_image).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"sequence image not found: {image_path}")
            if sample_id in sample_ids or image_path in rows:
                raise ValueError(f"duplicate sample or image on sequence manifest row {row_number}")
            try:
                day = float((row.get("day") or "").strip())
            except ValueError as exc:
                raise ValueError(f"invalid day on sequence manifest row {row_number}") from exc
            sample_ids.add(sample_id)
            rows[image_path] = {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "day": day,
            }
    return rows


DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
PINNED_MODEL_REVISIONS: dict[str, str] = {
    "mlx-community/Qwen3-VL-8B-Instruct-4bit": "9ba067a99fba636e053cbdb2ebaf2a417c8cfeb2",
    "mlx-community/Qwen3-VL-4B-Instruct-8bit": "0943db6e15185b86be368d3cf0704aec740b142b",
    "mlx-community/gemma-3-12b-it-4bit": "86cc6a8d799fa7e366052dc5e33bf7dc974e4776",
}


def _render_vlm_record_card(record: Any) -> str:
    from cycles.core.types import EstrousStage

    pred = getattr(record, "image_prediction", None)
    morph = getattr(record, "morphology", None)
    seq = getattr(record, "sequence_prediction", None)
    if pred is None or morph is None or seq is None:
        return ""

    sample_id = getattr(record, "sample_id", "sample")
    image_path = getattr(record, "image_path", "")
    subject_id = getattr(record, "subject_id", None)
    day = getattr(record, "day", None)

    lines = []
    lines.append("=" * 76)
    sample_info = f"SAMPLE: {sample_id}"
    if subject_id:
        day_str = f", Day: {day}" if day is not None else ""
        sample_info += f"  (Subject: {subject_id}{day_str})"
    lines.append(sample_info)
    lines.append(f"IMAGE:  {image_path}")

    stage_str = pred.primary_stage.display_name.upper() if pred.primary_stage else "UNGRADABLE"
    conf_str = pred.confidence_tier.value.upper()

    if seq.adjusted and seq.final_stage:
        lines.append(
            f"STAGE:  {seq.final_stage.display_name.upper()}  "
            f"[Reconciled by longitudinal sequence: {seq.reason}] (Single-Image: {stage_str})"
        )
    else:
        lines.append(f"STAGE:  {stage_str}  (Confidence: {conf_str})")

    if pred.probabilities:
        lines.append("\nPHASE PROBABILITIES:")
        for stage in (
            EstrousStage.DIESTRUS,
            EstrousStage.PROESTRUS,
            EstrousStage.ESTRUS,
            EstrousStage.METESTRUS,
        ):
            prob = pred.probabilities.get(stage, 0.0)
            bar_len = int(round(prob * 30))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {stage.display_name:<10}: {prob*100:>5.1f}%  |{bar}|")

    lines.append("\nCYTOMORPHOLOGY:")
    lines.append(
        f"  Leukocytes: {morph.leukocytes.value.upper():<9} | "
        f"Cornified: {morph.cornified_squames.value.upper():<9} | "
        f"Nucleated: {morph.nucleated_epithelial.value.upper():<9}"
    )
    lines.append(
        f"  Nuclear:    {morph.nuclear_state.value:<9} | "
        f"Layout:    {morph.arrangement.value:<9} | "
        f"QC:        {morph.qc_status.value.upper()}"
    )

    if pred.rationale:
        lines.append(f"\nRATIONALE:\n  {pred.rationale}")
    lines.append("=" * 76)
    return "\n".join(lines)


def _render_vlm_summary_table(records: list[Any]) -> str:
    from cycles.core.types import EstrousStage

    if not records or not hasattr(records[0], "image_prediction"):
        return ""

    headers = (
        f"{'Sample ID':<16} {'Assessed Stage':<14} {'Conf':<6} "
        f"{'P(D)':<6} {'P(P)':<6} {'P(E)':<6} {'P(M)':<6} {'Leukocytes':<10} {'Cornified':<10}"
    )
    divider = "-" * len(headers)
    rows = [headers, divider]
    for r in records:
        pred = getattr(r, "image_prediction", None)
        seq = getattr(r, "sequence_prediction", None)
        morph = getattr(r, "morphology", None)
        sample_id = getattr(r, "sample_id", "sample")
        if pred is None or seq is None or morph is None:
            continue
        final_stage = seq.final_stage or pred.primary_stage
        stage_text = final_stage.display_name.upper() if final_stage else "UNGRADABLE"
        if seq.adjusted:
            stage_text += "*"
        conf = pred.confidence_tier.value[:4].upper()
        probs = pred.probabilities or {}
        pd = f"{probs.get(EstrousStage.DIESTRUS, 0.0):.2f}"
        pp = f"{probs.get(EstrousStage.PROESTRUS, 0.0):.2f}"
        pe = f"{probs.get(EstrousStage.ESTRUS, 0.0):.2f}"
        pm = f"{probs.get(EstrousStage.METESTRUS, 0.0):.2f}"
        leuko = morph.leukocytes.value
        corn = morph.cornified_squames.value
        rows.append(
            f"{sample_id[:15]:<16} {stage_text:<14} {conf:<6} "
            f"{pd:<6} {pp:<6} {pe:<6} {pm:<6} {leuko:<10} {corn:<10}"
        )
    return "\n".join(rows)


def _export_vlm_records_csv(records: list[Any], output_path: Path) -> None:
    from cycles.core.types import EstrousStage

    if not records or not hasattr(records[0], "image_prediction"):
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_id",
            "image_path",
            "subject_id",
            "day",
            "predicted_stage",
            "secondary_stage",
            "confidence_tier",
            "p_diestrus",
            "p_proestrus",
            "p_estrus",
            "p_metestrus",
            "leukocytes",
            "cornified_squames",
            "nucleated_epithelial",
            "nuclear_state",
            "arrangement",
            "qc_status",
            "final_stage",
            "adjusted",
            "reason",
            "rationale",
        ])
        for r in records:
            pred = getattr(r, "image_prediction", None)
            morph = getattr(r, "morphology", None)
            seq = getattr(r, "sequence_prediction", None)
            if pred is None or morph is None or seq is None:
                continue
            probs = pred.probabilities or {}
            writer.writerow([
                getattr(r, "sample_id", ""),
                getattr(r, "image_path", ""),
                getattr(r, "subject_id", "") or "",
                getattr(r, "day", "") if getattr(r, "day", None) is not None else "",
                pred.primary_stage.value if pred.primary_stage else "",
                pred.secondary_stage.value if pred.secondary_stage else "",
                pred.confidence_tier.value,
                f"{probs.get(EstrousStage.DIESTRUS, 0.0):.4f}",
                f"{probs.get(EstrousStage.PROESTRUS, 0.0):.4f}",
                f"{probs.get(EstrousStage.ESTRUS, 0.0):.4f}",
                f"{probs.get(EstrousStage.METESTRUS, 0.0):.4f}",
                morph.leukocytes.value,
                morph.cornified_squames.value,
                morph.nucleated_epithelial.value,
                morph.nuclear_state.value,
                morph.arrangement.value,
                morph.qc_status.value,
                seq.final_stage.value if seq.final_stage else "",
                seq.adjusted,
                seq.reason,
                pred.rationale,
            ])


def _infer_metadata_from_path(image_path: Path) -> dict[str, Any]:
    stem = image_path.stem
    match = re.match(r"^([A-Za-z0-9_-]+?)[-_]?[dD](?:ay)?[-_]?(\d+(?:\.\d+)?)$", stem)
    if match:
        subject = match.group(1)
        try:
            day = float(match.group(2))
            return {"sample_id": stem, "subject_id": subject, "day": day}
        except ValueError:
            pass
    return {"sample_id": stem, "subject_id": None, "day": None}


def _cmd_vlm_local(args: argparse.Namespace) -> int:
    from cycles.vlm_local.schema import LocalVLMRecord

    images = _vlm_input_images(args.input)
    manifest = (
        _read_vlm_sequence_manifest(args.sequence_manifest)
        if args.sequence_manifest is not None
        else None
    )

    existing_by_sample: dict[str, LocalVLMRecord] = {}
    existing_by_path: dict[str, LocalVLMRecord] = {}
    if args.resume and args.output.is_file():
        with args.output.open(encoding="utf-8") as stream:
            for _line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record_dict = json.loads(line)
                    record = LocalVLMRecord.from_dict(record_dict)
                    existing_by_sample[record.sample_id] = record
                    existing_by_path[record.image_path] = record
                except Exception:
                    continue
        if existing_by_sample:
            print(
                f"Resuming run from {args.output}: {len(existing_by_sample)} sample(s) already processed.",
                file=sys.stderr,
            )

    pipeline = _build_local_vlm_pipeline(
        args.model,
        args.adapter,
        args.model_revision,
        args.calibrator,
        reuse_prompt_prefix=not args.no_prompt_prefix_reuse,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume or not args.output.is_file():
        args.output.write_text("", encoding="utf-8")

    records: list[LocalVLMRecord] = []
    append_stream = args.output.open("a", encoding="utf-8")
    try:
        for index, image_path in enumerate(images, start=1):
            metadata = manifest.get(image_path) if manifest is not None else None
            if manifest is not None and metadata is None:
                raise ValueError(f"input image is absent from sequence manifest: {image_path}")
            metadata = metadata or _infer_metadata_from_path(image_path)
            sample_id = metadata["sample_id"]
            existing = existing_by_sample.get(sample_id) or existing_by_path.get(str(image_path))
            if existing is not None:
                records.append(existing)
                _progress(index, len(images), f"Skipped (already processed) {image_path.name}")
            else:
                record = pipeline.classify_image(
                    image_path,
                    sample_id=metadata["sample_id"],
                    subject_id=metadata["subject_id"],
                    day=metadata["day"],
                )
                records.append(record)
                append_stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                append_stream.flush()
                _progress(index, len(images), f"Classified {image_path.name}")
    finally:
        append_stream.close()

    has_longitudinal_series = any(
        getattr(r, "subject_id", None) is not None and getattr(r, "day", None) is not None
        for r in records
    )
    if manifest is not None or has_longitudinal_series:
        records = _build_temporal_reconciler(
            args.margin_threshold,
            args.adjustment_threshold,
        ).reconcile(records)
        tmp_output = args.output.with_suffix(".tmp")
        with tmp_output.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        tmp_output.replace(args.output)
    elif existing_by_sample:
        tmp_output = args.output.with_suffix(".tmp")
        with tmp_output.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        tmp_output.replace(args.output)

    if getattr(args, "csv", None):
        _export_vlm_records_csv(records, args.csv)
        print(f"Exported tabular CSV to {args.csv}")

    if len(records) == 1:
        print(_render_vlm_record_card(records[0]))
    elif len(records) > 1:
        print(_render_vlm_summary_table(records))

    print(f"Classified {len(records)} image(s); wrote {args.output}")
    return 0


def _cmd_stage(args: argparse.Namespace) -> int:
    if args.engine == "vlm":
        model = args.model or DEFAULT_VLM_MODEL
        revision = args.model_revision or PINNED_MODEL_REVISIONS.get(model)
        if revision is None:
            raise ValueError(f"--model-revision is required for custom model: {model}")
        args.model = model
        args.model_revision = revision
        exit_code = _cmd_vlm_local(args)
        if exit_code != 0:
            return exit_code
        if args.plot:
            from cycles.core.cycle import fit_cyclicity, generate_cycle_plot_data

            timestamps, stages = _read_cycle_jsonl(args.output)
            if len(stages) >= 2:
                _write_cycle_plot(generate_cycle_plot_data(timestamps, stages), args.plot)
                fit = fit_cyclicity(timestamps, stages, mouse_id="Subject")
                reg = fit.get("regularity_score", 0.0)
                length = fit.get("cycle_length_days", 0.0)
                print(
                    f"Longitudinal cyclicity: Regularity={reg:.2f}, "
                    f"Mean Period={length:.1f}d"
                )
                print(f"Generated timeline plot at {args.plot}")
            else:
                print(
                    "Single image assessed; timeline plot skipped (requires >= 2 observations).",
                    file=sys.stderr,
                )
        return 0

    elif args.engine == "cell-centric":
        from cycles.stages.cell_centric import CellCentricPipeline

        pipeline = CellCentricPipeline(detector_mode=args.detector)
        if args.input.is_dir():
            results = pipeline.process_folder(
                args.input,
                save_overlays_dir=args.save_overlays,
                recursive=False,
                progress_callback=_progress,
            )
        else:
            res = pipeline.process_image(args.input, save_overlay_path=args.save_overlays)
            results = [res] if res else []
        csv_out = (
            args.csv
            or (args.output if args.output.suffix == ".csv" else args.output.with_suffix(".csv"))
        )
        pipeline.export_results_csv(results, csv_out)
        print(f"Processed {len(results)} image(s) via Cell-Centric morphometry; wrote {csv_out}")
        if args.plot:
            from cycles.core.cycle import fit_cyclicity, generate_cycle_plot_data

            timestamps, stages = _read_cycle_csv(csv_out)
            if len(stages) >= 2:
                _write_cycle_plot(generate_cycle_plot_data(timestamps, stages), args.plot)
                print(f"Generated timeline plot at {args.plot}")
        return 0

    elif args.engine == "cnn":
        model_spec = args.model or "runs/resnet50_estrousbank_finetuned.pt"
        service = _build_cnn_service(model_spec, _device(args.device))
        if args.input.is_dir():
            result = service.classify_folder(args.input, progress_callback=_progress)
        else:
            res = service.classify_image(args.input)
            from cycles.stages.cnn import BatchClassificationResult

            result = BatchClassificationResult(
                [res], Path(args.input).parent, datetime.now().isoformat(), "resnet50"
            )
        csv_out = (
            args.csv
            or (args.output if args.output.suffix == ".csv" else args.output.with_suffix(".csv"))
        )
        service.export_results_csv(result, csv_out)
        print(f"Classified {len(result.results)} image(s) via CNN; wrote {csv_out}")
        if args.plot:
            from cycles.core.cycle import fit_cyclicity, generate_cycle_plot_data

            timestamps, stages = _read_cycle_csv(csv_out)
            if len(stages) >= 2:
                _write_cycle_plot(generate_cycle_plot_data(timestamps, stages), args.plot)
                print(f"Generated timeline plot at {args.plot}")
        return 0

    elif args.engine == "mil":
        from cycles.stages.mil import AttentionMILPipeline

        pipeline = AttentionMILPipeline()
        folder = args.input if args.input.is_dir() else args.input.parent
        result = pipeline.process_folder(folder, save_heatmaps_dir=args.save_heatmaps)
        csv_out = (
            args.csv
            or (args.output if args.output.suffix == ".csv" else args.output.with_suffix(".csv"))
        )
        pipeline.export_results_csv(result, csv_out)
        print(f"Processed {len(result.results)} image(s) via Attention-MIL; wrote {csv_out}")
        return 0

    return 0


def _cmd_vlm_calibrate(args: argparse.Namespace) -> int:
    from cycles.vlm_local.calibration import fit_and_freeze_calibrator

    result = fit_and_freeze_calibrator(
        predictions_path=args.predictions,
        labels_path=args.labels,
        output_path=args.output,
    )
    print(f"Fitted temperature: {result['temperature']:.4f} across {result['sample_count']} sample(s)")
    print(f"NLL: {result['pre_nll']:.4f} -> {result['post_nll']:.4f}")
    print(f"Brier score: {result['pre_brier']:.4f} -> {result['post_brier']:.4f}")
    print(f"Calibrator SHA-256: {result['calibrator_hash']}")
    print(f"Wrote calibrator to {args.output}")
    return 0


def _cmd_vlm_prepare_sft(args: argparse.Namespace) -> int:
    from cycles.vlm_local.datasets import prepare_sft_dataset

    summary = prepare_sft_dataset(args.source, args.input, args.output)
    print(f"Prepared {sum(summary['samples_by_split'].values())} sample(s) in {args.output}")
    return 0


def _cmd_vlm_benchmark(args: argparse.Namespace) -> int:
    from cycles.vlm_local.benchmark import benchmark_predictions

    report = benchmark_predictions(
        args.predictions,
        args.labels,
        args.output,
        baseline_predictions=args.baseline_predictions,
        require_prefill_mode=args.require_prefill_mode,
    )
    mode = report["run_configuration"]["prompt_prefix_reuse"]
    print(f"Benchmarked {report['matched_samples']} sample(s); prompt_prefix_reuse={mode}")
    print(f"Wrote {args.output / 'report.json'}")
    return 0



def _cmd_gui(args: argparse.Namespace) -> int:
    from cycles.gui.app import main as gui_main

    return gui_main(checkpoint=args.checkpoint)


def build_parser() -> argparse.ArgumentParser:
    """Create the unified command-line parser."""
    parser = argparse.ArgumentParser(
        prog="cycles",
        description="Rodent estrous phase assessment and longitudinal cycle tracking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser(
        "stage", help="Unified end-to-end automated estrous cycle phase assessment"
    )
    stage.add_argument(
        "--input",
        type=_existing_path,
        required=True,
        help="Path to cytology image or directory of images",
    )
    stage.add_argument(
        "--engine",
        choices=("vlm", "cell-centric", "cnn", "mil"),
        default="vlm",
        help="Analysis engine (default: vlm)",
    )
    stage.add_argument(
        "--output",
        type=Path,
        default=Path("runs/staging_results.jsonl"),
        help="Output JSONL or CSV file path",
    )
    stage.add_argument("--csv", type=Path, help="Optional path to export tabular CSV results")
    stage.add_argument("--plot", type=Path, help="Optional path to generate timeline plot PNG")
    stage.add_argument("--model", help="Model checkpoint path or HuggingFace ID")
    stage.add_argument("--model-revision", help="Hugging Face model commit SHA")
    stage.add_argument("--adapter", type=Path, help="Fine-tuned LoRA adapter path")
    stage.add_argument("--calibrator", type=_existing_file, help="Temperature calibrator JSON")
    stage.add_argument("--sequence-manifest", type=_existing_file)
    stage.add_argument("--margin-threshold", type=float, default=0.15)
    stage.add_argument("--adjustment-threshold", type=float, default=0.0)
    stage.add_argument(
        "--no-prompt-prefix-reuse",
        action="store_true",
        help="Cold-prefill every pass instead of reusing prefix",
    )
    stage.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted inference by skipping completed samples",
    )
    stage.add_argument(
        "--detector",
        choices=("auto", "yolo", "morphometry"),
        default="auto",
        help="Detector mode for cell-centric engine",
    )
    stage.add_argument("--save-overlays", type=Path, help="Directory to save cell overlays")
    stage.add_argument("--save-heatmaps", type=Path, help="Directory to save MIL attention heatmaps")
    stage.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    stage.set_defaults(func=_cmd_stage)

    classify = subparsers.add_parser("classify", help="Run classical CNN staging")
    classify.add_argument("--folder", type=_existing_directory, required=True)
    classify.add_argument("--model", required=True, help="Checkpoint path or backbone architecture")
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    classify.add_argument("--recursive", action="store_true")
    classify.set_defaults(func=_cmd_classify)

    cell = subparsers.add_parser("cell-centric", help="Run explainable cell-centric staging")
    cell.add_argument("--folder", type=_existing_directory, required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--detector", choices=("auto", "yolo", "morphometry"), default="auto")
    cell.add_argument("--save-overlays", type=Path)
    cell.set_defaults(func=_cmd_cell_centric)

    mil = subparsers.add_parser("mil", help="Run attention multiple-instance learning staging")
    mil.add_argument("--folder", type=_existing_directory, required=True)
    mil.add_argument("--output", type=Path, required=True)
    mil.add_argument("--save-heatmaps", type=Path)
    mil.set_defaults(func=_cmd_mil)

    cycle_fit = subparsers.add_parser("cycle-fit", help="Fit longitudinal estrous cyclicity")
    cycle_fit.add_argument("--input", type=_existing_file, required=True)
    cycle_fit.add_argument("--output", type=Path, required=True)
    cycle_fit.add_argument("--mouse-id")
    cycle_fit.set_defaults(func=_cmd_cycle_fit)

    evaluate = subparsers.add_parser("evaluate", help="Benchmark staging models")
    evaluate.add_argument("--image-dir", type=_existing_directory, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--markdown-report", type=Path)
    evaluate.add_argument("--plot-dir", type=Path)
    evaluate.add_argument("--models", default="cnn,cell-centric,mil")
    evaluate.set_defaults(func=_cmd_evaluate)

    train = subparsers.add_parser("train", help="Fine-tune ResNet-50 CNN on stage folders")
    train.add_argument("--train-dir", type=_existing_directory, required=True)
    train.add_argument("--val-dir", type=_existing_directory, required=True)
    train.add_argument("--output", type=Path, default=Path("runs/resnet50_finetuned.pt"))
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--architecture", default="resnet50")
    train.add_argument("--no-freeze", action="store_true", help="Train all layers instead of freezing backbone")
    train.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    train.set_defaults(func=_cmd_train)

    vlm_local = subparsers.add_parser(
        "vlm-local", help="Run morphology-first local MLX-VLM staging"
    )
    vlm_local.add_argument("--input", type=_existing_path, required=True)
    vlm_local.add_argument("--model", required=True)
    vlm_local.add_argument("--output", type=Path, required=True)
    vlm_local.add_argument("--csv", type=Path, help="Optional path to export tabular CSV results")
    vlm_local.add_argument("--adapter", type=Path)
    vlm_local.add_argument(
        "--model-revision",
        required=True,
        help="Immutable 40-character Hugging Face model commit SHA",
    )
    vlm_local.add_argument("--calibrator", type=_existing_file)
    vlm_local.add_argument("--sequence-manifest", type=_existing_file)
    vlm_local.add_argument("--margin-threshold", type=float, default=0.15)
    vlm_local.add_argument("--adjustment-threshold", type=float, default=0.0)
    vlm_local.add_argument(
        "--no-prompt-prefix-reuse",
        action="store_true",
        help=(
            "Cold-prefill every pass instead of reusing the shared image prefix. "
            "Roughly 3x slower; use when a run must match records produced without reuse."
        ),
    )
    vlm_local.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted inference by skipping images already recorded in --output",
    )
    vlm_local.set_defaults(func=_cmd_vlm_local)

    vlm_calibrate = subparsers.add_parser(
        "vlm-calibrate",
        help="Fit and freeze temperature calibrator from validation predictions",
    )
    vlm_calibrate.add_argument("--predictions", type=_existing_file, required=True)
    vlm_calibrate.add_argument("--labels", type=_existing_file, required=True)
    vlm_calibrate.add_argument("--output", type=Path, required=True)
    vlm_calibrate.set_defaults(func=_cmd_vlm_calibrate)

    vlm_prepare = subparsers.add_parser(
        "vlm-prepare-sft", help="Prepare auditable single-image MLX-VLM SFT data"
    )
    vlm_prepare.add_argument(
        "--source", choices=("estrousbank", "blind-teacher"), required=True
    )
    vlm_prepare.add_argument("--input", type=_existing_path, required=True)
    vlm_prepare.add_argument("--output", type=Path, required=True)
    vlm_prepare.set_defaults(func=_cmd_vlm_prepare_sft)

    vlm_benchmark = subparsers.add_parser(
        "vlm-benchmark", help="Evaluate local VLM JSONL predictions"
    )
    vlm_benchmark.add_argument("--predictions", type=_existing_file, required=True)
    vlm_benchmark.add_argument("--baseline-predictions", type=_existing_file)
    vlm_benchmark.add_argument("--labels", type=_existing_file, required=True)
    vlm_benchmark.add_argument("--output", type=Path, required=True)
    vlm_benchmark.add_argument(
        "--require-prefill-mode",
        choices=("on", "off"),
        help=(
            "Refuse to score unless every prediction was produced with this "
            "prompt_prefix_reuse setting. Pin this for any scored bakeoff: reused and "
            "cold prefill are different numeric paths and must not be mixed."
        ),
    )
    vlm_benchmark.set_defaults(func=_cmd_vlm_benchmark)

    gui = subparsers.add_parser("gui", help="Launch the PySide6 desktop application")
    gui.add_argument("--checkpoint", type=Path)
    gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f"cycles: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
