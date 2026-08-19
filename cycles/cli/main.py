"""Unified command-line entry point for all cycles analysis modes."""

from __future__ import annotations

import argparse
import csv
import json
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

    timestamps, stages = _read_cycle_csv(args.input, mouse_id=args.mouse_id)
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
