"""Comparative benchmark harness for all estrous-stage model families."""

from __future__ import annotations

import csv
import json
import math
import platform
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sklearn.metrics import classification_report

from cyclonaut.eval.metrics import (
    EvaluationMetrics,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
)

CANONICAL_STAGES = ("diestrus", "proestrus", "estrus", "metestrus")
SUPPORTED_MODELS = (
    "resnet50",
    "inception_v3",
    "vgg19",
    "mobilenet_v2",
    "cell_centric",
    "mil",
)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_CNN_MODELS = frozenset(SUPPORTED_MODELS[:4])
_STAGE_ALIASES = {
    "d": "diestrus",
    "di": "diestrus",
    "diestrus": "diestrus",
    "p": "proestrus",
    "pro": "proestrus",
    "proestrus": "proestrus",
    "e": "estrus",
    "es": "estrus",
    "estrus": "estrus",
    "m": "metestrus",
    "met": "metestrus",
    "metestrus": "metestrus",
}


class _ImageProcessor(Protocol):
    def process_image(self, image_path: Path | str, *args: object, **kwargs: object) -> object: ...


@dataclass(slots=True)
class BenchmarkReport:
    """In-memory benchmark result and the artifacts written for it."""

    timestamp: str
    hardware_info: dict[str, str]
    total_slides: int
    labeled_slides: int
    metrics_by_model: dict[str, EvaluationMetrics]
    per_class: dict[str, dict[str, dict[str, float | int]]]
    predictions: list[dict[str, Any]]
    failures: dict[str, dict[str, str]] = field(default_factory=dict)
    csv_path: Path | None = None
    json_path: Path | None = None
    markdown_path: Path | None = None
    plot_paths: list[Path] = field(default_factory=list)

    def to_dict(self, *, include_predictions: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        models: dict[str, Any] = {}
        for model_name, metrics in self.metrics_by_model.items():
            model_data = asdict(metrics)
            model_data["per_class"] = self.per_class.get(model_name, {})
            models[model_name] = model_data

        data: dict[str, Any] = {
            "timestamp": self.timestamp,
            "hardware_info": self.hardware_info,
            "total_slides": self.total_slides,
            "labeled_slides": self.labeled_slides,
            "models": models,
            "failures": self.failures,
            "artifacts": {
                "csv": str(self.csv_path) if self.csv_path else None,
                "json": str(self.json_path) if self.json_path else None,
                "markdown": str(self.markdown_path) if self.markdown_path else None,
                "plots": [str(path) for path in self.plot_paths],
            },
        }
        if include_predictions:
            data["predictions"] = self.predictions
        return data


class BenchmarkHarness:
    """Run comparable image-level benchmarks and export complete reports.

    Model objects may be injected through ``model_instances``. This is useful for
    already-loaded services and deterministic external validation. Otherwise CNN
    services are loaded from ``checkpoints`` and the cell-centric/MIL pipelines
    are constructed lazily.
    """

    def __init__(
        self,
        image_dir: Path | str | None = None,
        output_dir: Path | str = Path("runs"),
        *,
        models: list[str] | tuple[str, ...] | None = None,
        annotations_csv: Path | str | None = None,
        checkpoints: Mapping[str, Path | str] | None = None,
        device: str | None = None,
        model_instances: Mapping[str, object] | None = None,
        cell_detector: str = "auto",
    ) -> None:
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.output_dir = Path(output_dir)
        self.models = list(models) if models is not None else list(SUPPORTED_MODELS)
        unknown_models = sorted(set(self.models) - set(SUPPORTED_MODELS))
        if unknown_models:
            raise ValueError(
                f"unsupported models {unknown_models}; supported models are {list(SUPPORTED_MODELS)}"
            )
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must not contain duplicates")
        self.annotations_csv = Path(annotations_csv) if annotations_csv is not None else None
        self.checkpoints = {name: Path(path) for name, path in (checkpoints or {}).items()}
        self.device = device
        self.model_instances = dict(model_instances or {})
        unknown_instances = sorted(set(self.model_instances) - set(self.models))
        if unknown_instances:
            raise ValueError(f"model_instances contains models not selected for the run: {unknown_instances}")
        self.cell_detector = cell_detector

    @staticmethod
    def parse_ground_truth(image_path: Path | str, dataset_root: Path | str | None = None) -> str | None:
        """Infer a stage from a stage folder or common filename conventions."""

        path = Path(image_path)
        root = Path(dataset_root) if dataset_root is not None else None
        parents = list(path.parents)
        if root is not None:
            try:
                relative = path.relative_to(root)
                parents = list(relative.parents)
            except ValueError:
                pass
        for parent in parents:
            stage = _STAGE_ALIASES.get(parent.name.strip().lower())
            if stage in CANONICAL_STAGES:
                return stage

        stem = path.stem.lower()
        if stem.endswith(".ome"):
            stem = Path(stem).stem
        tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
        for token in tokens:
            if token in CANONICAL_STAGES:
                return token
        if tokens:
            # Prefix conventions are deliberately restricted to a single-letter
            # first token so ordinary words beginning with p/d/e/m do not match.
            prefix = _STAGE_ALIASES.get(tokens[0])
            if len(tokens[0]) == 1 and prefix is not None:
                return prefix
        return None

    @staticmethod
    def load_annotations(csv_path: Path | str) -> dict[str, str]:
        """Load filename/path-to-stage mappings from a flexible annotations CSV."""

        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"annotations CSV not found: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            path_column = next(
                (
                    column
                    for column in reader.fieldnames
                    if column.strip().lower() in {"filename", "image", "file", "name", "path", "image_path"}
                ),
                None,
            )
            label_column = next(
                (
                    column
                    for column in reader.fieldnames
                    if column.strip().lower()
                    in {"stage", "label", "ground_truth", "target", "estrous_stage"}
                ),
                None,
            )
            if path_column is None or label_column is None:
                raise ValueError(
                    "annotations CSV requires an image column (filename/image/path) "
                    "and a label column (stage/label/ground_truth)"
                )

            annotations: dict[str, str] = {}
            for row_number, row in enumerate(reader, start=2):
                raw_path = (row.get(path_column) or "").strip()
                raw_label = (row.get(label_column) or "").strip().lower()
                if not raw_path and not raw_label:
                    continue
                stage = _STAGE_ALIASES.get(raw_label)
                if stage is None:
                    raise ValueError(f"invalid estrous stage {raw_label!r} on CSV row {row_number}")
                normalized = Path(raw_path).as_posix().lstrip("./").lower()
                annotations[normalized] = stage
                annotations.setdefault(Path(normalized).name, stage)
        return annotations

    def run(
        self,
        *,
        output_json: Path | str | None = None,
        markdown_report: Path | str | None = None,
        plot_dir: Path | str | None = None,
    ) -> BenchmarkReport:
        """Execute all selected models with per-image failure isolation."""

        if self.image_dir is None or not self.image_dir.is_dir():
            raise NotADirectoryError(f"benchmark image directory not found: {self.image_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images = self._images()
        annotations = self.load_annotations(self.annotations_csv) if self.annotations_csv else {}
        services, initialization_failures = self._initialize_models()
        failures: dict[str, dict[str, str]] = {
            model: {"<initialization>": error} for model, error in initialization_failures.items()
        }
        rows: list[dict[str, Any]] = []

        for image_path in images:
            relative_path = image_path.relative_to(self.image_dir).as_posix()
            ground_truth = (
                annotations.get(relative_path.lower())
                or annotations.get(image_path.name.lower())
                or self.parse_ground_truth(image_path, self.image_dir)
            )
            row: dict[str, Any] = {
                "image": image_path.name,
                "relative_path": relative_path,
                "path": str(image_path),
                "ground_truth": ground_truth,
            }
            for model_name in self.models:
                service = services.get(model_name)
                if service is None:
                    error = initialization_failures.get(model_name, "model unavailable")
                    row[f"{model_name}_error"] = error
                    continue
                started = time.perf_counter()
                try:
                    result = self._process_image(service, image_path)
                    elapsed = time.perf_counter() - started
                    stage, confidence, probabilities = self._unpack_result(result)
                    row[f"{model_name}_stage"] = stage
                    row[f"{model_name}_confidence"] = confidence
                    row[f"{model_name}_latency_seconds"] = elapsed
                    row[f"{model_name}_probabilities"] = json.dumps(
                        probabilities, sort_keys=True, separators=(",", ":")
                    )
                except Exception as exc:  # noqa: BLE001 - isolation is a harness contract
                    elapsed = time.perf_counter() - started
                    message = f"{type(exc).__name__}: {exc}"
                    row[f"{model_name}_latency_seconds"] = elapsed
                    row[f"{model_name}_error"] = message
                    failures.setdefault(model_name, {})[relative_path] = message
            rows.append(row)

        metrics_by_model, per_class = self._summarize(rows)
        timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        report = BenchmarkReport(
            timestamp=timestamp,
            hardware_info=self._hardware_info(),
            total_slides=len(rows),
            labeled_slides=sum(row["ground_truth"] is not None for row in rows),
            metrics_by_model=metrics_by_model,
            per_class=per_class,
            predictions=rows,
            failures=failures,
            csv_path=self.output_dir / "benchmark_report.csv",
            json_path=Path(output_json) if output_json is not None else self.output_dir / "benchmark_report.json",
            markdown_path=(
                Path(markdown_report)
                if markdown_report is not None
                else self.output_dir / "benchmark_report.md"
            ),
        )
        self._write_csv(report.csv_path, rows)
        report.plot_paths = self._write_plots(
            metrics_by_model,
            Path(plot_dir) if plot_dir is not None else self.output_dir / "plots",
        )
        self._write_json(report.json_path, report)
        self._write_markdown(report.markdown_path, report)
        return report

    def run_benchmark(
        self,
        image_dir: Path | str,
        models: list[str] | tuple[str, ...],
        output_json: Path | str,
        markdown_report: Path | str | None = None,
        plot_dir: Path | str | None = None,
    ) -> BenchmarkReport:
        """Compatibility entrypoint used by the unified command-line interface."""

        output_path = Path(output_json)
        selected_models = []
        model_aliases = {"cnn": "resnet50", "cellcentric": "cell_centric"}
        for model_name in models:
            normalized = str(model_name).strip().lower().replace("-", "_")
            selected_models.append(model_aliases.get(normalized, normalized))
        if not selected_models:
            raise ValueError("models must contain at least one model name")
        harness = type(self)(
            image_dir=image_dir,
            output_dir=output_path.parent,
            models=selected_models,
            annotations_csv=self.annotations_csv,
            checkpoints=self.checkpoints,
            device=self.device,
            model_instances={
                name: self.model_instances[name]
                for name in selected_models
                if name in self.model_instances
            },
            cell_detector=self.cell_detector,
        )
        return harness.run(
            output_json=output_path,
            markdown_report=markdown_report,
            plot_dir=plot_dir,
        )

    def _images(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.relative_to(self.image_dir).as_posix().lower(),
        )

    def _initialize_models(self) -> tuple[dict[str, object], dict[str, str]]:
        services: dict[str, object] = {}
        failures: dict[str, str] = {}
        for model_name in self.models:
            if model_name in self.model_instances:
                services[model_name] = self.model_instances[model_name]
                continue
            try:
                services[model_name] = self._create_model(model_name)
            except Exception as exc:  # noqa: BLE001 - one unavailable family must not stop others
                failures[model_name] = f"{type(exc).__name__}: {exc}"
        return services, failures

    def _create_model(self, model_name: str) -> object:
        if model_name in _CNN_MODELS:
            checkpoint = self.checkpoints.get(model_name)
            if checkpoint is None:
                raise ValueError(f"checkpoint required for CNN model {model_name!r}")
            from cyclonaut.stages.cnn import CNNClassifierService

            service = CNNClassifierService.from_checkpoint(checkpoint, device=self.device)
            loaded_architecture = getattr(service, "architecture", model_name)
            if str(loaded_architecture) != model_name:
                raise ValueError(
                    f"checkpoint architecture is {loaded_architecture!r}, expected {model_name!r}"
                )
            return service
        if model_name == "cell_centric":
            from cyclonaut.stages.cell_centric import CellCentricPipeline

            return CellCentricPipeline(
                detector_mode=self.cell_detector,
                yolo_weights_path=self.checkpoints.get(model_name),
                device=self.device,
            )
        if model_name == "mil":
            from cyclonaut.stages.mil import AttentionMILPipeline

            return AttentionMILPipeline(
                weights_path=self.checkpoints.get(model_name),
                device=self.device,
            )
        raise ValueError(f"unsupported model: {model_name}")

    @staticmethod
    def _process_image(service: object, image_path: Path) -> object:
        processor = getattr(service, "process_image", None)
        if callable(processor):
            return processor(image_path)
        classifier = getattr(service, "classify_image", None)
        if callable(classifier):
            return classifier(image_path)
        raise TypeError(f"{type(service).__name__} exposes neither process_image nor classify_image")

    @classmethod
    def _unpack_result(cls, result: object) -> tuple[str, float, dict[str, float]]:
        raw_stage = getattr(result, "predicted_stage", None)
        if raw_stage is None:
            raw_stage = getattr(result, "stage", None)
        if raw_stage is None:
            raise ValueError("inference result does not contain predicted_stage or stage")
        stage = cls._stage_value(raw_stage)
        confidence = float(getattr(result, "confidence", 0.0))
        if not math.isfinite(confidence):
            confidence = 0.0
        raw_probabilities = getattr(result, "probabilities", None)
        if raw_probabilities is None:
            raw_probabilities = getattr(result, "stage_probabilities", {})
        probabilities = {
            cls._stage_value(label): float(probability)
            for label, probability in dict(raw_probabilities or {}).items()
            if math.isfinite(float(probability))
        }
        return stage, confidence, probabilities

    @staticmethod
    def _stage_value(stage: object) -> str:
        value = getattr(stage, "value", stage)
        normalized = str(value).strip().lower()
        return _STAGE_ALIASES.get(normalized, normalized)

    def _summarize(
        self, rows: list[dict[str, Any]]
    ) -> tuple[
        dict[str, EvaluationMetrics],
        dict[str, dict[str, dict[str, float | int]]],
    ]:
        observed = set(CANONICAL_STAGES)
        for row in rows:
            if row.get("ground_truth"):
                observed.add(str(row["ground_truth"]))
            for model_name in self.models:
                if row.get(f"{model_name}_stage"):
                    observed.add(str(row[f"{model_name}_stage"]))
        labels = list(CANONICAL_STAGES) + sorted(observed - set(CANONICAL_STAGES))

        metrics_by_model: dict[str, EvaluationMetrics] = {}
        per_class: dict[str, dict[str, dict[str, float | int]]] = {}
        for model_name in self.models:
            successful = [
                row
                for row in rows
                if row.get("ground_truth") is not None and row.get(f"{model_name}_stage") is not None
            ]
            y_true = [str(row["ground_truth"]) for row in successful]
            y_pred = [str(row[f"{model_name}_stage"]) for row in successful]
            confidences = [float(row.get(f"{model_name}_confidence", 0.0)) for row in successful]
            latencies = [float(row.get(f"{model_name}_latency_seconds", 0.0)) for row in successful]
            metrics = compute_classification_metrics(
                y_true,
                y_pred,
                confidences=confidences,
                latencies=latencies,
                labels=labels,
            )
            metrics_by_model[model_name] = metrics
            if successful:
                raw_report = classification_report(
                    y_true,
                    y_pred,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
                per_class[model_name] = {
                    label: {
                        "precision": float(raw_report[label]["precision"]),
                        "recall": float(raw_report[label]["recall"]),
                        "f1_score": float(raw_report[label]["f1-score"]),
                        "support": int(raw_report[label]["support"]),
                    }
                    for label in labels
                }
            else:
                per_class[model_name] = {
                    label: {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}
                    for label in labels
                }
        return metrics_by_model, per_class

    @staticmethod
    def _write_plots(
        metrics_by_model: dict[str, EvaluationMetrics],
        plot_dir: Path,
    ) -> list[Path]:
        if not metrics_by_model:
            return []
        paths = [
            plot_model_comparison(metrics_by_model, plot_dir / "model_comparison.png")
        ]
        for model_name, metrics in metrics_by_model.items():
            paths.append(
                plot_confusion_matrix(
                    metrics.confusion_matrix,
                    metrics.class_labels,
                    plot_dir / f"{model_name}_confusion_matrix.png",
                    title=f"{model_name} Confusion Matrix",
                )
            )
        return paths

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["image", "relative_path", "path", "ground_truth"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, report: BenchmarkReport) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")

    @staticmethod
    def _write_markdown(path: Path, report: BenchmarkReport) -> None:
        lines = [
            "# Estrous Stage Benchmark Report",
            "",
            f"- **Execution timestamp (UTC):** {report.timestamp}",
            f"- **Slides discovered:** {report.total_slides}",
            f"- **Slides with ground truth:** {report.labeled_slides}",
            "",
            "## Summary",
            "",
            "| Model | Accuracy | Balanced accuracy | Cohen's kappa | Macro F1 | Weighted F1 | Mean confidence | Latency (ms/slide) | Throughput (slides/s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for model_name, metrics in report.metrics_by_model.items():
            lines.append(
                f"| {model_name} | {metrics.accuracy:.3f} | {metrics.balanced_accuracy:.3f} "
                f"| {metrics.cohens_kappa:.3f} | {metrics.macro_f1:.3f} "
                f"| {metrics.weighted_f1:.3f} | {metrics.mean_confidence:.3f} "
                f"| {metrics.latency_ms_per_slide:.2f} | {metrics.throughput_fps:.2f} |"
            )

        lines.extend(["", "## Per-class performance", ""])
        for model_name, stages in report.per_class.items():
            lines.extend(
                [
                    f"### {model_name}",
                    "",
                    "| Stage | Precision | Recall | F1 | Support |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for stage, values in stages.items():
                lines.append(
                    f"| {stage} | {float(values['precision']):.3f} "
                    f"| {float(values['recall']):.3f} | {float(values['f1_score']):.3f} "
                    f"| {int(values['support'])} |"
                )
            lines.append("")

        lines.extend(["## Hardware and runtime", ""])
        for key, value in report.hardware_info.items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        lines.extend(["", "## Artifacts", ""])
        if report.csv_path:
            lines.append(f"- Raw predictions: `{report.csv_path.name}`")
        if report.json_path:
            lines.append(f"- Machine-readable report: `{report.json_path.name}`")
        for plot_path in report.plot_paths:
            try:
                displayed_path = plot_path.relative_to(path.parent).as_posix()
            except ValueError:
                displayed_path = plot_path.as_posix()
            lines.append(f"- Plot: `{displayed_path}`")
        if report.failures:
            lines.extend(["", "## Failures", ""])
            for model_name, model_failures in report.failures.items():
                lines.append(f"- **{model_name}:** {len(model_failures)} failure(s)")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _hardware_info(self) -> dict[str, str]:
        info = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "requested_device": self.device or "auto",
        }
        try:
            import torch

            info["pytorch"] = torch.__version__
            if torch.cuda.is_available():
                info["accelerator"] = torch.cuda.get_device_name(0)
                info["cuda"] = torch.version.cuda or "unknown"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                info["accelerator"] = "Apple Metal Performance Shaders (MPS)"
            else:
                info["accelerator"] = "CPU"
        except ImportError:
            info["pytorch"] = "unavailable"
            info["accelerator"] = "unknown"
        info["executable"] = sys.executable
        return info
