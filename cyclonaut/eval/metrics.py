"""Metrics and publication-ready plots for estrous-stage evaluation."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(slots=True)
class EvaluationMetrics:
    """Aggregate classification quality and inference performance."""

    accuracy: float
    balanced_accuracy: float
    cohens_kappa: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    mean_confidence: float
    latency_ms_per_slide: float
    throughput_fps: float
    confusion_matrix: list[list[int]]
    class_labels: list[str]


def _validate_parallel_values(
    values: Sequence[object] | None,
    expected_length: int,
    name: str,
) -> None:
    if values is not None and len(values) != expected_length:
        raise ValueError(
            f"{name} must contain one value per prediction: "
            f"expected {expected_length}, received {len(values)}"
        )


def _finite_mean(values: Sequence[float] | None) -> float:
    if not values:
        return 0.0
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float] | None = None,
    latencies: list[float] | None = None,
    labels: list[str] | None = None,
) -> EvaluationMetrics:
    """Compute robust classification and timing metrics.

    ``latencies`` are elapsed seconds per slide (as returned by
    :func:`time.perf_counter`); the report converts their mean to milliseconds.
    Empty inputs return zero-valued scores and a correctly shaped zero confusion
    matrix. Parallel arrays must have equal lengths so that silently corrupted
    benchmark results cannot be produced.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length: "
            f"received {len(y_true)} and {len(y_pred)}"
        )
    _validate_parallel_values(confidences, len(y_pred), "confidences")
    _validate_parallel_values(latencies, len(y_pred), "latencies")

    if labels is None:
        class_labels = sorted(set(y_true) | set(y_pred))
    else:
        class_labels = list(dict.fromkeys(str(label) for label in labels))
        unknown = sorted((set(y_true) | set(y_pred)) - set(class_labels))
        if unknown:
            raise ValueError(f"labels does not include observed classes: {unknown}")

    if not y_true:
        size = len(class_labels)
        return EvaluationMetrics(
            accuracy=0.0,
            balanced_accuracy=0.0,
            cohens_kappa=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            mean_confidence=0.0,
            latency_ms_per_slide=0.0,
            throughput_fps=0.0,
            confusion_matrix=[[0 for _ in range(size)] for _ in range(size)],
            class_labels=class_labels,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = float(accuracy_score(y_true, y_pred))
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=class_labels))
        macro_f1 = float(
            f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)
        )
        macro_precision = float(
            precision_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        )
        macro_recall = float(
            recall_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        )
        matrix = confusion_matrix(y_true, y_pred, labels=class_labels).astype(int).tolist()

    # Kappa is undefined for a single constant class. Reporting 0.0 keeps JSON
    # standards-compliant while accurately conveying no beyond-chance evidence.
    cohens_kappa = kappa if math.isfinite(kappa) else 0.0
    mean_confidence = _finite_mean(confidences)
    valid_latencies = (
        [float(value) for value in latencies if math.isfinite(float(value)) and float(value) >= 0.0]
        if latencies
        else []
    )
    mean_latency_s = float(np.mean(valid_latencies)) if valid_latencies else 0.0

    return EvaluationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        cohens_kappa=cohens_kappa,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        mean_confidence=mean_confidence,
        latency_ms_per_slide=mean_latency_s * 1_000.0,
        throughput_fps=(1.0 / mean_latency_s) if mean_latency_s > 0.0 else 0.0,
        confusion_matrix=matrix,
        class_labels=class_labels,
    )


def plot_confusion_matrix(
    cm: list[list[int]],
    class_labels: list[str],
    output_path: Path | str,
    title: str = "Confusion Matrix",
) -> Path:
    """Render and save an annotated confusion-matrix heatmap."""

    matrix = np.asarray(cm, dtype=np.int64)
    expected_shape = (len(class_labels), len(class_labels))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"confusion matrix shape must be {expected_shape}, received {matrix.shape}"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    side = max(5.0, 0.85 * len(class_labels) + 2.5)
    figure, axis = plt.subplots(figsize=(side, side), dpi=150)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_labels))
    axis.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted stage",
        ylabel="True stage",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = float(matrix.max()) / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_model_comparison(
    metrics_by_model: dict[str, EvaluationMetrics],
    output_path: Path | str,
) -> Path:
    """Save grouped quality-score bars with latency on a secondary axis."""

    if not metrics_by_model:
        raise ValueError("metrics_by_model must contain at least one model")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(metrics_by_model)
    x = np.arange(len(model_names), dtype=float)
    width = 0.2

    figure_width = max(8.5, len(model_names) * 1.35 + 3.0)
    figure, score_axis = plt.subplots(figsize=(figure_width, 5.5), dpi=150)
    score_series = (
        ("Accuracy", [metrics_by_model[name].accuracy for name in model_names], "#2563eb"),
        (
            "Balanced Acc",
            [metrics_by_model[name].balanced_accuracy for name in model_names],
            "#059669",
        ),
        ("Macro F1", [metrics_by_model[name].macro_f1 for name in model_names], "#7c3aed"),
    )
    for index, (label, values, color) in enumerate(score_series):
        score_axis.bar(x + (index - 1.5) * width, values, width, label=label, color=color)

    latency_axis = score_axis.twinx()
    latency_values = [metrics_by_model[name].latency_ms_per_slide for name in model_names]
    latency_axis.bar(
        x + 1.5 * width,
        latency_values,
        width,
        label="Latency",
        color="#f59e0b",
        alpha=0.8,
    )

    score_axis.set_ylabel("Classification score")
    score_axis.set_ylim(0.0, 1.05)
    score_axis.set_xticks(x)
    score_axis.set_xticklabels(model_names, rotation=20, ha="right")
    score_axis.set_title("Model Performance Comparison")
    score_axis.grid(axis="y", linestyle="--", alpha=0.3)
    latency_axis.set_ylabel("Latency (ms / slide)")
    latency_axis.set_ylim(bottom=0.0)
    handles, labels = score_axis.get_legend_handles_labels()
    latency_handles, latency_labels = latency_axis.get_legend_handles_labels()
    score_axis.legend(handles + latency_handles, labels + latency_labels, loc="upper left")

    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination
