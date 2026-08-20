"""Evaluation and calibration reports for v3 JSONL predictions."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cycles.eval.metrics import compute_classification_metrics

CANONICAL = ["diestrus", "proestrus", "estrus", "metestrus"]
SUBGROUP_COLUMNS = ("species", "stain", "lab")


def benchmark_predictions(
    predictions_path: Path | str,
    labels_path: Path | str,
    output_dir: Path | str,
    *,
    baseline_predictions: Path | str | None = None,
    bootstrap_samples: int = 2_000,
    min_subgroup_size: int = 50,
    random_seed: int = 17,
) -> dict[str, Any]:
    predictions = _read_predictions(Path(predictions_path))
    labels = _read_labels(Path(labels_path))
    if set(predictions) != set(labels):
        missing = len(set(labels) - set(predictions))
        extra = len(set(predictions) - set(labels))
        raise ValueError(
            f"prediction and label sample_id coverage differs: {missing} missing, {extra} extra"
        )
    matched: list[tuple[dict[str, str], dict[str, Any]]] = []
    for sample_id, label in labels.items():
        prediction = predictions.get(sample_id)
        if prediction is not None:
            matched.append((label, prediction))
    if not matched:
        raise ValueError("predictions and labels have no matching sample_id values")

    y_true = [label["stage"] for label, _ in matched]
    y_pred = [prediction["primary_stage"] for _, prediction in matched]
    probabilities = [prediction["probabilities"] for _, prediction in matched]
    confidences = [max(values.values()) for values in probabilities]
    metrics = compute_classification_metrics(
        y_true, y_pred, confidences=confidences, labels=CANONICAL
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "matched_samples": len(matched),
        "unmatched_labels": len(labels) - len(matched),
        "unmatched_predictions": len(predictions) - len(matched),
        "metrics": asdict(metrics),
        "calibration": {
            "ece": round(_ece(y_true, y_pred, confidences), 10),
            "brier_score": round(_brier(y_true, probabilities), 10),
        },
        "subgroups": _subgroup_metrics(matched),
        "gates": {
            "calibration_ece": round(_ece(y_true, y_pred, confidences), 10) <= 0.10,
        },
    }
    if baseline_predictions is not None:
        baseline = _read_predictions(Path(baseline_predictions))
        if set(baseline) != set(labels):
            missing = len(set(labels) - set(baseline))
            extra = len(set(baseline) - set(labels))
            raise ValueError(
                f"baseline and label sample_id coverage differs: {missing} missing, {extra} extra"
            )
        baseline_pred = [baseline[label["sample_id"]] for label, _ in matched]
        baseline_metrics = compute_classification_metrics(
            y_true,
            [prediction["primary_stage"] for prediction in baseline_pred],
            labels=CANONICAL,
        )
        confidence_interval = _paired_group_bootstrap(
            matched,
            baseline,
            samples=bootstrap_samples,
            seed=random_seed,
        )
        subgroup_comparison = _subgroup_comparison(
            matched,
            baseline,
            min_size=min_subgroup_size,
        )
        report["comparison"] = {
            "baseline_metrics": asdict(baseline_metrics),
            "macro_f1_delta": round(metrics.macro_f1 - baseline_metrics.macro_f1, 10),
            "group_bootstrap_95_ci": confidence_interval,
            "subgroups": subgroup_comparison,
        }
        report["gates"].update(
            relative_improvement=confidence_interval[0] > 0,
            subgroup_safety=not subgroup_comparison["violations"],
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _paired_group_bootstrap(
    matched: list[tuple[dict[str, str], dict[str, Any]]],
    baseline: dict[str, dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (label, _) in enumerate(matched):
        grouped[label.get("group_id") or label["sample_id"]].append(index)
    group_names = sorted(grouped)
    generator = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(samples):
        selected_groups = generator.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in selected_groups for index in grouped[str(group)]]
        truth = [matched[index][0]["stage"] for index in indices]
        candidate_stages = [matched[index][1]["primary_stage"] for index in indices]
        baseline_stages = [
            baseline[matched[index][0]["sample_id"]]["primary_stage"] for index in indices
        ]
        candidate_f1 = compute_classification_metrics(
            truth, candidate_stages, labels=CANONICAL
        ).macro_f1
        baseline_f1 = compute_classification_metrics(
            truth, baseline_stages, labels=CANONICAL
        ).macro_f1
        deltas.append(candidate_f1 - baseline_f1)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return [round(float(lower), 10), round(float(upper), 10)]


def _subgroup_comparison(
    matched: list[tuple[dict[str, str], dict[str, Any]]],
    baseline: dict[str, dict[str, Any]],
    *,
    min_size: int,
) -> dict[str, Any]:
    if min_size < 1:
        raise ValueError("min_subgroup_size must be positive")
    comparisons: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    for column in SUBGROUP_COLUMNS:
        grouped: dict[str, list[tuple[dict[str, str], dict[str, Any]]]] = defaultdict(list)
        for row in matched:
            value = row[0].get(column, "").strip()
            if value:
                grouped[value].append(row)
        comparisons[column] = {}
        for value, rows in sorted(grouped.items()):
            if len(rows) < min_size:
                continue
            truth = [label["stage"] for label, _ in rows]
            candidate_stages = [prediction["primary_stage"] for _, prediction in rows]
            baseline_stages = [baseline[label["sample_id"]]["primary_stage"] for label, _ in rows]
            candidate_f1 = compute_classification_metrics(
                truth, candidate_stages, labels=CANONICAL
            ).macro_f1
            baseline_f1 = compute_classification_metrics(
                truth, baseline_stages, labels=CANONICAL
            ).macro_f1
            delta = candidate_f1 - baseline_f1
            item = {"count": len(rows), "macro_f1_delta": round(delta, 10)}
            comparisons[column][value] = item
            if delta < -0.03:
                violations.append({"column": column, "value": value, **item})
    return {"minimum_size": min_size, "comparisons": comparisons, "violations": violations}


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("sample_id", "")).strip()
            image_prediction = payload.get("image_prediction")
            if not sample_id or not isinstance(image_prediction, dict):
                raise ValueError(f"invalid prediction row {line_number}")
            if sample_id in predictions:
                raise ValueError(f"duplicate prediction sample_id on row {line_number}: {sample_id}")
            primary = image_prediction.get("primary_stage")
            probabilities = image_prediction.get("probabilities")
            if primary not in CANONICAL or not isinstance(probabilities, dict):
                raise ValueError(f"invalid stage prediction on row {line_number}")
            if set(probabilities) != set(CANONICAL):
                raise ValueError(f"prediction row {line_number} lacks all four probabilities")
            parsed = {stage: float(probabilities[stage]) for stage in CANONICAL}
            if any(not math.isfinite(value) for value in parsed.values()):
                raise ValueError(f"probabilities must be finite on row {line_number}")
            if any(value < 0 for value in parsed.values()) or sum(parsed.values()) <= 0:
                raise ValueError(f"invalid probabilities on row {line_number}")
            total = sum(parsed.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"probabilities must sum to 1 on row {line_number}")
            maximum = max(parsed.values())
            if not math.isclose(parsed[primary], maximum, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"primary_stage must match the largest probability on row {line_number}"
                )
            predictions[sample_id] = {
                "primary_stage": primary,
                "probabilities": parsed,
            }
    return predictions


def _read_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "stage"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("labels CSV requires sample_id and stage columns")
        labels: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            stage = (row.get("stage") or "").strip().lower()
            if not sample_id or stage not in CANONICAL:
                raise ValueError(f"invalid label on row {row_number}")
            if sample_id in labels:
                raise ValueError(f"duplicate label sample_id on row {row_number}: {sample_id}")
            labels[sample_id] = {key: (value or "").strip() for key, value in row.items()}
            labels[sample_id]["stage"] = stage
    return labels


def _ece(y_true: list[str], y_pred: list[str], confidences: list[float]) -> float:
    total = len(y_true)
    error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if (lower <= confidence <= upper)
            if upper >= 1.0
            or (lower <= confidence < upper)
        ]
        if not indices:
            continue
        accuracy = np.mean([y_true[index] == y_pred[index] for index in indices])
        confidence = np.mean([confidences[index] for index in indices])
        error += len(indices) / total * abs(float(accuracy) - float(confidence))
    return float(error)


def _brier(y_true: list[str], probabilities: list[dict[str, float]]) -> float:
    values = []
    for truth, predicted in zip(y_true, probabilities, strict=True):
        values.append(sum((predicted[stage] - float(stage == truth)) ** 2 for stage in CANONICAL))
    return float(np.mean(values))


def _subgroup_metrics(
    matched: list[tuple[dict[str, str], dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for column in SUBGROUP_COLUMNS:
        groups: dict[str, list[tuple[dict[str, str], dict[str, Any]]]] = defaultdict(list)
        for label, prediction in matched:
            value = label.get(column, "").strip()
            if value:
                groups[value].append((label, prediction))
        output[column] = {}
        for value, rows in sorted(groups.items()):
            metrics = compute_classification_metrics(
                [label["stage"] for label, _ in rows],
                [prediction["primary_stage"] for _, prediction in rows],
                labels=CANONICAL,
            )
            output[column][value] = {
                "count": len(rows),
                "macro_f1": metrics.macro_f1,
                "accuracy": metrics.accuracy,
            }
    return output
