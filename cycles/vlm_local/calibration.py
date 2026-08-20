"""Validation-fitted temperature calibration for local stage scores."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.optimize import minimize_scalar

from cycles.core.types import EstrousStage


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    @classmethod
    def fit(
        cls,
        scores: list[dict[EstrousStage, float]],
        labels: list[EstrousStage],
    ) -> TemperatureCalibrator:
        if not scores or len(scores) != len(labels):
            raise ValueError("scores and labels must be non-empty and have equal length")

        def objective(log_temperature: float) -> float:
            return cls(math.exp(log_temperature)).negative_log_likelihood(scores, labels)

        result = minimize_scalar(
            objective,
            bounds=(math.log(0.05), math.log(20.0)),
            method="bounded",
        )
        if not result.success:
            raise RuntimeError(f"temperature fitting failed: {result.message}")
        return cls(math.exp(float(result.x)))

    def transform(
        self,
        scores: dict[EstrousStage, float],
    ) -> dict[EstrousStage, float]:
        stages = EstrousStage.canonical_stages()
        if set(scores) != set(stages):
            raise ValueError("scores must contain exactly the four canonical stages")
        scaled = {stage: float(scores[stage]) / self.temperature for stage in stages}
        if any(not math.isfinite(value) for value in scaled.values()):
            raise ValueError("scores must be finite")
        maximum = max(scaled.values())
        exponentials = {stage: math.exp(value - maximum) for stage, value in scaled.items()}
        total = sum(exponentials.values())
        return {stage: value / total for stage, value in exponentials.items()}

    def negative_log_likelihood(
        self,
        scores: list[dict[EstrousStage, float]],
        labels: list[EstrousStage],
    ) -> float:
        if not scores or len(scores) != len(labels):
            raise ValueError("scores and labels must be non-empty and have equal length")
        losses = []
        for sample_scores, label in zip(scores, labels, strict=True):
            probability = self.transform(sample_scores)[label]
            losses.append(-math.log(max(probability, 1e-12)))
        return sum(losses) / len(losses)

    def brier_score(
        self,
        scores: list[dict[EstrousStage, float]],
        labels: list[EstrousStage],
    ) -> float:
        if not scores or len(scores) != len(labels):
            raise ValueError("scores and labels must be non-empty and have equal length")
        stages = EstrousStage.canonical_stages()
        losses = []
        for sample_scores, label in zip(scores, labels, strict=True):
            probabilities = self.transform(sample_scores)
            losses.append(
                sum((probabilities[stage] - float(stage == label)) ** 2 for stage in stages)
            )
        return sum(losses) / len(losses)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._encoded()).hexdigest()

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._encoded() + b"\n")

    @classmethod
    def load(cls, path: Path | str) -> TemperatureCalibrator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported calibrator schema")
        return cls(float(payload["temperature"]))

    def _encoded(self) -> bytes:
        return json.dumps(
            {"schema_version": "1.0", "temperature": self.temperature},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


def fit_and_freeze_calibrator(
    predictions_path: Path | str,
    labels_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    pred_path = Path(predictions_path)
    label_path = Path(labels_path)
    out_path = Path(output_path)

    scores_by_sample: dict[str, dict[EstrousStage, float]] = {}
    with pred_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"missing sample_id on line {line_number} of {pred_path}")
            img_pred = payload.get("image_prediction") or {}
            raw_scores = img_pred.get("raw_scores")
            if not isinstance(raw_scores, dict):
                raise ValueError(f"missing raw_scores on line {line_number} of {pred_path}")
            stages = EstrousStage.canonical_stages()
            if set(raw_scores) != {stage.value for stage in stages}:
                raise ValueError(f"raw_scores must contain all 4 canonical stages on line {line_number}")
            parsed = {stage: float(raw_scores[stage.value]) for stage in stages}
            if any(not math.isfinite(val) for val in parsed.values()):
                raise ValueError(f"raw_scores must be finite on line {line_number}")
            if sample_id in scores_by_sample:
                raise ValueError(f"duplicate sample_id on line {line_number}: {sample_id}")
            scores_by_sample[sample_id] = parsed

    labels_by_sample: dict[str, EstrousStage] = {}
    with label_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "stage"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"labels CSV requires sample_id and stage columns: {label_path}")
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            raw_stage = (row.get("stage") or "").strip().lower()
            if not sample_id or not raw_stage:
                raise ValueError(f"invalid label on row {row_number} of {label_path}")
            try:
                stage = EstrousStage(raw_stage)
            except ValueError as exc:
                raise ValueError(f"unknown stage {raw_stage!r} on row {row_number}") from exc
            if sample_id in labels_by_sample:
                raise ValueError(f"duplicate sample_id on row {row_number} of {label_path}: {sample_id}")
            labels_by_sample[sample_id] = stage

    if set(scores_by_sample) != set(labels_by_sample):
        missing = len(set(labels_by_sample) - set(scores_by_sample))
        extra = len(set(scores_by_sample) - set(labels_by_sample))
        raise ValueError(
            f"prediction and label sample_id coverage differs: {missing} missing, {extra} extra"
        )

    matched_samples = sorted(labels_by_sample.keys())
    scores_list = [scores_by_sample[s] for s in matched_samples]
    labels_list = [labels_by_sample[s] for s in matched_samples]

    uncalibrated = TemperatureCalibrator(temperature=1.0)
    pre_nll = uncalibrated.negative_log_likelihood(scores_list, labels_list)
    pre_brier = uncalibrated.brier_score(scores_list, labels_list)

    calibrator = TemperatureCalibrator.fit(scores_list, labels_list)
    post_nll = calibrator.negative_log_likelihood(scores_list, labels_list)
    post_brier = calibrator.brier_score(scores_list, labels_list)

    calibrator.save(out_path)

    return {
        "temperature": calibrator.temperature,
        "pre_nll": pre_nll,
        "post_nll": post_nll,
        "pre_brier": pre_brier,
        "post_brier": post_brier,
        "calibrator_hash": calibrator.sha256,
        "sample_count": len(matched_samples),
        "output_path": str(out_path),
    }
