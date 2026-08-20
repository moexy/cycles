"""Validation-fitted temperature calibration for local stage scores."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

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
