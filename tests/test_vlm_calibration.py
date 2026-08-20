from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cycles.core.types import EstrousStage
from cycles.vlm_local.calibration import TemperatureCalibrator


def test_temperature_calibrator_transforms_all_four_logits() -> None:
    calibrator = TemperatureCalibrator(temperature=2.0)
    scores = {
        EstrousStage.DIESTRUS: 0.0,
        EstrousStage.PROESTRUS: 0.0,
        EstrousStage.ESTRUS: 2.0,
        EstrousStage.METESTRUS: 0.0,
    }

    probabilities = calibrator.transform(scores)

    expected_estrus = math.e / (math.e + 3)
    assert probabilities[EstrousStage.ESTRUS] == pytest.approx(expected_estrus)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_temperature_calibrator_round_trip_is_versioned(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    original = TemperatureCalibrator(temperature=1.75)

    original.save(path)
    restored = TemperatureCalibrator.load(path)

    assert restored == original
    assert json.loads(path.read_text())["schema_version"] == "1.0"
    assert len(restored.sha256) == 64


def test_temperature_fit_reduces_validation_negative_log_likelihood() -> None:
    scores = [
        {EstrousStage.DIESTRUS: 8.0, EstrousStage.PROESTRUS: 0.0, EstrousStage.ESTRUS: 0.0, EstrousStage.METESTRUS: 0.0},
        {EstrousStage.DIESTRUS: 0.0, EstrousStage.PROESTRUS: 8.0, EstrousStage.ESTRUS: 0.0, EstrousStage.METESTRUS: 0.0},
        {EstrousStage.DIESTRUS: 8.0, EstrousStage.PROESTRUS: 0.0, EstrousStage.ESTRUS: 0.0, EstrousStage.METESTRUS: 0.0},
    ]
    labels = [EstrousStage.DIESTRUS, EstrousStage.PROESTRUS, EstrousStage.PROESTRUS]

    fitted = TemperatureCalibrator.fit(scores, labels)

    assert fitted.temperature > 1.0
    assert fitted.negative_log_likelihood(scores, labels) < TemperatureCalibrator().negative_log_likelihood(scores, labels)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_temperature_calibrator_rejects_non_finite_scores(invalid: float) -> None:
    scores = {
        EstrousStage.DIESTRUS: 0.0,
        EstrousStage.PROESTRUS: 0.0,
        EstrousStage.ESTRUS: invalid,
        EstrousStage.METESTRUS: 0.0,
    }

    with pytest.raises(ValueError, match="finite"):
        TemperatureCalibrator().transform(scores)
