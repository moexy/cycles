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


def test_fit_and_freeze_calibrator_saves_calibrator_file(tmp_path: Path) -> None:
    from cycles.vlm_local.calibration import fit_and_freeze_calibrator

    predictions_path = tmp_path / "val_predictions.jsonl"
    labels_path = tmp_path / "val_labels.csv"
    output_path = tmp_path / "fitted_calibrator.json"

    rows = [
        {"sample_id": "s1", "image_prediction": {"raw_scores": {"diestrus": 8.0, "proestrus": 0.0, "estrus": 0.0, "metestrus": 0.0}}},
        {"sample_id": "s2", "image_prediction": {"raw_scores": {"diestrus": 0.0, "proestrus": 8.0, "estrus": 0.0, "metestrus": 0.0}}},
        {"sample_id": "s3", "image_prediction": {"raw_scores": {"diestrus": 8.0, "proestrus": 0.0, "estrus": 0.0, "metestrus": 0.0}}},
    ]
    predictions_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    labels_path.write_text("sample_id,stage\ns1,diestrus\ns2,proestrus\ns3,proestrus\n", encoding="utf-8")

    result = fit_and_freeze_calibrator(predictions_path, labels_path, output_path)

    assert result["temperature"] > 1.0
    assert result["post_nll"] < result["pre_nll"]
    assert result["sample_count"] == 3
    assert output_path.is_file()

    loaded = TemperatureCalibrator.load(output_path)
    assert loaded.temperature == pytest.approx(result["temperature"])
    assert loaded.sha256 == result["calibrator_hash"]

