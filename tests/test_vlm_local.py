from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from cycles.core.types import EstrousStage
from cycles.vlm_local.calibration import TemperatureCalibrator
from cycles.vlm_local.pipeline import LocalVLMPipeline
from cycles.vlm_local.schema import ImagePrediction, QCStatus
from cycles.vlm_local.views import build_view_pack


class ScriptedBackend:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []
        self.view_counts: list[int] = []

    @property
    def provenance(self) -> dict[str, str]:
        return {"model_id": "test/model", "model_revision": "abc123", "adapter_hash": "none"}

    def generate(self, images: Sequence[Image.Image], prompt: str) -> str:
        self.prompts.append(prompt)
        self.view_counts.append(len(images))
        return next(self.responses)


def _sample_image(path: Path) -> Path:
    image = Image.new("RGB", (100, 80), (20, 80, 160))
    image.save(path, format="PNG")
    return path


def _morphology_response() -> str:
    return json.dumps(
        {
            "cornified_squames": "dominant",
            "nucleated_epithelial": "rare",
            "leukocytes": "absent",
            "nuclear_state": "anucleate",
            "arrangement": "sheets",
            "artifacts": ["mucus"],
            "qc_status": "usable",
            "qc_reasons": [],
            "evidence": ["Broad sheets of anucleate squames are visible."],
        }
    )


def _stage_response() -> str:
    return json.dumps(
        {
            "raw_scores": {
                "diestrus": -2.0,
                "proestrus": -1.0,
                "estrus": 3.0,
                "metestrus": 0.0,
            },
            "probabilities": {
                "diestrus": 0.02,
                "proestrus": 0.04,
                "estrus": 0.89,
                "metestrus": 0.05,
            },
            "primary_stage": "estrus",
            "secondary_stage": "metestrus",
            "confidence_tier": "high",
            "rationale": "Dominant cornified sheets with no convincing leukocytes support estrus.",
        }
    )


def test_view_pack_contains_overview_and_four_deterministic_quadrants(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")

    first = build_view_pack(image_path)
    second = build_view_pack(image_path)

    assert [view.label for view in first] == [
        "overview",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]
    assert [view.image.size for view in first] == [view.image.size for view in second]
    assert first[0].image.size == (100, 80)
    assert all(view.image.mode == "RGB" for view in first)


def test_pipeline_runs_stage_blind_morphology_before_stage_scoring(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")
    backend = ScriptedBackend([_morphology_response(), _stage_response()])
    pipeline = LocalVLMPipeline(backend=backend, prompt_version="test-v1")

    record = pipeline.classify_image(image_path, sample_id="mouse-1-day-3")

    assert backend.view_counts == [5, 5]
    assert "Do not assign an estrous stage" in backend.prompts[0]
    assert "four canonical stages" in backend.prompts[1]
    assert record.morphology.cornified_squames.value == "dominant"
    assert record.image_prediction.primary_stage.value == "estrus"
    assert record.image_prediction.secondary_stage.value == "metestrus"
    assert record.sequence_prediction.final_stage.value == "estrus"
    assert record.sequence_prediction.adjusted is False
    assert len(record.image_sha256) == 64
    assert record.provenance["model_id"] == "test/model"
    assert record.provenance["prompt_version"] == "test-v1"


def test_pipeline_repairs_malformed_stage_json_once(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")
    backend = ScriptedBackend([_morphology_response(), "not json", _stage_response()])

    record = LocalVLMPipeline(backend=backend).classify_image(image_path)

    assert len(backend.prompts) == 3
    assert "Repair the following response" in backend.prompts[-1]
    assert record.image_prediction.primary_stage.value == "estrus"


def test_pipeline_marks_persistently_invalid_output_ungradable(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")
    backend = ScriptedBackend([_morphology_response(), "not json", "still not json"])

    record = LocalVLMPipeline(backend=backend).classify_image(image_path)

    assert record.morphology.qc_status is QCStatus.UNGRADABLE
    assert record.image_prediction.primary_stage is None
    assert record.sequence_prediction.final_stage is None
    assert "invalid_model_output" in record.morphology.qc_reasons


def test_record_serialization_preserves_all_stage_probabilities(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")
    record = LocalVLMPipeline(
        backend=ScriptedBackend([_morphology_response(), _stage_response()])
    ).classify_image(image_path)

    payload = record.to_dict()

    assert payload["schema_version"] == "3.0"
    assert set(payload["image_prediction"]["probabilities"]) == {
        "diestrus",
        "proestrus",
        "estrus",
        "metestrus",
    }
    assert payload["morphology"]["qc_status"] == "usable"


def test_record_round_trip_restores_typed_schema(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")
    original = LocalVLMPipeline(
        backend=ScriptedBackend([_morphology_response(), _stage_response()])
    ).classify_image(image_path)

    restored = type(original).from_dict(original.to_dict())

    assert restored == original


def test_pipeline_applies_validation_calibrator_to_raw_scores(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")
    calibrator = TemperatureCalibrator(temperature=2.0)
    pipeline = LocalVLMPipeline(
        backend=ScriptedBackend([_morphology_response(), _stage_response()]),
        calibrator=calibrator,
    )

    record = pipeline.classify_image(image_path)

    assert record.image_prediction.probabilities[record.image_prediction.primary_stage] == pytest.approx(0.6942, abs=1e-4)
    assert record.image_prediction.primary_stage is not None
    assert record.image_prediction.primary_stage.value == "estrus"
    assert record.provenance["calibrator_hash"] == calibrator.sha256


def test_image_prediction_accepts_confident_one_hot_distribution() -> None:
    """A one-hot distribution ties three stages at zero.

    Which tied stage lands in a given sorted slot is arbitrary, so the secondary
    stage is validated against the second-largest probability *value*. Requiring
    identity with a particular tied key would reject every confident prediction.
    """
    payload = {
        "raw_scores": {"diestrus": 0.0, "proestrus": 0.0, "estrus": 0.0, "metestrus": 1.0},
        "probabilities": {"diestrus": 0.0, "proestrus": 0.0, "estrus": 0.0, "metestrus": 1.0},
        "primary_stage": "metestrus",
        "secondary_stage": "estrus",
        "confidence_tier": "high",
        "rationale": "Ghost nuclei with leukocyte infiltration.",
    }

    prediction = ImagePrediction.from_dict(payload)

    assert prediction.primary_stage is EstrousStage.METESTRUS
    assert prediction.secondary_stage is EstrousStage.ESTRUS


def test_image_prediction_still_rejects_primary_below_the_mode() -> None:
    payload = {
        "raw_scores": {"diestrus": 0.1, "proestrus": 0.1, "estrus": 0.1, "metestrus": 0.7},
        "probabilities": {"diestrus": 0.1, "proestrus": 0.1, "estrus": 0.1, "metestrus": 0.7},
        "primary_stage": "diestrus",
        "secondary_stage": "metestrus",
        "confidence_tier": "high",
        "rationale": "Inconsistent with the reported distribution.",
    }

    with pytest.raises(ValueError, match="largest probability"):
        ImagePrediction.from_dict(payload)


def test_image_prediction_rejects_secondary_that_is_not_runner_up() -> None:
    payload = {
        "raw_scores": {"diestrus": 0.05, "proestrus": 0.15, "estrus": 0.1, "metestrus": 0.7},
        "probabilities": {"diestrus": 0.05, "proestrus": 0.15, "estrus": 0.1, "metestrus": 0.7},
        "primary_stage": "metestrus",
        "secondary_stage": "diestrus",
        "confidence_tier": "medium",
        "rationale": "Diestrus is the weakest stage, not the runner-up.",
    }

    with pytest.raises(ValueError, match="second-largest probability"):
        ImagePrediction.from_dict(payload)


def test_quadrant_max_edge_caps_quadrants_and_leaves_overview_alone(tmp_path: Path) -> None:
    """Vision prefill scales with pixel count, so quadrant size is the main cost lever."""
    image_path = _sample_image(tmp_path / "slide.png")

    full = build_view_pack(image_path)
    capped = build_view_pack(image_path, quadrant_max_edge=20)

    assert full[0].image.size == capped[0].image.size, "overview must be unaffected"
    assert all(max(view.image.size) <= 20 for view in capped[1:])
    assert all(max(view.image.size) > 20 for view in full[1:])
    assert [view.label for view in capped] == [view.label for view in full]


def test_quadrant_max_edge_rejects_nonpositive_values(tmp_path: Path) -> None:
    image_path = _sample_image(tmp_path / "slide.png")

    with pytest.raises(ValueError, match="quadrant_max_edge"):
        build_view_pack(image_path, quadrant_max_edge=0)
