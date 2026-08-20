"""Two-pass, morphology-first local VLM inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from cycles.vlm_local.backend import VLMBackend
from cycles.vlm_local.calibration import TemperatureCalibrator
from cycles.vlm_local.prompts import MORPHOLOGY_PROMPT, PROMPT_VERSION, repair_prompt, stage_prompt
from cycles.vlm_local.schema import (
    ConfidenceTier,
    ImagePrediction,
    LocalVLMRecord,
    MorphologyObservation,
    SequencePrediction,
    StageEvidence,
)
from cycles.vlm_local.views import VIEW_PACK_VERSION, build_view_pack

Parsed = TypeVar("Parsed")


class LocalVLMPipeline:
    def __init__(
        self,
        backend: VLMBackend,
        *,
        prompt_version: str = PROMPT_VERSION,
        software_lock_hash: str = "unlocked",
        calibrator: TemperatureCalibrator | None = None,
    ) -> None:
        self.backend = backend
        self.prompt_version = prompt_version
        self.software_lock_hash = software_lock_hash
        self.calibrator = calibrator or TemperatureCalibrator()

    def classify_image(
        self,
        image_path: Path | str,
        *,
        sample_id: str | None = None,
        subject_id: str | None = None,
        day: float | None = None,
    ) -> LocalVLMRecord:
        path = Path(image_path).expanduser().resolve()
        views = build_view_pack(path)
        images = [view.image for view in views]
        morphology = self._request(
            images,
            MORPHOLOGY_PROMPT,
            MorphologyObservation.from_dict,
        )
        if morphology is None:
            return self._ungradable_record(path, sample_id, subject_id, day)

        evidence = self._request(
            images,
            stage_prompt(morphology),
            StageEvidence.from_dict,
        )
        if evidence is None:
            prediction = ImagePrediction.ungradable("Model output failed schema validation")
        else:
            prediction = calibrate_stage_evidence(evidence, self.calibrator)

        return LocalVLMRecord(
            sample_id=sample_id or path.stem,
            image_path=str(path),
            image_sha256=_sha256(path),
            subject_id=subject_id,
            day=day,
            morphology=morphology,
            image_prediction=prediction,
            sequence_prediction=SequencePrediction(
                final_stage=prediction.primary_stage,
                adjusted=False,
                reason="image_only",
            ),
            provenance={
                **self.backend.provenance,
                "prompt_version": self.prompt_version,
                "schema_version": "3.0",
                "view_pack_version": VIEW_PACK_VERSION,
                "software_lock_hash": self.software_lock_hash,
                "calibrator_hash": self.calibrator.sha256,
            },
        )

    def _request(
        self,
        images: list[Any],
        prompt: str,
        parser: Callable[[dict[str, Any]], Parsed],
    ) -> Parsed | None:
        response = self.backend.generate(images, prompt)
        try:
            return parser(_parse_json_object(response))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as first_error:
            repaired = self.backend.generate(images, repair_prompt(prompt, response, first_error))
            try:
                return parser(_parse_json_object(repaired))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return None

    def _ungradable_record(
        self,
        path: Path,
        sample_id: str | None,
        subject_id: str | None,
        day: float | None,
    ) -> LocalVLMRecord:
        prediction = ImagePrediction.ungradable("Model output failed schema validation")
        return LocalVLMRecord(
            sample_id=sample_id or path.stem,
            image_path=str(path),
            image_sha256=_sha256(path),
            subject_id=subject_id,
            day=day,
            morphology=MorphologyObservation.ungradable("invalid_model_output"),
            image_prediction=prediction,
            sequence_prediction=SequencePrediction(None, False, "ungradable"),
            provenance={
                **self.backend.provenance,
                "prompt_version": self.prompt_version,
                "schema_version": "3.0",
                "view_pack_version": VIEW_PACK_VERSION,
                "software_lock_hash": self.software_lock_hash,
                "calibrator_hash": self.calibrator.sha256,
            },
        )


def calibrate_stage_evidence(
    evidence: StageEvidence,
    calibrator: TemperatureCalibrator,
) -> ImagePrediction:
    probabilities = calibrator.transform(evidence.raw_scores)
    ranked = sorted(probabilities, key=probabilities.__getitem__, reverse=True)
    margin = probabilities[ranked[0]] - probabilities[ranked[1]]
    confidence = (
        ConfidenceTier.HIGH
        if margin >= 0.50
        else ConfidenceTier.MEDIUM
        if margin >= 0.20
        else ConfidenceTier.LOW
    )
    return ImagePrediction(
        primary_stage=ranked[0],
        secondary_stage=ranked[1],
        raw_scores=evidence.raw_scores,
        probabilities=probabilities,
        confidence_tier=confidence,
        rationale=evidence.rationale,
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
