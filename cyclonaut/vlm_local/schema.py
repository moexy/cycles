"""Versioned records for morphology-first local VLM staging."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

from cyclonaut.core.types import EstrousStage

SCHEMA_VERSION = "3.0"


class Abundance(StrEnum):
    ABSENT = "absent"
    RARE = "rare"
    PRESENT = "present"
    DOMINANT = "dominant"


class NuclearState(StrEnum):
    CLEAR_NUCLEI = "clear_nuclei"
    GHOST_NUCLEI = "ghost_nuclei"
    ANUCLEATE = "anucleate"
    MIXED = "mixed"
    NOT_ASSESSABLE = "not_assessable"


class Arrangement(StrEnum):
    ISOLATED = "isolated"
    CLUSTERS = "clusters"
    SHEETS = "sheets"
    MIXED = "mixed"
    NOT_ASSESSABLE = "not_assessable"


class QCStatus(StrEnum):
    USABLE = "usable"
    LOW_CELLULARITY = "low_cellularity"
    OUT_OF_FOCUS = "out_of_focus"
    OBSCURED = "obscured"
    UNGRADABLE = "ungradable"


class ConfidenceTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class StageEvidence:
    """Minimal model-authored stage evidence; derived fields stay deterministic."""

    raw_scores: dict[EstrousStage, float]
    rationale: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StageEvidence:
        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        return cls(
            raw_scores=_stage_map(payload["raw_scores"], normalise=False),
            rationale=rationale.strip(),
        )


@dataclass(frozen=True, slots=True)
class MorphologyObservation:
    cornified_squames: Abundance
    nucleated_epithelial: Abundance
    leukocytes: Abundance
    nuclear_state: NuclearState
    arrangement: Arrangement
    artifacts: tuple[str, ...]
    qc_status: QCStatus
    qc_reasons: tuple[str, ...]
    evidence: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MorphologyObservation:
        return cls(
            cornified_squames=Abundance(payload["cornified_squames"]),
            nucleated_epithelial=Abundance(payload["nucleated_epithelial"]),
            leukocytes=Abundance(payload["leukocytes"]),
            nuclear_state=NuclearState(payload["nuclear_state"]),
            arrangement=Arrangement(payload["arrangement"]),
            artifacts=_string_tuple(payload.get("artifacts", []), "artifacts"),
            qc_status=QCStatus(payload["qc_status"]),
            qc_reasons=_string_tuple(payload.get("qc_reasons", []), "qc_reasons"),
            evidence=_string_tuple(payload.get("evidence", []), "evidence"),
        )

    @classmethod
    def ungradable(cls, reason: str) -> MorphologyObservation:
        return cls(
            cornified_squames=Abundance.ABSENT,
            nucleated_epithelial=Abundance.ABSENT,
            leukocytes=Abundance.ABSENT,
            nuclear_state=NuclearState.NOT_ASSESSABLE,
            arrangement=Arrangement.NOT_ASSESSABLE,
            artifacts=(),
            qc_status=QCStatus.UNGRADABLE,
            qc_reasons=(reason,),
            evidence=(),
        )


@dataclass(frozen=True, slots=True)
class ImagePrediction:
    primary_stage: EstrousStage | None
    secondary_stage: EstrousStage | None
    raw_scores: dict[EstrousStage, float]
    probabilities: dict[EstrousStage, float]
    confidence_tier: ConfidenceTier
    rationale: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImagePrediction:
        primary = _optional_stage(payload.get("primary_stage"))
        secondary = _optional_stage(payload.get("secondary_stage"))
        if primary is None:
            if secondary is not None:
                raise ValueError("ungradable prediction cannot have a secondary_stage")
            if payload.get("raw_scores") != {} or payload.get("probabilities") != {}:
                raise ValueError("ungradable prediction cannot have stage scores")
            rationale = payload.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("rationale must be a non-empty string")
            return cls(
                None,
                None,
                {},
                {},
                ConfidenceTier(payload["confidence_tier"]),
                rationale.strip(),
            )
        if secondary == primary:
            # A maximally confident model names the same stage twice, meaning
            # "no distinct runner-up". That is an answer, not malformed output;
            # rejecting it forced a full repair round-trip on every confident
            # image, re-encoding the whole view pack to learn nothing.
            secondary = None
        raw_scores = _stage_map(payload["raw_scores"], normalise=False)
        probabilities = _stage_map(payload["probabilities"], normalise=True)
        # Compare probability values rather than sorted-key identity. A confident
        # model emits ties (one-hot leaves three stages at 0.0) and which tied key
        # sorts into a given slot is arbitrary, so demanding identity would reject
        # correct output that no model could be asked to guess.
        ordered = sorted(probabilities.values(), reverse=True)
        if not math.isclose(probabilities[primary], ordered[0], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("primary_stage must match the largest probability")
        if secondary is not None and not math.isclose(
            probabilities[secondary], ordered[1], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("secondary_stage must match the second-largest probability")
        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        return cls(
            primary_stage=primary,
            secondary_stage=secondary,
            raw_scores=raw_scores,
            probabilities=probabilities,
            confidence_tier=ConfidenceTier(payload["confidence_tier"]),
            rationale=rationale.strip(),
        )

    @classmethod
    def ungradable(cls, reason: str) -> ImagePrediction:
        return cls(None, None, {}, {}, ConfidenceTier.LOW, reason)


@dataclass(frozen=True, slots=True)
class SequencePrediction:
    final_stage: EstrousStage | None
    adjusted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class LocalVLMRecord:
    sample_id: str
    image_path: str
    image_sha256: str
    subject_id: str | None
    day: float | None
    morphology: MorphologyObservation
    image_prediction: ImagePrediction
    sequence_prediction: SequencePrediction
    provenance: dict[str, str]
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LocalVLMRecord:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported local VLM schema: {payload.get('schema_version')!r}")
        sequence = payload["sequence_prediction"]
        if not isinstance(sequence, dict):
            raise ValueError("sequence_prediction must be an object")
        provenance = payload["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        return cls(
            sample_id=str(payload["sample_id"]),
            image_path=str(payload["image_path"]),
            image_sha256=str(payload["image_sha256"]),
            subject_id=(str(payload["subject_id"]) if payload.get("subject_id") is not None else None),
            day=(float(payload["day"]) if payload.get("day") is not None else None),
            morphology=MorphologyObservation.from_dict(payload["morphology"]),
            image_prediction=ImagePrediction.from_dict(payload["image_prediction"]),
            sequence_prediction=SequencePrediction(
                final_stage=_optional_stage(sequence.get("final_stage")),
                adjusted=bool(sequence["adjusted"]),
                reason=str(sequence["reason"]),
            ),
            provenance={str(key): str(value) for key, value in provenance.items()},
            schema_version=SCHEMA_VERSION,
        )

    def with_sequence_prediction(self, prediction: SequencePrediction) -> LocalVLMRecord:
        return replace(self, sequence_prediction=prediction)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def _optional_stage(value: Any) -> EstrousStage | None:
    if value is None:
        return None
    stage = EstrousStage(value)
    if stage not in EstrousStage.canonical_stages():
        raise ValueError(f"{value!r} is not a canonical estrous stage")
    return stage


def _stage_map(value: Any, *, normalise: bool) -> dict[EstrousStage, float]:
    if not isinstance(value, dict):
        raise ValueError("stage scores must be an object")
    stages = EstrousStage.canonical_stages()
    if set(value) != {stage.value for stage in stages}:
        raise ValueError("stage scores must contain exactly the four canonical stages")
    parsed = {stage: float(value[stage.value]) for stage in stages}
    if any(not math.isfinite(score) for score in parsed.values()):
        raise ValueError("stage scores must be finite")
    if normalise:
        if any(score < 0 for score in parsed.values()):
            raise ValueError("probabilities cannot be negative")
        total = sum(parsed.values())
        if total <= 0:
            raise ValueError("probabilities must have positive mass")
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1")
    return parsed


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {_json_value(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
