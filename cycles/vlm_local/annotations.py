"""Append-only review events and explicit frozen teacher exports."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cycles.vlm_local.schema import LocalVLMRecord

ACTIONS = frozenset({"accept", "correct", "ungradable", "defer"})
CORRECTION_FIELDS = frozenset(
    {
        "primary_stage",
        "secondary_stage",
        "confidence_tier",
        "cornified_squames",
        "nucleated_epithelial",
        "leukocytes",
        "nuclear_state",
        "arrangement",
        "artifacts",
        "qc_status",
        "qc_reasons",
        "evidence",
    }
)


@dataclass(frozen=True, slots=True)
class AnnotationEvent:
    event_id: str
    schema_version: str
    timestamp: str
    reviewer_id: str
    sample_id: str
    record_hash: str
    action: str
    corrections: dict[str, Any]
    note: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationEvent:
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnnotationStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(
        self,
        record: LocalVLMRecord,
        *,
        reviewer_id: str,
        action: str,
        corrections: dict[str, Any] | None = None,
        note: str = "",
    ) -> AnnotationEvent:
        if action not in ACTIONS:
            raise ValueError(f"unsupported annotation action: {action}")
        if not reviewer_id.strip():
            raise ValueError("reviewer_id cannot be empty")
        corrections = dict(corrections or {})
        unknown = set(corrections) - CORRECTION_FIELDS
        if unknown:
            raise ValueError(f"unsupported correction field(s): {', '.join(sorted(unknown))}")
        if action == "correct" and not corrections:
            raise ValueError("correct action requires at least one corrected field")
        event = AnnotationEvent(
            event_id=str(uuid.uuid4()),
            schema_version="1.0",
            timestamp=datetime.now(UTC).isoformat(),
            reviewer_id=reviewer_id.strip(),
            sample_id=record.sample_id,
            record_hash=record_hash(record),
            action=action,
            corrections=corrections,
            note=note.strip(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            stream.flush()
        return event

    def events(self) -> list[AnnotationEvent]:
        if not self.path.exists():
            return []
        events: list[AnnotationEvent] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(AnnotationEvent.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid annotation event at {self.path}:{line_number}"
                    ) from exc
        return events

    def latest_by_sample(self) -> dict[str, AnnotationEvent]:
        latest: dict[str, AnnotationEvent] = {}
        for event in self.events():
            latest[event.sample_id] = event
        return latest

    def export_teacher(
        self,
        records: list[LocalVLMRecord],
        output_dir: Path | str,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"teacher export already exists: {destination}")
        destination.mkdir(parents=True)
        latest = self.latest_by_sample()
        exported = 0
        dataset_path = destination / "dataset.jsonl"
        with dataset_path.open("x", encoding="utf-8") as stream:
            for record in records:
                event = latest.get(record.sample_id)
                if event is None or event.action == "defer":
                    continue
                if event.record_hash != record_hash(record):
                    raise ValueError(
                        f"review for {record.sample_id} references a different model record"
                    )
                payload = {
                    "sample_id": record.sample_id,
                    "model_record": record.to_dict(),
                    "teacher_label": _teacher_label(record, event),
                    "review_event": event.to_dict(),
                }
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
                exported += 1
        digest = _file_hash(dataset_path)
        (destination / "manifest.sha256").write_text(
            f"{digest}  dataset.jsonl\n", encoding="utf-8"
        )
        summary = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source_annotation_log": str(self.path.resolve()),
            "input_records": len(records),
            "exported_samples": exported,
            "dataset_sha256": digest,
        }
        (destination / "metadata.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


def record_hash(record: LocalVLMRecord) -> str:
    encoded = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _teacher_label(record: LocalVLMRecord, event: AnnotationEvent) -> dict[str, Any]:
    prediction = record.image_prediction
    morphology = record.morphology
    label: dict[str, Any] = {
        "primary_stage": prediction.primary_stage.value if prediction.primary_stage else None,
        "secondary_stage": prediction.secondary_stage.value if prediction.secondary_stage else None,
        "confidence_tier": prediction.confidence_tier.value,
        "cornified_squames": morphology.cornified_squames.value,
        "nucleated_epithelial": morphology.nucleated_epithelial.value,
        "leukocytes": morphology.leukocytes.value,
        "nuclear_state": morphology.nuclear_state.value,
        "arrangement": morphology.arrangement.value,
        "artifacts": list(morphology.artifacts),
        "qc_status": morphology.qc_status.value,
        "qc_reasons": list(morphology.qc_reasons),
        "evidence": list(morphology.evidence),
        "review_action": event.action,
    }
    if event.action == "ungradable":
        label.update(primary_stage=None, secondary_stage=None, qc_status="ungradable")
    label.update(event.corrections)
    return label


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
