"""Guarded longitudinal tie-breaking for local VLM records."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from cycles.core.cycle import STAGE_CYCLE_ORDER, TRANSITION_MATRIX
from cycles.core.types import EstrousStage
from cycles.vlm_local.schema import LocalVLMRecord, SequencePrediction


class TemporalReconciler:
    """Apply Viterbi context only to uncertain, adjacent-stage image calls."""

    def __init__(
        self,
        *,
        margin_threshold: float = 0.15,
        adjustment_threshold: float = 0.0,
    ) -> None:
        if not 0 <= margin_threshold <= 1:
            raise ValueError("margin_threshold must be in [0, 1]")
        if adjustment_threshold < 0:
            raise ValueError("adjustment_threshold cannot be negative")
        self.margin_threshold = margin_threshold
        self.adjustment_threshold = adjustment_threshold

    def reconcile(self, records: list[LocalVLMRecord]) -> list[LocalVLMRecord]:
        output = list(records)
        groups: dict[str, list[tuple[int, LocalVLMRecord]]] = defaultdict(list)
        for index, record in enumerate(records):
            if record.subject_id is None or record.day is None:
                output[index] = _with_reason(record, "no_sequence_metadata")
            elif record.image_prediction.primary_stage is None:
                output[index] = _with_reason(record, "ungradable")
            else:
                groups[record.subject_id].append((index, record))

        for group in groups.values():
            ordered = sorted(group, key=lambda item: (float(item[1].day or 0), item[0]))
            if len(ordered) < 2:
                index, record = ordered[0]
                output[index] = _with_reason(record, "insufficient_sequence_context")
                continue
            sequence = [record for _, record in ordered]
            best_path, gain = _viterbi(sequence)
            for (index, record), suggested in zip(ordered, best_path, strict=True):
                output[index] = self._guarded_update(record, suggested, gain)
        return output

    def _guarded_update(
        self,
        record: LocalVLMRecord,
        suggested: EstrousStage,
        sequence_gain: float,
    ) -> LocalVLMRecord:
        prediction = record.image_prediction
        primary = prediction.primary_stage
        secondary = prediction.secondary_stage
        if primary is None:
            return _with_reason(record, "ungradable")
        ranked = sorted(prediction.probabilities.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
        if margin > self.margin_threshold:
            return _with_reason(record, "confident_image_call")
        if secondary is None or not _adjacent(primary, secondary):
            return _with_reason(record, "nonadjacent_runner_up")
        if suggested == primary:
            return _with_reason(record, "sequence_agrees")
        if suggested != secondary:
            return _with_reason(record, "sequence_not_runner_up")
        if sequence_gain < self.adjustment_threshold:
            return _with_reason(record, "below_adjustment_threshold")
        return record.with_sequence_prediction(
            SequencePrediction(suggested, True, "adjacent_sequence_tiebreak")
        )


def _transition_matrix_for_delta(delta_days: float) -> np.ndarray:
    dt = max(0.0, float(delta_days))
    steps = int(round(dt))
    if steps == 0:
        return np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ],
            dtype=np.float64,
        )
    if steps == 1:
        return TRANSITION_MATRIX
    return np.linalg.matrix_power(TRANSITION_MATRIX, steps)


def _viterbi(records: list[LocalVLMRecord]) -> tuple[list[EstrousStage], float]:
    stages = STAGE_CYCLE_ORDER
    epsilon = 1e-12
    scores = [
        math.log(max(records[0].image_prediction.probabilities.get(stage, 0.0), epsilon))
        for stage in stages
    ]
    backpointers: list[list[int]] = []
    for prev_record, record in zip(records[:-1], records[1:], strict=True):
        dt = (
            (record.day - prev_record.day)
            if (record.day is not None and prev_record.day is not None)
            else 1.0
        )
        trans = _transition_matrix_for_delta(dt)
        next_scores: list[float] = []
        pointers: list[int] = []
        for destination_index, destination in enumerate(stages):
            candidates = [
                previous_score
                + math.log(max(float(trans[source_index, destination_index]), epsilon))
                for source_index, previous_score in enumerate(scores)
            ]
            best_source = max(range(len(stages)), key=candidates.__getitem__)
            emission = record.image_prediction.probabilities.get(destination, 0.0)
            next_scores.append(candidates[best_source] + math.log(max(emission, epsilon)))
            pointers.append(best_source)
        scores = next_scores
        backpointers.append(pointers)

    final_index = max(range(len(stages)), key=scores.__getitem__)
    best_score = scores[final_index]
    path_indices = [final_index]
    for pointers in reversed(backpointers):
        path_indices.append(pointers[path_indices[-1]])
    path_indices.reverse()
    primary_path = [record.image_prediction.primary_stage for record in records]
    primary_score = _path_score(records, primary_path)
    return [stages[index] for index in path_indices], best_score - primary_score


def _path_score(
    records: list[LocalVLMRecord],
    path: list[EstrousStage | None],
) -> float:
    epsilon = 1e-12
    indices = {stage: index for index, stage in enumerate(STAGE_CYCLE_ORDER)}
    total = 0.0
    for index, (record, stage) in enumerate(zip(records, path, strict=True)):
        if stage is None:
            return float("-inf")
        total += math.log(max(record.image_prediction.probabilities.get(stage, 0.0), epsilon))
        if index:
            prev_record = records[index - 1]
            previous = path[index - 1]
            if previous is None:
                return float("-inf")
            dt = (
                (record.day - prev_record.day)
                if (record.day is not None and prev_record.day is not None)
                else 1.0
            )
            trans = _transition_matrix_for_delta(dt)
            total += math.log(
                max(float(trans[indices[previous], indices[stage]]), epsilon)
            )
    return total


def _adjacent(first: EstrousStage, second: EstrousStage) -> bool:
    order = STAGE_CYCLE_ORDER
    first_index, second_index = order.index(first), order.index(second)
    return (first_index - second_index) % len(order) in {1, len(order) - 1}


def _with_reason(record: LocalVLMRecord, reason: str) -> LocalVLMRecord:
    return record.with_sequence_prediction(
        SequencePrediction(record.image_prediction.primary_stage, False, reason)
    )
