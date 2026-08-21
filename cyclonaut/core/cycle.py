"""Longitudinal analytics for the rodent estrous cycle.

The canonical physiological progression used throughout this module is
``Proestrus -> Estrus -> Metestrus -> Diestrus -> Proestrus``.  Quality-control
labels such as :class:`~cyclonaut.core.types.EstrousStage.INSUFFICIENT_CELLS` are
retained in returned data but excluded from biological transition analyses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from cyclonaut.core.types import CycleFitResult, EstrousStage

STAGE_CYCLE_ORDER: list[EstrousStage] = [
    EstrousStage.PROESTRUS,
    EstrousStage.ESTRUS,
    EstrousStage.METESTRUS,
    EstrousStage.DIESTRUS,
]
"""Canonical forward stage order: P -> E -> M -> D -> P."""

# Rows and columns both follow STAGE_CYCLE_ORDER.  The diagonal permits a stage
# to persist across observations; the dominant off-diagonal value is always the
# next physiological stage.  Every row is a probability distribution.
TRANSITION_MATRIX: np.ndarray = np.array(
    [
        [0.20, 0.72, 0.04, 0.04],  # Proestrus
        [0.03, 0.18, 0.75, 0.04],  # Estrus
        [0.03, 0.04, 0.20, 0.73],  # Metestrus
        [0.55, 0.02, 0.03, 0.40],  # Diestrus
    ],
    dtype=np.float64,
)
"""Expected Markov transition probabilities between canonical stages."""

_STAGE_TO_ORDER = {stage: index for index, stage in enumerate(STAGE_CYCLE_ORDER)}
_STAGE_TO_VALUE: dict[EstrousStage, int] = {
    EstrousStage.DIESTRUS: 1,
    EstrousStage.PROESTRUS: 2,
    EstrousStage.ESTRUS: 3,
    EstrousStage.METESTRUS: 4,
}
_STAGE_TO_ANGLE = {
    stage: 2.0 * np.pi * index / len(STAGE_CYCLE_ORDER)
    for index, stage in enumerate(STAGE_CYCLE_ORDER)
}
_TRANSITION_CONFIDENCE_THRESHOLD = 0.25
_TRANSITION_RATIO_THRESHOLD = 0.75


def compute_confidence_index(
    probabilities: dict[EstrousStage, float],
) -> tuple[float, bool, EstrousStage | None]:
    """Return top-two confidence separation and transition ambiguity.

    The confidence index is ``P(top-1) - P(top-2)``.  A prediction is marked as
    transitional when this separation is below 0.25 or when the runner-up is at
    least 75% as probable as the leading stage.  Only canonical biological
    stages participate; QC labels are deliberately ignored.

    Empty mappings return ``(0.0, False, None)``.  With one usable stage the
    absent runner-up is treated as probability zero and no transition target is
    reported.  Non-finite scores are ignored and negative scores are treated as
    zero so that a malformed model output cannot propagate NaNs.
    """

    ranked: list[tuple[EstrousStage, float]] = []
    for stage in STAGE_CYCLE_ORDER:
        if stage not in probabilities:
            continue
        score = float(probabilities[stage])
        if np.isfinite(score):
            ranked.append((stage, max(0.0, score)))

    if not ranked:
        return 0.0, False, None

    # Canonical order provides deterministic tie-breaking.
    ranked.sort(key=lambda item: (-item[1], _STAGE_TO_ORDER[item[0]]))
    top_score = ranked[0][1]
    if len(ranked) == 1:
        return float(np.clip(top_score, 0.0, 1.0)), False, None

    second_stage, second_score = ranked[1]
    confidence_index = float(np.clip(top_score - second_score, 0.0, 1.0))
    ratio_is_close = top_score > 0.0 and second_score / top_score >= _TRANSITION_RATIO_THRESHOLD
    is_transition = top_score > 0.0 and (
        confidence_index < _TRANSITION_CONFIDENCE_THRESHOLD or ratio_is_close
    )
    return confidence_index, is_transition, second_stage if is_transition else None


def detect_transition_anomaly(
    prev_stage: EstrousStage,
    curr_stage: EstrousStage,
) -> bool:
    """Return whether a stage change violates forward biological progression.

    Persistence in the same stage and movement to the next canonical stage are
    normal.  Skipped or retrograde transitions are anomalous.  A transition
    involving a non-canonical QC label is not classifiable and therefore is not
    reported as a biological anomaly.
    """

    previous_index = _STAGE_TO_ORDER.get(prev_stage)
    current_index = _STAGE_TO_ORDER.get(curr_stage)
    if previous_index is None or current_index is None:
        return False
    step = (current_index - previous_index) % len(STAGE_CYCLE_ORDER)
    return step not in (0, 1)


def detect_pseudopregnancy(
    stage_sequence: list[EstrousStage],
    threshold_days: int = 10,
) -> tuple[bool, int]:
    """Detect prolonged Diestrus consistent with pseudopregnancy.

    Each stage observation represents one day.  QC/unknown stages interrupt a
    Diestrus run rather than being imputed.  The maximum run length is returned
    even when it does not cross ``threshold_days``.
    """

    if threshold_days < 1:
        raise ValueError("threshold_days must be at least 1")

    longest_run = 0
    current_run = 0
    for stage in stage_sequence:
        if stage == EstrousStage.DIESTRUS:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return longest_run >= threshold_days, longest_run


def fit_cyclicity(
    timestamps: list[str | datetime],
    stages: list[EstrousStage],
    mouse_id: str = "Mouse1",
) -> CycleFitResult:
    """Summarize estrous cyclicity for one longitudinal stage series.

    Cycle length is estimated with a circular periodogram.  Canonical stages
    are placed at quarter-cycle angles and candidate periods are scored by
    their phase coherence.  This avoids the discontinuity between Diestrus and
    Proestrus inherent in an ordinary integer encoding.  At least four usable
    observations, two distinct stages, and a positive time span are required;
    otherwise the estimated length is reported as ``0.0``.

    Regularity is the mean alignment of consecutive observations with the
    canonical progression: forward movement scores 1, persistence scores 0.5,
    and skipped/retrograde movement scores 0.  Pairs containing QC labels are
    not scored.

    Raises:
        ValueError: If timestamps and stages differ in length, a timestamp is
            invalid, or the timestamps are not chronological.
    """

    if len(timestamps) != len(stages):
        raise ValueError(
            "timestamps and stages must contain the same number of observations "
            f"({len(timestamps)} != {len(stages)})"
        )

    parsed_timestamps = [_parse_timestamp(value, index) for index, value in enumerate(timestamps)]
    numeric_days = _timestamps_to_days(parsed_timestamps)
    if numeric_days.size > 1 and np.any(np.diff(numeric_days) < 0.0):
        raise ValueError("timestamps must be in chronological order")

    timestamp_strings = [
        value if isinstance(value, str) else value.isoformat() for value in timestamps
    ]

    anomaly_records: list[tuple[int, str]] = []
    regularity_components: list[float] = []
    for index, (previous, current) in enumerate(zip(stages, stages[1:], strict=False), start=1):
        previous_index = _STAGE_TO_ORDER.get(previous)
        current_index = _STAGE_TO_ORDER.get(current)
        if previous_index is None or current_index is None:
            continue
        step = (current_index - previous_index) % len(STAGE_CYCLE_ORDER)
        if step == 1:
            regularity_components.append(1.0)
        elif step == 0:
            regularity_components.append(0.5)
        else:
            regularity_components.append(0.0)
            anomaly_records.append(
                (index, f"{previous.value} -> {current.value}")
            )

    regularity_score = (
        float(np.mean(regularity_components)) if regularity_components else 0.0
    )
    is_pseudopregnant, consecutive_diestrus = detect_pseudopregnancy(stages)
    estimated_cycle_length = _estimate_cycle_length_days(numeric_days, stages)

    return CycleFitResult(
        mouse_id=mouse_id,
        timestamps=timestamp_strings,
        stages=list(stages),
        regularity_score=float(np.clip(regularity_score, 0.0, 1.0)),
        estimated_cycle_length_days=estimated_cycle_length,
        is_pseudopregnant=is_pseudopregnant,
        consecutive_diestrus_days=consecutive_diestrus,
        anomalies=anomaly_records,
    )


def generate_cycle_plot_data(
    timestamps: list[str],
    stages: list[EstrousStage],
) -> dict[str, Any]:
    """Prepare stage coordinates, step-line points, and transition markers.

    The numerical y-axis follows the established display mapping ``D=1,
    P=2, E=3, M=4``.  Unknown/QC labels use ``None`` so GUI and plotting code
    can render a gap.  Explicit ``step_x``/``step_y`` arrays form a post-step
    line without requiring a plotting-library-specific draw style.
    """

    if len(timestamps) != len(stages):
        raise ValueError(
            "timestamps and stages must contain the same number of observations "
            f"({len(timestamps)} != {len(stages)})"
        )

    stage_values: list[int | None] = [_STAGE_TO_VALUE.get(stage) for stage in stages]
    stage_labels = [stage.value for stage in stages]

    step_x: list[str] = []
    step_y: list[int | None] = []
    if timestamps:
        step_x.append(timestamps[0])
        step_y.append(stage_values[0])
        for index in range(1, len(timestamps)):
            step_x.extend((timestamps[index], timestamps[index]))
            step_y.extend((stage_values[index - 1], stage_values[index]))

    transition_markers: list[dict[str, Any]] = []
    for index, (previous, current) in enumerate(zip(stages, stages[1:], strict=False), start=1):
        if previous == current:
            continue
        previous_value = _STAGE_TO_VALUE.get(previous)
        current_value = _STAGE_TO_VALUE.get(current)
        if previous_value is None or current_value is None:
            continue
        transition_markers.append(
            {
                "index": index,
                "timestamp": timestamps[index],
                "from_stage": previous.value,
                "to_stage": current.value,
                "from_value": previous_value,
                "to_value": current_value,
                "is_anomaly": detect_transition_anomaly(previous, current),
            }
        )

    return {
        "timestamps": list(timestamps),
        "stages": stage_labels,
        "stage_values": stage_values,
        "step_x": step_x,
        "step_y": step_y,
        "transition_markers": transition_markers,
        "stage_ticks": {
            value: stage.display_name for stage, value in _STAGE_TO_VALUE.items()
        },
        "canonical_order": [stage.value for stage in STAGE_CYCLE_ORDER],
    }


def _parse_timestamp(value: str | datetime, index: int) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"timestamp at index {index} must be a string or datetime")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp at index {index}: {value!r}") from exc


def _timestamps_to_days(timestamps: list[datetime]) -> np.ndarray:
    if not timestamps:
        return np.empty(0, dtype=np.float64)

    seconds: list[float] = []
    for value in timestamps:
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=UTC)
        else:
            normalized = normalized.astimezone(UTC)
        seconds.append(normalized.timestamp())
    numeric = np.asarray(seconds, dtype=np.float64)
    return (numeric - numeric[0]) / 86_400.0


def _estimate_cycle_length_days(
    numeric_days: np.ndarray,
    stages: list[EstrousStage],
) -> float:
    valid_indices = [
        index for index, stage in enumerate(stages) if stage in _STAGE_TO_ANGLE
    ]
    if len(valid_indices) < 4:
        return 0.0

    valid_stages = [stages[index] for index in valid_indices]
    if len(set(valid_stages)) < 2:
        return 0.0

    times = numeric_days[np.asarray(valid_indices, dtype=np.int64)]
    span = float(times[-1] - times[0])
    if span <= 0.0:
        return 0.0

    positive_intervals = np.diff(np.unique(times))
    positive_intervals = positive_intervals[positive_intervals > 0.0]
    if positive_intervals.size == 0:
        return 0.0

    median_interval = float(np.median(positive_intervals))
    minimum_period = max(2.0, 2.0 * median_interval)
    maximum_period = min(10.0, max(4.5, 1.5 * span))
    if maximum_period <= minimum_period:
        maximum_period = minimum_period + max(0.5, median_interval)

    candidate_periods = np.linspace(minimum_period, maximum_period, 4096)
    observed_phase = np.asarray(
        [_STAGE_TO_ANGLE[stage] for stage in valid_stages], dtype=np.float64
    )
    observed_circle = np.exp(1j * observed_phase)
    angular_frequencies = 2.0 * np.pi / candidate_periods
    coherence = np.empty_like(candidate_periods)
    # Bound temporary memory for long longitudinal series rather than forming
    # a candidates-by-observations matrix for the entire search grid.
    chunk_size = 256
    for start in range(0, candidate_periods.size, chunk_size):
        stop = min(start + chunk_size, candidate_periods.size)
        expected_circle = np.exp(
            1j * np.outer(angular_frequencies[start:stop], times)
        )
        coherence[start:stop] = np.abs(
            np.mean(
                observed_circle[np.newaxis, :] * np.conj(expected_circle),
                axis=1,
            )
        )
    best_index = int(np.argmax(coherence))
    best_period = float(candidate_periods[best_index])

    # A weakly phase-locked series does not support a meaningful period.
    if float(coherence[best_index]) < 0.35:
        return 0.0
    return best_period
