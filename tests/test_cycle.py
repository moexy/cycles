from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from cycles.core.cycle import (
    STAGE_CYCLE_ORDER,
    TRANSITION_MATRIX,
    compute_confidence_index,
    detect_pseudopregnancy,
    detect_transition_anomaly,
    fit_cyclicity,
    generate_cycle_plot_data,
)
from cycles.core.types import EstrousStage


def test_cycle_order_and_transition_matrix_are_valid() -> None:
    assert STAGE_CYCLE_ORDER == [
        EstrousStage.PROESTRUS,
        EstrousStage.ESTRUS,
        EstrousStage.METESTRUS,
        EstrousStage.DIESTRUS,
    ]
    assert TRANSITION_MATRIX.shape == (4, 4)
    np.testing.assert_allclose(
        TRANSITION_MATRIX.sum(axis=1),
        np.ones(4),
        err_msg="Every transition row must be a probability distribution",
    )
    dominant_destinations = np.argmax(TRANSITION_MATRIX, axis=1)
    assert dominant_destinations.tolist() == [1, 2, 3, 0]


def test_compute_confidence_index_distinguishes_clear_and_transition_predictions() -> None:
    clear = {
        EstrousStage.DIESTRUS: 0.80,
        EstrousStage.PROESTRUS: 0.10,
        EstrousStage.ESTRUS: 0.05,
        EstrousStage.METESTRUS: 0.05,
    }
    ambiguous = {
        EstrousStage.DIESTRUS: 0.46,
        EstrousStage.PROESTRUS: 0.44,
        EstrousStage.ESTRUS: 0.05,
        EstrousStage.METESTRUS: 0.05,
    }

    clear_index, clear_transition, clear_target = compute_confidence_index(clear)
    assert clear_index == pytest.approx(0.70)
    assert clear_transition is False and clear_target is None
    index, transition, target = compute_confidence_index(ambiguous)
    assert index == pytest.approx(0.02)
    assert transition is True and target is EstrousStage.PROESTRUS


def test_compute_confidence_index_handles_empty_single_and_nonfinite_values() -> None:
    assert compute_confidence_index({}) == (0.0, False, None)
    assert compute_confidence_index({EstrousStage.ESTRUS: 1.4}) == (1.0, False, None)
    assert compute_confidence_index(
        {EstrousStage.ESTRUS: float("nan"), EstrousStage.DIESTRUS: -2.0}
    ) == (0.0, False, None)


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (EstrousStage.PROESTRUS, EstrousStage.PROESTRUS, False),
        (EstrousStage.PROESTRUS, EstrousStage.ESTRUS, False),
        (EstrousStage.DIESTRUS, EstrousStage.PROESTRUS, False),
        (EstrousStage.ESTRUS, EstrousStage.PROESTRUS, True),
        (EstrousStage.PROESTRUS, EstrousStage.METESTRUS, True),
        (EstrousStage.INSUFFICIENT_CELLS, EstrousStage.ESTRUS, False),
    ],
)
def test_detect_transition_anomaly(
    previous: EstrousStage,
    current: EstrousStage,
    expected: bool,
) -> None:
    assert detect_transition_anomaly(previous, current) is expected


def test_detect_pseudopregnancy_uses_longest_consecutive_diestrus_run() -> None:
    normal = [EstrousStage.DIESTRUS] * 9 + [EstrousStage.PROESTRUS]
    prolonged = [EstrousStage.ESTRUS] + [EstrousStage.DIESTRUS] * 10

    assert detect_pseudopregnancy(normal) == (False, 9)
    assert detect_pseudopregnancy(prolonged) == (True, 10)
    assert detect_pseudopregnancy([]) == (False, 0)
    with pytest.raises(ValueError, match="at least 1"):
        detect_pseudopregnancy(prolonged, threshold_days=0)


def test_fit_cyclicity_scores_regular_progression_and_cycle_length() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    stages = STAGE_CYCLE_ORDER * 3
    timestamps = [(start + timedelta(days=index)).isoformat() for index in range(len(stages))]

    fit = fit_cyclicity(timestamps, stages, mouse_id="M42")

    assert fit.mouse_id == "M42"
    assert fit.regularity_score == pytest.approx(1.0)
    assert 3.0 <= fit.estimated_cycle_length_days <= 5.0, "A four-stage daily cycle should fit near 4 days"
    assert fit.anomalies == [] and not fit.is_pseudopregnant


def test_fit_cyclicity_reports_anomalies_and_edge_cases() -> None:
    single = fit_cyclicity(["2026-01-01"], [EstrousStage.DIESTRUS])
    empty = fit_cyclicity([], [])
    anomalous = fit_cyclicity(
        ["2026-01-01", "2026-01-02"],
        [EstrousStage.PROESTRUS, EstrousStage.METESTRUS],
    )

    assert single.estimated_cycle_length_days == 0.0 and single.regularity_score == 0.0
    assert empty.stages == [] and empty.estimated_cycle_length_days == 0.0
    assert anomalous.regularity_score == 0.0
    assert anomalous.anomalies == [(1, "proestrus -> metestrus")]
    with pytest.raises(ValueError, match="same number"):
        fit_cyclicity(["2026-01-01"], [])
    with pytest.raises(ValueError, match="chronological"):
        fit_cyclicity(
            ["2026-01-02", "2026-01-01"],
            [EstrousStage.DIESTRUS, EstrousStage.PROESTRUS],
        )
    with pytest.raises(ValueError, match="invalid ISO timestamp"):
        fit_cyclicity(["not-a-date"], [EstrousStage.DIESTRUS])


def test_generate_cycle_plot_data_builds_step_line_and_transition_markers() -> None:
    timestamps = ["d1", "d2", "d3", "d4"]
    stages = [
        EstrousStage.PROESTRUS,
        EstrousStage.ESTRUS,
        EstrousStage.PROESTRUS,
        EstrousStage.INSUFFICIENT_CELLS,
    ]

    plot = generate_cycle_plot_data(timestamps, stages)

    assert plot["stage_values"] == [2, 3, 2, None]
    assert plot["step_x"] == ["d1", "d2", "d2", "d3", "d3", "d4", "d4"]
    assert plot["step_y"] == [2, 2, 3, 3, 2, 2, None]
    assert len(plot["transition_markers"]) == 2, "QC transitions should be omitted"
    assert plot["transition_markers"][0]["is_anomaly"] is False
    assert plot["transition_markers"][1]["is_anomaly"] is True
    assert generate_cycle_plot_data([], [])["step_x"] == []
    with pytest.raises(ValueError, match="same number"):
        generate_cycle_plot_data(["d1"], [])
