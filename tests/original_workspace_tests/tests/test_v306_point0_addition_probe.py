import numpy as np

from analysis_v306_point0_addition_probe import (
    decision_label,
    select_point0_additions,
)


def test_budget_selector_picks_highest_positive_point0_margins_and_exact_budget():
    base = np.array([1, 2, 0, 3, 4, 5])
    prob = np.array(
        [
            [0.80, 0.10, 0.05, 0.02, 0.02, 0.01],
            [0.70, 0.05, 0.20, 0.02, 0.02, 0.01],
            [0.99, 0.00, 0.00, 0.00, 0.00, 0.01],
            [0.30, 0.05, 0.05, 0.40, 0.10, 0.10],
            [0.55, 0.05, 0.05, 0.05, 0.10, 0.20],
            [0.51, 0.05, 0.05, 0.05, 0.10, 0.20],
        ]
    )

    selected, margin = select_point0_additions(base, prob, budget=3)

    assert selected.tolist() == [True, True, False, False, True, False]
    assert np.allclose(margin[[0, 1, 4]], [0.70, 0.50, 0.45])
    assert int(selected.sum()) == 3


def test_decision_label_requires_delta_gate_and_row_limit():
    assert decision_label(0.0015, 18) == "REVIEW_P0"
    assert decision_label(0.00149, 18) == "DO_NOT_UPLOAD"
    assert decision_label(0.0100, 19) == "DO_NOT_UPLOAD"
