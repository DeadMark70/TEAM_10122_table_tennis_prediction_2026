import numpy as np

from analysis_v307_point0_dose_extension import (
    decision_label,
    select_exact_point0_additions,
)


def test_exact_budget_selector_respects_requested_row_count():
    base = np.array([1, 2, 0, 3, 4, 5, 6])
    prob = np.array(
        [
            [0.80, 0.10, 0.05, 0.02, 0.02, 0.01, 0.00],
            [0.70, 0.05, 0.20, 0.02, 0.02, 0.01, 0.00],
            [0.99, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00],
            [0.30, 0.05, 0.05, 0.40, 0.10, 0.10, 0.00],
            [0.55, 0.05, 0.05, 0.05, 0.10, 0.20, 0.00],
            [0.51, 0.05, 0.05, 0.05, 0.10, 0.20, 0.00],
            [0.52, 0.00, 0.00, 0.00, 0.13, 0.20, 0.15],
        ]
    )

    selected, margin = select_exact_point0_additions(base, prob, budget=4)

    assert selected.tolist() == [True, True, False, False, True, False, True]
    assert int(selected.sum()) == 4
    assert np.allclose(margin[[0, 1, 4, 6]], [0.70, 0.50, 0.45, 0.37])


def test_dose_decision_function_matches_thresholds():
    assert decision_label(0.0030, 24) == "REVIEW_STRONG"
    assert decision_label(0.0040, 25) == "REVIEW_EXPLORE"
    assert decision_label(0.0040, 36) == "REVIEW_EXPLORE"
    assert decision_label(0.00299, 24) == "DO_NOT_UPLOAD"
    assert decision_label(0.0035, 25) == "DO_NOT_UPLOAD"
    assert decision_label(0.0040, 37) == "DO_NOT_UPLOAD"
