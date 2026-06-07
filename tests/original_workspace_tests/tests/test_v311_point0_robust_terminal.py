import numpy as np
import pandas as pd

from analysis_v311_point0_robust_terminal import (
    decision_label,
    foldsafe_terminal_slice_prior,
    select_point0_rows,
)


def test_select_point0_rows_requires_nonzero_base_zero_candidate_and_positive_score():
    base = np.array([7, 0, 8, 9, 6, 5])
    candidate = np.array([0, 0, 7, 0, 0, 0])
    score = np.array([0.50, 0.99, 0.95, 0.40, -0.10, np.nan])

    selected = select_point0_rows(base, candidate, score, budget=4)

    assert selected.tolist() == [True, False, False, True, False, False]


def test_select_point0_rows_is_stable_for_tied_scores_and_respects_budget():
    base = np.array([7, 8, 9, 7])
    candidate = np.zeros(4, dtype=int)
    score = np.array([0.30, 0.30, 0.20, 0.10])

    selected = select_point0_rows(base, candidate, score, budget=2)

    assert selected.tolist() == [True, True, False, False]


def test_select_point0_rows_applies_optional_gate():
    base = np.array([7, 8, 9, 6])
    candidate = np.zeros(4, dtype=int)
    score = np.array([0.90, 0.80, 0.70, 0.60])
    gate = np.array([False, True, True, True])

    selected = select_point0_rows(base, candidate, score, budget=2, gate=gate)

    assert selected.tolist() == [False, True, True, False]


def test_decision_label_uses_strict_reference_delta_thresholds():
    v306_delta = 0.003578457165028276
    v307_budget24_delta = 0.00469246629968606

    assert decision_label(v306_delta + 0.000001, 24, v306_delta, v307_budget24_delta) == "REVIEW_SAFE"
    assert decision_label(v307_budget24_delta + 0.000001, 36, v306_delta, v307_budget24_delta) == "REVIEW_EXPLORE"
    assert decision_label(v306_delta, 24, v306_delta, v307_budget24_delta) == "DIAGNOSTIC"
    assert decision_label(v306_delta + 0.000001, 25, v306_delta, v307_budget24_delta) == "DIAGNOSTIC"
    assert decision_label(v307_budget24_delta + 0.010000, 37, v306_delta, v307_budget24_delta) == "DIAGNOSTIC"


def test_foldsafe_terminal_slice_prior_fills_unseen_slices():
    train = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "slice": [1, 1, 2, 2],
        }
    )
    test = pd.DataFrame({"slice": [1, 3]})
    target = np.array([1, 0, 0, 0])

    oof, test_prior = foldsafe_terminal_slice_prior(train, test, target, ["slice"])

    assert np.isfinite(oof).all()
    assert np.isfinite(test_prior).all()
    assert test_prior[1] == (target.sum() + 1.0) / (len(target) + 2.0)
