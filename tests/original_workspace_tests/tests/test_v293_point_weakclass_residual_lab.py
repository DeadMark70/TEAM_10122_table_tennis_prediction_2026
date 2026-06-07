import numpy as np
import pandas as pd

from analysis_v293_point_weakclass_residual_lab import (
    LOCAL_MOVES,
    POINT_GROUPS,
    apply_point_caps,
    apply_candidates,
    normalize_score01,
    point0_addition_allowed,
    point_depth,
    point_side,
    preserve_long_identity,
)


def test_point_depth_side_mapping():
    assert point_depth(0) == "zero"
    assert point_depth(1) == "short"
    assert point_depth(4) == "half"
    assert point_depth(7) == "long"
    assert point_side(1) == "forehand"
    assert point_side(5) == "middle"
    assert point_side(9) == "backhand"


def test_preserve_long_identity_only_allows_long_to_long():
    assert preserve_long_identity(7, 8)
    assert preserve_long_identity(9, 7)
    assert not preserve_long_identity(8, 4)
    assert not preserve_long_identity(4, 8)


def test_point0_addition_allowed_requires_high_confidence():
    assert point0_addition_allowed(
        base_point=7, p0_score=0.95, phase="rally", terminal_proxy=0.9
    )
    assert not point0_addition_allowed(
        base_point=7, p0_score=0.60, phase="rally", terminal_proxy=0.9
    )
    assert not point0_addition_allowed(
        base_point=0, p0_score=0.95, phase="rally", terminal_proxy=0.9
    )


def test_apply_point_caps_limits_changed_rows():
    base = np.array([8, 8, 8, 8, 8])
    candidates = pd.DataFrame(
        {"row_id": [0, 1, 2], "candidate_point": [7, 9, 7], "score": [0.9, 0.8, 0.7]}
    )
    pred, selected = apply_point_caps(base, candidates, max_churn=0.4)
    assert selected.sum() == 2
    assert (pred != base).sum() == 2


def test_apply_candidates_accepts_empty_candidate_frame():
    base = np.array([7, 8, 9])
    pred, selected, selected_rows = apply_candidates(base, pd.DataFrame(), cap=0.01)
    np.testing.assert_array_equal(pred, base)
    assert not selected.any()
    assert selected_rows.empty


def test_constants_and_normalize_score_are_available():
    assert POINT_GROUPS["long789"] == [7, 8, 9]
    assert LOCAL_MOVES["rare134"][4] == [1, 3, 7]
    got = normalize_score01(np.array([-1, 0.25, np.inf, np.nan, 2]))
    np.testing.assert_allclose(got, np.array([0.0, 0.25, 0.0, 0.0, 1.0]))
