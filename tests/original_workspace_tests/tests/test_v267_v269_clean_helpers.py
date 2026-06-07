import numpy as np
import pandas as pd
import pytest

from analysis_v267_macro_f1_action_teacher import (
    cap_by_score,
    class_prior_logit_adjustment,
    normalize_rows_safe,
    validate_submission_frame,
)
from analysis_v268_macro_f1_point_residual import point0_rate_ok


def test_validate_submission_frame_accepts_expected_shape():
    df = pd.DataFrame(
        {
            "rally_uid": range(1845),
            "actionId": [1] * 1845,
            "pointId": [8] * 1845,
            "serverGetPoint": [0.5] * 1845,
        }
    )
    validate_submission_frame(df)


def test_validate_submission_frame_rejects_wrong_columns():
    df = pd.DataFrame({"rally_uid": [1], "actionId": [1], "pointId": [8]})
    with pytest.raises(ValueError):
        validate_submission_frame(df, expected_rows=1)


def test_normalize_rows_safe_handles_nan_and_zero_rows():
    arr = normalize_rows_safe(np.array([[np.nan, 2.0], [0.0, 0.0]]))
    assert np.all(np.isfinite(arr))
    assert np.allclose(arr.sum(axis=1), 1.0)
    assert np.allclose(arr[1], [0.5, 0.5])


def test_cap_by_score_selects_top_budget():
    mask = cap_by_score(np.array([0.1, 0.9, 0.5, 0.2]), 0.5)
    assert mask.tolist() == [False, True, True, False]


def test_class_prior_logit_adjustment_boosts_rare_classes():
    adj = class_prior_logit_adjustment(np.array([100, 10, 1]), tau=0.5)
    assert adj[2] > adj[1] > adj[0]


def test_point0_rate_ok_bounds():
    assert point0_rate_ok(np.array([0] * 25 + [8] * 75), lower=0.20, upper=0.30)
    assert not point0_rate_ok(np.array([0] * 80 + [8] * 20), lower=0.20, upper=0.30)
