import numpy as np
import pandas as pd
import pytest

from analysis_v277_v272b_point_refinement import (
    agreement_mask,
    changed_mask,
    no_point0_add_mask,
    nonterminal_change_mask,
    validate_submission_frame,
)


def test_validate_submission_frame_accepts_expected():
    df = pd.DataFrame(
        {
            "rally_uid": range(1845),
            "actionId": [1] * 1845,
            "pointId": [8] * 1845,
            "serverGetPoint": [0.5] * 1845,
        }
    )
    validate_submission_frame(df)


def test_validate_submission_frame_rejects_bad_columns():
    df = pd.DataFrame({"rally_uid": [1], "actionId": [1], "pointId": [8]})
    with pytest.raises(ValueError):
        validate_submission_frame(df, expected_rows=1)


def test_masks():
    anchor = np.array([0, 8, 8, 7, 4])
    model = np.array([0, 0, 9, 7, 5])
    table = np.array([0, 9, 9, 8, 5])
    assert changed_mask(anchor, model).tolist() == [False, True, True, False, True]
    assert no_point0_add_mask(anchor, model).tolist() == [True, False, True, True, True]
    assert nonterminal_change_mask(anchor, model).tolist() == [False, False, True, False, True]
    assert agreement_mask(anchor, model, table).tolist() == [False, False, True, False, True]
