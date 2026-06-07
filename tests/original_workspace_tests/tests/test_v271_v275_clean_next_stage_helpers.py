import numpy as np
import pandas as pd
import pytest

from analysis_v275_public_like_validation_lab import (
    action_family,
    no_ttmatch_path_guard,
    point_depth,
    prefix_bin,
    safe_mad,
    validate_submission_frame,
)


def test_validate_submission_frame_accepts_clean_submission_shape():
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


def test_point_depth_mapping():
    assert [point_depth(i) for i in range(10)] == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_action_family_mapping():
    assert action_family(0) == 0
    assert all(action_family(i) == 1 for i in range(1, 8))
    assert all(action_family(i) == 2 for i in range(8, 12))
    assert all(action_family(i) == 3 for i in range(12, 15))
    assert all(action_family(i) == 4 for i in range(15, 19))


def test_prefix_bin_mapping():
    assert [prefix_bin(i) for i in [1, 2, 3, 4, 6, 7, 10]] == [1, 2, 3, 4, 4, 5, 5]


def test_safe_mad_and_ttmatch_guard():
    assert safe_mad(np.array([0.1, 0.4]), np.array([0.2, 0.1])) == pytest.approx(0.2)
    no_ttmatch_path_guard(["external_data/OpenTTGames/train.csv"])
    with pytest.raises(ValueError):
        no_ttmatch_path_guard(["external_data/TTMATCH/train.csv"])
