import numpy as np
import pandas as pd
import pytest

from analysis_v310_terminal_action_point_consistency import (
    EXPECTED_COLUMNS,
    terminal_compatibility_mask,
    validate_submission_schema,
)


def test_terminal_compatibility_mask_requires_action0_exactly_for_point0():
    point = np.array([0, 0, 1, 9, 0, 4])
    action = np.array([0, 13, 13, 0, 7, 2])

    mask = terminal_compatibility_mask(point, action)

    assert mask.tolist() == [True, False, True, False, False, True]


def test_terminal_compatibility_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        terminal_compatibility_mask(np.array([0, 1]), np.array([0]))


def test_validate_submission_schema_preserves_expected_column_order():
    frame = pd.DataFrame(
        {
            "rally_uid": [11, 12],
            "actionId": [0, 13],
            "pointId": [0, 2],
            "serverGetPoint": [1, 0],
        }
    )

    out = validate_submission_schema(frame)

    assert list(out.columns) == EXPECTED_COLUMNS
    assert out["actionId"].tolist() == [0, 13]


def test_validate_submission_schema_rejects_missing_or_extra_columns():
    missing = pd.DataFrame({"rally_uid": [1], "actionId": [0], "pointId": [0]})
    extra = pd.DataFrame(
        {
            "rally_uid": [1],
            "actionId": [0],
            "pointId": [0],
            "serverGetPoint": [1],
            "debug": [99],
        }
    )

    with pytest.raises(ValueError, match="submission columns"):
        validate_submission_schema(missing)
    with pytest.raises(ValueError, match="submission columns"):
        validate_submission_schema(extra)
