import numpy as np
import pandas as pd
import pytest

from analysis_v313_joint_terminal_consistency import (
    EXPECTED_COLUMNS,
    action_terminal_masks,
    decision_label,
    select_joint_point0_additions,
    validate_submission_schema,
)


def test_action_terminal_masks_flags_support_and_strong_nonterminal_veto():
    action_prob = np.array(
        [
            [0.42, 0.10, 0.05, 0.03],
            [0.03, 0.52, 0.20, 0.10],
            [0.12, 0.24, 0.20, 0.18],
            [0.20, 0.21, 0.18, 0.17],
        ]
    )
    base_action = np.array([2, 1, 1, 0])

    masks = action_terminal_masks(action_prob, base_action)

    assert masks["terminal_compatible"].tolist() == [True, False, True, True]
    assert masks["strong_nonterminal"].tolist() == [False, True, False, False]
    assert masks["eligible_action_source"].tolist() == [True, False, True, True]


def test_select_joint_point0_additions_requires_terminal_compatible_and_vetoes_nonterminal():
    base_point = np.array([9, 8, 7, 0, 5])
    point_prob = np.array(
        [
            [0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
            [0.80, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.10, 0.01],
            [0.70, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20, 0.01, 0.01],
            [0.99, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.60, 0.01, 0.01, 0.01, 0.01, 0.50, 0.01, 0.01, 0.01, 0.01],
        ]
    )
    action_prob = np.array(
        [
            [0.41, 0.10, 0.08],
            [0.02, 0.70, 0.10],
            [0.13, 0.20, 0.19],
            [0.50, 0.20, 0.10],
            [0.01, 0.40, 0.20],
        ]
    )
    base_action = np.array([2, 1, 1, 0, 2])

    pred, selected, audit = select_joint_point0_additions(base_point, point_prob, action_prob, base_action, budget=3)

    assert selected.tolist() == [True, False, True, False, False]
    assert pred.tolist() == [0, 8, 0, 0, 5]
    assert audit["positive_point0_margin"].tolist() == [True, True, True, False, True]
    assert audit["strong_nonterminal"].tolist() == [False, True, False, False, True]


def test_validate_submission_schema_preserves_columns_and_rejects_extra():
    frame = pd.DataFrame(
        {
            "rally_uid": [1],
            "actionId": [0],
            "pointId": [0],
            "serverGetPoint": [1],
        }
    )

    out = validate_submission_schema(frame)

    assert list(out.columns) == EXPECTED_COLUMNS
    with pytest.raises(ValueError, match="submission columns"):
        validate_submission_schema(frame.assign(debug=1))


def test_decision_label_compares_to_v307_budget24_delta_and_risk():
    assert decision_label(0.0048, 24, 0.00469246629968606, 24) == "REVIEW"
    assert decision_label(0.0047, 18, 0.00469246629968606, 24) == "REVIEW"
    assert decision_label(0.0040, 18, 0.00469246629968606, 24) == "DO_NOT_UPLOAD"
