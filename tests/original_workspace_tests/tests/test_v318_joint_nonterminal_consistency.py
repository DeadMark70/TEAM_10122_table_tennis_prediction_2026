import numpy as np
import pandas as pd
import pytest

from analysis_v318_joint_nonterminal_consistency import (
    EXPECTED_COLUMNS,
    build_support_tables,
    compatibility_score,
    select_joint_nonterminal_candidates,
    validate_submission_schema,
)


def test_build_support_tables_computes_action_point_conditionals():
    frame = pd.DataFrame(
        {
            "phase_id": [1, 1, 1, 1, 2],
            "lag0_pointId": [7, 8, 9, 4, 7],
            "lag0_actionId": [3, 3, 4, 8, 3],
            "next_actionId": [4, 4, 5, 8, 4],
            "next_pointId": [8, 8, 9, 5, 7],
        }
    )

    tables = build_support_tables(frame)

    pg = tables["point_given_action"]
    row = pg[
        pg["actionId"].eq(4)
        & pg["phase_id"].eq(1)
        & pg["lag0_depth"].eq(3)
        & pg["pointId"].eq(8)
    ].iloc[0]
    assert int(row["support"]) == 2
    assert float(row["prob"]) == pytest.approx(1.0)

    ag = tables["action_given_point"]
    row = ag[
        ag["pointId"].eq(9)
        & ag["phase_id"].eq(1)
        & ag["lag0_action_family"].eq(1)
        & ag["actionId"].eq(5)
    ].iloc[0]
    assert int(row["support"]) == 1
    assert float(row["prob"]) == pytest.approx(1.0)

    score = compatibility_score(
        tables,
        action=4,
        point=8,
        phase=1,
        lag0_depth=3,
        lag0_action_family=1,
    )
    assert score["score"] == pytest.approx(1.0)
    assert score["min_support"] == 2


def test_select_joint_candidates_requires_paired_nonterminal_improvement():
    frame = pd.DataFrame(
        {
            "phase_id": [1, 1, 1, 1],
            "lag0_pointId": [7, 7, 7, 7],
            "lag0_actionId": [3, 3, 3, 3],
        }
    )
    support_rows = pd.DataFrame(
        {
            "phase_id": [1, 1, 1, 1, 1, 1],
            "lag0_pointId": [7, 7, 7, 7, 7, 7],
            "lag0_actionId": [3, 3, 3, 3, 3, 3],
            "next_actionId": [4, 4, 5, 5, 6, 6],
            "next_pointId": [8, 8, 9, 9, 7, 7],
        }
    )
    tables = build_support_tables(support_rows)
    base_point = np.array([7, 8, 0, 4])
    base_action = np.array([3, 4, 4, 6])
    point_prob = np.array(
        [
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.10, 0.80],
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.80, 0.10],
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10, 0.80, 0.05],
            [0.01, 0.01, 0.01, 0.01, 0.80, 0.01, 0.01, 0.10, 0.03, 0.02],
        ]
    )
    action_prob = np.array(
        [
            [0.01, 0.01, 0.01, 0.01, 0.10, 0.80, 0.05],
            [0.01, 0.01, 0.01, 0.01, 0.80, 0.10, 0.05],
            [0.01, 0.01, 0.01, 0.01, 0.10, 0.80, 0.05],
            [0.01, 0.01, 0.01, 0.01, 0.10, 0.05, 0.80],
        ]
    )

    selected = select_joint_nonterminal_candidates(
        frame,
        base_point,
        base_action,
        point_prob,
        action_prob,
        tables,
        budget=4,
        min_score_gain=0.01,
        min_pair_support=2,
    )

    assert selected["row_id"].tolist() == [0]
    row = selected.iloc[0]
    assert int(row["old_pointId"]) == 7
    assert int(row["new_pointId"]) == 9
    assert int(row["old_actionId"]) == 3
    assert int(row["new_actionId"]) == 5
    assert float(row["compat_gain"]) > 0


def test_validate_submission_schema_preserves_columns_and_rejects_extra():
    frame = pd.DataFrame(
        {
            "rally_uid": [1],
            "actionId": [5],
            "pointId": [9],
            "serverGetPoint": [0.25],
        }
    )

    out = validate_submission_schema(frame)

    assert list(out.columns) == EXPECTED_COLUMNS
    with pytest.raises(ValueError, match="submission columns"):
        validate_submission_schema(frame.assign(debug=1))
