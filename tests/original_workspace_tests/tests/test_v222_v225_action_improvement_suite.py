import numpy as np
import pandas as pd

from analysis_v222_v225_action_improvement_suite import (
    build_style_tables,
    compat_delta,
    select_budgeted_changes,
    style_lift_for_candidate,
)


def test_style_lift_uses_player_specific_rates():
    rows = pd.DataFrame(
        {
            "player": [1, 1, 1, 2, 2],
            "next_action": [8, 8, 10, 10, 10],
        }
    )
    tables = build_style_tables(rows, smoothing=1.0)

    lift_8 = style_lift_for_candidate(tables, player=1, candidate_action=8)
    lift_10 = style_lift_for_candidate(tables, player=1, candidate_action=10)

    assert lift_8 > lift_10


def test_compat_delta_prefers_candidate_matching_point():
    table = np.ones((19, 10), dtype=float)
    table[12, 0] = 20.0
    table[10, 0] = 2.0
    table = table / table.sum(axis=1, keepdims=True)

    assert compat_delta(table, candidate_action=12, anchor_action=10, point_id=0) > 0


def test_select_budgeted_changes_respects_caps_and_forbidden_serves():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "anchor_action": [3, 10, 1, 6],
            "candidate_action": [5, 12, 15, 5],
            "score": [0.9, 0.8, 10.0, 0.7],
            "score_combined": [0.9, 0.8, 10.0, 0.7],
            "terminal_mismatch": [0, 0, 0, 0],
        }
    )
    anchor = np.array([3, 10, 1, 6])

    pred, selected = select_budgeted_changes(
        anchor,
        frame,
        score_col="score",
        total_cap=0.75,
        per_class_cap={5: 1, 12: 1},
    )

    assert selected.sum() == 2
    assert pred.tolist() == [5, 12, 1, 6]
