import pandas as pd

from analysis_v344_point0_swap_optimizer import build_submission, select_fixed_budget


def test_select_fixed_budget_point0_rows():
    bank = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "anchor_value": [7, 8, 9],
            "candidate_value": [0, 0, 0],
            "utility": [0.2, 0.9, 0.5],
        }
    )
    selected = select_fixed_budget(bank, budget=2)
    assert selected["row_id"].tolist() == [1, 2]


def test_build_submission_only_changes_selected_rows():
    base = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [1, 1],
            "pointId": [7, 8],
            "serverGetPoint": [0.1, 0.2],
        }
    )
    selected = pd.DataFrame({"row_id": [1], "candidate_value": [0]})
    out = build_submission(base, selected)
    assert out["pointId"].tolist() == [7, 0]
