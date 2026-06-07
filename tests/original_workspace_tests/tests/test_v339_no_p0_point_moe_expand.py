import numpy as np
import pandas as pd

from analysis_v339_no_p0_point_moe_expand import (
    build_export_frame,
    enforce_no_p0_add,
    select_budget,
)


def test_no_p0_export_blocks_nonzero_to_zero():
    base = np.array([7, 8, 0, 9])
    cand = np.array([0, 9, 0, 7])
    out = enforce_no_p0_add(base, cand)
    assert out.tolist() == [7, 9, 0, 7]


def test_budget_selector_monotonic_changes():
    base = np.array([8, 8, 8, 8])
    cand = np.array([7, 9, 4, 6])
    score = np.array([0.4, 0.9, 0.2, 0.7])
    assert (select_budget(base, cand, score, 2) != base).sum() == 2
    assert (select_budget(base, cand, score, 3) != base).sum() == 3


def test_export_frame_preserves_action_and_server_and_blocks_p0_add():
    anchor = pd.DataFrame(
        {
            "rally_uid": ["a", "b", "c"],
            "actionId": [4, 8, 15],
            "pointId": [8, 0, 9],
            "serverGetPoint": [0.2, 0.5, 0.8],
        }
    )

    out = build_export_frame(anchor, np.array([0, 7, 6]))

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [4, 8, 15]
    assert out["serverGetPoint"].tolist() == [0.2, 0.5, 0.8]
    assert out["pointId"].tolist() == [8, 7, 6]
