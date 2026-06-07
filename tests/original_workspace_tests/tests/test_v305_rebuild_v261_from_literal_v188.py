import numpy as np
import pandas as pd

from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column, select_top_margin_changes


def test_select_top_margin_changes_respects_cap():
    base = np.array([1, 1, 1, 1, 1])
    cand = np.array([2, 1, 3, 4, 1])
    margin = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    mask = select_top_margin_changes(base, cand, margin, cap=0.4)
    assert mask.sum() == 2
    assert mask.tolist() == [True, False, True, False, False]


def test_select_top_margin_changes_ignores_nonpositive_margin():
    base = np.array([1, 1, 1, 1])
    cand = np.array([2, 3, 1, 4])
    margin = np.array([0.2, -0.1, 0.9, 0.0])
    mask = select_top_margin_changes(base, cand, margin, cap=1.0)
    assert mask.tolist() == [True, False, False, False]


def test_point_column_accepts_v305_export_schema():
    frame = pd.DataFrame({"cap0p05_point_pred": [1, 2]})
    assert point_column(frame) == "cap0p05_point_pred"


def test_align_train_to_literal_meta_uses_unique_prefix_key():
    train = pd.DataFrame(
        {
            "rally_uid": [1, 1, 2],
            "prefix_len": [1, 2, 1],
            "next_actionId": [3, 4, 5],
            "next_pointId": [6, 7, 8],
            "value": [10, 20, 30],
        }
    )
    meta = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "prefix_len": [2, 1],
            "next_actionId": [4, 5],
            "next_pointId": [7, 8],
        }
    )
    out = align_train_to_literal_meta(train, meta)
    assert out["value"].tolist() == [20, 30]
