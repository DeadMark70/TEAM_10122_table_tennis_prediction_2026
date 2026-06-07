import numpy as np
import pandas as pd

from analysis_v287_weak_action_gated_ensemble import (
    apply_allowed_action_filter,
    apply_row_cap,
    changed_row_report,
    select_best_candidate_per_row,
)


def test_allowed_action_filter_keeps_only_requested_specialist_actions():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "anchor_action": [10, 10, 10, 10],
            "candidate_action": [5, 7, 8, 14],
            "specialist_score": [0.9, 0.8, 0.95, 0.99],
            "support_count": [20, 20, 20, 20],
        }
    )
    out = apply_allowed_action_filter(frame, {5, 7})
    assert out["candidate_action"].tolist() == [5, 7]


def test_select_best_candidate_per_row_prefers_high_score_then_support():
    frame = pd.DataFrame(
        {
            "row_id": [0, 0, 1],
            "anchor_action": [10, 10, 12],
            "candidate_action": [5, 7, 3],
            "specialist_score": [0.8, 0.8, 0.9],
            "support_count": [10, 30, 10],
        }
    )
    out = select_best_candidate_per_row(frame)
    assert out.sort_values("row_id")["candidate_action"].tolist() == [7, 3]


def test_apply_row_cap_changes_only_top_ranked_rows():
    anchor = np.array([10, 10, 10, 10])
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "candidate_action": [5, 7, 3],
            "specialist_score": [0.7, 0.9, 0.8],
            "support_count": [10, 10, 10],
        }
    )
    pred, selected = apply_row_cap(anchor, frame, max_rows=2)
    assert selected.tolist() == [False, True, True, False]
    assert pred.tolist() == [10, 7, 3, 10]


def test_changed_row_report_counts_changes_by_action():
    anchor = np.array([10, 10, 12, 13])
    pred = np.array([10, 5, 7, 13])
    report = changed_row_report(anchor, pred)
    assert report["changed_rows"] == 2
    assert report["changed_to_5"] == 1
    assert report["changed_to_7"] == 1
