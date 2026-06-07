import numpy as np
import pandas as pd

from analysis_v219_action_classwise_budget_search import (
    best_budget_for_class,
    select_classwise_budget_changes,
    top_candidates_for_class,
)


def test_top_candidates_for_class_keeps_best_candidate_per_row():
    frame = pd.DataFrame(
        {
            "row_id": [0, 0, 1, 2],
            "candidate_action": [8, 8, 8, 9],
            "anchor_action": [1, 1, 1, 1],
            "utility": [0.1, 0.3, 0.2, 0.9],
        }
    )
    out = top_candidates_for_class(frame, 8)
    assert out["row_id"].tolist() == [0, 1]
    assert out["utility"].tolist() == [0.3, 0.2]


def test_best_budget_for_class_can_choose_zero_when_no_gain():
    y = np.array([1, 1, 1, 1])
    anchor = np.array([1, 1, 1, 1])
    frame = pd.DataFrame(
        {
            "row_id": [0, 1],
            "candidate_action": [8, 8],
            "anchor_action": [1, 1],
            "utility": [0.9, 0.8],
        }
    )
    result = best_budget_for_class(y, anchor, frame, 8, max_k=2, labels=[1, 8])
    assert result["best_k"] == 0
    assert result["best_delta"] == 0.0


def test_select_classwise_budget_changes_resolves_row_conflicts_by_score():
    anchor = np.array([1, 1, 1])
    frame = pd.DataFrame(
        {
            "row_id": [0, 0, 1, 2],
            "candidate_action": [8, 9, 8, 9],
            "anchor_action": [1, 1, 1, 1],
            "utility": [0.4, 0.9, 0.8, 0.7],
        }
    )
    pred, changed = select_classwise_budget_changes(anchor, frame, {8: 2, 9: 2})
    assert changed.sum() == 3
    assert pred.tolist() == [9, 8, 9]
