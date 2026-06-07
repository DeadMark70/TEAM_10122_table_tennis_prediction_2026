import numpy as np
import pandas as pd

from analysis_v218_action_weakclass_booster import (
    apply_class_utility_weights,
    select_weighted_candidate_changes,
)


def test_apply_class_utility_weights_preserves_positive_order_inside_class():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "candidate_action": [8, 8, 1],
            "utility": [0.2, 0.1, 0.3],
        }
    )
    out = apply_class_utility_weights(frame, {8: 2.0, 1: 0.5})
    assert out.loc[0, "weighted_utility"] > out.loc[1, "weighted_utility"]
    assert out.loc[0, "weighted_utility"] > out.loc[2, "weighted_utility"]


def test_select_weighted_candidate_changes_respects_total_and_class_caps():
    anchor = np.array([1, 1, 1, 1, 1])
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "candidate_action": [8, 8, 9, 3, 1],
            "weighted_utility": [0.9, 0.8, 0.7, 0.6, 1.0],
        }
    )
    pred, changed = select_weighted_candidate_changes(
        anchor,
        frame,
        total_cap=0.8,
        per_class_cap={8: 1, 9: 1, 3: 1},
        allowed_classes={3, 8, 9},
        min_score=0.0,
    )
    assert changed.sum() == 3
    assert pred.tolist().count(8) == 1
    assert pred.tolist().count(9) == 1
    assert pred.tolist().count(3) == 1
    assert pred[4] == 1
