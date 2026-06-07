import numpy as np
import pandas as pd

from analysis_v209_action_selector_reranker import (
    action_point_compatibility,
    best_non_anchor_by_score,
    build_action_candidate_frame,
    geometric_log_blend,
    select_capped_action_changes,
)


def test_geometric_log_blend_uses_soft_anchor_not_onehot_lock():
    anchor = np.array([[0.55, 0.40, 0.05]])
    model = np.array([[0.05, 0.90, 0.05]])
    out = geometric_log_blend(anchor, model, 0.25)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 1] > anchor[0, 1]
    assert out[0, 0] < anchor[0, 0]


def test_action_point_compatibility_prefers_seen_action_point_pairs():
    action = np.array([1, 1, 1, 10, 10])
    point = np.array([8, 8, 9, 2, 2])
    compat = action_point_compatibility(action, point, smoothing=0.1)
    assert compat[1, 8] > compat[1, 2]
    assert compat[10, 2] > compat[10, 8]
    assert np.allclose(compat.sum(axis=1), 1.0)


def test_build_action_candidate_frame_marks_correct_source_and_agreement():
    rows = pd.DataFrame({"fold": [0, 0], "prefix_len": [2, 3], "audit_phase": ["receive", "rally"]})
    sources = {
        "v173": np.array([1, 10]),
        "v166": np.array([4, 10]),
        "v208": np.array([4, 12]),
    }
    frame = build_action_candidate_frame(rows, sources, truth=np.array([4, 10]), anchor_name="v173")
    assert len(frame) == 6
    row0_v166 = frame[(frame["row_id"].eq(0)) & (frame["source"].eq("v166"))].iloc[0]
    assert row0_v166["candidate_action"] == 4
    assert row0_v166["is_correct"] == 1
    assert row0_v166["agreement_count"] == 2
    row1_anchor = frame[(frame["row_id"].eq(1)) & (frame["source"].eq("v173"))].iloc[0]
    assert row1_anchor["is_anchor"] == 1
    assert row1_anchor["agreement_count"] == 2


def test_select_capped_action_changes_keeps_best_positive_deltas_only():
    anchor = np.array([1, 1, 1, 1])
    candidate = np.array([4, 5, 6, 7])
    delta = np.array([0.3, -0.2, 0.8, 0.1])
    pred, mask = select_capped_action_changes(anchor, candidate, delta, max_churn=0.5, min_delta=0.0)
    assert mask.tolist() == [True, False, True, False]
    assert pred.tolist() == [4, 1, 6, 1]


def test_best_non_anchor_by_score_preserves_non_contiguous_row_ids():
    frame = pd.DataFrame(
        {
            "row_id": [2, 2, 5, 5],
            "is_anchor": [1, 0, 1, 0],
            "differs_anchor": [0, 1, 0, 1],
            "anchor_action": [1, 1, 10, 10],
            "candidate_action": [1, 4, 10, 12],
        }
    )
    best, delta, anchor_score = best_non_anchor_by_score(frame, np.array([0.4, 0.8, 0.7, 0.6]))
    assert best[2] == 4
    assert best[5] == 12
    assert np.isclose(delta[2], 0.4)
    assert np.isclose(delta[5], -0.1)
    assert np.isclose(anchor_score[5], 0.7)
