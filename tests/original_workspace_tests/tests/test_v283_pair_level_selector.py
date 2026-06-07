import pandas as pd

from analysis_v283_pair_level_selector import (
    build_pair_training_candidates,
    select_improvements,
)


def test_build_pair_training_candidates_includes_true_pair_and_negatives():
    examples = pd.DataFrame(
        [
            {
                "rally_uid": 1,
                "phase": "receive",
                "last_action": 15,
                "last_point": 4,
                "candidate_action": 10,
                "candidate_point": 2,
            },
            {
                "rally_uid": 2,
                "phase": "receive",
                "last_action": 15,
                "last_point": 4,
                "candidate_action": 1,
                "candidate_point": 8,
            },
        ]
    )
    pairs = [(10, 2), (1, 8), (13, 0)]
    candidates = build_pair_training_candidates(examples, pairs, max_negative_pairs=2)
    assert len(candidates) >= 4
    assert candidates["label"].sum() == 2
    true_pairs = candidates[candidates["label"] == 1][["candidate_action", "candidate_point"]]
    assert set(map(tuple, true_pairs.to_numpy())) == {(10, 2), (1, 8)}


def test_select_improvements_requires_margin_and_filters_point0_additions():
    scored = pd.DataFrame(
        [
            {
                "rally_uid": 1,
                "candidate_action": 4,
                "candidate_point": 8,
                "anchor_action": 4,
                "anchor_point": 8,
                "pair_changed": False,
                "action_changed": False,
                "point_changed": False,
                "pred_correct_prob": 0.40,
                "compatibility_score": 0.95,
            },
            {
                "rally_uid": 1,
                "candidate_action": 10,
                "candidate_point": 2,
                "anchor_action": 4,
                "anchor_point": 8,
                "pair_changed": True,
                "action_changed": True,
                "point_changed": True,
                "pred_correct_prob": 0.48,
                "compatibility_score": 0.90,
            },
            {
                "rally_uid": 2,
                "candidate_action": 5,
                "candidate_point": 7,
                "anchor_action": 5,
                "anchor_point": 7,
                "pair_changed": False,
                "action_changed": False,
                "point_changed": False,
                "pred_correct_prob": 0.50,
                "compatibility_score": 0.95,
            },
            {
                "rally_uid": 2,
                "candidate_action": 3,
                "candidate_point": 0,
                "anchor_action": 5,
                "anchor_point": 7,
                "pair_changed": True,
                "action_changed": True,
                "point_changed": True,
                "pred_correct_prob": 0.90,
                "compatibility_score": 0.90,
            },
        ]
    )
    selected = select_improvements(scored, max_rows=5, margin=0.05, require_both_changed=True)
    assert selected["rally_uid"].tolist() == [1]
