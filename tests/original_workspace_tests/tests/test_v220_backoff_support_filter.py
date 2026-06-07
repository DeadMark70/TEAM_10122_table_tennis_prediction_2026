import numpy as np
import pandas as pd

from analysis_v220_action_backoff_support_filter import (
    backoff_support_score,
    build_next_action_examples,
    filter_supported_changes,
)


def test_backoff_support_score_prefers_more_common_candidate():
    examples = pd.DataFrame(
        {
            "phase": ["rally"] * 6,
            "lag0_action": [10] * 6,
            "lag0_point": [5] * 6,
            "lag0_depth": [2] * 6,
            "lag0_spin": [2] * 6,
            "lag0_strength": [2] * 6,
            "next_action": [12, 12, 12, 10, 1, 1],
        }
    )
    score, details = backoff_support_score(
        examples,
        phase="rally",
        lag0_action=10,
        lag0_point=5,
        lag0_depth=2,
        lag0_spin=2,
        lag0_strength=2,
        base_action=10,
        cand_action=12,
        min_support=3,
    )
    assert score > 0
    assert details[0]["cand_rate"] > details[0]["base_rate"]


def test_filter_supported_changes_keeps_positive_score_only():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1],
            "base_action": [10, 6],
            "cand_action": [12, 5],
            "support_score": [2, -1],
            "support_margin": [0.2, -0.1],
        }
    )
    out = filter_supported_changes(frame, mode="balanced")
    assert out["row_id"].tolist() == [0]


def test_build_next_action_examples_has_expected_rows():
    train = pd.DataFrame(
        {
            "rally_uid": [1, 1, 1],
            "strikeNumber": [1, 2, 3],
            "actionId": [15, 10, 12],
            "pointId": [5, 6, 0],
            "spinId": [2, 2, 0],
            "strengthId": [2, 2, 0],
        }
    )
    out = build_next_action_examples(train)
    assert out.shape[0] == 2
    assert out["next_action"].tolist() == [10, 12]
