import pandas as pd

from analysis_v282_joint_context_support_optimizer import (
    build_transition_tables,
    context_support_features,
    latest_prefix_context,
)


def test_context_support_uses_exact_then_backoff_counts():
    train = pd.DataFrame(
        [
            {"rally_uid": 1, "strikeNumber": 1, "actionId": 15, "pointId": 4},
            {"rally_uid": 1, "strikeNumber": 2, "actionId": 10, "pointId": 2},
            {"rally_uid": 1, "strikeNumber": 3, "actionId": 1, "pointId": 8},
            {"rally_uid": 2, "strikeNumber": 1, "actionId": 15, "pointId": 4},
            {"rally_uid": 2, "strikeNumber": 2, "actionId": 10, "pointId": 2},
            {"rally_uid": 2, "strikeNumber": 3, "actionId": 3, "pointId": 0},
        ]
    )
    tables = build_transition_tables(train)

    exact = context_support_features(
        tables,
        phase="receive",
        last_action=15,
        last_point=4,
        candidate_action=10,
        candidate_point=2,
    )
    assert exact["support_level"] == "phase_action_point"
    assert exact["support_count"] == 2
    assert exact["pair_count"] == 2
    assert exact["pair_prob"] == 1.0

    backoff = context_support_features(
        tables,
        phase="receive",
        last_action=15,
        last_point=9,
        candidate_action=10,
        candidate_point=2,
    )
    assert backoff["support_level"] == "phase_action"
    assert backoff["support_count"] == 2
    assert backoff["pair_prob"] == 1.0


def test_context_support_returns_global_for_unseen_context():
    train = pd.DataFrame(
        [
            {"rally_uid": 1, "strikeNumber": 1, "actionId": 15, "pointId": 4},
            {"rally_uid": 1, "strikeNumber": 2, "actionId": 10, "pointId": 2},
        ]
    )
    tables = build_transition_tables(train)

    features = context_support_features(
        tables,
        phase="rally",
        last_action=13,
        last_point=8,
        candidate_action=10,
        candidate_point=2,
    )
    assert features["support_level"] == "global"
    assert features["support_count"] == 1
    assert features["pair_count"] == 1
    assert features["pair_prob"] == 1.0


def test_latest_prefix_context_keeps_last_observed_stroke():
    test = pd.DataFrame(
        [
            {"rally_uid": 10, "strikeNumber": 1, "actionId": 15, "pointId": 4},
            {"rally_uid": 10, "strikeNumber": 2, "actionId": 10, "pointId": 2},
            {"rally_uid": 11, "strikeNumber": 1, "actionId": 16, "pointId": 5},
        ]
    )
    context = latest_prefix_context(test)
    row = context.set_index("rally_uid").loc[10]
    assert int(row["last_action"]) == 10
    assert int(row["last_point"]) == 2
    assert row["phase"] == "third_ball"
