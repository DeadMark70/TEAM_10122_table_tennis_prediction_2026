import pandas as pd

from analysis_v353_v338_row_causal_audit import (
    build_candidate_summary,
    build_group_audit,
    build_leave_group_submission,
    candidate_metrics,
    identify_v338_changed_rows,
)


def _submission(points, actions=None, servers=None):
    n = len(points)
    return pd.DataFrame(
        {
            "rally_uid": [f"r{i}" for i in range(n)],
            "actionId": actions or list(range(1, n + 1)),
            "pointId": points,
            "serverGetPoint": servers or [round(0.1 * (i + 1), 3) for i in range(n)],
        }
    )


def test_identify_v338_changed_rows_includes_grouping_fields_and_trust_quantiles():
    v306 = _submission([8, 8, 7, 4, 5, 5])
    v338 = _submission([9, 7, 9, 4, 5, 2])
    trust = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 5],
            "final_trust_score": [0.1, 2.0, 3.0, 1.0],
            "trust_score": [0.1, 2.0, 3.0, 1.0],
        }
    )

    changed = identify_v338_changed_rows(v306, v338, trust_scores=trust)

    assert changed["row_id"].tolist() == [0, 1, 2, 5]
    assert changed["transition"].tolist() == ["8->9", "8->7", "7->9", "5->2"]
    assert changed["old_point"].tolist() == [8, 8, 7, 5]
    assert changed["new_point"].tolist() == [9, 7, 9, 2]
    assert changed["depth_change"].tolist() == [0, 0, 0, -1]
    assert set(changed["trust_quantile"]) == {"q1_low", "q2_midlow", "q3_midhigh", "q4_high"}


def test_identify_v338_changed_rows_does_not_duplicate_on_gate_alternatives():
    v306 = _submission([8, 8, 7])
    v338 = _submission([9, 7, 7])
    gate = pd.DataFrame(
        {
            "row_id": [0, 0, 1],
            "anchor_value": [8, 8, 8],
            "candidate_value": [9, 7, 7],
            "trust_score": [3.0, -1.0, 2.0],
            "risk_score": [0.0, 2.0, 0.0],
        }
    )

    changed = identify_v338_changed_rows(v306, v338, gate_scores=gate)

    assert changed["row_id"].tolist() == [0, 1]
    assert changed["v348_trust_score"].tolist() == [3.0, 2.0]


def test_leave_group_builder_only_reverts_v338_rows_and_preserves_action_server():
    v306 = _submission([8, 8, 7, 4], actions=[9, 8, 7, 6], servers=[0.9, 0.8, 0.7, 0.6])
    v338 = _submission([9, 7, 9, 4], actions=[4, 3, 2, 1], servers=[0.4, 0.3, 0.2, 0.1])

    out = build_leave_group_submission(v306, v338, rows_to_revert=[1])

    assert out["pointId"].tolist() == [9, 8, 9, 4]
    assert out["actionId"].tolist() == v338["actionId"].tolist()
    assert out["serverGetPoint"].tolist() == v338["serverGetPoint"].tolist()


def test_candidate_metrics_show_no_new_rows_beyond_v338_or_point0_additions():
    v306 = _submission([8, 8, 7, 4, 0])
    v338 = _submission([9, 7, 9, 4, 0])
    candidate = build_leave_group_submission(v306, v338, rows_to_revert=[1])

    metrics = candidate_metrics(v306, v338, candidate)

    assert metrics["remaining_v338_rows"] == 2
    assert metrics["new_rows_beyond_v338"] == 0
    assert metrics["point0_additions_vs_v306"] == 0
    assert metrics["action_preserved"] is True
    assert metrics["server_preserved"] is True


def test_group_audit_and_candidate_summary_skip_groups_with_too_few_remaining_rows():
    v306 = _submission([8, 8, 7, 5])
    v338 = _submission([9, 7, 9, 2])
    trust = pd.DataFrame({"row_id": [0, 1, 2, 3], "final_trust_score": [0.1, 0.2, 0.3, 0.4]})
    changed = identify_v338_changed_rows(v306, v338, trust_scores=trust)

    groups = build_group_audit(changed)
    candidates = build_candidate_summary(v306, v338, groups, top_n=10)

    assert {"transition:8->9", "old_point:8", "new_point:9", "depth_change:0"}.issubset(set(groups["group_key"]))
    assert all(candidates["remaining_v338_rows"] >= 2)
    assert all(candidates["new_rows_beyond_v338"] == 0)
    assert all(candidates["point0_additions_vs_v306"] == 0)
