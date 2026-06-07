def test_submission_schema_contract_rejects_bad_columns():
    import pandas as pd
    from analysis_v360_hierarchical_specialist_gate import validate_submission_schema

    bad = pd.DataFrame({"rally_uid": [1], "actionId": [2], "pointId": [8]})
    result = validate_submission_schema(bad, expected_rows=1)
    assert result["ok"] is False
    assert "serverGetPoint" in result["errors"][0]


def test_anchor_diff_counts_point_action_server_changes():
    import pandas as pd
    from analysis_v360_hierarchical_specialist_gate import compute_anchor_diff

    anchor = pd.DataFrame({
        "rally_uid": [1, 2],
        "actionId": [10, 11],
        "pointId": [8, 9],
        "serverGetPoint": [0.4, 0.6],
    })
    cand = pd.DataFrame({
        "rally_uid": [1, 2],
        "actionId": [10, 12],
        "pointId": [0, 9],
        "serverGetPoint": [0.4, 0.7],
    })
    out = compute_anchor_diff(anchor, cand)
    assert out["action_churn"] == 1
    assert out["point_churn"] == 1
    assert out["server_changed"] == 1
    assert out["point0_additions"] == 1


def test_candidate_policy_penalizes_point0_and_new_rows():
    from analysis_v360_hierarchical_specialist_gate import score_candidate_policy

    score = score_candidate_policy({
        "candidate": "bad",
        "point_churn_vs_v338": 20,
        "action_churn_vs_v173": 0,
        "point0_additions": 5,
        "new_rows_beyond_v338": 10,
        "public_like_delta": 0.002,
        "ordinary_delta": 0.003,
        "weak_class_delta": 0.001,
    })
    safer = score_candidate_policy({
        "candidate": "safe",
        "point_churn_vs_v338": 4,
        "action_churn_vs_v173": 0,
        "point0_additions": 0,
        "new_rows_beyond_v338": 0,
        "public_like_delta": 0.001,
        "ordinary_delta": 0.002,
        "weak_class_delta": 0.0,
    })
    assert safer > score
