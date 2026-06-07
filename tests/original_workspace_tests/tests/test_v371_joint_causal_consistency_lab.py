def test_action_family_point_depth_support_score():
    from analysis_v371_joint_causal_consistency_lab import compatibility_score

    assert compatibility_score("control", "short") > compatibility_score("control", "long")
    assert compatibility_score("attack", "long") > compatibility_score("attack", "short")


def test_terminal_inconsistency_flags_nonzero_point_with_zero_action():
    import pandas as pd
    from analysis_v371_joint_causal_consistency_lab import terminal_inconsistency_flags

    rows = pd.DataFrame({"actionId": [0, 10, 0], "pointId": [0, 0, 8]})
    out = terminal_inconsistency_flags(rows)
    assert out.tolist() == [False, True, True]


def test_correction_does_not_change_server():
    import pandas as pd
    from analysis_v371_joint_causal_consistency_lab import package_joint_candidate

    anchor = pd.DataFrame({
        "rally_uid": [1, 2],
        "actionId": [10, 11],
        "pointId": [8, 9],
        "serverGetPoint": [0.2, 0.8],
    })
    out = package_joint_candidate(anchor, action_pred=[10, 12], point_pred=[7, 9])
    assert out["serverGetPoint"].tolist() == [0.2, 0.8]
