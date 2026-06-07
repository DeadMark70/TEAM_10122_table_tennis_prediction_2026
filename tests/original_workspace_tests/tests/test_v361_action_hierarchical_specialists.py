def test_action_family_mapping():
    from analysis_v361_action_hierarchical_specialists import action_to_family

    assert action_to_family(0) == "zero"
    assert action_to_family(3) == "attack"
    assert action_to_family(10) == "control"
    assert action_to_family(13) == "defensive"
    assert action_to_family(16) == "serve"


def test_serve_like_actions_blocked_on_hidden_next():
    import pandas as pd
    from analysis_v361_action_hierarchical_specialists import block_serve_like_actions

    base = pd.Series([10, 11, 12])
    proposed = pd.Series([15, 14, 18])
    out = block_serve_like_actions(base, proposed)
    assert list(out) == [10, 14, 12]


def test_package_preserves_point_and_server():
    import pandas as pd
    from analysis_v361_action_hierarchical_specialists import package_action_candidate

    anchor = pd.DataFrame({
        "rally_uid": [1, 2],
        "actionId": [10, 11],
        "pointId": [8, 9],
        "serverGetPoint": [0.4, 0.6],
    })
    out = package_action_candidate(anchor, pd.Series([7, 11]))
    assert list(out["actionId"]) == [7, 11]
    assert list(out["pointId"]) == [8, 9]
    assert list(out["serverGetPoint"]) == [0.4, 0.6]
