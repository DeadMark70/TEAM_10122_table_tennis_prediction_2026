def test_point_depth_mapping():
    from analysis_v362_point_hierarchical_specialists import point_to_depth

    assert point_to_depth(0) == "terminal"
    assert point_to_depth(1) == "short"
    assert point_to_depth(4) == "half"
    assert point_to_depth(9) == "long"


def test_no_p0_policy_blocks_nonterminal_to_zero():
    import pandas as pd
    from analysis_v362_point_hierarchical_specialists import apply_no_p0_policy

    base = pd.Series([8, 7, 0], name="base")
    proposed = pd.Series([0, 9, 0], name="proposed")
    out = apply_no_p0_policy(base, proposed, allow_existing_zero=True)
    assert list(out) == [8, 9, 0]


def test_package_preserves_action_and_server():
    import pandas as pd
    from analysis_v362_point_hierarchical_specialists import package_point_candidate

    anchor = pd.DataFrame({
        "rally_uid": [1, 2],
        "actionId": [10, 11],
        "pointId": [8, 9],
        "serverGetPoint": [0.4, 0.6],
    })
    out = package_point_candidate(anchor, pd.Series([7, 9]))
    assert list(out["actionId"]) == [10, 11]
    assert list(out["pointId"]) == [7, 9]
    assert list(out["serverGetPoint"]) == [0.4, 0.6]
