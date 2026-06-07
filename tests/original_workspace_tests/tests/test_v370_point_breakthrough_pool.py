def test_collect_point_candidates_dedupes_sources():
    import pandas as pd
    from analysis_v370_point_breakthrough_pool import collect_row_candidates

    anchor = pd.DataFrame({"rally_uid": [1, 2], "pointId": [8, 9]})
    sources = {
        "a": pd.DataFrame({"rally_uid": [1, 2], "pointId": [7, 9]}),
        "b": pd.DataFrame({"rally_uid": [1, 2], "pointId": [7, 9]}),
    }
    bank = collect_row_candidates(anchor, sources)
    assert len(bank) == 1
    assert bank.iloc[0]["candidate_point"] == 7
    assert bank.iloc[0]["support_count"] == 2


def test_no_p0_policy_blocks_unsupported_zero():
    import pandas as pd
    from analysis_v370_point_breakthrough_pool import apply_point0_support_policy

    rows = pd.DataFrame({
        "base_point": [8, 7, 0],
        "candidate_point": [0, 9, 0],
        "point0_support_count": [1, 0, 0],
    })
    out = apply_point0_support_policy(rows, min_point0_support=2)
    assert out["allowed"].tolist() == [False, True, True]


def test_package_preserves_action_server():
    import pandas as pd
    from analysis_v370_point_breakthrough_pool import package_point_submission

    anchor = pd.DataFrame({
        "rally_uid": [1, 2],
        "actionId": [10, 11],
        "pointId": [8, 9],
        "serverGetPoint": [0.2, 0.8],
    })
    out = package_point_submission(anchor, pd.Series([7, 9]))
    assert out["actionId"].tolist() == [10, 11]
    assert out["serverGetPoint"].tolist() == [0.2, 0.8]
