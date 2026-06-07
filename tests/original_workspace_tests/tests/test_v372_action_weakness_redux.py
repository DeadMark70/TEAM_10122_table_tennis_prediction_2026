import pandas as pd


def test_weak_class_priority_marks_known_targets():
    from analysis_v372_action_weakness_redux import weak_class_priority

    assert weak_class_priority(8) > 0
    assert weak_class_priority(14) > 0
    assert weak_class_priority(10) == 0


def test_action_candidate_blocks_serve_like():
    from analysis_v372_action_weakness_redux import block_serve_like_changes

    base = pd.Series([10, 11, 12])
    pred = pd.Series([15, 8, 18])
    out = block_serve_like_changes(base, pred)
    assert out.tolist() == [10, 8, 12]


def test_action_pack_preserves_point_server():
    from analysis_v372_action_weakness_redux import package_action_submission

    anchor = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [10, 11],
            "pointId": [8, 9],
            "serverGetPoint": [0.2, 0.8],
        }
    )
    out = package_action_submission(anchor, pd.Series([8, 11]))
    assert out["pointId"].tolist() == [8, 9]
    assert out["serverGetPoint"].tolist() == [0.2, 0.8]


def test_collect_action_candidates_scores_support_and_weak_priority():
    from analysis_v372_action_weakness_redux import collect_action_candidates

    anchor = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3],
            "actionId": [10, 8, 14],
            "pointId": [7, 8, 9],
            "serverGetPoint": [0.1, 0.2, 0.3],
        }
    )
    sources = {
        "a": pd.DataFrame({"rally_uid": [1, 2, 3], "actionId": [8, 8, 12]}),
        "b": pd.DataFrame({"rally_uid": [1, 2, 3], "actionId": [8, 9, 12]}),
    }

    bank = collect_action_candidates(anchor, sources)

    row = bank.loc[(bank["rally_uid"] == 1) & (bank["candidate_action"] == 8)].iloc[0]
    assert row["support_count"] == 2
    assert row["weak_priority"] > 0
    assert row["score"] > 0
    assert not bank["candidate_action"].isin([15, 16, 17, 18]).any()
