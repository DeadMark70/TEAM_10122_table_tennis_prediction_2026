import pandas as pd

from analysis_v435_residual_packager import (
    apply_ranked_candidates,
    package_residual_submission,
)


def _anchor():
    return pd.DataFrame(
        {
            "rally_uid": ["r1", "r2", "r3"],
            "actionId": [1, 10, 13],
            "pointId": [7, 4, 0],
            "serverGetPoint": [0.2, 0.7, 0.4],
        }
    )


def test_apply_ranked_candidates_blocks_point0_additions_by_default():
    anchor = _anchor()
    candidates = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "candidate_pointId": [0, 8],
            "utility": [10.0, 2.0],
        }
    )

    out, report = apply_ranked_candidates(
        anchor,
        candidates,
        target_col="pointId",
        candidate_col="candidate_pointId",
        max_changes=2,
    )

    assert out["pointId"].tolist() == [7, 8, 0]
    assert report["applied_changes"] == 1
    assert report["blocked_point0_additions"] == 1


def test_apply_ranked_candidates_blocks_serve_action_explosion():
    anchor = _anchor()
    candidates = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "candidate_actionId": [15, 3],
            "utility": [9.0, 1.0],
        }
    )

    out, report = apply_ranked_candidates(
        anchor,
        candidates,
        target_col="actionId",
        candidate_col="candidate_actionId",
        max_changes=2,
    )

    assert out["actionId"].tolist() == [1, 3, 13]
    assert report["applied_changes"] == 1
    assert report["blocked_serve_additions"] == 1


def test_package_residual_submission_preserves_schema_and_server():
    anchor = _anchor()
    action_candidates = pd.DataFrame(
        {"rally_uid": ["r1"], "candidate_actionId": [3], "utility": [1.5]}
    )
    point_candidates = pd.DataFrame(
        {"rally_uid": ["r2"], "candidate_pointId": [8], "utility": [1.0]}
    )

    submission, report = package_residual_submission(
        anchor,
        action_candidates=action_candidates,
        point_candidates=point_candidates,
        action_top=1,
        point_top=1,
        name="tiny",
    )

    assert list(submission.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert submission["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
    assert submission["actionId"].tolist() == [3, 10, 13]
    assert submission["pointId"].tolist() == [7, 8, 0]
    assert report["total_changed_rows"] == 2


def test_package_accepts_v434_generic_candidate_value_schema():
    anchor = _anchor()
    action_candidates = pd.DataFrame(
        {
            "rally_uid": ["r1"],
            "target": ["action"],
            "candidate_value": [3],
            "score": [1.5],
        }
    )
    point_candidates = pd.DataFrame(
        {
            "rally_uid": ["r2"],
            "target": ["point"],
            "candidate_value": [8],
            "score": [1.0],
        }
    )

    submission, report = package_residual_submission(
        anchor,
        action_candidates=action_candidates,
        point_candidates=point_candidates,
        action_top=1,
        point_top=1,
        name="v434_generic",
    )

    assert submission["actionId"].tolist() == [3, 10, 13]
    assert submission["pointId"].tolist() == [7, 8, 0]
    assert report["total_changed_rows"] == 2


def test_package_normalizes_float_like_rally_uid_from_iterrows():
    anchor = _anchor().assign(rally_uid=[18310, 20014, 16571])
    proposals = pd.DataFrame(
        {
            "rally_uid": [18310.0],
            "candidate_pointId": [8],
            "utility": [1.2],
        }
    )

    submission, report = package_residual_submission(
        anchor,
        point_candidates=proposals,
        point_top=1,
        name="float_uid",
    )

    assert report["point"]["applied_changes"] == 1
    assert submission["pointId"].tolist() == [8, 4, 0]
