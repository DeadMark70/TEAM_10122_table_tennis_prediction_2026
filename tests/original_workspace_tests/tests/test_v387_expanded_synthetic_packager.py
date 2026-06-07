import pandas as pd

from analysis_v387_expanded_synthetic_packager import (
    package_candidate,
    run_pipeline,
    select_contrastive_action_updates,
    select_contrastive_point_updates,
)


def test_select_contrastive_point_updates_blocks_point0_additions():
    scores = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3],
            "candidate_point": [0, 9, 7],
            "is_point0_addition": [True, False, False],
            "synthetic_allowed": [False, True, True],
            "contrastive_score": [99.0, 20.0, 30.0],
        }
    )
    updates = select_contrastive_point_updates(scores, budget=2)
    assert updates == {3: 7, 2: 9}


def test_package_candidate_preserves_action_and_server_when_only_point_updates():
    anchor = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [10, 11],
            "pointId": [8, 9],
            "serverGetPoint": [0.2, 0.8],
        }
    )
    out = package_candidate(anchor, point_updates={1: 7}, action_updates={})
    assert out["actionId"].tolist() == [10, 11]
    assert out["serverGetPoint"].tolist() == [0.2, 0.8]
    assert out["pointId"].tolist() == [7, 9]


def test_select_contrastive_action_updates_blocks_new_serve_classes():
    scores = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3],
            "candidate_action": [15, 3, 11],
            "synthetic_allowed": [True, True, False],
            "contrastive_score": [99.0, 20.0, 30.0],
        }
    )
    updates = select_contrastive_action_updates(scores, budget=2)
    assert updates == {2: 3}


def test_run_pipeline_reports_missing_v386_scores_without_candidates(tmp_path):
    report = run_pipeline(
        outdir=tmp_path,
        point_scores_path=tmp_path / "missing_point.csv",
        action_scores_path=tmp_path / "missing_action.csv",
    )
    assert report["v386_scores_available"] == {"point": False, "action": False}
    assert report["generated_submission_count"] == 0
    assert (tmp_path / "ranked_candidates.csv").exists()
