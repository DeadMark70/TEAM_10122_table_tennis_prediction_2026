import pandas as pd

from analysis_v455_professor_full_packager import (
    expert_candidates_from_scores,
    filter_full_professor_candidates,
    summarize_submission_safety,
)


def test_v455_filter_blocks_point0_and_serve_additions():
    rows = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3],
            "target": ["point", "action", "point"],
            "candidate_value": [0, 16, 7],
            "anchor_value": [8, 10, 8],
            "utility": [10.0, 10.0, 1.0],
            "source": ["ok", "ok", "ok"],
        }
    )
    filtered = filter_full_professor_candidates(rows)
    assert filtered["rally_uid"].tolist() == [3]


def test_v455_safety_summary_reports_changed_rows_and_preserved_server():
    anchor = pd.DataFrame({"rally_uid": [1], "actionId": [1], "pointId": [8], "serverGetPoint": [0.2]})
    candidate = pd.DataFrame({"rally_uid": [1], "actionId": [1], "pointId": [7], "serverGetPoint": [0.2]})
    summary = summarize_submission_safety(anchor, candidate)
    assert summary["total_changed_rows"] == 1
    assert summary["server_preserved"] is True


def test_v455_expert_loader_accepts_v450_score_columns():
    anchor = pd.DataFrame({"rally_uid": [1], "actionId": [10], "pointId": [8]})
    scores = pd.DataFrame(
        {
            "rally_uid": [1],
            "rare_action_8_9_12_14_score": [0.7],
            "rare_action_8_9_12_14_candidate_actionId": [12],
        }
    )
    candidates = expert_candidates_from_scores(anchor, scores, target="action")
    assert candidates.loc[0, "candidate_value"] == 12
    assert candidates.loc[0, "utility"] > 0.7
