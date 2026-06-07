import pandas as pd

from analysis_v351_v338_pruning_trust_model import (
    build_pruned_submission,
    build_candidates,
    candidate_metrics,
    score_v338_rows,
)


def _submission(points):
    return pd.DataFrame(
        {
            "rally_uid": [f"r{i}" for i in range(len(points))],
            "actionId": [1, 2, 3, 4][: len(points)],
            "pointId": points,
            "serverGetPoint": [0.1, 0.2, 0.3, 0.4][: len(points)],
        }
    )


def test_score_v338_rows_prefers_gate_trust_and_subset_support():
    v306 = _submission([8, 8, 7])
    v338 = _submission([9, 7, 9])
    gate_scores = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "anchor_value": [8, 8, 7],
            "candidate_value": [9, 7, 9],
            "trust_score": [1.0, 2.0, 3.0],
            "risk_score": [0.0, 0.4, 1.0],
            "transition": ["8->9", "8->7", "7->9"],
        }
    )
    selected_sets = {
        "b12": {1},
        "b18": {0, 1},
        "b24": {0, 1, 2},
    }
    transition_penalty = {"7->9": 2}

    scored = score_v338_rows(v306, v338, gate_scores, selected_sets, transition_penalty)

    assert scored["row_id"].tolist()[0] == 1
    assert scored.loc[scored["row_id"].eq(1), "in_b12"].iloc[0]
    assert scored.loc[scored["row_id"].eq(2), "v341_transition_extra_count"].iloc[0] == 2


def test_build_pruned_submission_reverts_only_selected_point_rows():
    v306 = _submission([8, 8, 7])
    v338 = _submission([9, 7, 9])

    out = build_pruned_submission(v306, v338, rows_to_revert=[1])

    assert out["pointId"].tolist() == [9, 8, 9]
    assert out["actionId"].tolist() == v338["actionId"].tolist()
    assert out["serverGetPoint"].tolist() == v338["serverGetPoint"].tolist()


def test_candidate_metrics_marks_no_new_rows_or_point0_additions():
    v306 = _submission([8, 8, 7, 4])
    v338 = _submission([9, 7, 9, 4])
    candidate = _submission([9, 8, 9, 4])

    metrics = candidate_metrics(v306, v338, candidate)

    assert metrics["point_churn_vs_v306"] == 2
    assert metrics["point_churn_vs_v338"] == 1
    assert metrics["new_rows_beyond_v338"] == 0
    assert metrics["point0_additions_vs_v306"] == 0
    assert metrics["action_preserved"] is True
    assert metrics["server_preserved"] is True


def test_build_candidates_writes_safe_local_submissions():
    v306 = _submission([8, 8, 7])
    v338 = _submission([9, 7, 9])
    scored = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "final_trust_score": [3.0, 2.0, 1.0],
            "trust_score": [3.0, 2.0, 1.0],
            "subset_support": [1.0, 0.5, 0.25],
        }
    )

    records = build_candidates(v306, v338, scored)

    assert records
    assert all(record["new_rows_beyond_v338"] == 0 for record in records)
    assert all(record["point0_additions_vs_v306"] == 0 for record in records)
    assert all("v351_v338_pruning_trust_model/" in record["path"] for record in records)
