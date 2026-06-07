import pandas as pd

from analysis_v449_intent_gru_point_full import (
    _package_point_submission,
    build_intent_conditioned_sequence_features,
    point_residual_candidates_from_probs,
)


def test_v449_sequence_features_include_intent_without_future_point_leakage():
    frame = pd.DataFrame({
        "rally_uid": [1, 1],
        "strikeNumber": [1, 2],
        "actionId": [1, 10],
        "pointId": [7, 0],
        "pred_action": [10, 0],
        "target_pointId": [0, 8],
    })
    feats = build_intent_conditioned_sequence_features(frame)
    assert "target_pointId" not in feats.columns
    assert any(col.startswith("intent_") for col in feats.columns)


def test_v449_candidate_builder_blocks_point0_additions():
    anchor = pd.DataFrame({"rally_uid": [1, 2], "pointId": [8, 0]})
    probs = pd.DataFrame({"rally_uid": [1, 2], "prob_0": [0.9, 0.1], "prob_7": [0.1, 0.9]})
    cand = point_residual_candidates_from_probs(anchor, probs, block_point0_additions=True)
    assert not ((cand["rally_uid"] == 1) & (cand["candidate_pointId"] == 0)).any()


def test_v449_submission_packaging_matches_numeric_uid_forms_and_preserves_action_server():
    anchor = pd.DataFrame(
        {"rally_uid": [1], "actionId": [4], "pointId": [8], "serverGetPoint": [0.25]}
    )
    candidates = pd.DataFrame(
        {"rally_uid": [1.0], "candidate_pointId": [7], "utility": [1.0]}
    )
    submission, report = _package_point_submission(anchor, candidates, top_k=1, expected_rows=1)
    assert submission["pointId"].tolist() == [7]
    assert submission["actionId"].tolist() == [4]
    assert submission["serverGetPoint"].tolist() == [0.25]
    assert report["applied_changes"] == 1
