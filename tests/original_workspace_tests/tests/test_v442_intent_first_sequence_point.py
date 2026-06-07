import pandas as pd

from analysis_v442_intent_first_sequence_point import (
    condition_point_features_on_action_intent,
    package_point_residual_candidates,
)


def test_v442_point_features_include_predicted_action_intent_not_true_future_label():
    frame = pd.DataFrame(
        {
            "rally_uid": ["r1"],
            "actionId": [4],
            "pointId": [2],
            "pred_action": [7],
            "action_confidence": [0.8],
            "target_actionId": [3],
            "target_pointId": [8],
        }
    )

    features = condition_point_features_on_action_intent(frame)

    assert "pred_intent_drive" in features.columns
    assert "target_actionId" not in features.columns
    assert "target_pointId" not in features.columns


def test_v442_residual_packaging_blocks_point0_additions():
    anchor = pd.DataFrame(
        {"rally_uid": ["r1", "r2"], "actionId": [4, 7], "pointId": [5, 0], "serverGetPoint": [0.2, 0.7]}
    )
    proposals = pd.DataFrame({"rally_uid": ["r1", "r2"], "candidate_pointId": [0, 2], "utility": [10.0, 1.0]})

    out, report = package_point_residual_candidates(anchor, proposals, top_k=2)

    assert out["pointId"].tolist() == [5, 2]
    assert report["blocked_point0_additions"] == 1
    assert out["serverGetPoint"].tolist() == [0.2, 0.7]
