import pandas as pd

from analysis_v448_neural_response_style_contrastive import (
    build_response_context_table,
    make_safe_style_features,
)


def test_v448_safe_style_features_do_not_export_raw_player_or_future_labels():
    frame = pd.DataFrame({
        "rally_uid": [1, 1, 2],
        "strikeNumber": [1, 2, 1],
        "gamePlayerId": ["a", "b", "a"],
        "actionId": [1, 10, 3],
        "pointId": [7, 0, 8],
        "target_actionId": [10, 0, 11],
    })
    table = build_response_context_table(frame)
    features = make_safe_style_features(table, embedding_dim=4)
    forbidden = {"gamePlayerId", "gamePlayerOtherId", "target_actionId", "target_pointId", "target_serverGetPoint"}
    assert forbidden.isdisjoint(features.columns)
    assert any(col.startswith("neural_style_") for col in features.columns)
