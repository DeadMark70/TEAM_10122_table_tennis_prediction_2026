import pandas as pd

from analysis_v443_response_style_contrastive import (
    build_contrastive_pairs,
    compute_response_style_embeddings,
)


def test_v443_builds_positive_pairs_for_same_actor_context_without_exact_future_labels():
    frame = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2", "r3"],
            "gamePlayerId": [10, 10, 11],
            "phase": ["receive", "receive", "receive"],
            "actionId": [4, 4, 7],
            "pointId": [2, 2, 8],
            "target_actionId": [7, 8, 9],
        }
    )

    pairs = build_contrastive_pairs(frame, max_pairs=10)

    assert {"left_row", "right_row", "pair_label"}.issubset(pairs.columns)
    assert "target_actionId" not in pairs.columns
    assert 1 in set(pairs["pair_label"])


def test_v443_builds_negative_pairs_for_different_actor_or_context():
    frame = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2", "r3", "r4"],
            "gamePlayerId": [10, 10, 11, 12],
            "phase": ["receive", "receive", "serve", "receive"],
            "actionId": [4, 4, 7, 4],
            "pointId": [2, 2, 8, 2],
        }
    )

    pairs = build_contrastive_pairs(frame, max_pairs=20)

    assert 0 in set(pairs["pair_label"])


def test_v443_embeddings_have_one_row_per_test_rally_uid():
    test = pd.DataFrame(
        {
            "rally_uid": ["a", "a", "b"],
            "gamePlayerId": [1, 1, 2],
            "strikeNumber": [1, 2, 1],
        }
    )

    emb = compute_response_style_embeddings(test, embedding_dim=4)

    assert emb["rally_uid"].tolist() == ["a", "b"]
    assert len([c for c in emb.columns if c.startswith("style_emb_")]) == 4


def test_v443_embeddings_exclude_exact_future_targets_and_raw_player_ids():
    frame = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "gamePlayerId": [10, 10],
            "phase": ["receive", "receive"],
            "actionId": [4, 7],
            "pointId": [2, 8],
            "target_actionId": [1, 2],
            "target_pointId": [3, 4],
            "serverGetPoint": [0, 1],
        }
    )

    emb = compute_response_style_embeddings(frame, embedding_dim=3)

    forbidden = {"gamePlayerId", "target_actionId", "target_pointId", "serverGetPoint"}
    assert forbidden.isdisjoint(emb.columns)
    assert len([c for c in emb.columns if c.startswith("style_emb_")]) == 3
