from pathlib import Path

import pandas as pd

from analysis_v416_external_embedding_aicup_finetune import build_feature_frame, build_test_rows, build_train_transition_rows, run_pipeline


def _aicup_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rally_uid": 1,
                "sex": 1,
                "match": 100,
                "numberGame": 1,
                "rally_id": 1,
                "strikeNumber": 2,
                "scoreSelf": 3,
                "scoreOther": 2,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 2,
                "spinId": 3,
                "pointId": 4,
                "actionId": 15,
                "positionId": 1,
            },
            {
                "rally_uid": 1,
                "sex": 1,
                "match": 100,
                "numberGame": 1,
                "rally_id": 1,
                "strikeNumber": 1,
                "scoreSelf": 2,
                "scoreOther": 2,
                "strikeId": 2,
                "handId": 2,
                "strengthId": 1,
                "spinId": 2,
                "pointId": 9,
                "actionId": 4,
                "positionId": 2,
            },
            {
                "rally_uid": 2,
                "sex": 2,
                "match": 101,
                "numberGame": 1,
                "rally_id": 2,
                "strikeNumber": 1,
                "scoreSelf": 0,
                "scoreOther": 0,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 3,
                "spinId": 5,
                "pointId": 0,
                "actionId": 18,
                "positionId": 1,
            },
            {
                "rally_uid": 2,
                "sex": 2,
                "match": 101,
                "numberGame": 1,
                "rally_id": 2,
                "strikeNumber": 3,
                "scoreSelf": 1,
                "scoreOther": 0,
                "strikeId": 2,
                "handId": 1,
                "strengthId": 2,
                "spinId": 4,
                "pointId": 6,
                "actionId": 7,
                "positionId": 2,
            },
        ]
    )


def _embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token": [
                "fam=badminton_net_shot",
                "phase=rally",
                "depth=long",
                "side=right",
                "spin=medium",
                "speed=high",
            ],
            "emb_0": [1.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "emb_1": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )


def test_transition_builder_uses_only_next_row_inside_same_rally():
    rows = build_train_transition_rows(_aicup_rows())

    assert rows["rally_uid"].tolist() == [1, 2]
    assert rows["strikeNumber"].tolist() == [1, 1]
    assert rows["target_actionId"].tolist() == [15, 7]
    assert rows["target_pointId"].tolist() == [4, 6]


def test_test_rows_use_last_observed_prefix_aligned_to_anchor_order():
    observed = _aicup_rows().drop(columns=["serverGetPoint"], errors="ignore")
    anchor = pd.DataFrame({"rally_uid": [2, 1], "actionId": [18, 15], "pointId": [0, 4], "serverGetPoint": [0.2, 0.8]})

    rows = build_test_rows(observed, anchor)

    assert rows["rally_uid"].tolist() == [2, 1]
    assert rows["strikeNumber"].tolist() == [3, 2]


def test_feature_builder_uses_clean_coarse_tokens_and_zero_fills_missing_embeddings():
    rows = build_train_transition_rows(_aicup_rows()).head(1)

    features, meta = build_feature_frame(rows, _embeddings())

    assert {"v416_emb_mean_0", "v416_emb_sum_1", "family_code", "phase_code", "depth_code", "side_code"}.issubset(
        features.columns
    )
    assert features.loc[0, "v416_emb_mean_0"] > 0
    assert "fam=badminton_net_shot" in meta.loc[0, "tokens_json"]
    assert "phase=rally" in meta.loc[0, "tokens_json"]
    assert "spin=medium" in meta.loc[0, "tokens_json"]
    assert "speed=high" in meta.loc[0, "tokens_json"]
    assert meta.loc[0, "matched_token_count"] >= 4
    forbidden = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
    assert forbidden.isdisjoint(_embeddings().columns)


def test_pipeline_writes_oof_prediction_columns_and_anchor_aligned_test_row_count(tmp_path):
    train = pd.concat([_aicup_rows(), _aicup_rows().assign(rally_uid=[3, 3, 4, 4], match=[102, 102, 103, 103])], ignore_index=True)
    test = _aicup_rows().drop(columns=["serverGetPoint"], errors="ignore")
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test_new.csv"
    emb_path = tmp_path / "token_embeddings.csv"
    anchor_path = tmp_path / "anchor.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    _embeddings().to_csv(emb_path, index=False)
    pd.DataFrame(
        {
            "rally_uid": [2, 1],
            "actionId": [15, 15],
            "pointId": [4, 4],
            "serverGetPoint": [0.5, 0.5],
        }
    ).to_csv(anchor_path, index=False)

    report = run_pipeline(
        train_path=train_path,
        test_path=test_path,
        token_embedding_path=emb_path,
        anchor_path=anchor_path,
        outdir=tmp_path / "out",
    )

    oof = pd.read_csv(tmp_path / "out" / "oof_predictions.csv")
    pred = pd.read_csv(tmp_path / "out" / "test_predictions.csv")
    assert {"pred_actionId", "pred_pointId", "action_confidence", "point_confidence"}.issubset(oof.columns)
    assert pd.api.types.is_integer_dtype(oof["pred_actionId"])
    assert pd.api.types.is_integer_dtype(oof["pred_pointId"])
    assert pred["rally_uid"].tolist() == [2, 1]
    assert len(pred) == 2
    assert report["test_rows"] == 2
