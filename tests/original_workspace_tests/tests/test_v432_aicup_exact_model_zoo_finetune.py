import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v432_aicup_exact_model_zoo_finetune import (
    build_intent_features,
    build_test_rows,
    build_train_transition_rows,
    discover_embedding_sources,
    run_pipeline,
)


def _tiny_token_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token": [
                "fam=badminton_drive",
                "fam=badminton_net_shot",
                "phase=early",
                "phase=middle",
                "depth=half",
                "depth=long",
                "side=middle",
                "side=right",
                "speed=high",
                "speed=medium",
                "spin=medium",
                "terminal=nonterminal",
            ],
            "emb_0": [0.4, 0.2, 0.1, 0.3, 0.6, 0.8, 0.5, 0.7, 0.9, 0.2, 0.1, 0.4],
            "emb_1": [0.1, 0.3, 0.2, 0.4, 0.5, 0.7, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5],
        }
    )


def _tiny_train_rows() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        rally_uid = f"tr{idx}"
        target_action = 6 if idx % 2 == 0 else 7
        target_point = 5 if idx % 2 == 0 else 8
        rows.append(
            {
                "rally_uid": rally_uid,
                "match": idx // 2,
                "sex": 1 + (idx % 2),
                "numberGame": 1,
                "strikeNumber": 1,
                "scoreSelf": idx % 4,
                "scoreOther": (idx + 1) % 4,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 1 + (idx % 2),
                "spinId": 2,
                "pointId": 4,
                "actionId": 4,
                "positionId": 1,
                "serverGetPoint": 0.0,
            }
        )
        rows.append(
            {
                "rally_uid": rally_uid,
                "match": idx // 2,
                "sex": 1 + (idx % 2),
                "numberGame": 1,
                "strikeNumber": 2,
                "scoreSelf": idx % 4,
                "scoreOther": (idx + 1) % 4,
                "strikeId": 2,
                "handId": 2,
                "strengthId": 2,
                "spinId": 3,
                "pointId": target_point,
                "actionId": target_action,
                "positionId": 2,
                "serverGetPoint": float(idx % 2),
            }
        )
    return pd.DataFrame(rows)


def _tiny_test_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rally_uid": "te0",
                "sex": 1,
                "numberGame": 1,
                "strikeNumber": 1,
                "scoreSelf": 0,
                "scoreOther": 0,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 1,
                "spinId": 2,
                "pointId": 4,
                "actionId": 4,
                "positionId": 1,
            },
            {
                "rally_uid": "te0",
                "sex": 1,
                "numberGame": 1,
                "strikeNumber": 3,
                "scoreSelf": 1,
                "scoreOther": 0,
                "strikeId": 2,
                "handId": 2,
                "strengthId": 2,
                "spinId": 3,
                "pointId": 5,
                "actionId": 6,
                "positionId": 2,
            },
            {
                "rally_uid": "te1",
                "sex": 2,
                "numberGame": 1,
                "strikeNumber": 2,
                "scoreSelf": 2,
                "scoreOther": 1,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 2,
                "spinId": 2,
                "pointId": 4,
                "actionId": 4,
                "positionId": 1,
            },
        ]
    )


def test_v432_uses_next_row_transition_and_anchor_aligned_test_rows():
    train = pd.DataFrame(
        {"rally_uid": ["r1", "r1"], "strikeNumber": [1, 2], "actionId": [4, 7], "pointId": [2, 8]}
    )
    transitions = build_train_transition_rows(train)
    assert transitions["target_actionId"].tolist() == [7]
    assert transitions["target_pointId"].tolist() == [8]

    test = pd.DataFrame({"rally_uid": ["a", "a", "b"], "strikeNumber": [1, 3, 2]})
    anchor = pd.DataFrame({"rally_uid": ["b", "a"], "actionId": [0, 0], "pointId": [0, 0], "serverGetPoint": [0.5, 0.5]})
    aligned = build_test_rows(test, anchor)
    assert aligned["rally_uid"].tolist() == ["b", "a"]
    assert aligned["strikeNumber"].tolist() == [2, 3]


def test_v432_intent_features_drop_leak_columns_and_include_intents():
    frame = pd.DataFrame(
        {
            "actionId": [7],
            "pointId": [8],
            "target_actionId": [3],
            "target_pointId": [1],
            "serverGetPoint": [1.0],
            "strikeNumber": [3],
            "spinId": [2],
            "strengthId": [1],
        }
    )
    features = build_intent_features(frame)
    assert any(col.startswith("intent_family_") for col in features.columns)
    assert any(col.startswith("point_depth_") for col in features.columns)
    assert any(col.startswith("point_side_") for col in features.columns)
    assert {"target_actionId", "target_pointId", "serverGetPoint"}.isdisjoint(features.columns)


def test_v432_embedding_source_selection_prefers_v431_then_v418_then_v415(tmp_path: Path):
    v418 = tmp_path / "v418_clean_external_sequence_pretrain"
    v415 = tmp_path / "v415_clean_external_representation"
    v418.mkdir()
    v415.mkdir()
    _tiny_token_embeddings().to_csv(v418 / "token_embeddings.csv", index=False)
    _tiny_token_embeddings().assign(emb_0=0.0).to_csv(v415 / "token_embeddings.csv", index=False)

    fallback_sources = discover_embedding_sources(tmp_path)
    assert [source.source_version for source in fallback_sources] == ["V418"]

    model_dir = tmp_path / "v431_external_sequence_model_zoo" / "gru_small"
    model_dir.mkdir(parents=True)
    _tiny_token_embeddings().to_csv(model_dir / "token_embeddings.csv", index=False)
    pd.DataFrame({"sequence_id": ["s1"], "emb_0": [0.25], "emb_1": [0.75]}).to_csv(
        model_dir / "sequence_embeddings.csv", index=False
    )

    v431_sources = discover_embedding_sources(tmp_path)
    assert [source.source_version for source in v431_sources] == ["V431"]
    assert v431_sources[0].name == "gru_small"
    assert v431_sources[0].sequence_path is not None


def test_v432_pipeline_exports_normalized_probabilities_and_no_submissions(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test_new.csv"
    anchor_path = tmp_path / "anchor.csv"
    emb_dir = tmp_path / "v418_clean_external_sequence_pretrain"
    outdir = tmp_path / "v432"
    emb_dir.mkdir()

    _tiny_train_rows().to_csv(train_path, index=False)
    _tiny_test_rows().to_csv(test_path, index=False)
    _tiny_token_embeddings().to_csv(emb_dir / "token_embeddings.csv", index=False)
    pd.DataFrame(
        {"rally_uid": ["te1", "te0"], "actionId": [4, 6], "pointId": [4, 5], "serverGetPoint": [0.25, 0.75]}
    ).to_csv(anchor_path, index=False)

    summary = run_pipeline(
        train_path=train_path,
        test_path=test_path,
        anchor_path=anchor_path,
        root=tmp_path,
        outdir=outdir,
        expected_rows=2,
        model_names=("logistic",),
        quick=True,
    )

    action = np.load(outdir / "oof_action_probs_v418__logistic.npy")
    point = np.load(outdir / "test_point_probs_v418__logistic.npy")
    assert action.shape == (8, 2)
    assert point.shape == (2, 2)
    assert np.allclose(action.sum(axis=1), 1.0)
    assert np.allclose(point.sum(axis=1), 1.0)

    test_csv = pd.read_csv(outdir / "test_action_probs_v418__logistic.csv")
    assert test_csv["rally_uid"].tolist() == ["te1", "te0"]
    assert not list(outdir.glob("submission*.csv"))
    assert summary["embedding_sources"][0]["source_version"] == "V418"
    assert json.loads((outdir / "summary.json").read_text(encoding="utf-8"))["submission_exports"] == 0


def test_v432_pipeline_can_write_partial_bounded_summary(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test_new.csv"
    anchor_path = tmp_path / "anchor.csv"
    emb_dir = tmp_path / "v431_external_sequence_model_zoo" / "gru_small"
    outdir = tmp_path / "v432"
    emb_dir.mkdir(parents=True)

    _tiny_train_rows().to_csv(train_path, index=False)
    _tiny_test_rows().to_csv(test_path, index=False)
    _tiny_token_embeddings().to_csv(emb_dir / "token_embeddings.csv", index=False)
    pd.DataFrame({"sequence_id": ["s1"], "emb_0": [0.1], "emb_1": [0.2]}).to_csv(
        emb_dir / "sequence_embeddings.csv", index=False
    )
    pd.DataFrame(
        {"rally_uid": ["te1", "te0"], "actionId": [4, 6], "pointId": [4, 5], "serverGetPoint": [0.25, 0.75]}
    ).to_csv(anchor_path, index=False)

    summary = run_pipeline(
        train_path=train_path,
        test_path=test_path,
        anchor_path=anchor_path,
        root=tmp_path,
        outdir=outdir,
        expected_rows=2,
        model_names=("extratrees",),
        quick=True,
        max_train_transitions=4,
        max_embedding_sources=1,
    )

    reports = pd.read_csv(outdir / "model_reports.csv")
    assert summary["partial_run"] is True
    assert summary["train_transition_rows_used"] == 4
    assert reports.loc[0, "source_version"] == "V431"
    assert (outdir / "summary.json").exists()
