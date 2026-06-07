import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

import analysis_v326_masked_family_pretrain as v326


def test_canonical_event_table_skips_ttmatch_and_has_required_columns():
    raw = pd.DataFrame(
        [
            {
                "source_dataset": "openttgames",
                "source_path": "external_data/openttgames/processed/events.csv",
                "sequence_id": "seq1",
                "event_index": 0,
                "coarse_family": "Attack",
                "phase": "serve_like",
                "landing_x": -0.5,
                "landing_y": 2.0,
                "speed": 4.0,
                "spin": None,
            },
            {
                "source_dataset": "TTMATCH",
                "source_path": "external_data/TTMATCH/train.csv",
                "sequence_id": "bad",
                "event_index": 0,
                "coarse_family": "Serve",
                "phase": "serve_like",
            },
        ]
    )

    events = v326.build_canonical_event_table(raw)

    assert events.columns.tolist() == v326.CANONICAL_COLUMNS
    assert len(events) == 1
    row = events.iloc[0]
    assert row["corpus"] == "openttgames"
    assert row["sequence_id"] == "seq1"
    assert int(row["step_idx"]) == 0
    assert row["phase_code"] == "serve_like"
    assert row["coarse_family"] == "Attack"
    assert row["landing_depth"] in {"near", "mid", "far", "unknown"}
    assert row["landing_side"] in {"left", "middle", "right", "unknown"}
    assert bool(row["has_speed"])
    assert not bool(row["has_spin"])
    assert float(row["source_weight"]) == pytest.approx(1.0)
    assert not events.astype(str).apply(lambda col: col.str.contains("TTMATCH", case=False)).any().any()


def test_masked_training_reports_accuracy_and_macro_f1_by_corpus():
    external = pd.DataFrame(
        [
            {
                "corpus": "openttgames",
                "sequence_id": "a",
                "step_idx": i,
                "phase_code": "serve_like" if i % 3 == 0 else "rally_like",
                "coarse_family": ["Serve", "Attack", "Control"][i % 3],
                "landing_depth": "near",
                "landing_side": "left",
                "has_spin": i % 2 == 0,
                "has_speed": True,
                "source_weight": 1.0,
            }
            for i in range(12)
        ]
        + [
            {
                "corpus": "CoachAI-Projects-main",
                "sequence_id": "b",
                "step_idx": i,
                "phase_code": "rally_like",
                "coarse_family": ["Attack", "Defensive", "Control"][i % 3],
                "landing_depth": "far",
                "landing_side": "right",
                "has_spin": False,
                "has_speed": False,
                "source_weight": 0.65,
            }
            for i in range(12)
        ]
    )

    model, metrics, feature_cols = v326.train_masked_family_model(external, random_state=7, min_folds=3)

    assert model is not None
    assert "prev_family" in feature_cols
    assert "next_family" in feature_cols
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0
    assert 0.0 <= metrics["overall_macro_f1"] <= 1.0
    assert set(metrics["by_corpus"]) == {"openttgames", "CoachAI-Projects-main"}
    for corpus_metrics in metrics["by_corpus"].values():
        assert {"rows", "accuracy", "macro_f1"}.issubset(corpus_metrics)


def test_aicup_prefix_features_have_stable_keys_and_no_exact_action_columns():
    train_raw = pd.DataFrame(
        [
            [1, 1, 10, 1, 1, 1, 0, 0, 1, 11, 12, 1, 1, 2, 3, 9, 15, 1],
            [1, 1, 10, 1, 1, 2, 0, 0, 1, 12, 11, 2, 2, 1, 2, 5, 2, 1],
            [2, 2, 11, 1, 1, 1, 0, 0, 0, 13, 14, 1, 1, 1, 1, 4, 16, 2],
            [2, 2, 11, 1, 1, 2, 0, 0, 0, 14, 13, 2, 2, 2, 2, 6, 12, 2],
        ],
        columns=[
            "rally_uid",
            "sex",
            "match",
            "numberGame",
            "rally_id",
            "strikeNumber",
            "scoreSelf",
            "scoreOther",
            "serverGetPoint",
            "gamePlayerId",
            "gamePlayerOtherId",
            "strikeId",
            "handId",
            "strengthId",
            "spinId",
            "pointId",
            "actionId",
            "positionId",
        ],
    )
    test_raw = train_raw.drop(columns=["serverGetPoint"]).copy()
    test_raw["rally_uid"] = [101, 101, 102, 102]
    test_raw["match"] = [12, 12, 13, 13]
    model, _, _ = v326.train_masked_family_model(
        pd.DataFrame(
            [
                {
                    "corpus": "openttgames",
                    "sequence_id": "s",
                    "step_idx": i,
                    "phase_code": "rally_like",
                    "coarse_family": ["Serve", "Attack", "Control", "Defensive"][i % 4],
                    "landing_depth": "unknown",
                    "landing_side": "unknown",
                    "has_spin": False,
                    "has_speed": False,
                    "source_weight": 1.0,
                }
                for i in range(16)
            ]
        ),
        random_state=9,
        min_folds=2,
    )

    features = v326.build_aicup_prefix_family_features(train_raw, test_raw, model)

    assert set(features["split"]) == {"train", "test"}
    assert features[["split", "rally_uid", "match", "prefix_len"]].duplicated().sum() == 0
    assert not {"actionId", "next_actionId"}.intersection(features.columns)
    prob_cols = [c for c in features.columns if c.startswith("v326_family_p_")]
    assert prob_cols
    assert (features[prob_cols].sum(axis=1).round(6) == 1.0).all()
    assert features["v326_pred_family"].notna().all()


def test_run_pipeline_writes_only_v326_artifacts():
    workspace = v326.OUTDIR / f"pytest_tmp_{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=True)
    canonical_path = workspace / "v255.csv"
    train_path = workspace / "train.csv"
    test_path = workspace / "test_new.csv"
    outdir = workspace / "out"
    pd.DataFrame(
        [
            {
                "source_dataset": "openttgames",
                "source_path": "external_data/openttgames/processed/events.csv",
                "sequence_id": f"g{i // 4}",
                "event_index": i,
                "coarse_family": ["Serve", "Attack", "Control", "Defensive"][i % 4],
                "phase": "serve_like" if i % 4 == 0 else "rally_like",
                "landing_x": i % 3,
                "landing_y": i % 5,
                "speed": i + 0.1,
                "spin": None,
            }
            for i in range(24)
        ]
    ).to_csv(canonical_path, index=False)
    train_frame = pd.DataFrame(
        [
            [1, 1, 10, 1, 1, 1, 0, 0, 1, 11, 12, 1, 1, 2, 3, 9, 15, 1],
            [1, 1, 10, 1, 1, 2, 0, 0, 1, 12, 11, 2, 2, 1, 2, 5, 2, 1],
            [2, 2, 11, 1, 1, 1, 0, 0, 0, 13, 14, 1, 1, 1, 1, 4, 16, 2],
            [2, 2, 11, 1, 1, 2, 0, 0, 0, 14, 13, 2, 2, 2, 2, 6, 12, 2],
        ],
        columns=[
            "rally_uid",
            "sex",
            "match",
            "numberGame",
            "rally_id",
            "strikeNumber",
            "scoreSelf",
            "scoreOther",
            "serverGetPoint",
            "gamePlayerId",
            "gamePlayerOtherId",
            "strikeId",
            "handId",
            "strengthId",
            "spinId",
            "pointId",
            "actionId",
            "positionId",
        ],
    )
    train_frame.to_csv(train_path, index=False)
    test_frame = train_frame.drop(columns=["serverGetPoint"]).copy()
    test_frame["rally_uid"] = [101, 101, 102, 102]
    test_frame["match"] = [12, 12, 13, 13]
    test_frame.to_csv(test_path, index=False)

    summary = v326.run_pipeline(
        canonical_path=canonical_path,
        train_path=train_path,
        test_path=test_path,
        outdir=outdir,
    )

    assert summary["submissions_written"] == 0
    assert summary["ttmatch_rows"] == 0
    assert (outdir / "v326_external_event_table.csv").exists()
    assert (outdir / "v326_family_model.pkl").exists()
    assert (outdir / "v326_aicup_prefix_family_features.csv").exists()
    assert (outdir / "v326_report.json").exists()
    assert (outdir / "v326_report.md").exists()
    report = json.loads((outdir / "v326_report.json").read_text(encoding="utf-8"))
    assert report["submissions_written"] == 0
    assert report["external_rows"] == 24
    assert not any(path.name.startswith("submission") for path in outdir.iterdir())
