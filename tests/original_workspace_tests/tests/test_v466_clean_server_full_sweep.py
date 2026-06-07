from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest

from analysis_v466_clean_server_full_sweep import (
    ModelConfig,
    build_anchor_context,
    build_candidate_servers,
    build_model_configs,
    build_specialist_masks,
    fit_model_zoo,
    package_server_only,
    run_pipeline,
)


def tiny_anchor(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": list(range(10, 10 + rows)),
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14][:rows],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9][:rows],
            "serverGetPoint": np.linspace(0.2, 0.8, rows),
        }
    )


def tiny_train(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": np.repeat(np.arange(1, 7), 2)[:rows],
            "match": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3][:rows],
            "numberGame": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2][:rows],
            "rally_id": list(range(1, rows + 1)),
            "scoreSelf": [0, 1, 2, 10, 0, 1, 9, 11, 0, 3, 10, 12][:rows],
            "scoreOther": [0, 0, 1, 10, 1, 1, 9, 10, 0, 2, 9, 11][:rows],
            "strikeNumber": [1, 2, 3, 6, 1, 4, 5, 7, 1, 2, 8, 9][:rows],
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14, 0, 4, 10, 13][:rows],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9, 0, 3, 6, 9][:rows],
            "serverGetPoint": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1][:rows],
        }
    )


def tiny_test_new(anchor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in anchor.iterrows():
        for prefix in range(2 if idx % 2 == 0 else 1):
            rows.append(
                {
                    "rally_uid": row["rally_uid"],
                    "match": 1,
                    "numberGame": 1,
                    "rally_id": idx + 1,
                    "scoreSelf": idx + prefix,
                    "scoreOther": idx,
                    "strikeNumber": prefix + 1,
                    "actionId": row["actionId"],
                    "pointId": row["pointId"],
                }
            )
    return pd.DataFrame(rows)


def test_model_config_names_are_unique_and_multi_family():
    configs = build_model_configs(seed=466, runtime="fast")
    names = [config.name for config in configs]
    families = {config.family for config in configs}

    assert len(names) == len(set(names))
    assert {"linear", "tree", "boosting"}.issubset(families)
    assert all(isinstance(config, ModelConfig) for config in configs)


def test_thread_detection_is_pinned_for_sandboxed_boosting():
    assert os.environ.get("LOKY_MAX_CPU_COUNT") == "1"
    assert os.environ.get("OMP_NUM_THREADS") == "1"


def test_fit_model_zoo_returns_oof_and_row_level_test_predictions():
    x = pd.DataFrame({"a": list(range(12)), "b": [0, 1] * 6})
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1])
    x_test = pd.DataFrame({"a": [1, 2, 8], "b": [0, 1, 1]})
    groups = np.repeat(np.arange(6), 2)
    configs = build_model_configs(seed=466, runtime="test")

    signals = fit_model_zoo(x, y, x_test, groups=groups, configs=configs)

    assert signals
    assert {signal.family for signal in signals}.issuperset({"linear", "tree", "boosting"})
    for signal in signals:
        assert len(signal.oof) == len(y)
        assert len(signal.test) == len(x_test)
        assert np.isfinite(signal.auc)
        assert signal.test.min() >= 0.0
        assert signal.test.max() <= 1.0


def test_specialist_masks_are_anchor_aligned_without_rally_uid_sorting():
    anchor = tiny_anchor(4).iloc[[2, 0, 3, 1]].reset_index(drop=True)
    test_new = tiny_test_new(anchor).sample(frac=1.0, random_state=466).reset_index(drop=True)

    context = build_anchor_context(anchor, test_new)
    masks = build_specialist_masks(context)

    assert context["rally_uid"].tolist() == anchor["rally_uid"].tolist()
    assert set(masks) == {
        "score_pressure",
        "early_phase",
        "terminal_like",
        "action_point_conditioned",
    }
    assert all(len(mask) == len(anchor) for mask in masks.values())
    assert masks["early_phase"].dtype == bool


def test_candidate_generation_preserves_v362_action_point_and_caps_mad():
    anchor = tiny_anchor(6)
    model_targets = {
        "global_all_model_rankmean": np.linspace(0.7, 0.3, len(anchor)),
        "tree_rankmean": np.linspace(0.6, 0.4, len(anchor)),
        "boosting_rankmean": np.linspace(0.65, 0.35, len(anchor)),
        "linear_rankmean": np.linspace(0.55, 0.45, len(anchor)),
        "mlp_rankmean": np.linspace(0.45, 0.55, len(anchor)),
        "clean_source_rankmean": np.linspace(0.35, 0.65, len(anchor)),
    }
    masks = build_specialist_masks(build_anchor_context(anchor, tiny_test_new(anchor)))

    candidates = build_candidate_servers(anchor, model_targets, masks=masks)
    candidate = candidates[0]
    packaged = package_server_only(anchor, candidate.server, expected_rows=len(anchor))

    assert packaged["actionId"].tolist() == anchor["actionId"].tolist()
    assert packaged["pointId"].tolist() == anchor["pointId"].tolist()
    assert max(row.actual_mad for row in candidates) <= 0.0100001
    assert {row.decision for row in candidates} <= {"review", "hold"}


def test_run_pipeline_writes_valid_server_only_submissions(tmp_path: Path):
    root = tmp_path
    rows = 8
    anchor = tiny_anchor(rows)
    anchor["rally_uid"] = range(10, 10 + rows)
    train = tiny_train(12)
    test_new = tiny_test_new(anchor)
    anchor_dir = root / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir()
    train.to_csv(root / "train.csv", index=False)
    test_new.to_csv(root / "test_new.csv", index=False)
    anchor.to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    report = run_pipeline(root=root, outdir=root / "v466_clean_server_full_sweep", expected_rows=rows, runtime="test")
    board = pd.read_csv(root / "v466_clean_server_full_sweep" / "v466_server_search.csv")

    assert report["policy"]["no_upload_candidates_20260519"] is True
    assert report["row_test_prediction"] == "test_new_row_level_then_rally_uid_mean"
    assert not board.empty
    assert {"linear", "tree", "boosting"}.issubset(set("|".join(board["families"].astype(str)).split("|")))
    for path in board["path"].head(4):
        sub = pd.read_csv(path)
        assert list(sub.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
        assert len(sub) == rows
        assert sub["actionId"].equals(anchor["actionId"])
        assert sub["pointId"].equals(anchor["pointId"])
        assert sub["serverGetPoint"].between(0, 1).all()
