from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from analysis_v468_server_full_run import (
    FINE_TARGET_MADS,
    build_full_model_configs,
    build_specialist_anchor_masks,
    build_specialist_train_masks,
    fit_calibrator,
    run_pipeline,
    train_true_specialists,
)


def tiny_anchor(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": list(range(100, 100 + rows)),
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14][:rows],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9][:rows],
            "serverGetPoint": np.linspace(0.2, 0.8, rows),
        }
    )


def tiny_train(rows: int = 40) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "rally_uid": np.repeat(np.arange(1, 21), 2),
            "match": np.repeat(np.arange(1, 11), 4),
            "numberGame": np.tile([1, 1, 2, 2], 10),
            "rally_id": np.arange(1, 41),
            "scoreSelf": np.tile([0, 1, 9, 11, 3, 10, 12, 5], 5),
            "scoreOther": np.tile([0, 0, 9, 10, 2, 9, 11, 6], 5),
            "strikeNumber": np.tile([1, 2, 3, 7, 1, 4, 8, 9], 5),
            "actionId": np.tile([0, 1, 8, 12, 15, 2, 9, 14], 5),
            "pointId": np.tile([0, 1, 4, 7, 2, 5, 8, 9], 5),
            "serverGetPoint": np.tile([0, 0, 1, 1, 0, 1, 1, 0], 5),
        }
    )
    return base.head(rows).copy()


def tiny_test_new(anchor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in anchor.iterrows():
        for prefix in range(1 + (idx % 3)):
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


def test_fine_target_mads_include_safe_and_diagnostic_caps():
    assert 0.0010 in FINE_TARGET_MADS
    assert 0.0120 in FINE_TARGET_MADS
    assert max(mad for mad in FINE_TARGET_MADS if mad <= 0.0100) == 0.0100


def test_full_model_config_names_are_unique_and_include_available_full_families():
    configs = build_full_model_configs(seed=468, runtime="fast")
    names = [config.name for config in configs]

    assert len(names) == len(set(names))
    assert any("lightgbm" in name or "hist_gradient" in name for name in names)
    assert any("mlp_large" in name for name in names)
    assert any("random_forest_large" in name for name in names)
    assert any("extra_trees_large" in name for name in names)
    if importlib.util.find_spec("xgboost") is not None:
        assert any("xgboost" in name for name in names)
    if importlib.util.find_spec("catboost") is not None:
        assert any("catboost" in name for name in names)


def test_specialist_masks_exist_and_are_aligned():
    anchor = tiny_anchor()
    train = tiny_train()
    test_new = tiny_test_new(anchor).sample(frac=1.0, random_state=468).reset_index(drop=True)

    train_masks = build_specialist_train_masks(train)
    anchor_masks = build_specialist_anchor_masks(anchor, test_new)

    expected = {"score_pressure", "phase_early", "terminal_like", "action_point_conditioned"}
    assert set(train_masks) == expected
    assert set(anchor_masks) == expected
    assert all(mask.dtype == bool and len(mask) == len(train) for mask in train_masks.values())
    assert all(mask.dtype == bool and len(mask) == len(anchor) for mask in anchor_masks.values())


def test_calibrators_are_finite_and_clipped():
    y = np.array([0, 0, 1, 1, 0, 1])
    oof = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6])
    test = np.array([0.15, 0.85])

    for kind in ["identity", "platt", "isotonic"]:
        calibrator = fit_calibrator(kind, oof, y)
        out = calibrator(test)
        assert np.isfinite(out).all()
        assert ((0 <= out) & (out <= 1)).all()


def test_true_specialists_fallback_on_tiny_support():
    anchor = tiny_anchor()
    train = tiny_train(12)
    test_new = tiny_test_new(anchor)
    base_target = np.linspace(0.8, 0.2, len(anchor))

    targets, report = train_true_specialists(
        train,
        test_new,
        anchor,
        base_target=base_target,
        runtime="test",
        min_rows=300,
    )

    assert set(targets) == {
        "true_score_pressure_specialist",
        "true_phase_early_specialist",
        "true_terminal_like_specialist",
        "true_action_point_specialist",
    }
    assert all(len(values) == len(anchor) for values in targets.values())
    assert all(item["status"] == "fallback" for item in report.values())


def test_run_pipeline_writes_valid_server_only_submissions(tmp_path: Path):
    anchor = tiny_anchor()
    train = tiny_train()
    test_new = tiny_test_new(anchor)
    anchor_dir = tmp_path / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    test_new.to_csv(tmp_path / "test_new.csv", index=False)
    anchor.to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    report = run_pipeline(
        root=tmp_path,
        outdir=tmp_path / "v468_server_full_run",
        expected_rows=len(anchor),
        runtime="test",
        sequence_enabled=False,
    )
    board = pd.read_csv(tmp_path / "v468_server_full_run" / "v468_server_search.csv")

    assert report["policy"]["no_old_server_direct_labels"] is True
    assert report["candidate_count"] > 0
    for path in board["path"].head(5):
        sub = pd.read_csv(path)
        assert list(sub.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
        assert sub["actionId"].equals(anchor["actionId"])
        assert sub["pointId"].equals(anchor["pointId"])
        assert sub["serverGetPoint"].between(0, 1).all()
