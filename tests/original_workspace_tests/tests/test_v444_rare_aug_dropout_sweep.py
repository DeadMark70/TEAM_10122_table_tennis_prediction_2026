from pathlib import Path

import pandas as pd

from analysis_v444_rare_aug_dropout_sweep import (
    ACTION_WEAK_CLASSES,
    POINT_WEAK_CLASSES,
    _target_metrics,
    apply_train_only_smote_like_augmentation,
    build_sweep_grid,
    write_sweep_outputs,
)


def test_v444_sweep_grid_contains_class_weight_resampling_smote_and_masking():
    grid = build_sweep_grid()
    names = {cfg["name"] for cfg in grid}
    assert {"class_weight", "resample_rare", "smote_rare", "mask_dropout", "class_weight_plus_mask"}.issubset(names)
    assert ACTION_WEAK_CLASSES == {0, 3, 4, 5, 7, 8, 9, 12, 14}
    assert POINT_WEAK_CLASSES == {1, 3, 4, 7, 8, 9}
    assert all("use_class_weight" in cfg for cfg in grid)
    assert any(cfg["mask_probability"] > 0 for cfg in grid)


def test_v444_smote_like_augmentation_train_only_keeps_validation_unchanged():
    train_x = pd.DataFrame({"x": [0.0, 1.0, 2.0], "target": [1, 9, 9]})
    val_x = pd.DataFrame({"x": [3.0], "target": [9]})
    aug_train, aug_val, report = apply_train_only_smote_like_augmentation(
        train_x,
        val_x,
        target_col="target",
        rare_classes={1},
        multiplier=3,
    )
    assert len(aug_train) > len(train_x)
    assert len(aug_val) == len(val_x)
    assert aug_val.equals(val_x)
    assert set(aug_train.loc[len(train_x) :, "target"].astype(int)) == {1}
    assert report["validation_rows_added"] == 0
    assert report["train_only"] is True


def test_v444_outputs_score_tables_and_diagnostics_without_submissions(tmp_path: Path):
    action_scores = pd.DataFrame({"rally_uid": ["a", "b"], "class_weight_pred": [0, 3]})
    point_scores = pd.DataFrame({"rally_uid": ["a", "b"], "class_weight_pred": [1, 7]})
    oof_report = pd.DataFrame({"target": ["action"], "variant": ["class_weight"], "macro_f1": [0.1]})
    summary = {"version": "V444", "submission_exports": 0}

    paths = write_sweep_outputs(action_scores, point_scores, oof_report, summary, outdir=tmp_path)

    assert {path.name for path in paths.values()} == {
        "action_sweep_scores_test.csv",
        "point_sweep_scores_test.csv",
        "oof_sweep_report.csv",
        "summary.json",
    }
    assert not list(tmp_path.glob("submission*.csv"))


def test_v444_weak_class_metric_support_uses_observed_class_count():
    metrics = _target_metrics(
        y_true=pd.Series([1, 1, 9]).to_numpy(),
        pred=pd.Series([1, 9, 9]).to_numpy(),
        prob=pd.DataFrame([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]).to_numpy(),
        classes=[1, 9],
        weak_classes={1, 9},
    )

    assert metrics["class_1_support"] == 2
    assert metrics["class_9_support"] == 1
