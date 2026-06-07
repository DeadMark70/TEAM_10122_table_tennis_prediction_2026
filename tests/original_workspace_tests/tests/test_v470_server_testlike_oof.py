from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_v470_server_testlike_oof import (
    compute_model_metrics,
    compute_slice_metrics,
    run_pipeline,
    weighted_auc,
)


def test_weighted_auc_differs_when_weights_shift_pair_importance():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0.9, 0.2, 0.8, 0.7])
    unweighted = weighted_auc(y, pred, np.ones_like(pred, dtype=float))
    weighted = weighted_auc(y, pred, np.array([10.0, 1.0, 1.0, 1.0]))

    assert 0 <= unweighted <= 1
    assert 0 <= weighted <= 1
    assert weighted < unweighted


def test_compute_model_metrics_has_ordinary_and_testlike_auc():
    y = np.array([0, 0, 1, 1, 0, 1])
    weights = np.array([1, 2, 1, 2, 1, 2], dtype=float)
    predictions = {
        "good": np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]),
        "weak": np.array([0.4, 0.5, 0.6, 0.55, 0.45, 0.52]),
    }
    metrics = compute_model_metrics(y, weights, predictions)

    assert set(metrics["model"]) == {"good", "weak"}
    assert metrics["ordinary_auc"].between(0, 1).all()
    assert metrics["testlike_weighted_auc"].between(0, 1).all()
    assert metrics.iloc[0]["testlike_weighted_auc"] >= metrics.iloc[1]["testlike_weighted_auc"]


def test_compute_slice_metrics_reports_each_slice_value():
    y = np.array([0, 1, 0, 1, 0, 1])
    weights = np.ones(6)
    pred = {"m": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])}
    slices = pd.DataFrame({"phase_bin": ["a", "a", "b", "b", "b", "b"], "pressure": ["n", "n", "n", "p", "p", "p"]})
    out = compute_slice_metrics(y, weights, pred, slices, ["phase_bin", "pressure"])

    assert {"model", "slice_name", "slice_value", "rows", "ordinary_auc", "weighted_auc"}.issubset(out.columns)
    assert set(out["slice_name"]) == {"phase_bin", "pressure"}


def test_run_pipeline_tiny(tmp_path):
    train = pd.DataFrame(
        {
            "rally_uid": np.arange(1, 13),
            "strikeNumber": [1, 2, 3, 4, 7, 8, 1, 2, 5, 6, 9, 10],
            "scoreSelf": [0, 1, 9, 10, 3, 4, 11, 8, 2, 6, 10, 12],
            "scoreOther": [0, 0, 9, 9, 5, 4, 10, 8, 1, 7, 11, 11],
            "actionId": [0, 1, 8, 12, 14, 10, 15, 2, 7, 9, 13, 4],
            "pointId": [0, 1, 4, 7, 9, 6, 2, 5, 8, 3, 0, 9],
            "serverGetPoint": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        }
    )
    test_new = train.drop(columns=["serverGetPoint"]).copy()
    anchor_dir = tmp_path / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    test_new.to_csv(tmp_path / "test_new.csv", index=False)
    pd.DataFrame(
        {
            "rally_uid": np.arange(1, 13),
            "actionId": train["actionId"],
            "pointId": train["pointId"],
            "serverGetPoint": np.linspace(0.2, 0.8, 12),
        }
    ).to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    report = run_pipeline(root=tmp_path, outdir=tmp_path / "v470", expected_rows=12, runtime="test")
    assert report["model_count"] > 0
    assert (tmp_path / "v470" / "v470_model_oof_metrics.csv").exists()
