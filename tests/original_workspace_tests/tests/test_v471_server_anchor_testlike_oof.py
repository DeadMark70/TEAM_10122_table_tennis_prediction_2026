from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_v471_server_anchor_testlike_oof import (
    candidate_auc_rows,
    derive_teacher_from_blend,
    write_report_files,
)


def test_derive_teacher_from_blend_recovers_original_teacher():
    proxy = np.array([0.2, 0.4, 0.7, 0.9])
    teacher = np.array([0.8, 0.3, 0.6, 0.1])
    blend = (1.0 - 0.02) * proxy + 0.02 * teacher

    recovered = derive_teacher_from_blend(proxy, blend, 0.02)

    assert np.allclose(recovered, teacher)


def test_candidate_auc_rows_reports_deltas_vs_v470():
    y = np.array([0, 0, 1, 1, 0, 1])
    weights = np.array([1, 3, 1, 3, 1, 3], dtype=float)
    candidates = {
        "old_anchor": np.array([0.2, 0.3, 0.6, 0.7, 0.4, 0.8]),
        "weak": np.array([0.6, 0.5, 0.4, 0.3, 0.7, 0.2]),
    }

    metrics = candidate_auc_rows(y, weights, candidates, compare_auc=1.0)

    assert {"candidate", "ordinary_auc", "testlike_weighted_auc", "delta_vs_v470_best_testlike"}.issubset(metrics.columns)
    assert metrics.iloc[0]["candidate"] == "old_anchor"
    assert metrics.loc[metrics["candidate"].eq("old_anchor"), "delta_vs_v470_best_testlike"].iloc[0] <= 0


def test_write_report_files_creates_csv_json_and_markdown(tmp_path):
    metrics = pd.DataFrame(
        [
            {
                "candidate": "old_anchor",
                "ordinary_auc": 0.62,
                "testlike_weighted_auc": 0.60,
                "delta_vs_v470_best_testlike": -0.02,
                "prediction_mean": 0.5,
                "prediction_std": 0.1,
            },
            {
                "candidate": "v470_best",
                "ordinary_auc": 0.70,
                "testlike_weighted_auc": 0.63,
                "delta_vs_v470_best_testlike": 0.0,
                "prediction_mean": 0.52,
                "prediction_std": 0.2,
            },
        ]
    )

    report = write_report_files(tmp_path, metrics, v470_best_model="v470_best", v470_best_auc=0.63)

    assert (tmp_path / "v471_anchor_family_oof_metrics.csv").exists()
    assert (tmp_path / "v471_report.json").exists()
    assert (tmp_path / "v471_report.md").exists()
    assert report["best_anchor_family_candidate"] == "old_anchor"
