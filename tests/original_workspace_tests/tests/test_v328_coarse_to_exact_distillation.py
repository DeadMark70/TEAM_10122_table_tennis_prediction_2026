from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis_v328_coarse_to_exact_distillation import (
    CANDIDATE_SPECS,
    ExportSpec,
    build_export_frame,
    changed_row_precision,
    evidence_passes,
    merge_optional_external_features,
    protected_output_path,
    require_rebuilt_v173_anchor,
    select_low_churn_predictions,
)


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "rally_uid": [10, 11, 12],
            "prefix_len": [2, 3, 4],
            "next_actionId": [4, 5, 6],
            "lag0_actionId": [1, 2, 3],
        }
    )
    test = pd.DataFrame(
        {
            "rally_uid": [20, 21],
            "prefix_len": [2, 5],
            "lag0_actionId": [1, 9],
        }
    )
    return train, test


def _unit_tmp() -> Path:
    path = Path("v328_coarse_to_exact_distillation") / "unit_tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_optional_external_features_are_rerunnable_and_drop_label_like_columns():
    train, test = _rows()
    feature_path = _unit_tmp() / "v326_aicup_prefix_family_features.csv"
    pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "rally_uid": [10, 12, 21],
            "prefix_len": [2, 4, 5],
            "family_score": [0.8, 0.2, 0.7],
            "next_actionId": [99, 99, 99],
            "actionId": [88, 88, 88],
            "external_label": [77, 77, 77],
        }
    ).to_csv(feature_path, index=False)

    merged_train, merged_test, report = merge_optional_external_features(
        train,
        test,
        [("v326", feature_path), ("v327", _unit_tmp() / "missing.csv")],
    )

    assert "v328_ext_v326_family_score" in merged_train
    assert merged_train["v328_ext_v326_family_score"].tolist() == [0.8, 0.0, 0.2]
    assert merged_test["v328_ext_v326_family_score"].tolist() == [0.0, 0.7]
    assert "v328_ext_v326_next_actionId" not in merged_train
    assert "v328_ext_v326_actionId" not in merged_train
    assert "v328_ext_v326_external_label" not in merged_train
    assert report["v326"]["status"] == "loaded"
    assert report["v327"]["status"] == "missing"


def test_missing_optional_external_features_leave_frames_unchanged():
    train, test = _rows()

    merged_train, merged_test, report = merge_optional_external_features(
        train,
        test,
        [("v326", _unit_tmp() / "missing_v326.csv")],
    )

    assert merged_train.equals(train)
    assert merged_test.equals(test)
    assert report["v326"]["status"] == "missing"


def test_changed_row_precision_counts_only_action_edits():
    y = np.array([0, 4, 5, 6])
    anchor = np.array([1, 4, 0, 6])
    pred = np.array([0, 4, 5, 7])

    report = changed_row_precision(y, anchor, pred)

    assert report["changed_rows"] == 3
    assert report["changed_correct"] == 2
    assert report["changed_precision"] == pytest.approx(2 / 3)


def test_low_churn_selector_uses_margin_budget_and_preserves_anchor_when_not_positive():
    anchor = np.array([1, 2, 3, 4])
    prob = np.array(
        [
            [0.1, 0.2, 0.7, 0.0, 0.0],
            [0.1, 0.1, 0.3, 0.5, 0.0],
            [0.1, 0.1, 0.1, 0.7, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.6],
        ]
    )

    pred, selected, margin = select_low_churn_predictions(anchor, prob, budget=2)

    assert pred.tolist() == [2, 3, 3, 4]
    assert selected.tolist() == [True, True, False, False]
    assert margin[0] > 0
    assert margin[1] > 0
    assert margin[2] == pytest.approx(0.0)


def test_evidence_gate_requires_action_delta_and_precision_above_prior_action_lines():
    assert evidence_passes(
        {
            "action_oof_delta": 0.002,
            "changed_row_oof_precision": 0.451,
            "changed_action_rows": 8,
            "serve_action_rows": 0,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.00199,
            "changed_row_oof_precision": 0.9,
            "changed_action_rows": 8,
            "serve_action_rows": 0,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.01,
            "changed_row_oof_precision": 0.45,
            "changed_action_rows": 8,
            "serve_action_rows": 0,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.01,
            "changed_row_oof_precision": 0.9,
            "changed_action_rows": 0,
            "serve_action_rows": 0,
        }
    )


def test_build_export_frame_preserves_v306_point_and_v300_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102],
            "actionId": [1, 5],
            "pointId": [0, 9],
            "serverGetPoint": [0.125, 0.875],
        }
    )

    out = build_export_frame(anchor, np.array([4, 12]))

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [4, 12]
    assert out["pointId"].tolist() == [0, 9]
    assert out["serverGetPoint"].tolist() == [0.125, 0.875]


def test_protected_output_path_blocks_upload_selected_and_parent_traversal():
    outdir = Path("v328_coarse_to_exact_distillation")
    spec = CANDIDATE_SPECS["family_feature"]

    path = protected_output_path(outdir, spec)

    assert path.parent == outdir
    assert "upload_candidates" not in str(path)
    assert "selected" not in str(path)
    assert "submissions" not in str(path)

    with pytest.raises(ValueError, match="refusing non-local V328 export path"):
        protected_output_path(outdir, ExportSpec("../submissions/bad.csv", "bad"))


def test_require_rebuilt_v173_anchor_rejects_fallback_sources():
    require_rebuilt_v173_anchor({"anchor_oof_source": "rebuilt_v173_pred_oof"})

    with pytest.raises(RuntimeError, match="strict V173 action anchor required"):
        require_rebuilt_v173_anchor({"anchor_oof_source": "fallback_lag0_actionId"})

    with pytest.raises(RuntimeError, match="strict V173 action anchor required"):
        require_rebuilt_v173_anchor({})
