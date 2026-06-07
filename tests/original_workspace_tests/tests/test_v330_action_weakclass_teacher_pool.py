from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis_v330_action_weakclass_teacher_pool import (
    ExportSpec,
    build_export_frame,
    changed_row_precision,
    evidence_passes,
    protected_output_path,
    require_rebuilt_v173_anchor,
)


def test_strict_anchor_requires_rebuilt_v173_oof_source():
    require_rebuilt_v173_anchor(
        {
            "row_source": "analysis_r184_receiver_affordance_refiner.rebuild_v173_best_actions",
            "anchor_oof_source": "rebuilt_v173_pred_oof",
        }
    )

    with pytest.raises(RuntimeError, match="strict V173 action anchor required"):
        require_rebuilt_v173_anchor(
            {
                "row_source": "baseline_lgbm_prefix_tables",
                "anchor_oof_source": "fallback_lag0_actionId",
            }
        )

    with pytest.raises(RuntimeError, match="strict V173 action anchor required"):
        require_rebuilt_v173_anchor({})


def test_build_export_frame_preserves_fixed_point_and_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [1, 5, 7],
            "pointId": [0, 8, 4],
            "serverGetPoint": [0.11111111, 0.50340328, 0.875],
        }
    )

    out = build_export_frame(anchor, np.array([4, 12, 0]))

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [4, 12, 0]
    assert out["pointId"].tolist() == [0, 8, 4]
    assert out["serverGetPoint"].tolist() == [0.11111111, 0.50340328, 0.875]


def test_changed_row_precision_counts_only_action_edits():
    y = np.array([0, 4, 5, 6])
    anchor = np.array([1, 4, 0, 6])
    pred = np.array([0, 4, 5, 7])

    report = changed_row_precision(y, anchor, pred)

    assert report["changed_rows"] == 3
    assert report["changed_correct"] == 2
    assert report["changed_precision"] == pytest.approx(2 / 3)


def test_evidence_gate_blocks_weak_fallback_and_low_evidence_candidates():
    good = {
        "action_oof_delta": 0.0015,
        "changed_row_oof_precision": 0.45,
        "changed_action_rows": 5,
        "serve_action_rows": 0,
        "source_family": "v291_style_fold_safe_weak_ovr_extratrees",
    }
    assert evidence_passes(good)

    fallback = dict(good, source_family="fallback_weak_candidate")
    assert not evidence_passes(fallback)

    low_delta = dict(good, action_oof_delta=0.00149)
    assert not evidence_passes(low_delta)

    low_precision = dict(good, changed_row_oof_precision=0.449)
    assert not evidence_passes(low_precision)

    too_few_rows = dict(good, changed_action_rows=4)
    assert not evidence_passes(too_few_rows)

    too_many_rows = dict(good, changed_action_rows=81)
    assert not evidence_passes(too_many_rows)

    serve_explosion = dict(good, serve_action_rows=1)
    assert not evidence_passes(serve_explosion)


def test_protected_output_path_cannot_escape_v330_dir():
    outdir = Path("v330_action_weakclass_teacher_pool")
    path = protected_output_path(outdir, ExportSpec("submission_v330_safe.csv"))

    assert path.parent == outdir
    assert "upload_candidates" not in str(path)
    assert "selected" not in str(path)
    assert "submissions" not in str(path)

    with pytest.raises(ValueError, match="refusing non-local V330 export path"):
        protected_output_path(outdir, ExportSpec("../submissions/bad.csv"))

    with pytest.raises(ValueError, match="refusing non-local V330 export path"):
        protected_output_path(outdir, ExportSpec("nested/bad.csv"))
