from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v323_action_disagreement_mining import (
    ACTION_GROUPS,
    ExportSpec,
    action_group,
    build_export_frame,
    changed_row_precision_by_slice,
    evidence_passes,
    markdown_table,
    protected_output_path,
)


def test_action_group_maps_known_classes_to_stable_groups():
    assert action_group(0) == "terminal"
    assert action_group(3) == "terminal"
    assert action_group(5) == "attack"
    assert action_group(8) == "control"
    assert action_group(12) == "defensive"
    assert action_group(15) == "serve"
    assert action_group(999) == "other"
    assert set(ACTION_GROUPS) == {"terminal", "attack", "control", "defensive", "serve", "other"}


def test_changed_row_precision_by_slice_reports_source_phase_lag_and_target_group():
    y = np.array([0, 3, 5, 7, 8, 9])
    anchor = np.array([1, 3, 5, 0, 8, 2])
    source = np.array([0, 3, 7, 7, 8, 9])
    meta = pd.DataFrame(
        {
            "phase": ["receive", "receive", "third_ball", "third_ball", "rally", "rally"],
            "lag_action_family": ["serve", "serve", "attack", "attack", "control", "control"],
        }
    )

    report = changed_row_precision_by_slice("demo", y, anchor, source, meta)
    overall = report[(report["slice_type"] == "overall") & (report["slice_value"] == "all")].iloc[0]
    third_ball = report[(report["slice_type"] == "phase") & (report["slice_value"] == "third_ball")].iloc[0]
    target_terminal = report[
        (report["slice_type"] == "target_action_group") & (report["slice_value"] == "terminal")
    ].iloc[0]

    assert overall["changed_rows"] == 4
    assert overall["changed_correct"] == 3
    assert overall["changed_row_oof_precision"] == 0.75
    assert third_ball["changed_rows"] == 2
    assert third_ball["changed_correct"] == 1
    assert target_terminal["changed_rows"] == 1
    assert target_terminal["changed_correct"] == 1


def test_evidence_gate_requires_strong_delta_or_far_above_prior_precision_and_small_test_slice():
    assert evidence_passes(
        {
            "action_oof_delta": 0.002,
            "changed_row_oof_precision": 0.31,
            "oof_changed_rows": 25,
            "test_changed_rows": 20,
        }
    )
    assert evidence_passes(
        {
            "action_oof_delta": 0.0001,
            "changed_row_oof_precision": 0.45,
            "oof_changed_rows": 25,
            "test_changed_rows": 20,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.0019,
            "changed_row_oof_precision": 0.44,
            "oof_changed_rows": 25,
            "test_changed_rows": 20,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.002,
            "changed_row_oof_precision": 0.31,
            "oof_changed_rows": 25,
            "test_changed_rows": 21,
        }
    )


def test_build_export_frame_preserves_v306_point_and_v300_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [1, 5, 9],
            "pointId": [0, 7, 12],
            "serverGetPoint": [0.123456789, 0.5, 0.999999999],
        }
    )
    pred_action = np.array([0, 5, 12])

    out = build_export_frame(anchor, pred_action)

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [0, 5, 12]
    assert out["pointId"].tolist() == anchor["pointId"].tolist()
    assert out["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()


def test_protected_output_path_blocks_nonlocal_upload_selected_and_submissions_writes():
    outdir = Path("v323_action_disagreement_mining")
    spec = ExportSpec("submission_v323_best_disagreement_slice__v306point_v300server.csv")

    path = protected_output_path(outdir, spec)

    assert path.parent == outdir
    assert "upload_candidates" not in str(path)
    assert "selected" not in str(path)
    assert "submissions" not in str(path)

    bad = ExportSpec("../submissions/submission_v323_bad.csv")
    try:
        protected_output_path(outdir, bad)
    except ValueError as exc:
        assert "refusing non-local V323 export path" in str(exc)
    else:
        raise AssertionError("expected protected_output_path to reject parent traversal")


def test_markdown_table_formats_selected_columns():
    rows = pd.DataFrame(
        [
            {"source": "a", "delta": 0.001234, "decision": "AUDIT"},
            {"source": "b", "delta": 0.010000, "decision": "REVIEW"},
        ]
    )

    text = markdown_table(rows, ["source", "delta", "decision"])

    assert text.splitlines()[0] == "| source | delta | decision |"
    assert "| --- | --- | --- |" in text
    assert "| a | 0.001234 | AUDIT |" in text
