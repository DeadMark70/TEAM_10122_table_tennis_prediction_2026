from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from analysis_v329_point_distributional_selector import (
    CandidateSpec,
    OUTDIR,
    SUBMISSION_COLUMNS,
    build_export_frame,
    configured_input_paths,
    ensure_clean_input_paths,
    evidence_passes,
    path_has_banned_input_token,
    protected_output_path,
    write_submission_if_evidence,
)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [17, 4, 12],
            "pointId": [0, 7, 9],
            "serverGetPoint": [0.25, 0.75, 0.50],
        }
    )


def test_configured_inputs_do_not_use_old_or_external_match_paths():
    assert all(not path_has_banned_input_token(path) for path in configured_input_paths())

    with pytest.raises(ValueError, match="banned input paths"):
        ensure_clean_input_paths([Path("external_data") / "TTMATCH" / "train.csv"])
    with pytest.raises(ValueError, match="banned input paths"):
        ensure_clean_input_paths([Path("artifacts") / "old_server_scores.csv"])


def test_build_export_frame_changes_only_point_and_preserves_action_server():
    anchor = _anchor()
    out = build_export_frame(anchor, np.array([1, 8, 4]), expected_rows=3)

    assert out.columns.tolist() == SUBMISSION_COLUMNS
    assert out["rally_uid"].tolist() == [101, 102, 103]
    assert out["actionId"].tolist() == [17, 4, 12]
    assert out["pointId"].tolist() == [1, 8, 4]
    assert out["serverGetPoint"].tolist() == [0.25, 0.75, 0.50]


def test_protected_output_path_blocks_exports_outside_v329_dir():
    good = protected_output_path(OUTDIR, "submission_v329_unit__v173action_v300server.csv")

    assert good.parent == OUTDIR
    assert "upload_candidates" not in str(good)

    with pytest.raises(ValueError, match="refusing non-local V329 export path"):
        protected_output_path(OUTDIR, "../upload_candidates_20260519/bad.csv")
    with pytest.raises(ValueError, match="refusing non-local V329 export path"):
        protected_output_path(OUTDIR, "../submissions/bad.csv")
    with pytest.raises(ValueError, match="refusing non-local V329 export path"):
        protected_output_path(OUTDIR, "nested/bad.csv")


def _unit_outdir() -> Path:
    return OUTDIR / f"unit_no_export_{uuid.uuid4().hex}"


def test_failed_evidence_gate_prevents_export():
    outdir = _unit_outdir()
    spec = CandidateSpec("unit", "depth_side_table", 3, "submission_v329_unit.csv")
    evidence = {
        "local_delta_vs_anchor": -0.0001,
        "test_changed_rows": 3,
        "point0_additions": 0,
        "point0_removals": 0,
    }

    path = write_submission_if_evidence(outdir, spec, _anchor(), np.array([1, 8, 4]), evidence)

    assert path is None
    assert not outdir.exists()
    assert not evidence_passes(evidence)


def test_point0_evidence_gate_blocks_otherwise_positive_export():
    outdir = _unit_outdir()
    spec = CandidateSpec("unit", "depth_side_table", 3, "submission_v329_unit.csv")
    evidence = {
        "local_delta_vs_anchor": 0.001,
        "test_changed_rows": 3,
        "point0_additions": 1,
        "point0_removals": 0,
    }

    path = write_submission_if_evidence(outdir, spec, _anchor(), np.array([1, 8, 4]), evidence)

    assert path is None
    assert not outdir.exists()
    assert not evidence_passes(evidence)
