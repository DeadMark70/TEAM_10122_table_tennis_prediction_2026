from pathlib import Path

import pandas as pd
import pytest

import analysis_v334_joint_hierarchical_action_point as v334


def _unit_tmp() -> Path:
    path = Path("v334_joint_hierarchical_action_point") / "unit_tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_joint_refuses_negative_component_candidates():
    assert v334.passing_v332({"reviewable_candidates": [{"evidence_pass": 1, "action_oof_delta": -0.1}]}) == []


def test_compatibility_gate_blocks_terminal_incompatible_or_negative_action():
    point = {"test_changed_rows": 12, "point_oof_delta_vs_v306": 0.002}
    assert not v334.compatibility_allows({"changed_action_rows": 0, "action_oof_delta": 0.01}, point)
    assert not v334.compatibility_allows({"changed_action_rows": 10, "action_oof_delta": -0.01}, point)
    assert not v334.compatibility_allows({"changed_action_rows": 10, "action_oof_delta": 0.01, "serve_action_rows": 1}, point)
    assert v334.compatibility_allows({"changed_action_rows": 10, "action_oof_delta": 0.01, "serve_action_rows": 0}, point)


def test_protected_output_path_blocks_upload_dirs():
    path = v334.protected_output_path(Path("v334_joint_hierarchical_action_point"), "submission_v334_ok.csv")
    assert path.parent == Path("v334_joint_hierarchical_action_point")

    with pytest.raises(ValueError):
        v334.protected_output_path(Path("upload_candidates"), "bad.csv")


def test_copy_candidate_preserves_schema(monkeypatch):
    tmp = _unit_tmp()
    monkeypatch.setattr(v334, "OUTDIR", tmp)
    src = tmp / "src.csv"
    pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [3, 4],
            "pointId": [0, 8],
            "serverGetPoint": [0.1, 0.9],
        }
    ).to_csv(src, index=False)

    out_rel = v334.copy_candidate(src, "submission_v334_copy.csv")
    out = Path(out_rel)
    if not out.exists():
        out = tmp / "submission_v334_copy.csv"
    frame = pd.read_csv(out)

    assert frame.columns.tolist() == v334.SUBMISSION_COLUMNS
    assert frame["serverGetPoint"].tolist() == [0.1, 0.9]


def test_no_export_when_no_component_passes(monkeypatch):
    tmp = _unit_tmp()
    monkeypatch.setattr(v334, "OUTDIR", tmp)
    monkeypatch.setattr(v334, "ANCHOR_SUBMISSION", tmp / "anchor.csv")
    pd.DataFrame(
        {
            "rally_uid": [1],
            "actionId": [4],
            "pointId": [8],
            "serverGetPoint": [0.5],
        }
    ).to_csv(tmp / "anchor.csv", index=False)
    monkeypatch.setattr(v334, "load_json", lambda path: {"decision": "DO_NOT_UPLOAD", "verdict": "NO_EXPORT"})

    report = v334.run_pipeline()

    assert report["generated_submission_count"] == 0
    assert report["recommendation"] == "DO_NOT_UPLOAD"
