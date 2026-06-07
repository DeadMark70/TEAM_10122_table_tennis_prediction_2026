import json
from pathlib import Path

import pandas as pd

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS


def _write_fixture_inputs(root: Path) -> pd.DataFrame:
    anchor = pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2", "r3", "r4"],
            "actionId": [1, 2, 3, 15, 5],
            "pointId": [1, 0, 3, 4, 5],
            "serverGetPoint": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )
    anchor_dir = root / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir(parents=True)
    anchor.to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    v416_dir = root / "v416_external_embedding_aicup_finetune"
    v416_dir.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2", "r3", "r4"],
            "pred_actionId": [4, 6, 15, 16, 7],
            "pred_pointId": [2, 8, 0, 9, 6],
            "action_confidence": [0.92, 0.72, 0.99, 0.96, 0.82],
            "point_confidence": [0.91, 0.70, 0.99, 0.84, 0.83],
            "joint_confidence": [0.89, 0.61, 0.98, 0.80, 0.79],
        }
    )
    predictions.to_csv(v416_dir / "test_predictions.csv", index=False)
    predictions.to_csv(v416_dir / "oof_predictions.csv", index=False)
    (v416_dir / "local_metrics.json").write_text(json.dumps({"action_macro_f1": 0.1}), encoding="utf-8")
    return anchor


def test_packaging_preserves_schema_row_count_and_exports_reports(tmp_path):
    from analysis_v417_external_embedding_lowchurn_pack import run_pipeline

    anchor = _write_fixture_inputs(tmp_path)

    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=len(anchor))

    assert report["anchor_rows"] == len(anchor)
    assert (tmp_path / "out" / "candidate_summary.csv").exists()
    assert (tmp_path / "out" / "packaging_report.json").exists()
    for item in report["generated_submissions"]:
        frame = pd.read_csv(item["path"])
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert len(frame) == len(anchor)


def test_point_only_does_not_change_action_or_server(tmp_path):
    from analysis_v417_external_embedding_lowchurn_pack import run_pipeline

    anchor = _write_fixture_inputs(tmp_path)
    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=len(anchor))
    point_top5 = next(item for item in report["generated_submissions"] if item["candidate"] == "point_top5")
    frame = pd.read_csv(point_top5["path"])

    assert frame["actionId"].astype(int).tolist() == anchor["actionId"].astype(int).tolist()
    assert frame["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
    assert point_top5["action_churn"] == 0
    assert point_top5["server_changed"] == 0


def test_action_only_does_not_change_point_or_server(tmp_path):
    from analysis_v417_external_embedding_lowchurn_pack import run_pipeline

    anchor = _write_fixture_inputs(tmp_path)
    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=len(anchor))
    action_top5 = next(item for item in report["generated_submissions"] if item["candidate"] == "action_top5")
    frame = pd.read_csv(action_top5["path"])

    assert frame["pointId"].astype(int).tolist() == anchor["pointId"].astype(int).tolist()
    assert frame["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
    assert action_top5["point_churn"] == 0
    assert action_top5["server_changed"] == 0


def test_point0_additions_are_blocked_unless_anchor_already_point0(tmp_path):
    from analysis_v417_external_embedding_lowchurn_pack import run_pipeline

    anchor = _write_fixture_inputs(tmp_path)
    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=len(anchor))
    point_top5 = next(item for item in report["generated_submissions"] if item["candidate"] == "point_top5")
    frame = pd.read_csv(point_top5["path"])

    assert int(frame.loc[2, "pointId"]) == int(anchor.loc[2, "pointId"])
    assert int(frame.loc[1, "pointId"]) == 8
    assert point_top5["point0_additions"] == 0


def test_serve_action_15_to_18_additions_are_blocked(tmp_path):
    from analysis_v417_external_embedding_lowchurn_pack import run_pipeline

    anchor = _write_fixture_inputs(tmp_path)
    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=len(anchor))
    action_top5 = next(item for item in report["generated_submissions"] if item["candidate"] == "action_top5")
    frame = pd.read_csv(action_top5["path"])

    assert int(frame.loc[2, "actionId"]) == int(anchor.loc[2, "actionId"])
    assert int(frame.loc[3, "actionId"]) == 16
    assert action_top5["serve_15_18_additions"] == 0
