from pathlib import Path

import pandas as pd

from analysis_v434_anchor_aware_moe_gate import (
    build_moe_candidate_tables,
    moe_change_score,
    run_pipeline,
)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2", "r3"],
            "actionId": [4, 6, 10, 12],
            "pointId": [5, 6, 7, 0],
            "serverGetPoint": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_v434_moe_gate_never_forces_raw_model_when_anchor_confidence_is_high():
    row = {"anchor_confidence": 0.99, "expert_confidence": 0.51, "source_support": 0}

    score = moe_change_score(row)

    assert score < 0


def test_v434_preserves_anchor_when_no_candidate_has_positive_utility():
    anchor = _anchor()
    weak_source = pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"],
            "pred_actionId": [7, 7, 7, 7],
            "pred_pointId": [8, 8, 8, 8],
            "action_confidence": [0.50, 0.49, 0.48, 0.47],
            "point_confidence": [0.50, 0.49, 0.48, 0.47],
            "action_margin": [0.01, 0.01, 0.01, 0.01],
            "point_margin": [0.01, 0.01, 0.01, 0.01],
        }
    )

    action_candidates, point_candidates, report = build_moe_candidate_tables(anchor, [("weak", weak_source)])

    assert action_candidates.empty
    assert point_candidates.empty
    assert report["positive_action_candidates"] == 0
    assert report["positive_point_candidates"] == 0


def test_v434_penalizes_point0_additions_from_nonzero_anchor():
    anchor = _anchor()
    source = pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"],
            "pred_actionId": anchor["actionId"],
            "pred_pointId": [0, 8, 8, 0],
            "action_confidence": [0.99, 0.99, 0.99, 0.99],
            "point_confidence": [0.99, 0.90, 0.88, 0.99],
            "action_margin": [0.90, 0.90, 0.90, 0.90],
            "point_margin": [0.90, 0.80, 0.78, 0.90],
        }
    )

    _action_candidates, point_candidates, report = build_moe_candidate_tables(anchor, [("point_model", source)])

    assert 0 not in point_candidates["candidate_value"].tolist()
    assert report["blocked_point0_candidates"] == 1


def test_v434_blocks_serve_15_18_explosion():
    anchor = _anchor()
    source = pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"],
            "pred_actionId": [15, 16, 17, 18],
            "pred_pointId": anchor["pointId"],
            "action_confidence": [0.99, 0.98, 0.97, 0.96],
            "point_confidence": [0.99, 0.99, 0.99, 0.99],
            "action_margin": [0.90, 0.88, 0.86, 0.84],
            "point_margin": [0.90, 0.90, 0.90, 0.90],
        }
    )

    action_candidates, _point_candidates, report = build_moe_candidate_tables(anchor, [("serve_model", source)])

    assert action_candidates.empty
    assert report["blocked_serve_15_18_candidates"] == 4


def test_v434_pipeline_writes_candidate_tables_and_no_submission_export(tmp_path: Path):
    anchor_path = tmp_path / "anchor.csv"
    pred_dir = tmp_path / "preds"
    outdir = tmp_path / "v434"
    pred_dir.mkdir()
    anchor = _anchor()
    anchor.to_csv(anchor_path, index=False)
    pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"],
            "pred_actionId": [7, 6, 10, 12],
            "pred_pointId": [5, 8, 8, 0],
            "action_confidence": [0.96, 0.95, 0.95, 0.95],
            "point_confidence": [0.95, 0.96, 0.94, 0.95],
            "action_margin": [0.88, 0.80, 0.80, 0.80],
            "point_margin": [0.80, 0.86, 0.82, 0.80],
        }
    ).to_csv(pred_dir / "test_predictions_demo.csv", index=False)

    summary = run_pipeline(
        anchor_path=anchor_path,
        outdir=outdir,
        source_dirs=[pred_dir],
        expected_rows=len(anchor),
    )

    assert (outdir / "moe_action_candidates.csv").exists()
    assert (outdir / "moe_point_candidates.csv").exists()
    assert (outdir / "moe_gate_report.csv").exists()
    assert (outdir / "summary.json").exists()
    assert not list(outdir.glob("submission*.csv"))
    assert summary["submission_exports"] == 0


def test_v434_pipeline_loads_v432_probability_tables(tmp_path: Path):
    anchor_path = tmp_path / "anchor.csv"
    pred_dir = tmp_path / "v432"
    outdir = tmp_path / "v434"
    pred_dir.mkdir()
    anchor = _anchor()
    anchor.to_csv(anchor_path, index=False)
    pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"],
            "pred_action": [7, 6, 10, 12],
            "action_confidence": [0.96, 0.95, 0.95, 0.95],
            "action_margin": [0.88, 0.80, 0.80, 0.80],
        }
    ).to_csv(pred_dir / "test_action_probs_demo.csv", index=False)
    pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"],
            "pred_point": [5, 8, 8, 0],
            "point_confidence": [0.95, 0.96, 0.94, 0.95],
            "point_margin": [0.80, 0.86, 0.82, 0.80],
        }
    ).to_csv(pred_dir / "test_point_probs_demo.csv", index=False)

    summary = run_pipeline(
        anchor_path=anchor_path,
        outdir=outdir,
        source_dirs=[pred_dir],
        expected_rows=len(anchor),
    )

    assert summary["source_count"] >= 1
    assert summary["action_candidate_count"] >= 1
    assert summary["point_candidate_count"] >= 1
