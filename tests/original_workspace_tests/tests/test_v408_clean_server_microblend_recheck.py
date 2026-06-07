import json

import numpy as np
import pandas as pd
import pytest

from analysis_v408_clean_server_microblend_recheck import (
    SUBMISSION_COLUMNS,
    build_microblend_candidates,
    clip_prob,
    run_pipeline,
    should_emit_candidate,
)


def _submission(rows: int = 1845) -> pd.DataFrame:
    idx = np.arange(rows)
    return pd.DataFrame(
        {
            "rally_uid": 1000 + idx,
            "actionId": idx % 19,
            "pointId": idx % 10,
            "serverGetPoint": np.linspace(0.05, 0.95, rows),
        }
    )


def _write_source(root, dirname, name, anchor, server):
    path = root / dirname
    path.mkdir(parents=True, exist_ok=True)
    out = anchor.copy()
    out["serverGetPoint"] = server
    out.to_csv(path / name, index=False)


def test_microblends_preserve_action_point_and_include_mad_corr_metadata(tmp_path):
    anchor = _submission(8)
    source_a = anchor["serverGetPoint"].to_numpy(dtype=float)[::-1]
    source_b = np.linspace(0.9, 0.1, len(anchor))

    rows, submissions = build_microblend_candidates(
        anchor,
        [("a", source_a), ("b", source_b)],
        outdir=tmp_path,
        expected_rows=8,
    )

    assert {"mean_w0p005", "mean_w0p010", "rankavg_w0p005", "rankavg_w0p010"} <= set(submissions)
    assert all(row["action_churn"] == 0 and row["point_churn"] == 0 for row in rows)
    assert all("server_mad" in row and "server_corr" in row for row in rows)
    for frame in submissions.values():
        assert frame[["rally_uid", "actionId", "pointId"]].equals(anchor[["rally_uid", "actionId", "pointId"]])


def test_clip_prob_uses_required_public_probability_bounds():
    clipped = clip_prob(np.array([-10.0, 0.25, np.nan, 10.0]))

    assert clipped.tolist() == pytest.approx([0.001, 0.25, 0.5, 0.999])


def test_high_mad_candidates_are_blocked():
    assert should_emit_candidate(server_mad=0.02, action_churn=0, point_churn=0)
    assert not should_emit_candidate(server_mad=0.02001, action_churn=0, point_churn=0)
    assert not should_emit_candidate(server_mad=0.001, action_churn=1, point_churn=0)
    assert not should_emit_candidate(server_mad=0.001, action_churn=0, point_churn=1)


def test_duplicate_score_feature_rows_skip_scorestate_safe(tmp_path):
    anchor = _submission(4)
    source = np.array([0.9, 0.8, 0.2, 0.1])
    pd.DataFrame(
        {
            "rally_uid": [1000, 1000, 1001, 1002, 1003],
            "scoreSelf": [0, 1, 2, 3, 4],
            "scoreOther": [0, 1, 1, 1, 1],
        }
    ).to_csv(tmp_path / "test_new.csv", index=False)

    rows, submissions = build_microblend_candidates(
        anchor,
        [("source", source)],
        outdir=tmp_path,
        root=tmp_path,
        expected_rows=4,
    )

    assert "scorestate_safe" not in submissions
    assert all(row["candidate"] != "scorestate_safe" for row in rows)


def test_run_pipeline_writes_schema_1845_rows_ranked_candidates_and_report(tmp_path):
    root = tmp_path / "root"
    anchor_dir = root / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir(parents=True)
    anchor = _submission()
    anchor_path = anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv"
    anchor.to_csv(anchor_path, index=False)

    base_server = anchor["serverGetPoint"].to_numpy(dtype=float)
    _write_source(
        root,
        "v300_clean_server_blend_recycler",
        "submission_v300_mean_w0p005__v173action_v261point_server.csv",
        anchor,
        1.0 - base_server,
    )
    _write_source(
        root,
        "v321_server_robust_rankblend",
        "submission_v321_server_rankblend_mad0p001__v173action_v306point.csv",
        anchor,
        np.roll(base_server, 7),
    )

    report = run_pipeline(root=root, outdir=root / "v408_clean_server_microblend_recheck")

    ranked = pd.read_csv(root / "v408_clean_server_microblend_recheck" / "ranked_candidates.csv")
    assert not ranked.empty
    assert set(SUBMISSION_COLUMNS) == set(pd.read_csv(ranked.loc[0, "path"]).columns)
    for path in ranked["path"]:
        submission = pd.read_csv(path)
        assert list(submission.columns) == SUBMISSION_COLUMNS
        assert len(submission) == 1845
        assert submission["actionId"].astype(int).equals(anchor["actionId"].astype(int))
        assert submission["pointId"].astype(int).equals(anchor["pointId"].astype(int))
        assert submission["serverGetPoint"].between(0.001, 0.999).all()
    assert (ranked["server_mad"] <= 0.02).all()
    assert (ranked["action_churn"] == 0).all()
    assert (ranked["point_churn"] == 0).all()
    assert report["generated_submission_count"] == len(ranked)

    saved_report = json.loads((root / "v408_clean_server_microblend_recheck" / "search_report.json").read_text())
    assert saved_report["anchor"] == str(anchor_path)
    assert saved_report["source_count"] == 2
