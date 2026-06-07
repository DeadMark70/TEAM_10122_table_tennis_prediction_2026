from pathlib import Path

import numpy as np
import pandas as pd


def _write_submission(path: Path, rows: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "rally_uid": list(range(100, 100 + rows)),
            "actionId": [4, 10, 12, 6][:rows],
            "pointId": [8, 5, 7, 9][:rows],
            "serverGetPoint": [0.45, 0.55, 0.48, 0.52][:rows],
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _write_candidates(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "rally_uid": [100, 101, 102, 103],
            "base_point": [8, 5, 7, 9],
            "candidate_point": [9, 0, 8, 6],
            "agreement_count": [5, 99, 7, 3],
            "source_dir_count": [2, 5, 3, 1],
            "score": [9.5, 99.0, 10.0, 2.0],
            "is_point0_addition": [False, True, False, False],
            "depth_agree": [True, False, True, False],
            "side_agree": [False, True, False, True],
            "bank_agree": [False, True, True, False],
        }
    )
    frame.to_csv(path, index=False)


def test_fallback_mode_when_labels_cannot_align(tmp_path):
    from analysis_v403_neural_posterior_gate import run_pipeline

    anchor_path = tmp_path / "anchor.csv"
    candidate_path = tmp_path / "scored_candidates.csv"
    train_path = tmp_path / "train_missing_labels.csv"
    _write_submission(anchor_path)
    _write_candidates(candidate_path)
    pd.DataFrame({"rally_uid": [1, 2], "actionId": [4, 5]}).to_csv(train_path, index=False)

    report = run_pipeline(
        outdir=tmp_path / "out",
        anchor_path=anchor_path,
        train_path=train_path,
        candidate_paths=(candidate_path,),
        expected_rows=4,
    )

    assert report["model_used"] == "fallback_evidence_scorer"
    assert report["decision"] == "HAS_EXPORT"


def test_no_raw_argmax_submission_is_produced(tmp_path):
    from analysis_v403_neural_posterior_gate import run_pipeline

    anchor_path = tmp_path / "anchor.csv"
    candidate_path = tmp_path / "scored_candidates.csv"
    train_path = tmp_path / "train_missing_labels.csv"
    _write_submission(anchor_path)
    _write_candidates(candidate_path)
    pd.DataFrame({"rally_uid": [1], "actionId": [4]}).to_csv(train_path, index=False)

    run_pipeline(
        outdir=tmp_path / "out",
        anchor_path=anchor_path,
        train_path=train_path,
        candidate_paths=(candidate_path,),
        expected_rows=4,
    )

    names = {path.name for path in (tmp_path / "out").glob("*.csv")}
    assert "submission_v403_raw_argmax__v173action_v300server.csv" not in names
    assert all("argmax" not in name.lower() for name in names)


def test_candidate_posterior_scores_are_finite_and_block_point0(tmp_path):
    from analysis_v403_neural_posterior_gate import run_pipeline

    anchor_path = tmp_path / "anchor.csv"
    candidate_path = tmp_path / "scored_candidates.csv"
    train_path = tmp_path / "train.csv"
    _write_submission(anchor_path)
    _write_candidates(candidate_path)
    pd.DataFrame(
        {
            "rally_uid": range(20),
            "strikeNumber": range(1, 21),
            "actionId": [4, 10, 12, 6] * 5,
            "pointId": [9, 5, 8, 6] * 5,
            "strengthId": [1, 2] * 10,
            "spinId": [1, 2, 3, 4] * 5,
        }
    ).to_csv(train_path, index=False)

    run_pipeline(
        outdir=tmp_path / "out",
        anchor_path=anchor_path,
        train_path=train_path,
        candidate_paths=(candidate_path,),
        expected_rows=4,
    )

    scores = pd.read_csv(tmp_path / "out" / "candidate_posterior_scores.csv")
    assert not scores.empty
    assert np.isfinite(scores["posterior_score"].to_numpy(dtype=float)).all()
    assert (scores["candidate_point"] != 0).all()


def test_output_schema_and_row_count_on_real_anchor(tmp_path):
    from analysis_v403_neural_posterior_gate import SUBMISSION_COLUMNS, run_pipeline

    report = run_pipeline(outdir=tmp_path / "out")

    assert report["anchor_rows"] == 1845
    for filename in (
        "submission_v403_posterior_top9__v173action_v300server.csv",
        "submission_v403_posterior_top15__v173action_v300server.csv",
    ):
        frame = pd.read_csv(tmp_path / "out" / filename)
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert len(frame) == 1845

    ranked = pd.read_csv(tmp_path / "out" / "ranked_candidates.csv")
    assert set(["candidate", "path", "selected_row_count", "action_churn", "point_churn"]).issubset(
        ranked.columns
    )
