from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v469_server_public_like_validation import (
    build_anchor_slices,
    fit_density_weights,
    load_and_validate_submission,
    make_public_like_bins,
    rank_candidates,
    run_pipeline,
)


def tiny_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": np.arange(1, 13),
            "strikeNumber": [1, 2, 3, 4, 7, 8, 1, 2, 5, 6, 9, 10],
            "scoreSelf": [0, 1, 9, 10, 3, 4, 11, 8, 2, 6, 10, 12],
            "scoreOther": [0, 0, 9, 9, 5, 4, 10, 8, 1, 7, 11, 11],
            "actionId": [0, 1, 8, 12, 14, 10, 15, 2, 7, 9, 13, 4],
            "pointId": [0, 1, 4, 7, 9, 6, 2, 5, 8, 3, 0, 9],
            "serverGetPoint": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        }
    )


def tiny_anchor(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": np.arange(100, 100 + rows),
            "actionId": [0, 1, 8, 12, 14, 10][:rows],
            "pointId": [0, 1, 4, 7, 9, 6][:rows],
            "serverGetPoint": np.linspace(0.2, 0.8, rows),
        }
    )


def tiny_test_new(anchor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in anchor.iterrows():
        for prefix in range(1 + idx % 3):
            rows.append(
                {
                    "rally_uid": row["rally_uid"],
                    "strikeNumber": prefix + 1,
                    "scoreSelf": idx + prefix,
                    "scoreOther": idx,
                    "actionId": row["actionId"],
                    "pointId": row["pointId"],
                }
            )
    return pd.DataFrame(rows)


def test_public_like_bins_and_weights_are_finite():
    train_bins = make_public_like_bins(tiny_train())
    test_bins = make_public_like_bins(tiny_test_new(tiny_anchor()))
    weights = fit_density_weights(train_bins, test_bins)

    assert len(weights) == len(train_bins)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert weights.max() <= 10.0


def test_anchor_slices_are_anchor_aligned():
    anchor = tiny_anchor()
    slices = build_anchor_slices(anchor, tiny_test_new(anchor))

    assert len(slices) == len(anchor)
    assert slices["rally_uid"].equals(anchor["rally_uid"])
    assert {"prefix_bin", "phase_bin", "score_pressure", "terminal_like"}.issubset(slices.columns)


def test_load_and_validate_submission_rejects_action_change(tmp_path: Path):
    anchor = tiny_anchor()
    bad = anchor.copy()
    bad.loc[0, "actionId"] = 99
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)

    try:
        load_and_validate_submission(path, anchor)
    except ValueError as exc:
        assert "actionId" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_rank_candidates_penalizes_diagnostic_and_concentrated_changes():
    frame = pd.DataFrame(
        {
            "candidate": ["safe_ensemble", "diagnostic_big", "spiky"],
            "server_mad": [0.003, 0.015, 0.003],
            "server_corr": [0.999, 0.990, 0.999],
            "risk": ["safe", "diagnostic", "safe"],
            "decision": ["review", "diagnostic_hold", "review"],
            "family_diversity": [4, 4, 1],
            "top20_share": [0.20, 0.20, 0.95],
            "max_slice_mad_ratio": [1.2, 1.1, 3.5],
            "path": ["a", "b", "c"],
        }
    )
    ranked = rank_candidates(frame)

    assert ranked.iloc[0]["candidate"] == "safe_ensemble"
    assert ranked.iloc[-1]["candidate"] in {"diagnostic_big", "spiky"}


def test_run_pipeline_with_tiny_candidates(tmp_path: Path):
    anchor = tiny_anchor()
    train = tiny_train()
    test_new = tiny_test_new(anchor)
    anchor_dir = tmp_path / "v362_point_hierarchical_specialists"
    candidate_dir = tmp_path / "v468_server_full_run"
    anchor_dir.mkdir()
    candidate_dir.mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    test_new.to_csv(tmp_path / "test_new.csv", index=False)
    anchor_path = anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv"
    anchor.to_csv(anchor_path, index=False)
    cand = anchor.copy()
    cand["serverGetPoint"] = np.clip(cand["serverGetPoint"] + np.linspace(-0.01, 0.01, len(cand)), 0, 1)
    cand_path = candidate_dir / "submission_v468_tiny.csv"
    cand.to_csv(cand_path, index=False)
    pd.DataFrame(
        {
            "candidate": ["tiny"],
            "path": [str(cand_path)],
            "server_mad": [float(np.mean(np.abs(cand["serverGetPoint"] - anchor["serverGetPoint"])))],
            "server_corr": [0.999],
            "risk": ["safe"],
            "decision": ["review"],
            "family_diversity": [2],
        }
    ).to_csv(candidate_dir / "v468_server_search.csv", index=False)

    report = run_pipeline(root=tmp_path, outdir=tmp_path / "v469", expected_rows=len(anchor))
    board = pd.read_csv(tmp_path / "v469" / "v469_candidate_rank.csv")

    assert report["candidate_count"] == 1
    assert not board.empty
    assert board.iloc[0]["candidate"] == "tiny"
