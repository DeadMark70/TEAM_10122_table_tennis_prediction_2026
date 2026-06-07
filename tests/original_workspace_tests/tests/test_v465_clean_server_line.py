from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis_v465_clean_server_line import (
    aggregate_signals_to_anchor,
    build_action_point_conditioned_features,
    build_scoreboard_features,
    blend_to_target_mad,
    fit_server_signals,
    load_existing_clean_sources,
    no_banned_input_guard,
    package_server_only,
    run_pipeline,
    ServerSignal,
)


def tiny_submission(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": list(range(100, 100 + rows)),
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14][:rows],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9][:rows],
            "serverGetPoint": np.linspace(0.25, 0.75, rows),
        }
    )


def test_clean_guards_reject_old_ttmatch_and_upload_candidate_paths():
    for bad in ["oldserver/file.csv", "TTMATCH/file.csv", "upload_candidates_20260519/file.csv"]:
        with pytest.raises(ValueError):
            no_banned_input_guard([Path(bad)])


def test_blend_to_target_mad_hits_requested_mad():
    anchor = np.array([0.2, 0.8, 0.5, 0.3])
    target = np.array([0.8, 0.2, 0.3, 0.5])
    blended = blend_to_target_mad(anchor, target, target_mad=0.05)
    assert np.mean(np.abs(blended - anchor)) == pytest.approx(0.05)


def test_package_submission_preserves_action_and_point():
    anchor = tiny_submission()
    packaged = package_server_only(anchor, np.array([0.3, 0.4, 0.5, 0.6]), expected_rows=4)
    assert packaged["actionId"].tolist() == anchor["actionId"].tolist()
    assert packaged["pointId"].tolist() == anchor["pointId"].tolist()


def test_build_score_features_uses_rally_id_not_rally_uid_order():
    frame = pd.DataFrame(
        {
            "rally_uid": [300, 100, 200],
            "match": [1, 1, 1],
            "numberGame": [1, 1, 1],
            "rally_id": [3, 1, 2],
            "scoreSelf": [2, 0, 1],
            "scoreOther": [1, 0, 1],
            "strikeNumber": [2, 1, 3],
        }
    )
    features = build_scoreboard_features(frame)
    assert "rally_uid_rank" not in features.columns
    assert "rally_id" in features.columns
    assert "score_margin" in features.columns


def test_action_point_conditioned_features_use_anchor_predictions():
    anchor = tiny_submission()
    features = build_action_point_conditioned_features(anchor)
    assert "anchor_action_family" in features.columns
    assert "anchor_point_depth" in features.columns
    assert "anchor_point0" in features.columns


def test_fit_server_models_returns_oof_and_test_predictions():
    x = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [1, 1, 0, 0, 1, 1, 0, 0]})
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    x_test = pd.DataFrame({"a": [1, 4], "b": [0, 1]})
    signals = fit_server_signals(x, y, x_test, random_state=465)
    assert signals
    for signal in signals:
        assert len(signal.oof) == len(y)
        assert len(signal.test) == len(x_test)
        assert np.isfinite(signal.auc)


def test_aggregate_signals_to_anchor_handles_multiple_prefix_rows_per_rally():
    anchor = tiny_submission(3)
    test_new = pd.DataFrame({"rally_uid": [100, 100, 101, 102, 102, 102]})
    signal = ServerSignal(
        name="unit",
        oof=np.array([0.1, 0.2]),
        test=np.array([0.2, 0.4, 0.8, 0.1, 0.2, 0.3]),
        auc=0.75,
    )

    aggregated = aggregate_signals_to_anchor([signal], test_new, anchor)

    assert len(aggregated) == 1
    assert aggregated[0].test.tolist() == pytest.approx([0.3, 0.8, 0.2])


def test_existing_clean_sources_align_server_by_rally_uid_even_if_action_point_differ(tmp_path):
    anchor = tiny_submission(4)
    source = anchor.copy()
    source["rally_uid"] = source["rally_uid"].iloc[::-1].to_numpy()
    source["actionId"] = [18, 18, 18, 18]
    source["pointId"] = [9, 9, 9, 9]
    source["serverGetPoint"] = [0.9, 0.8, 0.7, 0.6]
    source_dir = tmp_path / "v408_clean_server_microblend_recheck"
    source_dir.mkdir()
    source_path = source_dir / "submission_source.csv"
    source.to_csv(source_path, index=False)
    pd.DataFrame({"candidate": ["unit"], "path": [str(source_path)]}).to_csv(
        source_dir / "ranked_candidates.csv",
        index=False,
    )

    sources = load_existing_clean_sources(tmp_path, anchor, expected_rows=4)

    assert len(sources) == 1
    assert sources[0].server.tolist() == pytest.approx([0.6, 0.7, 0.8, 0.9])


def test_run_pipeline_writes_server_only_candidates(tmp_path):
    root = tmp_path
    rows = 8
    train = pd.DataFrame(
        {
            "rally_uid": range(1, rows + 1),
            "match": [1, 1, 1, 1, 2, 2, 2, 2],
            "numberGame": [1, 1, 2, 2, 1, 1, 2, 2],
            "rally_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "scoreSelf": [0, 1, 2, 4, 0, 1, 6, 8],
            "scoreOther": [0, 0, 1, 3, 1, 1, 4, 7],
            "strikeNumber": [1, 2, 3, 4, 1, 2, 3, 4],
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9],
            "serverGetPoint": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )
    test_new = pd.concat(
        [
            train.drop(columns=["serverGetPoint"]).copy(),
            train.drop(columns=["serverGetPoint"]).iloc[[0, 1, 2, 3]].copy(),
        ],
        ignore_index=True,
    )
    anchor = tiny_submission(rows)
    anchor["rally_uid"] = range(1, rows + 1)

    anchor_dir = root / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir()
    train.to_csv(root / "train.csv", index=False)
    test_new.to_csv(root / "test_new.csv", index=False)
    anchor.to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    report = run_pipeline(root=root, outdir=root / "v465_clean_server_line", expected_rows=rows)
    ranked = pd.read_csv(root / "v465_clean_server_line" / "v465_server_search.csv")
    assert not ranked.empty
    for path in ranked["path"]:
        sub = pd.read_csv(path)
        assert sub["actionId"].equals(anchor["actionId"])
        assert sub["pointId"].equals(anchor["pointId"])
        assert sub["serverGetPoint"].between(0, 1).all()
    assert report["policy"]["no_old_server"] is True
