from pathlib import Path

import pandas as pd

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS


def _write_submission(path: Path, points: list[int], actions: list[int] | None = None) -> pd.DataFrame:
    actions = actions or [1] * len(points)
    frame = pd.DataFrame(
        {
            "rally_uid": [f"r{i}" for i in range(len(points))],
            "actionId": actions,
            "pointId": points,
            "serverGetPoint": [0.25 + (i * 0.01) for i in range(len(points))],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def test_missing_predecessor_falls_back_to_v362_selected_metadata(tmp_path):
    from analysis_v405_v362_pruning_lab import run_pipeline

    v362_dir = tmp_path / "v362_point_hierarchical_specialists"
    _write_submission(v362_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", [8, 6, 9])
    pd.DataFrame(
        {
            "row_id": [0, 2],
            "rally_uid": ["r0", "r2"],
            "base_point": [7, 0],
            "candidate_point": [8, 9],
            "depth_support": [2, 0],
        }
    ).to_csv(v362_dir / "selected_v362_depth_agree_only.csv", index=False)

    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=3)

    assert report["predecessor"]["kind"] == "pseudo_from_v362_selected_rows"
    assert report["v362_change_count"] == 2
    assert (tmp_path / "out" / "ranked_candidates.csv").exists()


def test_remove_low_support_preserves_v362_action_and_server(tmp_path):
    from analysis_v405_v362_pruning_lab import run_pipeline

    v362 = _write_submission(
        tmp_path / "v362_point_hierarchical_specialists" / "submission_v362_depth_agree_only__v173action_v300server.csv",
        [8, 6, 9],
        actions=[3, 4, 5],
    )
    _write_submission(tmp_path / "v338_joint_moe_pack" / "submission_v338_candidate.csv", [7, 4, 8])

    report = run_pipeline(root=tmp_path, outdir=tmp_path / "out", expected_rows=3)
    item = next(row for row in report["generated_submissions"] if row["candidate"] == "v405_v362_remove_low_support")
    out = pd.read_csv(item["path"])

    assert out["actionId"].astype(int).tolist() == v362["actionId"].astype(int).tolist()
    assert out["serverGetPoint"].tolist() == v362["serverGetPoint"].tolist()
    assert item["action_churn"] == 0
    assert item["server_changed"] == 0


def test_point0_additions_are_blocked(tmp_path):
    from analysis_v405_v362_pruning_lab import package_candidate

    predecessor = _write_submission(tmp_path / "pred.csv", [5, 8, 0])
    v362 = _write_submission(tmp_path / "v362.csv", [7, 9, 8])
    changes = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "old_point": [5, 8, 0],
            "new_point": [7, 9, 8],
            "point0_related": [False, False, True],
            "public_agreement": [False, False, False],
            "low_support": [True, True, True],
        }
    )

    out, selected = package_candidate(
        candidate="test",
        v362=v362,
        predecessor=predecessor,
        changes=changes,
        mode="remove_mask",
        remove_mask=changes["low_support"],
    )

    assert 2 not in selected["row_id"].astype(int).tolist()
    assert out["pointId"].astype(int).tolist() == [5, 8, 8]


def test_real_run_outputs_submission_schema_and_1845_rows(tmp_path):
    from analysis_v405_v362_pruning_lab import run_pipeline

    report = run_pipeline(outdir=tmp_path)

    assert report["anchor_rows"] == 1845
    assert report["generated_submission_count"] >= 1
    for item in report["generated_submissions"]:
        frame = pd.read_csv(item["path"])
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert len(frame) == 1845
        assert item["point0_additions"] == 0
        assert item["action_churn"] == 0
        assert item["server_changed"] == 0


def test_ranked_candidates_include_churn_and_selected_rows(tmp_path):
    from analysis_v405_v362_pruning_lab import run_pipeline

    report = run_pipeline(outdir=tmp_path)
    ranked = pd.read_csv(tmp_path / "ranked_candidates.csv")

    required = {
        "candidate",
        "path",
        "selected_rows",
        "selected_row_count",
        "action_churn",
        "point_churn",
        "point0_additions",
        "server_changed",
        "risk",
        "evidence",
    }
    assert required.issubset(ranked.columns)
    assert ranked["selected_row_count"].ge(0).all()
    assert ranked["point_churn"].ge(0).all()
    assert set(report["classification_counts"]).issuperset(
        {"long_side", "half_boundary", "short_control", "point0_related", "public_agreement", "low_support"}
    )
