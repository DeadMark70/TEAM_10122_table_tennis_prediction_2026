import json

import pandas as pd

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS
from analysis_v388_large_synthetic_candidate_pool import (
    ACTION_POOL_COLUMNS,
    POINT_POOL_COLUMNS,
    aggregate_action_pool,
    aggregate_point_pool,
    run_pipeline,
    validate_anchor_frame,
)


def _submission(rows):
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def test_pool_builders_aggregate_duplicate_rally_candidate_pairs():
    anchor = _submission(
        [
            [101, 3, 8, 0.2],
            [102, 4, 7, 0.8],
            [103, 5, 6, 0.4],
        ]
    )
    first = _submission(
        [
            [101, 3, 9, 0.2],
            [102, 6, 7, 0.8],
            [103, 5, 6, 0.4],
        ]
    )
    second = _submission(
        [
            [101, 3, 9, 0.2],
            [102, 6, 7, 0.8],
            [103, 5, 6, 0.4],
        ]
    )

    point_pool = aggregate_point_pool(
        anchor,
        [(first, "submission_a.csv", "source_a"), (second, "submission_b.csv", "source_b")],
    )
    action_pool = aggregate_action_pool(
        anchor,
        [(first, "submission_a.csv", "source_a"), (second, "submission_b.csv", "source_b")],
    )

    point_row = point_pool.loc[point_pool["rally_uid"] == 101].iloc[0]
    assert point_row["candidate_point"] == 9
    assert point_row["support_count"] == 2
    assert point_row["source_family_count"] == 2
    assert point_row["source_file"] == "submission_a.csv|submission_b.csv"
    assert point_row["source_dir"] == "source_a|source_b"

    action_row = action_pool.loc[action_pool["rally_uid"] == 102].iloc[0]
    assert action_row["candidate_action"] == 6
    assert action_row["support_count"] == 2
    assert action_row["source_family_count"] == 2
    assert action_row["source_file"] == "submission_a.csv|submission_b.csv"


def test_point0_additions_are_marked_explicitly():
    anchor = _submission([[201, 3, 8, 0.3], [202, 4, 0, 0.7]])
    candidate = _submission([[201, 3, 0, 0.3], [202, 4, 9, 0.7]])

    point_pool = aggregate_point_pool(anchor, [(candidate, "submission_p0.csv", "v306_point0_addition_probe")])

    add_row = point_pool.loc[point_pool["rally_uid"] == 201].iloc[0]
    removal_row = point_pool.loc[point_pool["rally_uid"] == 202].iloc[0]
    assert add_row["is_point0_addition"] is True
    assert add_row["is_point0_removal"] is False
    assert removal_row["is_point0_addition"] is False
    assert removal_row["is_point0_removal"] is True


def test_serve_15_18_action_additions_are_marked_explicitly():
    anchor = _submission([[301, 3, 8, 0.3], [302, 16, 7, 0.7]])
    candidate = _submission([[301, 15, 8, 0.3], [302, 3, 7, 0.7]])

    action_pool = aggregate_action_pool(anchor, [(candidate, "submission_action.csv", "action_source")])

    serve_add = action_pool.loc[action_pool["rally_uid"] == 301].iloc[0]
    serve_remove = action_pool.loc[action_pool["rally_uid"] == 302].iloc[0]
    assert serve_add["is_serve_15_18_addition"] is True
    assert serve_remove["is_serve_15_18_addition"] is False


def test_pipeline_preserves_anchor_schema_and_writes_expected_outputs(tmp_path):
    anchor_dir = tmp_path / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir()
    anchor = _submission([[401, 3, 8, 0.2], [402, 4, 7, 0.8]])
    anchor.to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    source_dir = tmp_path / "v261_action_conditioned_point_residual"
    source_dir.mkdir()
    _submission([[401, 3, 9, 0.2], [402, 15, 7, 0.8]]).to_csv(
        source_dir / "submission_fixture.csv", index=False
    )

    report = run_pipeline(root=tmp_path, source_dirs=["v261_action_conditioned_point_residual"])

    outdir = tmp_path / "v388_large_synthetic_candidate_pool"
    assert report["anchor_columns"] == SUBMISSION_COLUMNS
    assert validate_anchor_frame(anchor) == SUBMISSION_COLUMNS
    assert report["skipped_dirs"] == []
    assert sorted(report["outputs"]) == [
        "action_change_pool.csv",
        "candidate_source_summary.csv",
        "point_change_pool.csv",
        "search_report.json",
    ]

    point_pool = pd.read_csv(outdir / "point_change_pool.csv")
    action_pool = pd.read_csv(outdir / "action_change_pool.csv")
    summary = pd.read_csv(outdir / "candidate_source_summary.csv")
    stored = json.loads((outdir / "search_report.json").read_text())

    assert list(point_pool.columns) == POINT_POOL_COLUMNS
    assert list(action_pool.columns) == ACTION_POOL_COLUMNS
    assert summary.loc[0, "source_dir"] == "v261_action_conditioned_point_residual"
    assert stored["point_pool_rows"] == 1
    assert stored["action_pool_rows"] == 1
