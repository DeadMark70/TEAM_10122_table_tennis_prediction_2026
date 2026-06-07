from pathlib import Path

import pandas as pd

from analysis_v400_public_component_recombination import (
    SUBMISSION_COLUMNS,
    build_ranked_point_votes,
    discover_submission_paths,
    load_anchor_submission,
    load_public_positive_sources,
    run_pipeline,
)


def _write_submission(path: Path, point_values: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "rally_uid": [f"r{i}" for i in range(len(point_values))],
            "actionId": [1] * len(point_values),
            "pointId": point_values,
            "serverGetPoint": [0.5] * len(point_values),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def test_source_discovery_skips_missing_directories(tmp_path):
    existing = tmp_path / "v338_joint_moe_pack"
    csv_path = existing / "submission_v338_candidate.csv"
    _write_submission(csv_path, [1, 2, 3])

    paths = discover_submission_paths(
        root=tmp_path,
        source_dirs=("v338_joint_moe_pack", "does_not_exist"),
        extra_globs=(),
    )

    assert paths == [csv_path]


def test_duplicate_and_malformed_csvs_are_ignored(tmp_path):
    anchor_path = tmp_path / "anchor.csv"
    anchor = _write_submission(anchor_path, [1, 2, 3])
    good = tmp_path / "source_a" / "submission_good.csv"
    good_frame = _write_submission(good, [1, 4, 3])
    duplicate = tmp_path / "source_b" / "submission_duplicate.csv"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    good_frame.to_csv(duplicate, index=False)
    malformed = tmp_path / "source_b" / "submission_bad.csv"
    pd.DataFrame({"rally_uid": ["r0"], "pointId": [1]}).to_csv(malformed, index=False)

    sources, ignored = load_public_positive_sources(
        anchor=anchor,
        paths=[good, duplicate, malformed],
        expected_rows=3,
    )

    assert [source.path for source in sources] == [good]
    assert {item["reason"] for item in ignored} == {"duplicate_content", "bad_schema"}


def test_point0_additions_are_blocked_from_ranked_votes(tmp_path):
    anchor = _write_submission(tmp_path / "anchor.csv", [5, 5, 0])
    src1 = tmp_path / "s1" / "submission_s1.csv"
    src2 = tmp_path / "s2" / "submission_s2.csv"
    src3 = tmp_path / "s3" / "submission_s3.csv"
    _write_submission(src1, [0, 7, 7])
    _write_submission(src2, [0, 7, 8])
    _write_submission(src3, [5, 7, 7])
    sources, ignored = load_public_positive_sources(anchor=anchor, paths=[src1, src2, src3], expected_rows=3)

    ranked = build_ranked_point_votes(anchor=anchor, sources=sources, ignored_sources=ignored)

    assert ranked["rally_uid"].tolist() == ["r1", "r2"]
    assert ranked["new_point"].tolist() == [7, 7]
    assert int(ranked["point0_additions"].sum()) == 0


def test_run_pipeline_real_anchor_writes_schema_and_1845_rows(tmp_path):
    report = run_pipeline(outdir=tmp_path)

    assert report["anchor_rows"] == 1845
    assert report["generated_submission_count"] == 3
    for item in report["generated_submissions"]:
        frame = pd.read_csv(item["path"])
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert len(frame) == 1845
        assert item["point0_additions"] == 0
        assert item["action_churn"] == 0
        assert item["server_changed"] == 0


def test_top15_selected_rows_extend_top9(tmp_path):
    report = run_pipeline(outdir=tmp_path)
    top9_path = next(item["selected_rows"] for item in report["generated_submissions"] if "top9" in item["candidate"])
    top15_path = next(item["selected_rows"] for item in report["generated_submissions"] if "top15" in item["candidate"])

    top9 = pd.read_csv(top9_path)
    top15 = pd.read_csv(top15_path)

    assert set(top9["rally_uid"]).issubset(set(top15["rally_uid"]))
    assert len(top9) <= len(top15)


def test_real_anchor_is_v362_public_proven_anchor():
    anchor = load_anchor_submission()

    assert list(anchor.columns) == SUBMISSION_COLUMNS
    assert len(anchor) == 1845
