from pathlib import Path

import pandas as pd

from analysis_v421_ttmatch_risky_contrastive_audit import audit_ttmatch, run_pipeline, write_risky_report


def test_write_risky_report_marks_not_clean_and_creates_no_submission_csv(tmp_path):
    outdir = tmp_path / "v421"

    report = write_risky_report(outdir, {"rows": 3})

    assert report["clean_eligible"] is False
    assert report["reason"] == "TTMATCH has AICUP-like schema/overlap risk"
    assert report["submission_exports"] == 0
    assert (outdir / "risky_ttmatch_audit_report.json").exists()
    assert not list(outdir.glob("submission_*.csv"))


def test_audit_detects_aicup_like_schema_on_tiny_fixture(tmp_path):
    ttmatch_dir = tmp_path / "external_data" / "TTMATCH"
    ttmatch_dir.mkdir(parents=True)
    ttmatch_file = ttmatch_dir / "train.csv"
    aicup_train = tmp_path / "train.csv"
    aicup_test = tmp_path / "test_new.csv"

    rows = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "strikeNumber": [1, 2],
            "actionId": [6, 7],
            "pointId": [4, 5],
            "spinId": [1, 2],
            "strengthId": [2, 3],
        }
    )
    rows.to_csv(ttmatch_file, index=False)
    rows.iloc[[0]].to_csv(aicup_train, index=False)
    rows.iloc[[1]].drop(columns=["strengthId"]).to_csv(aicup_test, index=False)

    summary = audit_ttmatch(ttmatch_dir=ttmatch_dir, train_path=aicup_train, test_path=aicup_test)

    assert summary["ttmatch_present"] is True
    assert summary["aicup_like_file_count"] == 1
    assert summary["overlap_summary"]["max_shared_column_count"] >= 5
    assert summary["dedup_summary"]["row_signature_overlaps_train"] == 1
    assert summary["dedup_summary"]["row_signature_overlaps_test"] == 0
    assert summary["contrastive_pair_counts"]["coarse_context_pairs"] == 1


def test_audit_treats_ttmatch_stricknumber_as_aicup_like_schema(tmp_path):
    ttmatch_dir = tmp_path / "external_data" / "TTMATCH"
    ttmatch_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "rally_uid": ["r1"],
            "strickNumber": [1],
            "actionId": [6],
            "pointId": [4],
            "spinId": [1],
            "strengthId": [2],
        }
    ).to_csv(ttmatch_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "rally_uid": ["r1"],
            "strikeNumber": [1],
            "actionId": [6],
            "pointId": [4],
            "spinId": [1],
            "strengthId": [2],
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"rally_uid": ["x"]}).to_csv(tmp_path / "test_new.csv", index=False)

    summary = audit_ttmatch(
        ttmatch_dir=ttmatch_dir,
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test_new.csv",
    )

    assert summary["aicup_like_file_count"] == 1
    assert summary["dedup_summary"]["row_signature_overlaps_train"] == 1


def test_run_pipeline_handles_absent_or_empty_ttmatch_directory(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test_new.csv"
    pd.DataFrame({"rally_uid": ["r1"], "actionId": [1], "pointId": [2]}).to_csv(train_path, index=False)
    pd.DataFrame({"rally_uid": ["r2"], "actionId": [3], "pointId": [4]}).to_csv(test_path, index=False)

    absent_report = run_pipeline(
        ttmatch_dir=tmp_path / "missing" / "TTMATCH",
        train_path=train_path,
        test_path=test_path,
        outdir=tmp_path / "absent_out",
    )
    empty_dir = tmp_path / "empty" / "TTMATCH"
    empty_dir.mkdir(parents=True)
    empty_report = run_pipeline(
        ttmatch_dir=empty_dir,
        train_path=train_path,
        test_path=test_path,
        outdir=tmp_path / "empty_out",
    )

    assert absent_report["ttmatch_present"] is False
    assert empty_report["ttmatch_present"] is True
    assert empty_report["file_count"] == 0
    assert not list((tmp_path / "absent_out").glob("submission_*.csv"))
    assert not list((tmp_path / "empty_out").glob("submission_*.csv"))
