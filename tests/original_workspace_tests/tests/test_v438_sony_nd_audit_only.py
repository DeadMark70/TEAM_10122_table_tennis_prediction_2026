from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis_v438_sony_nd_audit_only import (
    build_sony_audit_report,
    classify_sony_license_policy,
    run_pipeline,
    scan_sony_sources,
)


def test_v438_sony_cc_by_nc_nd_is_audit_only_and_not_train_allowed():
    policy = classify_sony_license_policy("Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 CC BY-NC-ND")

    assert policy["tier"] == "audit_only"
    assert policy["train_allowed_by_default"] is False
    assert policy["clean_train_allowed"] is False
    assert policy["submission_exports"] == 0


def test_v438_sony_policy_defaults_to_cc_by_nc_nd_when_local_license_text_missing():
    policy = classify_sony_license_policy("")

    assert policy["license_family"] == "CC BY-NC-ND"
    assert policy["license_evidence_detected"] is False
    assert policy["train_allowed_by_default"] is False


def test_v438_scan_sony_sources_reads_targeted_sony_files_only(tmp_path: Path):
    sony_root = tmp_path / "external_data" / "sonytabletennis"
    other_root = tmp_path / "external_data" / "TTMATCH"
    sony_root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    (sony_root / "LICENSE").write_text("CC BY-NC-ND 4.0", encoding="utf-8")
    pd.DataFrame({"sequence_id": ["sony_1"], "coarse_feature": ["rally"]}).to_csv(sony_root / "events.csv", index=False)
    pd.DataFrame({"sequence_id": ["risk_1"]}).to_csv(other_root / "events.csv", index=False)

    summary = scan_sony_sources(tmp_path / "external_data")

    assert len(summary) == 2
    assert {row["relative_path"] for row in summary} == {"sonytabletennis/LICENSE", "sonytabletennis/events.csv"}
    assert sum(row["row_count"] for row in summary) == 1


def test_v438_tiny_fake_sony_source_produces_no_trainable_rows_or_submission_files(tmp_path: Path):
    sony_root = tmp_path / "external_data" / "sonytabletennis"
    outdir = tmp_path / "v438_sony_nd_audit_only"
    sony_root.mkdir(parents=True)
    (sony_root / "LICENSE").write_text("CC BY-NC-ND 4.0", encoding="utf-8")
    pd.DataFrame(
        {
            "sequence_id": ["sony_1", "sony_2"],
            "coarse_family": ["serve", "rally"],
            "actionId": [15, 7],
            "pointId": [0, 8],
        }
    ).to_csv(sony_root / "events.csv", index=False)

    report = run_pipeline(root=tmp_path / "external_data", outdir=outdir)
    disk_report = json.loads((outdir / "sony_nd_audit_report.json").read_text(encoding="utf-8"))

    assert report == disk_report
    assert report["tier"] == "audit_only"
    assert report["clean_train_allowed"] is False
    assert report["train_allowed_by_default"] is False
    assert report["trainable_rows"] == 0
    assert report["submission_exports"] == 0
    assert report["source_row_count"] == 2
    assert not list(outdir.glob("submission_*.csv"))
    assert not list(outdir.glob("*canonical*.csv"))


def test_v438_build_report_lists_possible_coarse_features_only_after_approval(tmp_path: Path):
    sony_root = tmp_path / "external_data" / "sonytabletennis"
    sony_root.mkdir(parents=True)
    (sony_root / "README.md").write_text("Sony table tennis data under CC BY-NC-ND 4.0", encoding="utf-8")
    pd.DataFrame(
        {
            "sequence_id": ["sony_1"],
            "coarse_family": ["drive"],
            "phase": ["rally"],
            "speed": ["fast"],
        }
    ).to_csv(sony_root / "sony_events.csv", index=False)

    report = build_sony_audit_report(tmp_path / "external_data")

    assert report["possible_coarse_features_if_later_approved"] == ["coarse_family", "phase", "speed"]
    assert report["blocked_uses"] == ["clean training rows", "canonical training exports", "submission exports"]
