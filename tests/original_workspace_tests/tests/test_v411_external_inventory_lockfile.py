import json
from pathlib import Path

import pandas as pd

from analysis_v411_external_inventory_lockfile import build_manifest, run_pipeline


def test_manifest_records_csv_hash_and_row_count(tmp_path):
    root = tmp_path / "external_data"
    source = root / "openttgames"
    source.mkdir(parents=True)
    csv_path = source / "events.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    manifest = build_manifest(external_root=root)

    row = manifest.iloc[0]
    assert row["source_dataset"] == "openttgames"
    assert row["row_count"] == 2
    assert len(row["sha256"]) == 64
    assert json.loads(row["columns_json"]) == ["a", "b"]


def test_ttmatch_is_excluded_from_first_version(tmp_path):
    root = tmp_path / "external_data"
    source = root / "TTMATCH"
    source.mkdir(parents=True)
    (source / "train.csv").write_text("rally_uid,actionId\n1,2\n", encoding="utf-8")

    manifest = build_manifest(external_root=root)

    assert manifest.iloc[0]["allowed_first_version"] is False
    assert manifest.iloc[0]["risk_tier"] == "excluded_overlap_risk"


def test_sony_is_audit_only(tmp_path):
    root = tmp_path / "external_data"
    source = root / "sonytabletennis"
    source.mkdir(parents=True)
    (source / "match_data.csv").write_text("type,timestamp\nshot,0.1\n", encoding="utf-8")

    manifest = build_manifest(external_root=root)

    assert manifest.iloc[0]["allowed_first_version"] is False
    assert "ND" in manifest.iloc[0]["license_tag"]


def test_missing_optional_aimy_spindoe_does_not_fail(tmp_path):
    root = tmp_path / "external_data"
    root.mkdir()

    manifest = build_manifest(external_root=root)

    assert manifest.empty


def test_run_pipeline_writes_outputs(tmp_path):
    root = tmp_path / "external_data"
    (root / "AIMY").mkdir(parents=True)
    (root / "AIMY" / "sample.hdf5").write_bytes(b"fake")

    report = run_pipeline(outdir=tmp_path / "out", external_root=root)

    manifest = pd.read_csv(tmp_path / "out" / "external_file_manifest.csv")
    assert report["file_count"] == 1
    assert (tmp_path / "out" / "dataset_summary.csv").exists()
    assert manifest.iloc[0]["license_tag"] == "DL-DE-BY-2-0"
