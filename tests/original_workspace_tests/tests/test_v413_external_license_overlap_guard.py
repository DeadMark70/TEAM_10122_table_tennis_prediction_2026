import json
from pathlib import Path

import pandas as pd

from analysis_v413_external_license_overlap_guard import apply_guard, run_pipeline


def test_ttmatch_like_source_is_blocked():
    canonical = pd.DataFrame(
        [
            {"source_dataset": "TTMATCH", "source_file": "external_data/TTMATCH/train.csv", "license_tag": "excluded_overlap_risk", "risk_tier": "excluded_overlap_risk", "raw_payload_hash": "a"},
            {"source_dataset": "TT3D", "source_file": "external_data/TT3D/001.csv", "license_tag": "CC-BY-4.0", "risk_tier": "clean_physics", "raw_payload_hash": "b"},
        ]
    )

    clean, report, blocked, allowed = apply_guard(canonical, pd.DataFrame())

    assert set(clean["source_dataset"]) == {"TT3D"}
    assert "TTMATCH" in set(blocked["source_dataset"])
    assert report["blocked_rows"] == 1
    assert "TT3D" in set(allowed["source_dataset"])


def test_sony_nd_rows_are_blocked_from_clean_output():
    canonical = pd.DataFrame(
        [
            {"source_dataset": "sonytabletennis", "source_file": "sony.csv", "license_tag": "CC-BY-NC-ND-4.0_audit_only", "risk_tier": "audit_only_nd", "raw_payload_hash": "a"},
            {"source_dataset": "openttgames", "source_file": "open.json", "license_tag": "CC-BY-NC-SA-4.0", "risk_tier": "clean_nc_sa", "raw_payload_hash": "b"},
        ]
    )

    clean, report, blocked, _ = apply_guard(canonical, pd.DataFrame())

    assert set(clean["source_dataset"]) == {"openttgames"}
    assert "sonytabletennis" in set(blocked["source_dataset"])
    assert "report_required_share_alike_or_nc" in report["warnings"]


def test_manifest_schema_signature_blocks_aicup_like_sources():
    canonical = pd.DataFrame(
        [
            {"source_dataset": "custom", "source_file": "custom.csv", "license_tag": "CC-BY-4.0", "risk_tier": "clean_physics", "raw_payload_hash": "a"},
            {"source_dataset": "TT3D", "source_file": "001.csv", "license_tag": "CC-BY-4.0", "risk_tier": "clean_physics", "raw_payload_hash": "b"},
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "source_dataset": "custom",
                "columns_json": json.dumps(["rally_uid", "actionId", "pointId", "serverGetPoint", "spinId", "strengthId"]),
            }
        ]
    )

    clean, report, blocked, _ = apply_guard(canonical, manifest)

    assert set(clean["source_dataset"]) == {"TT3D"}
    assert "custom" in set(blocked["source_dataset"])
    assert "custom" in report["blocked_sources"]


def test_run_pipeline_writes_guard_outputs(tmp_path):
    canonical_path = tmp_path / "canonical.csv"
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"source_dataset": "TT3D", "source_file": "001.csv", "license_tag": "CC-BY-4.0", "risk_tier": "clean_physics", "raw_payload_hash": "b"}
        ]
    ).to_csv(canonical_path, index=False)
    pd.DataFrame(columns=["source_dataset", "columns_json"]).to_csv(manifest_path, index=False)

    report = run_pipeline(canonical_path=canonical_path, manifest_path=manifest_path, outdir=tmp_path / "out")

    assert report["clean_rows"] == 1
    assert (tmp_path / "out" / "canonical_clean_events.csv").exists()
    assert (tmp_path / "out" / "blocked_sources.csv").exists()
