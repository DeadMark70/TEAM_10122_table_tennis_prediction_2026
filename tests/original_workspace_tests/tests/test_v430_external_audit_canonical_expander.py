import json
from pathlib import Path

import pandas as pd

from analysis_v430_external_audit_canonical_expander import (
    canonicalize_allowed_external_rows,
    classify_external_source,
    run_pipeline,
)


def test_v430_source_policy_places_ttmatchdynamics_in_clean_candidate_and_sony_audit_only():
    policy = classify_external_source("TT-MatchDynamics", license_text="Apache License 2.0")

    assert policy["tier"] == "clean_candidate"
    assert policy["train_allowed_by_default"] is True
    assert policy["requires_overlap_audit"] is True

    sony = classify_external_source("sonytabletennis", license_text="CC BY-NC-ND 4.0")

    assert sony["tier"] == "audit_only"
    assert sony["train_allowed_by_default"] is False
    assert sony["requires_overlap_audit"] is True


def test_v430_drops_exact_aicup_columns_from_external_canonical():
    rows = pd.DataFrame(
        {
            "source_dataset": ["TT-MatchDynamics"],
            "sequence_id": ["s1"],
            "event_index": [0],
            "actionId": [7],
            "pointId": [9],
            "serverGetPoint": [1],
            "spinId": [2],
            "strengthId": [3],
            "positionId": [4],
            "coarse_family": ["attack"],
            "phase": ["rally"],
        }
    )

    clean = canonicalize_allowed_external_rows(rows, source_policy={"TT-MatchDynamics": "clean_candidate"})

    forbidden = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
    assert forbidden.isdisjoint(clean.columns)
    assert clean.loc[0, "coarse_family"] == "attack"
    assert clean.loc[0, "phase"] == "rally"


def test_v430_run_pipeline_excludes_ttmatch_and_sony_from_training_rows_and_includes_clean_candidate(tmp_path):
    base_path = tmp_path / "v413_clean.csv"
    pretrain_path = tmp_path / "v414_pretrain.csv"
    external_root = tmp_path / "external_data"
    outdir = tmp_path / "out"

    pd.DataFrame(
        [
            {
                "source_dataset": "TT3D",
                "sequence_id": "base_1",
                "event_index": 0,
                "phase": "rally",
                "coarse_family": "table_tennis_trajectory",
                "landing_depth_bin": "long",
                "landing_side_bin": "left",
            }
        ]
    ).to_csv(base_path, index=False)
    pd.DataFrame(
        [
            {
                "source_dataset": "openttgames",
                "sequence_id": "pre_1",
                "event_index": 0,
                "token_family": "table_tennis_loop",
                "phase": "rally",
                "target_terminal": "unknown",
                "landing_depth_bin": "short",
                "landing_side_bin": "right",
            }
        ]
    ).to_csv(pretrain_path, index=False)

    ttmd = external_root / "TT-MatchDynamics"
    sony = external_root / "sonytabletennis"
    ttmatch = external_root / "TTMATCH"
    ttmd.mkdir(parents=True)
    sony.mkdir(parents=True)
    ttmatch.mkdir(parents=True)
    (ttmd / "LICENSE").write_text("Apache License 2.0", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "sequence_id": "clean_candidate_1",
                "event_index": 0,
                "phase": "rally",
                "coarse_family": "counter",
                "actionId": 7,
                "pointId": 8,
            }
        ]
    ).to_csv(ttmd / "events.csv", index=False)
    (sony / "LICENSE").write_text("CC BY-NC-ND 4.0", encoding="utf-8")
    pd.DataFrame([{"sequence_id": "sony_1", "coarse_family": "audit"}]).to_csv(sony / "events.csv", index=False)
    pd.DataFrame([{"sequence_id": "risk_1", "coarse_family": "risky"}]).to_csv(ttmatch / "events.csv", index=False)

    report = run_pipeline(
        v413_clean_path=base_path,
        v414_pretrain_path=pretrain_path,
        external_root=external_root,
        outdir=outdir,
    )

    expanded = pd.read_csv(outdir / "canonical_expanded_events.csv")
    audit = pd.read_csv(outdir / "external_source_audit.csv")
    overlap = json.loads((outdir / "license_overlap_report.json").read_text(encoding="utf-8"))

    assert "TT-MatchDynamics" in set(expanded["source_dataset"])
    assert "sonytabletennis" not in set(expanded["source_dataset"])
    assert "TTMATCH" not in set(expanded["source_dataset"])
    assert {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}.isdisjoint(expanded.columns)
    assert audit.loc[audit["source_dataset"].eq("sonytabletennis"), "tier"].item() == "audit_only"
    assert audit.loc[audit["source_dataset"].eq("TTMATCH"), "tier"].item() == "high_risk_quarantine"
    assert audit.loc[audit["source_dataset"].eq("sonytabletennis"), "parsed_rows"].item() == 1
    assert audit.loc[audit["source_dataset"].eq("TTMATCH"), "parsed_rows"].item() == 1
    assert audit.loc[audit["source_dataset"].eq("sonytabletennis"), "trainable_rows"].item() == 0
    assert audit.loc[audit["source_dataset"].eq("TTMATCH"), "trainable_rows"].item() == 0
    assert report["v413_base_rows"] == 1
    assert report["v414_base_rows"] == 1
    assert report["expanded_rows"] == len(expanded)
    assert overlap["source_counts"]["TT-MatchDynamics"] == 1
