import json
from pathlib import Path

import pandas as pd

from analysis_v440_professor_corpus_weighting import (
    compute_source_weights,
    deduplicate_external_events,
    forbid_exact_external_columns,
    run_pipeline,
)


def test_v440_forbids_exact_external_columns():
    frame = pd.DataFrame({"source_dataset": ["x"], "actionId": [1], "pointId": [2], "coarse_family": ["attack"]})
    clean = forbid_exact_external_columns(frame)
    assert {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}.isdisjoint(clean.columns)
    assert clean.loc[0, "coarse_family"] == "attack"


def test_v440_deduplicates_by_source_sequence_event_and_payload_hash():
    frame = pd.DataFrame({
        "source_dataset": ["a", "a", "b"],
        "sequence_id": ["s1", "s1", "s1"],
        "event_index": [0, 0, 0],
        "coarse_family": ["drive", "drive", "drive"],
        "landing_depth_bin": ["long", "long", "short"],
    })
    deduped, report = deduplicate_external_events(frame)
    assert len(deduped) == 2
    assert report["duplicate_rows_removed"] == 1


def test_v440_source_weights_cap_dominant_sources_and_keep_ttmatchdynamics_small():
    counts = pd.Series({"CoachAI-Projects-main": 138390, "openttgames": 52993, "TT-MatchDynamics": 1500})
    weights = compute_source_weights(counts, max_weight=2.0, min_weight=0.35)
    assert 0.35 <= weights["CoachAI-Projects-main"] <= 2.0
    assert weights["TT-MatchDynamics"] <= 2.0
    assert weights["TT-MatchDynamics"] > weights["CoachAI-Projects-main"]


def test_v440_pipeline_writes_weighted_rows_without_sony_or_ttmatch(tmp_path: Path):
    input_path = tmp_path / "canonical_expanded_events.csv"
    outdir = tmp_path / "v440_professor_corpus_weighting"
    pd.DataFrame(
        {
            "source_dataset": ["CoachAI-Projects-main", "CoachAI-Projects-main", "TT-MatchDynamics", "sonytabletennis", "TTMATCH"],
            "sequence_id": ["s1", "s1", "s2", "s3", "s4"],
            "event_index": [0, 0, 1, 2, 3],
            "coarse_family": ["drive", "drive", "counter", "audit", "risk"],
            "landing_depth_bin": ["long", "long", "short", "short", "long"],
            "actionId": [1, 1, 2, 3, 4],
            "pointId": [5, 5, 6, 7, 8],
        }
    ).to_csv(input_path, index=False)

    report = run_pipeline(input_path=input_path, outdir=outdir)
    weighted = pd.read_csv(outdir / "v440_weighted_external_events.csv")
    weight_table = pd.read_csv(outdir / "source_weight_table.csv")
    disk_report = json.loads((outdir / "corpus_weighting_report.json").read_text(encoding="utf-8"))

    assert report == disk_report
    assert len(weighted) == 2
    assert set(weighted["source_dataset"]) == {"CoachAI-Projects-main", "TT-MatchDynamics"}
    assert "TT-MatchDynamics" in set(weight_table["source_dataset"])
    assert {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}.isdisjoint(weighted.columns)
    assert report["blocked_source_rows"]["sony"] == 1
    assert report["blocked_source_rows"]["ttmatch"] == 1
