from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis_v436_ttmatch_quarantined_contrastive import (
    build_quarantine_report,
    row_signature_set,
    run_pipeline,
)


def test_v436_outputs_are_never_clean_eligible():
    report = build_quarantine_report(row_overlap_train=1, row_overlap_test=1)

    assert report["clean_eligible"] is False
    assert report["submission_exports"] == 0


def test_v436_run_pipeline_writes_report_and_no_submission_csv(tmp_path: Path):
    ttmatch_dir = tmp_path / "external_data" / "TTMATCH"
    ttmatch_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "rally_uid": ["tt1", "tt1", "tt2"],
            "strikeNumber": [1, 2, 1],
            "match": ["m1", "m1", "m1"],
            "numberGame": [1, 1, 1],
            "scoreSelf": [0, 1, 0],
            "scoreOther": [0, 0, 0],
            "gamePlayerId": [10, 10, 20],
            "gamePlayerOtherId": [11, 11, 21],
            "strikeId": [5, 6, 5],
            "handId": [1, 1, 2],
            "strengthId": [3, 4, 3],
            "spinId": [2, 2, 1],
            "pointId": [8, 9, 8],
            "actionId": [7, 3, 7],
            "positionId": [1, 2, 1],
        }
    ).to_csv(ttmatch_dir / "train.csv", index=False)

    aicup_train = pd.DataFrame(
        {
            "rally_uid": ["a1"],
            "strikeNumber": [1],
            "strikeId": [5],
            "handId": [1],
            "strengthId": [3],
            "spinId": [2],
            "pointId": [8],
            "actionId": [7],
            "positionId": [1],
        }
    )
    aicup_test = pd.DataFrame(
        {
            "rally_uid": ["b1"],
            "strikeNumber": [2],
            "strikeId": [6],
            "handId": [1],
            "strengthId": [4],
            "spinId": [2],
            "pointId": [9],
            "actionId": [3],
            "positionId": [2],
        }
    )

    report = run_pipeline(
        ttmatch_root=ttmatch_dir,
        aicup_train=aicup_train,
        aicup_test=aicup_test,
        outdir=tmp_path / "v436_ttmatch_quarantined_contrastive",
    )

    report_path = tmp_path / "v436_ttmatch_quarantined_contrastive" / "quarantine_report.json"
    assert report_path.exists()
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk_report["clean_eligible"] is False
    assert disk_report["submission_exports"] == 0
    assert disk_report["row_overlap_train"] == 1
    assert disk_report["row_overlap_test"] == 1
    assert disk_report["context_pair_count"] > 0
    assert report == disk_report
    assert not list((tmp_path / "v436_ttmatch_quarantined_contrastive").glob("submission_*.csv"))


def test_v436_stricknumber_alias_is_treated_as_strikenumber_in_signature():
    canonical = pd.DataFrame(
        {
            "rally_uid": ["a1"],
            "strikeNumber": [1],
            "strikeId": [5],
            "handId": [1],
            "strengthId": [3],
            "spinId": [2],
            "pointId": [8],
            "actionId": [7],
            "positionId": [1],
        }
    )
    typo = canonical.rename(columns={"strikeNumber": "strickNumber", "strikeId": "strickId"})

    assert row_signature_set(canonical) == row_signature_set(typo)
