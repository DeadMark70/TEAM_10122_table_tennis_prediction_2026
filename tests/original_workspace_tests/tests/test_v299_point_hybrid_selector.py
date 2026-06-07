from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v299_point_hybrid_selector import (
    SUBMISSION_COLUMNS,
    apply_candidates,
    build_variant_candidates,
    write_submission,
)


def test_agreement_selection_counts_distinct_sources_once():
    audits = {
        "v295": pd.DataFrame(
            {
                "row_id": [1, 1, 2, 3],
                "candidate_point": [8, 8, 7, 0],
                "score": [0.9, 0.8, 0.2, 0.7],
                "candidate": ["v295_a", "v295_b", "v295_c", "v295_d"],
                "base_point": [7, 7, 8, 9],
            }
        ),
        "v297": pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "candidate_point": [8, 9, 0],
                "score": [0.4, 0.6, 0.2],
                "mode": ["all_strong", "all_strong", "all_strong"],
                "candidate": ["v297_a", "v297_b", "v297_c"],
                "base_point": [7, 8, 9],
            }
        ),
        "v298": pd.DataFrame(
            {
                "row_id": [2, 4],
                "candidate_point": [6, 8],
                "score": [0.5, 0.3],
                "mode": ["all", "all"],
                "candidate": ["v298_a", "v298_b"],
                "base_point": [8, 7],
            }
        ),
    }

    selected = build_variant_candidates(audits, "agreement_2sources")

    assert selected["row_id"].tolist() == [1, 3]
    assert selected["candidate_point"].tolist() == [8, 0]
    assert selected["source_agreement_count"].tolist() == [2, 2]
    assert selected.loc[selected["row_id"] == 1, "sources"].iloc[0] == "v295+v297"


def test_no_point0_variant_filters_zero_candidates():
    audits = {
        "v295": pd.DataFrame({"row_id": [0, 1], "candidate_point": [0, 8], "score": [1.0, 0.5]}),
        "v297": pd.DataFrame({"row_id": [0, 1], "candidate_point": [0, 8], "score": [1.0, 0.4]}),
    }

    selected = build_variant_candidates(audits, "no_point0")

    assert selected["row_id"].tolist() == [1]
    assert selected["candidate_point"].tolist() == [8]


def test_long789_only_uses_v297_and_v298_long_votes():
    audits = {
        "v297": pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "candidate_point": [8, 0, 8],
                "score": [0.5, 0.9, 0.7],
                "mode": ["long789", "all_strong", "long789"],
                "base_point": [7, 9, 5],
            }
        ),
        "v298": pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "candidate_point": [8, 0, 8],
                "score": [0.4, 0.8, 0.6],
                "mode": ["long789", "long789", "long789"],
                "base_point": [7, 9, 5],
            }
        ),
    }

    selected = build_variant_candidates(audits, "long789_only")

    assert selected["row_id"].tolist() == [0]
    assert selected["candidate_point"].tolist() == [8]
    assert selected["sources"].tolist() == ["v297+v298"]


def test_apply_candidates_respects_floor_cap_and_score_order():
    base = np.zeros(100, dtype=int)
    candidates = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "candidate_point": [7, 8, 9],
            "score": [0.5, 0.9, 0.6],
            "source_agreement_count": [2, 2, 2],
        }
    )

    pred, selected = apply_candidates(base, candidates, cap=0.02)

    assert len(selected) == 2
    assert selected["row_id"].tolist() == [1, 2]
    assert pred[1] == 8
    assert pred[2] == 9
    assert pred[0] == 0


def test_write_submission_preserves_action_and_server_exactly():
    anchor = pd.DataFrame(
        {
            "rally_uid": [10, 11, 12],
            "actionId": [4, 8, 13],
            "pointId": [7, 8, 9],
            "serverGetPoint": [0.2, 0.8, 0.5],
        }
    )
    out = Path("v299_point_hybrid_selector") / "test_outputs" / "submission_v299_unit.csv"

    try:
        write_submission(out, np.array([8, 7, 9]), anchor, expected_rows=3)
        written = pd.read_csv(out)

        assert list(written.columns) == SUBMISSION_COLUMNS
        assert written["pointId"].tolist() == [8, 7, 9]
        assert written["actionId"].tolist() == anchor["actionId"].tolist()
        assert written["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
    finally:
        if out.exists():
            out.unlink()
