from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis_v340_no_p0_point_agreement_ensemble import (
    agreement_score,
    build_export_frame,
    enforce_no_p0_add,
    safe_output_path,
    select_candidate,
)


def test_agreement_requires_same_nonzero_target():
    base = np.array([8, 8, 8])
    sources = {
        "a": np.array([7, 9, 4]),
        "b": np.array([7, 8, 4]),
        "c": np.array([6, 9, 4]),
    }
    score, target = agreement_score(base, sources)
    assert target.tolist() == [7, 9, 4]
    assert score.tolist() == [2, 2, 3]


def test_agreement_blocks_p0_additions():
    base = np.array([8, 9])
    sources = {"a": np.array([0, 7]), "b": np.array([0, 7])}
    score, target = agreement_score(base, sources)
    assert target.tolist() == [8, 7]
    assert score.tolist() == [0, 2]


def test_select_candidate_enforces_agreement_and_budget_order():
    base = np.array([8, 8, 8, 8])
    target = np.array([7, 6, 5, 4])
    agreement = np.array([2, 3, 2, 1])

    selected = select_candidate(base, target, agreement, min_agreement=2, budget=2)

    assert selected.tolist() == [7, 6, 8, 8]


def test_no_p0_export_blocks_nonzero_to_zero():
    base = np.array([7, 8, 0, 9])
    cand = np.array([0, 9, 0, 7])
    out = enforce_no_p0_add(base, cand)
    assert out.tolist() == [7, 9, 0, 7]


def test_export_frame_preserves_action_and_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [4, 15],
            "pointId": [8, 0],
            "serverGetPoint": [0.25, 0.75],
        }
    )

    out = build_export_frame(anchor, np.array([7, 0]))

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [4, 15]
    assert out["serverGetPoint"].tolist() == [0.25, 0.75]
    assert out["pointId"].tolist() == [7, 0]


def test_safe_output_path_blocks_upload_candidates_writes():
    banned = Path("v340_no_p0_point_agreement_ensemble") / "unit_tmp" / "upload_candidates_20260519"
    with pytest.raises(ValueError):
        safe_output_path(banned, "candidate.csv")
