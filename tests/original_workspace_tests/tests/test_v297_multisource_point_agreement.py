import numpy as np
import pandas as pd
from pathlib import Path

from analysis_v297_multisource_point_agreement import (
    SUBMISSION_COLUMNS,
    apply_candidates,
    normalize_rows_safe,
    source_vote_candidates,
    write_submission,
)


def test_normalize_rows_safe_handles_bad_rows():
    x = np.array([[1.0, 1.0], [0.0, 0.0], [np.nan, np.inf]])
    y = normalize_rows_safe(x)
    assert y.shape == x.shape
    assert np.allclose(y.sum(axis=1), 1.0)
    assert np.isfinite(y).all()


def test_source_vote_candidates_long789_only():
    base = np.array([7, 8, 4])
    p1 = np.zeros((3, 10)); p2 = np.zeros((3, 10)); p3 = np.zeros((3, 10))
    p1[[0, 1, 2], [8, 9, 7]] = 1.0
    p2[[0, 1, 2], [8, 9, 7]] = 1.0
    p3[[0, 1, 2], [8, 7, 7]] = 1.0
    cands = source_vote_candidates(base, {"a": p1, "b": p2, "c": p3}, "long789")
    assert cands["row_id"].tolist() == [0]
    assert cands["candidate_point"].tolist() == [8]


def test_apply_candidates_respects_cap_order():
    base = np.zeros(100, dtype=int)
    candidates = pd.DataFrame(
        {"row_id": [1, 2, 3], "candidate_point": [7, 8, 9], "score": [0.1, 0.9, 0.5], "agree": [2, 2, 2]}
    )
    pred, selected = apply_candidates(base, candidates, 0.02)
    assert len(selected) == 2
    assert pred[2] == 8
    assert pred[3] == 9
    assert pred[1] == 0


def test_write_submission_preserves_schema():
    anchor = pd.DataFrame(
        {"rally_uid": [1, 2], "actionId": [4, 5], "pointId": [7, 8], "serverGetPoint": [0.2, 0.8]}
    )
    out = Path("v297_unit_submission.csv")
    try:
        write_submission(out, np.array([8, 9]), anchor, expected_rows=2)
        df = pd.read_csv(out)
        assert list(df.columns) == SUBMISSION_COLUMNS
        assert df["actionId"].tolist() == [4, 5]
        assert df["serverGetPoint"].tolist() == [0.2, 0.8]
        assert df["pointId"].tolist() == [8, 9]
    finally:
        if out.exists():
            out.unlink()
