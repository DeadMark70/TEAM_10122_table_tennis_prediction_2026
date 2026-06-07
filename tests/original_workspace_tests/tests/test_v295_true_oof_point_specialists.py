import numpy as np
import pandas as pd
from pathlib import Path

from analysis_v295_true_oof_point_specialists import (
    SUBMISSION_COLUMNS,
    apply_candidates,
    long789_candidates,
    point0_conservative_candidates,
    rare134_ovr_candidates,
    write_submission,
)


def _proba(rows: int) -> np.ndarray:
    out = np.full((rows, 10), 0.01, dtype=float)
    out[:, 5] = 0.91
    return out


def test_long789_candidates_only_move_inside_long_group():
    base = np.array([7, 8, 9, 6, 0])
    prob = _proba(len(base))
    prob[0, 8] = 0.80
    prob[0, 7] = 0.10
    prob[1, 9] = 0.70
    prob[1, 8] = 0.20
    prob[2, 7] = 0.65
    prob[2, 9] = 0.30
    prob[3, 7] = 0.95
    prob[4, 9] = 0.95

    candidates = long789_candidates(base, prob, mode="proba")
    pred, selected, _ = apply_candidates(base, candidates, cap=1.0)

    assert selected.tolist() == [True, True, True, False, False]
    assert set(pred[selected]).issubset({7, 8, 9})
    assert pred[3] == 6
    assert pred[4] == 0


def test_rare134_ovr_never_changes_base_8_or_9_to_rare():
    base = np.array([8, 9, 5, 1])
    prob = _proba(len(base))
    prob[:, 1] = [0.99, 0.98, 0.97, 0.01]
    prob[:, 3] = [0.02, 0.02, 0.01, 0.96]

    candidates = rare134_ovr_candidates(base, prob)
    pred, selected, _ = apply_candidates(base, candidates, cap=1.0)

    assert selected.tolist() == [False, False, True, True]
    assert pred[0] == 8
    assert pred[1] == 9
    assert set(pred[selected]).issubset({1, 3, 4})


def test_point0_conservative_only_adds_zero():
    base = np.array([0, 5, 8, 3])
    prob = _proba(len(base))
    prob[:, 0] = [0.99, 0.95, 0.91, 0.20]
    prob[:, 5] = [0.01, 0.02, 0.03, 0.70]

    candidates = point0_conservative_candidates(base, prob)
    pred, selected, _ = apply_candidates(base, candidates, cap=1.0)

    assert selected.tolist() == [False, True, True, False]
    assert pred[0] == 0
    assert pred[1] == 0
    assert pred[2] == 0
    assert pred[3] == 3


def test_apply_candidates_respects_bank_one_percent_churn_cap():
    base = np.full(200, 7, dtype=int)
    candidates = pd.DataFrame(
        {
            "row_id": np.arange(200),
            "candidate_point": np.full(200, 8, dtype=int),
            "score": np.linspace(1.0, 0.001, 200),
            "specialist": "long789",
        }
    )

    _pred, selected, selected_rows = apply_candidates(base, candidates, cap=0.01)

    assert int(selected.sum()) == 2
    assert len(selected_rows) == 2


def test_write_submission_preserves_schema_action_and_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [10, 11, 12],
            "actionId": [4, 8, 13],
            "pointId": [7, 8, 9],
            "serverGetPoint": [0.2, 0.8, 0.5],
        }
    )
    pred = np.array([8, 8, 7])
    out_dir = Path("v295_true_oof_point_specialists") / "test_outputs"

    path = write_submission(
        out_dir,
        "submission_v295_unit.csv",
        pred,
        anchor,
        expected_rows=3,
    )
    written = pd.read_csv(path)

    assert list(written.columns) == SUBMISSION_COLUMNS
    assert written["pointId"].tolist() == [8, 8, 7]
    assert written["actionId"].tolist() == anchor["actionId"].tolist()
    assert np.allclose(written["serverGetPoint"], anchor["serverGetPoint"])
