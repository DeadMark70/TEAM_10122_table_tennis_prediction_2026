import numpy as np

from analysis_v298_action_point_support_prior import (
    apply_candidates,
    build_action_point_prior,
    normalize_rows_safe,
    support_candidates,
)


def test_action_point_prior_rows_normalize():
    prior = build_action_point_prior(np.array([1, 1, 2]), np.array([7, 8, 0]), smoothing=1.0)
    assert prior.shape == (19, 10)
    assert np.allclose(prior.sum(axis=1), 1.0)
    assert prior[1, 7] > prior[1, 0]


def test_support_candidates_long_mode_only_long_to_long():
    base = np.array([7, 4])
    actions = np.array([1, 1])
    proba = np.zeros((2, 10))
    proba[:, 8] = 1.0
    prior = build_action_point_prior(np.array([1, 1]), np.array([8, 8]), smoothing=0.1)
    cands = support_candidates(base, actions, proba, prior, "long789")
    assert cands["row_id"].tolist() == [0]
    assert cands["candidate_point"].tolist() == [8]


def test_apply_candidates_uses_cap():
    base = np.zeros(100, dtype=int)
    import pandas as pd
    cands = pd.DataFrame({"row_id": [0, 1, 2], "candidate_point": [7, 8, 9], "score": [0.9, 0.8, 0.7]})
    pred, selected = apply_candidates(base, cands, 0.02)
    assert len(selected) == 2
    assert pred[0] == 7
    assert pred[1] == 8
    assert pred[2] == 0


def test_normalize_rows_safe_repairs_zero_row():
    out = normalize_rows_safe(np.array([[0.0, 0.0], [2.0, 1.0]]))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
