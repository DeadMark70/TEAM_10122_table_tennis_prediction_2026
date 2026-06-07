import numpy as np

from analysis_v263_questionnaire_baseline_helpers import (
    cap_by_score,
    class_f1_table,
    log_loss_safe_prob,
    normalize_rows,
    point_depth,
    point_side,
)


def test_normalize_rows_handles_bad_rows():
    x = np.array([[1.0, 1.0], [0.0, 0.0], [np.nan, 2.0]])
    out = normalize_rows(x)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()


def test_cap_by_score_selects_top_rows():
    scores = np.array([0.2, 0.9, 0.4, 0.7])
    mask = cap_by_score(scores, cap=0.5)
    assert mask.tolist() == [False, True, False, True]


def test_point_geometry_maps_ids():
    assert [point_depth(i) for i in range(10)] == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [point_side(i) for i in range(10)] == [0, 1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_class_f1_table_contains_delta():
    y = np.array([0, 1, 1, 2])
    base = np.array([0, 1, 2, 2])
    cand = np.array([0, 1, 1, 2])
    table = class_f1_table(y, base, cand, classes=[0, 1, 2])
    assert list(table["class_id"]) == [0, 1, 2]
    assert table["delta_f1"].sum() > 0


def test_log_loss_safe_prob_normalizes():
    base = np.array([[0.9, 0.1], [0.5, 0.5]])
    model = np.array([[0.2, 0.8], [0.8, 0.2]])
    out = log_loss_safe_prob(base, model, weight=0.1)
    assert np.allclose(out.sum(axis=1), 1.0)
