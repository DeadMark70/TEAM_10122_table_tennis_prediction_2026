import numpy as np

from train_v203_tt_shuttlenet import (
    apply_point_residual,
    depth_from_point,
    normalize_rows,
    side_from_point,
    split_type_area_static,
)


def test_depth_and_side_mapping_matches_aicup_grid():
    assert [depth_from_point(i) for i in range(10)] == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [side_from_point(i) for i in range(10)] == [0, 1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_normalize_rows_handles_zero_rows_without_nan():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    out = normalize_rows(x)
    assert np.isfinite(out).all()
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[0], [1 / 3, 1 / 3, 1 / 3])


def test_split_type_area_static_returns_stable_nonoverlapping_views():
    x = np.arange(24, dtype=float).reshape(3, 8)
    type_x, area_x = split_type_area_static(x)
    assert type_x.shape == (3, 4)
    assert area_x.shape == (3, 4)
    assert np.array_equal(type_x, x[:, 0::2])
    assert np.array_equal(area_x, x[:, 1::2])


def test_apply_point_residual_respects_cap_and_gate():
    base = np.array([8, 8, 8, 8, 8])
    prob = np.full((5, 10), 0.01)
    prob[:, 0] = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
    prob[np.arange(5), base] = np.array([0.20, 0.30, 0.40, 0.50, 0.60])
    prob = normalize_rows(prob)
    gate = np.array([True, False, True, True, True])
    out, changed = apply_point_residual(base, prob, max_churn=0.4, gate=gate)
    assert changed.sum() == 2
    assert not changed[1]
    assert np.all(out[changed] == 0)
