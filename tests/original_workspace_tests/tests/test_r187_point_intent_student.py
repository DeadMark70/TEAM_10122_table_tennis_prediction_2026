import numpy as np

from analysis_r187_point_intent_student import (
    R187_TEACHER_COLUMNS,
    apply_residual_with_churn_cap,
    depth_distribution_from_point_prob,
    normalize_teacher_weights,
)


def test_r187_teacher_columns_do_not_include_direct_pointid():
    assert R187_TEACHER_COLUMNS
    assert all("pointId" not in col for col in R187_TEACHER_COLUMNS)
    assert any(col.startswith("T_depth_") for col in R187_TEACHER_COLUMNS)
    assert any(col.startswith("T_safety_") for col in R187_TEACHER_COLUMNS)


def test_depth_distribution_from_point_prob_uses_only_1_to_9_depth_groups():
    prob = np.zeros((2, 10), dtype=float)
    prob[0, [0, 1, 2, 7]] = [0.2, 0.3, 0.1, 0.4]
    prob[1, [4, 5, 9]] = [0.2, 0.3, 0.5]
    depth = depth_distribution_from_point_prob(prob)
    np.testing.assert_allclose(depth[0], [0.4, 0.0, 0.4])
    np.testing.assert_allclose(depth[1], [0.0, 0.5, 0.5])


def test_normalize_teacher_weights_keeps_only_auxiliary_heads():
    weights = normalize_teacher_weights({"terminal": 0.02, "depth": 0.01, "width": 0.005, "safety": 0.0025})
    assert set(weights) == {"terminal", "depth", "width", "safety"}
    assert weights["terminal"] == 0.02
    assert "point" not in weights


def test_apply_residual_with_churn_cap_preserves_probabilities_and_limits_argmax_changes():
    base = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.51, 0.49, 0.0],
            [0.34, 0.33, 0.33],
            [0.9, 0.1, 0.0],
        ],
        dtype=float,
    )
    residual = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.0, 0.9],
            [0.0, 0.1, 0.9],
        ],
        dtype=float,
    )
    out, changed = apply_residual_with_churn_cap(base, residual, alpha=0.9, max_churn=0.25)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-12)
    assert np.isfinite(out).all()
    assert changed.sum() <= 1
