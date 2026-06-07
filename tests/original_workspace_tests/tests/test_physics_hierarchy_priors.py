import numpy as np

from analysis_r179_action_physics_hierarchy import (
    action_family,
    apply_logit_bias,
    normalize_rows_safe,
    phase_name,
    point_depth,
    point_side,
)
from analysis_r180_point_physics_calibration import (
    apply_long_side_redistribution,
    apply_point_hierarchy_calibration,
)


def test_action_family_mapping():
    assert action_family(0) == "Zero"
    assert {action_family(i) for i in range(1, 8)} == {"Attack"}
    assert {action_family(i) for i in range(8, 12)} == {"Control"}
    assert {action_family(i) for i in range(12, 15)} == {"Defensive"}
    assert {action_family(i) for i in range(15, 19)} == {"Serve"}


def test_point_depth_and_side_mapping():
    assert point_depth(0) == 0
    assert [point_depth(i) for i in [1, 2, 3]] == [1, 1, 1]
    assert [point_depth(i) for i in [4, 5, 6]] == [2, 2, 2]
    assert [point_depth(i) for i in [7, 8, 9]] == [3, 3, 3]
    assert [point_side(i) for i in [1, 4, 7]] == [1, 1, 1]
    assert [point_side(i) for i in [2, 5, 8]] == [2, 2, 2]
    assert [point_side(i) for i in [3, 6, 9]] == [3, 3, 3]


def test_phase_name_uses_prefix_when_phase_id_unknown():
    assert phase_name(1, 1) == "receive"
    assert phase_name(2, 2) == "third_ball"
    assert phase_name(3, 3) == "fourth_ball"
    assert phase_name(4, 8) == "rally"
    assert phase_name(0, 2) == "third_ball"


def test_apply_logit_bias_normalizes_and_caps_serve_classes():
    base = normalize_rows_safe(np.ones((2, 19)))
    prior = normalize_rows_safe(np.ones((2, 19)))
    prior[:, 15:19] = 0.9
    prior = normalize_rows_safe(prior)

    out = apply_logit_bias(base, prior, weight=0.5, class_caps={15: 1e-4, 16: 1e-4, 17: 1e-4, 18: 1e-4})

    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out[:, 15:19] <= 1e-4 + 1e-12)


def test_long_side_redistribution_preserves_long_mass():
    base = normalize_rows_safe(
        np.array(
            [
                [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.25, 0.15],
                [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.35, 0.15],
            ]
        )
    )
    q_long = normalize_rows_safe(np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]))

    out = apply_long_side_redistribution(base, q_long, alpha=0.2, long_thr=0.3)

    assert np.allclose(out[:, 7:10].sum(axis=1), base[:, 7:10].sum(axis=1))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_point_hierarchy_calibration_keeps_direct_decoder_shape():
    base = normalize_rows_safe(np.ones((3, 10)))
    terminal_prior = np.array([0.9, 0.1, 0.5])
    depth_prior = normalize_rows_safe(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
    side_prior = normalize_rows_safe(np.ones((3, 3)))

    out = apply_point_hierarchy_calibration(
        base,
        terminal_prior=terminal_prior,
        depth_prior=depth_prior,
        side_prior=side_prior,
        terminal_weight=0.01,
        depth_weight=0.005,
        side_weight=0.005,
    )

    assert out.shape == base.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.max(np.abs(out - base)) < 0.01
