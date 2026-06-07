import numpy as np
import pandas as pd

from analysis_v237_deep_phase_style_action import (
    ACTION_FAMILY_IDS,
    append_prior_features,
    class_balanced_sample_weight,
    family_targets,
    hierarchical_action_probability,
)


def test_family_targets_map_actions_to_family_ids():
    y = np.array([0, 1, 7, 8, 11, 12, 14, 15, 18])
    assert family_targets(y).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_hierarchical_action_probability_preserves_rows_and_family_mass():
    exact = np.full((2, 19), 1 / 19, dtype=float)
    family = np.array(
        [
            [0.05, 0.75, 0.10, 0.08, 0.02],
            [0.60, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=float,
    )
    out = hierarchical_action_probability(exact, family)

    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[0, ACTION_FAMILY_IDS[1]].sum(), 0.75)
    assert np.allclose(out[1, ACTION_FAMILY_IDS[0]].sum(), 0.60)


def test_class_balanced_sample_weight_boosts_rare_class():
    y = np.array([1] * 20 + [8] * 2)
    base = np.ones(len(y))
    w = class_balanced_sample_weight(y, base, power=0.5, cap=5.0)

    assert w[y == 8].mean() > w[y == 1].mean()
    assert np.isfinite(w).all()


def test_append_prior_features_adds_family_and_weak_columns():
    rows = pd.DataFrame({"x": [1, 2]})
    prior = np.full((2, 19), 1 / 19, dtype=float)
    out = append_prior_features(rows, prior, prefix="resp")

    assert "resp_family_attack" in out.columns
    assert "resp_action_8" in out.columns
    assert out.shape[0] == 2
