import numpy as np
import pandas as pd

from analysis_v250_v253_action_source_helpers import (
    confidence_margin,
    logit_adjust_probability,
    phase_family_one_hot,
    standardize_train_test,
    weighted_neighbor_action_prob,
)


def test_weighted_neighbor_action_prob_normalizes_and_prefers_close_labels():
    labels = np.array([[1, 2, 2]])
    distances = np.array([[0.0, 1.0, 2.0]])

    out = weighted_neighbor_action_prob(labels, distances, n_classes=4, temperature=1.0)

    assert out.shape == (1, 4)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 1] > out[0, 2]


def test_standardize_train_test_uses_train_statistics():
    train = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    test = np.array([[3.0, 4.0]])

    tr, te = standardize_train_test(train, test)

    assert np.allclose(tr[:, 0].mean(), 0.0)
    assert np.all(np.isfinite(te))
    assert np.allclose(tr[:, 1], 0.0)


def test_phase_family_one_hot_returns_numeric_columns():
    rows = pd.DataFrame(
        {
            "audit_phase": ["receive", "rally"],
            "audit_lag0_action_family": ["Attack", "Control"],
            "audit_lag0_depth": ["short", "long"],
        }
    )

    out = phase_family_one_hot(rows)

    assert len(out) == 2
    assert all(np.issubdtype(dtype, np.number) for dtype in out.dtypes)
    assert "phase=receive" in out.columns


def test_logit_adjust_probability_is_safe():
    prob = np.array([[0.8, 0.2, 0.0]])
    counts = np.array([100, 10, 1])

    out = logit_adjust_probability(prob, counts, tau=0.5)

    assert out.shape == prob.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] > 0.0


def test_confidence_margin_uses_top_two_gap():
    prob = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])

    margin = confidence_margin(prob)

    assert np.allclose(margin, [0.5, 0.05])
