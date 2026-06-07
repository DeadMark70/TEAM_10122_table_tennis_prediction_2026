import numpy as np
import pandas as pd

from analysis_v233_public_like_validation_lab import (
    bin_prefix_len,
    density_ratio_weights,
    family_tv_distance,
    weighted_macro_f1,
    worst_group_macro_f1,
)
from analysis_v235_player_conditional_response_teacher import (
    smoothed_action_prior,
    response_prior_from_counts,
)
from analysis_v236_distributional_action_calibrator import (
    class_prior_logit_adjust,
    classwise_temperature,
)


def test_density_ratio_weights_are_clipped_and_positive():
    train = pd.DataFrame({"prefix_bin": ["1", "1", "2", "rally"], "phase": ["receive", "receive", "third", "rally"]})
    test = pd.DataFrame({"prefix_bin": ["rally", "rally", "2"], "phase": ["rally", "rally", "third"]})
    w = density_ratio_weights(train, test, ["prefix_bin", "phase"], clip=(0.5, 2.0))

    assert w.shape == (4,)
    assert np.all(w >= 0.5)
    assert np.all(w <= 2.0)
    assert w[3] > w[0]


def test_weighted_and_worst_group_macro_f1():
    y = np.array([1, 1, 2, 2])
    pred = np.array([1, 2, 2, 2])
    weights = np.array([2.0, 2.0, 1.0, 1.0])
    groups = pd.Series(["a", "a", "b", "b"])

    assert 0.0 <= weighted_macro_f1(y, pred, weights, labels=[1, 2]) <= 1.0
    assert worst_group_macro_f1(y, pred, groups, labels=[1, 2], min_rows=1) <= weighted_macro_f1(y, pred, weights, labels=[1, 2])


def test_family_tv_distance_zero_for_equal_distributions():
    a = np.array([0, 1, 1, 8, 12, 13])
    b = a.copy()
    c = np.array([1, 1, 1, 1, 1, 1])

    assert family_tv_distance(a, b) == 0.0
    assert family_tv_distance(a, c) > 0.0


def test_smoothed_response_prior_normalizes_and_uses_counts():
    counts = np.zeros((2, 19), dtype=float)
    counts[0, 4] = 9
    counts[1, 10] = 4
    global_prior = np.full(19, 1 / 19)
    prior = response_prior_from_counts(np.array([0, 1]), counts, global_prior, smoothing=1.0)

    assert np.allclose(prior.sum(axis=1), 1.0)
    assert prior[0, 4] > prior[0, 10]
    assert prior[1, 10] > prior[1, 4]


def test_smoothed_action_prior_fallback_normalizes():
    y = np.array([1, 1, 10, 12])
    prior = smoothed_action_prior(y, smoothing=2.0)

    assert prior.shape == (19,)
    assert np.allclose(prior.sum(), 1.0)
    assert prior[1] > prior[3]


def test_distributional_calibration_preserves_probability_rows():
    prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=float)
    counts = np.array([80, 10, 10], dtype=float)
    adjusted = class_prior_logit_adjust(prob, counts, tau=0.5)
    temped = classwise_temperature(prob, np.array([1.2, 0.8, 0.8]))

    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert np.allclose(temped.sum(axis=1), 1.0)
    assert np.isfinite(adjusted).all()
    assert np.isfinite(temped).all()


def test_bin_prefix_len_labels():
    assert bin_prefix_len(1) == "1"
    assert bin_prefix_len(2) == "2"
    assert bin_prefix_len(3) == "3"
    assert bin_prefix_len(5) == "4_6"
    assert bin_prefix_len(8) == "7_plus"
