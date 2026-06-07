import numpy as np
import pandas as pd

from analysis_v243_v247_action_augmentation_helpers import (
    balanced_softmax_adjustment,
    build_context_key_frame,
    clip_density_weights,
    js_distance,
    label_smoothed_targets,
    mix_probabilities,
)


def test_build_context_key_frame_has_stable_bins():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 2, 3, 5, 8],
            "audit_phase": ["receive", "third_ball", "rally", "rally", "receive"],
            "audit_lag0_action_family": ["Attack", "Control", "Attack", "Zero", "Defensive"],
            "audit_lag0_depth": ["short", "half", "long", "zero", "long"],
        }
    )

    out = build_context_key_frame(rows)

    assert list(out.columns) == ["prefix_bin", "phase", "lag0_family", "lag0_depth"]
    assert out["prefix_bin"].tolist() == ["1", "2", "3", "4_6", "7_plus"]


def test_clip_density_weights_is_positive_and_normalized():
    weights = clip_density_weights(np.array([0.01, 1.0, 99.0]), low=0.25, high=4.0)

    assert np.all(weights >= 0.25)
    assert np.all(weights <= 4.0)
    assert np.isclose(weights.mean(), 1.0)


def test_balanced_softmax_adjustment_normalizes_rows():
    prob = np.array([[0.8, 0.2, 0.0], [0.1, 0.1, 0.8]])
    counts = np.array([100, 10, 1])

    out = balanced_softmax_adjustment(prob, counts, strength=0.5)

    assert out.shape == prob.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] > 0.0


def test_label_smoothed_targets_preserve_rows():
    y = np.array([0, 2])

    out = label_smoothed_targets(y, n_classes=4, smoothing=0.1)

    assert out.shape == (2, 4)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 0] > out[0, 1]


def test_mix_probabilities_and_js_distance_are_safe():
    a = np.array([[1.0, 0.0], [0.2, 0.8]])
    b = np.array([[0.0, 1.0], [0.8, 0.2]])

    mixed = mix_probabilities(a, b, weight=0.25)
    d = js_distance(a, b)

    assert np.allclose(mixed.sum(axis=1), 1.0)
    assert d.shape == (2,)
    assert np.all(d >= 0)
