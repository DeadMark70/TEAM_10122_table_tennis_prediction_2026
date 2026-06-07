import numpy as np
import torch

from analysis_v208_action_ttshuttlenet import (
    apply_action_residual,
    action_family_targets,
    blend_action_probs,
    class_gated_action_labels,
    weak_action_mask,
)


def test_action_family_targets_maps_all_action_groups():
    y = np.array([0, 1, 7, 8, 11, 12, 14, 15, 18])
    assert action_family_targets(y).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_blend_action_probs_preserves_normalization():
    base = np.eye(3)[[0, 1]]
    model = np.eye(3)[[1, 2]]
    out = blend_action_probs(base, model, 0.25)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out.argmax(axis=1).tolist() == [0, 1]


def test_class_gated_action_labels_only_accepts_allowed_targets():
    base_labels = np.array([1, 1, 1, 1])
    blended = np.full((4, 19), 0.01)
    blended[0, 4] = 0.8
    blended[1, 10] = 0.8
    blended[2, 12] = 0.8
    blended[3, 2] = 0.8
    out = class_gated_action_labels(base_labels, blended, allowed_targets={4, 12})
    assert out.tolist() == [4, 1, 12, 1]


def test_weak_action_mask_marks_style_sensitive_classes():
    labels = torch.tensor([0, 3, 4, 7, 8, 9, 10, 11, 12, 14])
    mask = weak_action_mask(labels)
    assert mask.tolist() == [False, True, True, True, True, True, False, True, True, True]


def test_apply_action_residual_uses_confidence_gain_and_cap():
    base = np.array([1, 1, 1, 1])
    prob = np.full((4, 19), 0.01)
    prob[:, 1] = np.array([0.20, 0.30, 0.40, 0.50])
    prob[0, 4] = 0.90
    prob[1, 4] = 0.40
    prob[2, 10] = 0.80
    prob[3, 12] = 0.70
    out, changed = apply_action_residual(base, prob, max_churn=0.5, allowed_targets={4, 12})
    assert changed.tolist() == [True, False, False, True]
    assert out.tolist() == [4, 1, 1, 12]
