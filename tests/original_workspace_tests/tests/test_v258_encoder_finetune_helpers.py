import numpy as np
import torch

from analysis_v258_encoder_finetune_helpers import (
    ACTION_FAMILY_ID,
    action_family_id,
    blend_probabilities,
    kd_cross_entropy,
    normalize_rows_safe,
    pad_sequence,
)


def test_action_family_id_mapping():
    assert action_family_id(0) == ACTION_FAMILY_ID["Zero"]
    for action in range(1, 8):
        assert action_family_id(action) == ACTION_FAMILY_ID["Attack"]
    for action in range(8, 12):
        assert action_family_id(action) == ACTION_FAMILY_ID["Control"]
    for action in range(12, 15):
        assert action_family_id(action) == ACTION_FAMILY_ID["Defensive"]
    for action in range(15, 19):
        assert action_family_id(action) == ACTION_FAMILY_ID["Serve"]


def test_pad_sequence_truncates_and_pads():
    assert pad_sequence([1, 2, 3], max_len=5, pad=0).tolist() == [1, 2, 3, 0, 0]
    assert pad_sequence([1, 2, 3, 4], max_len=2, pad=0).tolist() == [1, 2]


def test_normalize_rows_safe_handles_zero_nan():
    matrix = np.array([[1.0, 1.0], [0.0, 0.0], [np.nan, 5.0]])
    out = normalize_rows_safe(matrix)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert not np.isnan(out).any()
    assert np.allclose(out[1], [0.5, 0.5])


def test_blend_probabilities_keeps_normalization():
    anchor = np.array([0, 1])
    teacher = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = blend_probabilities(anchor, teacher, weight=0.25)
    assert out.shape == teacher.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 0] > teacher[0, 0]


def test_kd_cross_entropy_is_finite():
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.3]], dtype=torch.float32)
    teacher = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    loss = kd_cross_entropy(logits, teacher, temperature=2.0)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
