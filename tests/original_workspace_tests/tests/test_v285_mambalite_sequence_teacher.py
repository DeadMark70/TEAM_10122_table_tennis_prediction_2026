import numpy as np
import torch

from train_v285_mambalite_sequence_teacher import (
    MambaLiteBlock,
    geometric_logit_blend,
    normalize_rows_safe,
)


def test_mambalite_block_preserves_shape_and_is_finite():
    torch.manual_seed(7)
    block = MambaLiteBlock(dim=16, kernel_size=3)
    x = torch.randn(4, 6, 16)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_normalize_rows_safe_handles_zero_and_nan_rows():
    matrix = np.array([[0.0, 0.0, 0.0], [np.nan, 2.0, 0.0]])
    out = normalize_rows_safe(matrix)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
    assert np.allclose(out[0], [1 / 3, 1 / 3, 1 / 3])


def test_geometric_logit_blend_keeps_rows_normalized():
    anchor = np.array([[0.8, 0.2], [0.4, 0.6]])
    teacher = np.array([[0.1, 0.9], [0.7, 0.3]])
    out = geometric_logit_blend(anchor, teacher, weight=0.05)
    assert out.shape == anchor.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
    assert out[0, 0] > out[0, 1]
