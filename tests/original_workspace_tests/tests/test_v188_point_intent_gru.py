import numpy as np
import torch

from analysis_v188_point_intent_gru import (
    V188Batch,
    build_padded_stroke_tensor,
    capped_residual_labels,
    point_aux_targets,
    soft_kl_loss,
)


def test_build_padded_stroke_tensor_keeps_recent_prefix_and_padding_zero():
    seqs = [
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        np.array([[7, 8]], dtype=np.int64),
    ]
    x, lengths = build_padded_stroke_tensor(seqs, max_len=2, pad_value=0)
    assert x.shape == (2, 2, 2)
    assert lengths.tolist() == [2, 1]
    np.testing.assert_array_equal(x[0], np.array([[3, 4], [5, 6]]))
    np.testing.assert_array_equal(x[1], np.array([[7, 8], [0, 0]]))


def test_point_aux_targets_split_terminal_depth_side_safety():
    targets = point_aux_targets(np.array([0, 1, 5, 9], dtype=np.int64))
    assert targets["terminal"].tolist() == [1, 0, 0, 0]
    assert targets["depth"].tolist() == [0, 0, 1, 2]
    assert targets["side"].tolist() == [0, 0, 1, 2]
    assert targets["safety"].tolist() == [0, 1, 0, 2]
    assert targets["nonterminal"].tolist() == [False, True, True, True]


def test_soft_kl_loss_is_finite_and_ignores_zero_weight_teacher():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    loss = soft_kl_loss(logits, teacher, weight=0.01)
    assert torch.isfinite(loss)
    assert soft_kl_loss(logits, teacher, weight=0.0).item() == 0.0


def test_capped_residual_labels_limits_changes_by_gain():
    base = np.array([0, 0, 0, 0])
    prob = np.array(
        [
            [0.4, 0.6],
            [0.45, 0.55],
            [0.49, 0.51],
            [0.9, 0.1],
        ],
        dtype=float,
    )
    labels, changed = capped_residual_labels(base, prob, max_churn=0.25)
    assert changed.sum() == 1
    assert labels.tolist() == [1, 0, 0, 0]


def test_v188_batch_keeps_tensor_shapes():
    batch = V188Batch(
        strokes=torch.zeros((3, 4, 7), dtype=torch.long),
        lengths=torch.tensor([4, 3, 1]),
        static=torch.zeros((3, 5)),
        point=torch.tensor([0, 5, 9]),
        teacher=torch.zeros((3, 10)),
    )
    assert batch.strokes.shape == (3, 4, 7)
    assert batch.static.shape == (3, 5)
