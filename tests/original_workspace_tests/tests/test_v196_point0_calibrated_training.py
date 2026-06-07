import torch

from analysis_v196_point0_calibrated_gru import (
    point0_confidence_penalty,
    point0_rate_penalty,
    terminal_consistency_penalty,
)


def test_point0_rate_penalty_is_zero_near_target_and_positive_when_far():
    logits_near = torch.log(torch.tensor([[0.30, 0.70], [0.28, 0.72]], dtype=torch.float32))
    logits_far = torch.log(torch.tensor([[0.99, 0.01], [0.98, 0.02]], dtype=torch.float32))
    near = point0_rate_penalty(logits_near, target=0.29, weight=2.0)
    far = point0_rate_penalty(logits_far, target=0.29, weight=2.0)
    assert near.item() < 0.001
    assert far.item() > near.item()


def test_point0_confidence_penalty_only_penalizes_above_threshold():
    low = torch.log(torch.tensor([[0.40, 0.60], [0.50, 0.50]], dtype=torch.float32))
    high = torch.log(torch.tensor([[0.99, 0.01], [0.98, 0.02]], dtype=torch.float32))
    assert point0_confidence_penalty(low, threshold=0.85, weight=1.0).item() == 0.0
    assert point0_confidence_penalty(high, threshold=0.85, weight=1.0).item() > 0.0


def test_terminal_consistency_penalty_is_finite():
    point_logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    terminal_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss = terminal_consistency_penalty(point_logits, terminal_logits, weight=0.5)
    assert torch.isfinite(loss)
