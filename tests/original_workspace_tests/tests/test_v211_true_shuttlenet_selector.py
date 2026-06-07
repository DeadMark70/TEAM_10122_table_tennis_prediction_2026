import numpy as np
import torch

from analysis_v211_true_shuttlenet_selector import (
    combine_pgfn_contexts,
    split_player_subsequences,
)


def test_split_player_subsequences_uses_next_hitter_parity():
    strokes = np.arange(2 * 5 * 2).reshape(2, 5, 2)
    lengths = np.array([3, 4])
    hitter, receiver, hitter_len, receiver_len = split_player_subsequences(strokes, lengths)

    # row 0: prefix len 3 -> next hitter parity 1, so past hitter positions 1.
    assert hitter_len[0] == 1
    assert receiver_len[0] == 2
    assert hitter[0, 0].tolist() == strokes[0, 1].tolist()
    assert receiver[0, 0].tolist() == strokes[0, 0].tolist()
    assert receiver[0, 1].tolist() == strokes[0, 2].tolist()

    # row 1: prefix len 4 -> next hitter parity 0, so past hitter positions 0,2.
    assert hitter_len[1] == 2
    assert receiver_len[1] == 2
    assert hitter[1, 0].tolist() == strokes[1, 0].tolist()
    assert hitter[1, 1].tolist() == strokes[1, 2].tolist()


def test_split_player_subsequences_keeps_minimum_length_one():
    strokes = np.arange(1 * 3 * 2).reshape(1, 3, 2)
    hitter, receiver, hitter_len, receiver_len = split_player_subsequences(strokes, np.array([1]))
    assert hitter_len.tolist() == [1]
    assert receiver_len.tolist() == [1]
    assert hitter.shape[1] >= 1
    assert receiver[0, 0].tolist() == strokes[0, 0].tolist()


def test_combine_pgfn_contexts_normalizes_alpha_beta_weights():
    contexts = torch.ones(4, 5, 3)
    alpha = torch.softmax(torch.randn(4, 5), dim=1)
    beta = torch.sigmoid(torch.randn(4, 5))
    fused, weights = combine_pgfn_contexts(contexts, alpha, beta)
    assert fused.shape == (4, 3)
    assert weights.shape == (4, 5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)
