import torch

from analysis_v189_v190_point_intent_seq_models import (
    CausalTransformerBackbone,
    LSTMBackbone,
    causal_padding_mask,
    subsequent_mask,
)


def test_lstm_backbone_outputs_all_point_intent_heads():
    model = LSTMBackbone(vocab_sizes=[20, 11, 7, 4, 4, 5, 8], static_dim=12, hidden=32, layers=1, dropout=0.1)
    out = model(
        torch.ones((5, 6, 7), dtype=torch.long),
        torch.tensor([6, 5, 4, 2, 1]),
        torch.zeros((5, 12), dtype=torch.float32),
    )
    assert out["point"].shape == (5, 10)
    assert out["terminal"].shape == (5, 2)
    assert out["depth"].shape == (5, 3)
    assert out["side"].shape == (5, 3)
    assert out["width"].shape == (5, 2)
    assert out["safety"].shape == (5, 3)


def test_transformer_backbone_outputs_all_point_intent_heads():
    model = CausalTransformerBackbone(
        vocab_sizes=[20, 11, 7, 4, 4, 5, 8],
        static_dim=12,
        d_model=32,
        heads=4,
        layers=1,
        dropout=0.1,
        max_len=8,
    )
    out = model(
        torch.ones((4, 8, 7), dtype=torch.long),
        torch.tensor([8, 7, 4, 1]),
        torch.zeros((4, 12), dtype=torch.float32),
    )
    assert out["point"].shape == (4, 10)
    assert out["terminal"].shape == (4, 2)
    assert out["depth"].shape == (4, 3)


def test_subsequent_mask_blocks_future_positions():
    mask = subsequent_mask(4)
    assert mask.shape == (4, 4)
    assert mask[0, 1]
    assert mask[0, 3]
    assert not mask[3, 0]
    assert not mask[2, 2]


def test_causal_padding_mask_marks_tokens_after_length():
    lengths = torch.tensor([4, 2, 1])
    mask = causal_padding_mask(lengths, max_len=4)
    assert mask.tolist() == [
        [False, False, False, False],
        [False, False, True, True],
        [False, True, True, True],
    ]
