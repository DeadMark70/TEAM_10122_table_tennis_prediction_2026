"""R106 remaining-aware causal GRU.

This keeps the R101/R103 causal decoder but adds an early remaining-length
gate. The model predicts a coarse remaining distribution from the sequence
state, then softly biases action/terminal logits:
  - short remaining: boosts zero/finalizing actions and point0 terminal gate
  - long remaining: boosts transition/control actions

The gate is intentionally soft and differentiable; it is not a hard mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn

import baseline_v5_gru as v5
from baseline_lgbm import validate_raw_data
from baseline_r97_style_gru import add_style_columns, build_player_style


OUTDIR = Path("r106_remaining_gate_gru")


class RemainingGateGRUModel(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_dim: int,
        emb_dim: int,
        numeric_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim, padding_idx=0) for card in cat_cardinalities])
        self.numeric = nn.Sequential(nn.Linear(num_dim, numeric_dim), nn.LayerNorm(numeric_dim), nn.GELU())
        input_dim = emb_dim * len(cat_cardinalities) + numeric_dim
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.pre_remaining_head = nn.Linear(hidden_dim, 7)
        self.action_head = nn.Linear(hidden_dim, 19)
        self.action_embed = nn.Sequential(nn.Linear(19, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        self.terminal_head = nn.Linear(hidden_dim + 32, 1)
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 9),
        )
        self.server_head = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.parity_head = nn.Linear(hidden_dim + 32 + 10, 1)

        action_bias = torch.zeros(19)
        action_bias[[0, 3, 12, 14]] = 0.55
        action_bias[[6, 8, 9, 10, 11, 13]] = -0.20
        long_bias = torch.zeros(19)
        long_bias[[6, 8, 9, 10, 11, 13]] = 0.25
        long_bias[[0, 3, 14]] = -0.20
        self.register_buffer("short_action_bias", action_bias)
        self.register_buffer("long_action_bias", long_bias)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        embs = [emb(cat[:, :, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [self.numeric(num)], dim=-1)
        x = self.input_proj(x)
        packed = rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        h = self.head(hidden[-1])
        remaining_logits = self.pre_remaining_head(h)
        rem_prob = F.softmax(remaining_logits, dim=-1)
        p_short = rem_prob[:, 0] + rem_prob[:, 1]
        p_long = rem_prob[:, 3:].sum(dim=1)

        action_logits = self.action_head(h)
        action_logits = action_logits + p_short[:, None] * self.short_action_bias[None, :]
        action_logits = action_logits + p_long[:, None] * self.long_action_bias[None, :]
        action_prob = F.softmax(action_logits, dim=-1)
        action_emb = self.action_embed(action_prob)
        hp = torch.cat([h, action_emb], dim=-1)
        terminal_logits = self.terminal_head(hp).squeeze(-1) + 0.60 * (p_short - 0.25)
        point_logits = self.point_head(hp)
        terminal_prob = torch.sigmoid(terminal_logits)
        point_nonterm = F.softmax(point_logits, dim=-1)
        point_full = torch.cat([terminal_prob[:, None], (1.0 - terminal_prob[:, None]) * point_nonterm], dim=-1)
        hs = torch.cat([h, action_emb, point_full], dim=-1)
        return {
            "action": action_logits,
            "terminal": terminal_logits,
            "point": point_logits,
            "server": self.server_head(hs).squeeze(-1),
            "parity": self.parity_head(hs).squeeze(-1),
            "remaining": remaining_logits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R106 remaining-gated GRU.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-full-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    observed = pd.concat([train, test], ignore_index=True)
    styles, style_names = build_player_style(observed, train)
    train_aug = add_style_columns(train, styles, style_names)
    test_aug = add_style_columns(test, styles, style_names)
    train_aug_path = outdir / "train_r106_style_aug.csv"
    test_aug_path = outdir / "test_r106_style_aug.csv"
    train_aug.to_csv(train_aug_path, index=False)
    test_aug.to_csv(test_aug_path, index=False)

    style_num_fields = [f"{prefix}_{name}" for name in style_names for prefix in ("h", "r", "d")]
    v5.NUM_FIELDS = v5.NUM_FIELDS + style_num_fields
    v5.GRUModel = RemainingGateGRUModel

    sys.argv = [
        "baseline_v5_gru.py",
        "--train",
        str(train_aug_path),
        "--test",
        str(test_aug_path),
        "--submission",
        str(outdir / "submission_r106_remaining_gate_gru.csv"),
        "--cv-report",
        str(outdir / "cv_report_r106.csv"),
        "--prefix-len-report",
        str(outdir / "prefix_len_report_r106.csv"),
        "--class-report-action",
        str(outdir / "class_report_r106_action.csv"),
        "--class-report-point",
        str(outdir / "class_report_r106_point.csv"),
        "--feature-report",
        str(outdir / "feature_report_r106.json"),
        "--oof-proba",
        str(outdir / "oof_proba_r106.pkl"),
        "--test-proba",
        str(outdir / "test_proba_r106.pkl"),
        "--tabular-oof",
        "",
        "--epochs",
        str(args.epochs),
        "--folds",
        str(args.folds),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--multiplier-bins",
        "two",
    ]
    if args.skip_full_train:
        sys.argv.append("--skip-full-train")
    v5.main()


if __name__ == "__main__":
    main()
