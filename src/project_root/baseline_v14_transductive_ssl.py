"""V14 transductive point-aware SSL + short-prefix kNN point prior.

V14 extends the V12/V13 experiment by adding public test prefixes to the SSL
pretraining corpus. Test rows are used only for observed-prefix masked-field,
view-consistency, and within-prefix next action/point objectives. Outcome,
terminal, remaining-length, and server labels remain train-only.

This script is intentionally conservative:
- no retrieval for prefix_len >= 3
- no point multiplier retuning in the primary selection
- no player IDs or player-history features
- no validation/test prefixes in the retrieval database
- no test hidden target or test prefix-boundary/EOS objective
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, sample_validation_prefixes, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers
from baseline_v5_gru import (
    CAT_FIELDS,
    NUM_FIELDS,
    SequenceArrays,
    build_sequence_arrays,
    build_test_meta,
    build_train_meta,
    fit_numeric_stats,
)
from generate_r1_submission import compose_v3, compose_v3_full


POINT_NONTERMINAL_CLASSES = list(range(1, 10))
REMAINING_CLASSES = list(range(1, 8))
MASK_FIELD_RATES = {
    "pointId": 0.30,
    "actionId": 0.20,
    "spinId": 0.20,
    "strengthId": 0.15,
    "handId": 0.15,
    "positionId": 0.15,
}


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V14 transductive SSL + short-prefix kNN retrieval.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--v10b-r1-selected", default="v10b_r1_selected.json")
    parser.add_argument("--cv-report", default="cv_report_v14.csv")
    parser.add_argument("--search-report", default="v14_knn_search_report.csv")
    parser.add_argument("--prefix-report", default="prefix_len_report_v14.csv")
    parser.add_argument("--feature-report", default="feature_report_v14.json")
    parser.add_argument("--oof-proba", default="oof_proba_v14.pkl")
    parser.add_argument("--submission", default="submission_v14.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--retrieval-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr-pretrain", type=float, default=7e-4)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--view-loss-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-full-train", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def cat_cardinalities_with_mask(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    cards, mask_ids, class_sizes = [], [], []
    for field in CAT_FIELDS:
        max_value = int(max(train[field].max(), test[field].max()))
        class_size = max_value + 1
        card = max_value + 3
        cards.append(card)
        mask_ids.append(card - 1)
        class_sizes.append(class_size)
    return cards, mask_ids, class_sizes


def class_weights(values: pd.Series, classes: list[int], beta: float = 0.25) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    weights = np.array([float(counts.get(cls, 1)) ** (-beta) for cls in classes], dtype=np.float32)
    return torch.from_numpy(weights / weights.mean())


def mark_train_ssl_meta(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["has_next_label"] = 1.0
    meta["has_outcome_label"] = 1.0
    meta["ssl_source"] = "train"
    return meta


def build_test_internal_meta(test: pd.DataFrame) -> pd.DataFrame:
    """Build safe public-test SSL rows from observed within-prefix transitions.

    For a test prefix of length L, rows are created for t=1..L-1 to predict the
    observed t+1 stroke. No row asks the model to predict beyond the observed
    prefix, and outcome/terminal/parity/remaining labels are masked out.
    """
    rows: list[dict[str, int | float | str]] = []
    for _, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        for t_idx in range(len(group) - 1):
            nxt = group.iloc[t_idx + 1]
            rows.append(
                {
                    "rally_uid": int(group.iloc[0]["rally_uid"]),
                    "match": int(group.iloc[0]["match"]),
                    "prefix_index": int(t_idx),
                    "prefix_len": int(group.iloc[t_idx]["strikeNumber"]),
                    "next_actionId": int(nxt["actionId"]),
                    "next_pointId": int(nxt["pointId"]),
                    "next_is_terminal": 0,
                    "serverGetPoint": 0,
                    "final_parity_even": 0,
                    "remaining_len": 1,
                    "remaining_len_bucket": 1,
                    "num_prefixes_in_rally": int(max(1, len(group) - 1)),
                    "server_weight": 0.0,
                    "has_next_label": 1.0,
                    "has_outcome_label": 0.0,
                    "ssl_source": "public_test_prefix",
                }
            )
    return pd.DataFrame(rows)


def concat_sequence_arrays(parts: list[SequenceArrays]) -> SequenceArrays:
    parts = [p for p in parts if len(p.meta) > 0]
    if len(parts) == 1:
        return parts[0]
    return SequenceArrays(
        cat=np.concatenate([p.cat for p in parts], axis=0),
        num=np.concatenate([p.num for p in parts], axis=0),
        lengths=np.concatenate([p.lengths for p in parts], axis=0),
        meta=pd.concat([p.meta for p in parts], ignore_index=True),
    )


def crop_preserve_last(cat: np.ndarray, num: np.ndarray, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    if length < 4:
        return cat, num, length
    start = int(rng.integers(0, max(1, length - 2)))
    if start <= 0:
        return cat, num, length
    new_len = length - start
    cat2 = np.zeros_like(cat)
    num2 = np.zeros_like(num)
    cat2[:new_len] = cat[start:length]
    num2[:new_len] = num[start:length]
    return cat2, num2, new_len


def apply_field_masks(cat: np.ndarray, length: int, mask_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    out = cat.copy()
    targets = np.full((cat.shape[0], cat.shape[1]), -100, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for field, rate in MASK_FIELD_RATES.items():
        idx = CAT_FIELDS.index(field)
        if length <= 0:
            continue
        mask = rng.random(length) < rate
        if not mask.any():
            mask[int(rng.integers(0, length))] = True
        original = out[:length, idx].copy()
        targets[:length, idx][mask] = original[mask] - 1
        out[:length, idx][mask] = mask_ids[idx]
    return out, targets


class V12PretrainDataset(Dataset):
    def __init__(self, arrays: SequenceArrays, mask_ids: list[int], seed: int) -> None:
        self.cat = arrays.cat
        self.num = arrays.num
        self.lengths = arrays.lengths
        self.meta = arrays.meta.reset_index(drop=True)
        self.mask_ids = np.asarray(mask_ids, dtype=np.int64)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base_cat = self.cat[idx].copy()
        base_num = self.num[idx].copy()
        base_len = int(self.lengths[idx])
        row = self.meta.iloc[idx]
        rng = np.random.default_rng(self.seed + idx * 13)

        cat1, num1, len1 = crop_preserve_last(base_cat.copy(), base_num.copy(), base_len, rng)
        cat2, num2, len2 = crop_preserve_last(base_cat.copy(), base_num.copy(), base_len, rng)
        cat1, targets = apply_field_masks(cat1, len1, self.mask_ids, self.seed + idx * 17)
        cat2, _ = apply_field_masks(cat2, len2, self.mask_ids, self.seed + idx * 19)
        next_point = int(row.get("next_pointId", 0))
        has_next_label = float(row.get("has_next_label", 1.0))
        has_outcome_label = float(row.get("has_outcome_label", 1.0))
        return {
            "cat": torch.from_numpy(cat1).long(),
            "num": torch.from_numpy(num1).float(),
            "lengths": torch.tensor(len1, dtype=torch.long),
            "cat_view": torch.from_numpy(cat2).long(),
            "num_view": torch.from_numpy(num2).float(),
            "lengths_view": torch.tensor(len2, dtype=torch.long),
            "mask_targets": torch.from_numpy(targets).long(),
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point10": torch.tensor(next_point, dtype=torch.long),
            "point_nonterminal": torch.tensor(next_point - 1 if next_point > 0 else 0, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
            "has_next_label": torch.tensor(has_next_label, dtype=torch.float32),
            "has_outcome_label": torch.tensor(has_outcome_label, dtype=torch.float32),
        }


class V12EvalDataset(Dataset):
    def __init__(self, arrays: SequenceArrays) -> None:
        self.cat = torch.from_numpy(arrays.cat).long()
        self.num = torch.from_numpy(arrays.num).float()
        self.lengths = torch.from_numpy(arrays.lengths).long()
        self.meta = arrays.meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        next_point = int(row.get("next_pointId", 0))
        has_next_label = float(row.get("has_next_label", 1.0))
        has_outcome_label = float(row.get("has_outcome_label", 1.0))
        return {
            "cat": self.cat[idx],
            "num": self.num[idx],
            "lengths": self.lengths[idx],
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point10": torch.tensor(next_point, dtype=torch.long),
            "point_nonterminal": torch.tensor(next_point - 1 if next_point > 0 else 0, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
            "has_next_label": torch.tensor(has_next_label, dtype=torch.float32),
            "has_outcome_label": torch.tensor(has_outcome_label, dtype=torch.float32),
        }


class V12Transformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        cat_class_sizes: list[int],
        num_dim: int,
        max_len: int,
        d_model: int,
        emb_dim: int,
        numeric_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        retrieval_dim: int,
    ) -> None:
        super().__init__()
        self.mask_field_indices = [CAT_FIELDS.index(f) for f in MASK_FIELD_RATES]
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim, padding_idx=0) for card in cat_cardinalities])
        self.numeric = nn.Sequential(nn.Linear(num_dim, numeric_dim), nn.LayerNorm(numeric_dim), nn.GELU())
        input_dim = emb_dim * len(cat_cardinalities) + numeric_dim
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_gate = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Dropout(dropout))
        self.retrieval_proj = nn.Linear(d_model * 2, retrieval_dim)
        self.action_head = nn.Linear(d_model * 2, 19)
        self.terminal_head = nn.Linear(d_model * 2, 1)
        self.point_head = nn.Linear(d_model * 2, 9)
        self.point10_head = nn.Linear(d_model * 2, 10)
        self.server_head = nn.Linear(d_model * 2, 1)
        self.parity_head = nn.Linear(d_model * 2, 1)
        self.remaining_head = nn.Linear(d_model * 2, 7)
        self.mask_heads = nn.ModuleDict(
            {CAT_FIELDS[i]: nn.Linear(d_model, cat_class_sizes[i]) for i in self.mask_field_indices}
        )

    def encode(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = cat.shape
        embs = [emb(cat[:, :, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [self.numeric(num)], dim=-1)
        x = self.input_proj(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        pad_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        last_idx = (lengths - 1).clamp(min=0)
        h_last = x[torch.arange(bsz, device=x.device), last_idx]
        gate = self.pool_gate(x).squeeze(-1).masked_fill(pad_mask, -1e9)
        attn = torch.softmax(gate, dim=1)
        h_pool = (x * attn.unsqueeze(-1)).sum(dim=1)
        h = self.head(torch.cat([h_last, h_pool], dim=-1))
        z = F.normalize(self.retrieval_proj(h), dim=-1)
        return x, h, z

    def forward(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        token_h, h, z = self.encode(cat, num, lengths)
        mask_logits = {field: self.mask_heads[field](token_h) for field in self.mask_heads}
        return {
            "token_h": token_h,
            "embedding": z,
            "mask": mask_logits,
            "action": self.action_head(h),
            "terminal": self.terminal_head(h).squeeze(-1),
            "point": self.point_head(h),
            "point10": self.point10_head(h),
            "server": self.server_head(h).squeeze(-1),
            "parity": self.parity_head(h).squeeze(-1),
            "remaining": self.remaining_head(h),
        }


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(loss.device).float()
    denom = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denom


def supervised_loss(outputs, batch, action_w, point_w, device) -> torch.Tensor:
    next_mask = batch.get("has_next_label", torch.ones_like(batch["terminal"])).to(device)
    outcome_mask = batch.get("has_outcome_label", torch.ones_like(batch["terminal"])).to(device)
    action_loss_raw = F.cross_entropy(outputs["action"], batch["action"], weight=action_w.to(device), reduction="none")
    action_loss = weighted_mean(action_loss_raw, next_mask)
    terminal_loss_raw = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"], reduction="none")
    terminal_loss = weighted_mean(terminal_loss_raw, outcome_mask)
    point10_loss_raw = F.cross_entropy(outputs["point10"], batch["point10"], reduction="none")
    point10_loss = weighted_mean(point10_loss_raw, next_mask)
    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        point_loss_raw = F.cross_entropy(
            outputs["point"][point_mask],
            batch["point_nonterminal"][point_mask],
            weight=point_w.to(device),
            reduction="none",
        )
        point_loss = weighted_mean(point_loss_raw, next_mask[point_mask])
    else:
        point_loss = outputs["point"].sum() * 0.0
    server_loss_raw = F.binary_cross_entropy_with_logits(outputs["server"], batch["server"], reduction="none")
    server_loss = weighted_mean(server_loss_raw, batch["server_weight"].to(device) * outcome_mask)
    parity_loss_raw = F.binary_cross_entropy_with_logits(outputs["parity"], batch["parity"], reduction="none")
    parity_loss = weighted_mean(parity_loss_raw, outcome_mask)
    remaining_loss_raw = F.cross_entropy(outputs["remaining"], batch["remaining"], reduction="none")
    remaining_loss = weighted_mean(remaining_loss_raw, outcome_mask)
    return (
        0.25 * action_loss
        + 0.12 * terminal_loss
        + 0.22 * point_loss
        + 0.18 * point10_loss
        + 0.13 * server_loss
        + 0.05 * parity_loss
        + 0.05 * remaining_loss
    )


def mask_loss(outputs, batch, mask_field_indices: list[int]) -> torch.Tensor:
    losses = []
    targets = batch["mask_targets"]
    for field_idx in mask_field_indices:
        field = CAT_FIELDS[field_idx]
        target = targets[:, :, field_idx]
        if target.ge(0).any():
            logits = outputs["mask"][field]
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100))
    if not losses:
        return outputs["action"].sum() * 0.0
    return torch.stack(losses).mean()


def train_epoch(model, loader, opt, action_w, point_w, args, device, pretrain: bool) -> dict[str, float]:
    model.train()
    losses, mask_losses, sup_losses, view_losses = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["cat"], batch["num"], batch["lengths"])
        sup = supervised_loss(out, batch, action_w, point_w, device)
        if pretrain:
            m = mask_loss(out, batch, model.mask_field_indices)
            out_view = model(batch["cat_view"], batch["num_view"], batch["lengths_view"])
            view = (1.0 - (out["embedding"] * out_view["embedding"]).sum(dim=-1)).mean()
            loss = 0.45 * m + 0.50 * sup + args.view_loss_weight * view
            mask_losses.append(float(m.detach().cpu()))
            view_losses.append(float(view.detach().cpu()))
        else:
            m = out["action"].sum() * 0.0
            view = out["action"].sum() * 0.0
            loss = sup
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        sup_losses.append(float(sup.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "mask_loss": float(np.mean(mask_losses)) if mask_losses else 0.0,
        "supervised_loss": float(np.mean(sup_losses)),
        "view_loss": float(np.mean(view_losses)) if view_losses else 0.0,
    }


def predict_model(model, arrays: SequenceArrays, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(V12EvalDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    action_parts, point_parts, server_parts, emb_parts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["cat"].to(device), batch["num"].to(device), batch["lengths"].to(device))
            action = F.softmax(out["action"], dim=-1).cpu().numpy()
            terminal = torch.sigmoid(out["terminal"]).cpu().numpy()
            point_nonterm = F.softmax(out["point"], dim=-1).cpu().numpy()
            point = np.zeros((len(action), 10), dtype=np.float32)
            point[:, 0] = terminal
            point[:, 1:] = (1.0 - terminal[:, None]) * point_nonterm
            point = point / point.sum(axis=1, keepdims=True)
            server = torch.sigmoid(out["server"]).cpu().numpy()
            emb = out["embedding"].cpu().numpy()
            action_parts.append(action)
            point_parts.append(point)
            server_parts.append(server)
            emb_parts.append(emb)
    return np.vstack(action_parts), np.vstack(point_parts), np.concatenate(server_parts), np.vstack(emb_parts).astype(np.float32)


def evaluate(meta, action_prob, point_prob, server_prob, action_mult=None, point_mult=None, bins_mode="global") -> dict[str, float]:
    if action_mult is None:
        action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    else:
        action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, bins_mode)
    if point_mult is None:
        point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]
    else:
        point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, bins_mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def load_v3_subset(path: str, valid_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, V3Tuning]:
    with open(path, "rb") as f:
        oof = pickle.load(f)
    tuning = oof["tuning"]
    meta = oof["valid_meta"].reset_index(drop=True)
    key_cols = ["rally_uid", "prefix_len"]
    merged = valid_meta[key_cols].reset_index().merge(meta[key_cols].reset_index(), on=key_cols, how="left", suffixes=("", "_v3"))
    if merged["index_v3"].isna().any():
        raise ValueError("Could not align V12 valid rows to V3 OOF.")
    idx = merged["index_v3"].astype(int).to_numpy()
    action = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)[idx]
    point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)[idx]
    sw = tuning.server_weights
    server = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )[idx]
    return action, point, server, tuning


def train_model(train_arrays, valid_arrays, cat_cards, class_sizes, args, seed, action_w, point_w, pretrain_arrays=None):
    set_seed(seed)
    device = torch.device(args.device)
    model = V12Transformer(
        cat_cards,
        class_sizes,
        len(NUM_FIELDS),
        args.max_len,
        args.d_model,
        args.emb_dim,
        args.numeric_dim,
        args.num_layers,
        args.num_heads,
        args.dropout,
        args.retrieval_dim,
    ).to(device)
    ssl_arrays = train_arrays if pretrain_arrays is None else pretrain_arrays
    pre_ds = V12PretrainDataset(ssl_arrays, [card - 1 for card in cat_cards], seed + 9000)
    pre_loader = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
    for epoch in range(1, args.pretrain_epochs + 1):
        losses = train_epoch(model, pre_loader, opt, action_w, point_w, args, device, pretrain=True)
        print(
            f"  pretrain {epoch:02d}: loss={losses['loss']:.5f} mask={losses['mask_loss']:.5f} "
            f"sup={losses['supervised_loss']:.5f} view={losses['view_loss']:.5f}"
        )
    ft_loader = DataLoader(V12EvalDataset(train_arrays), batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed + 77))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
    best_state, best_score = None, -1.0
    bad = 0
    for epoch in range(1, args.finetune_epochs + 1):
        losses = train_epoch(model, ft_loader, opt, action_w, point_w, args, device, pretrain=False)
        a, p, s, _ = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(valid_arrays.meta, a, p, s)
        print(f"  finetune {epoch:02d}: loss={losses['loss']:.5f} point={metrics['point_macro_f1']:.6f} overall={metrics['overall']:.6f}")
        if metrics["point_macro_f1"] > best_score + 1e-6:
            best_score = metrics["point_macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def prior_by_prefix(meta: pd.DataFrame, prefix_len: int, alpha: float = 1.0) -> np.ndarray:
    sub = meta[meta["prefix_len"].eq(prefix_len)]
    counts = sub["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=np.float64)
    prior = (counts + alpha) / (counts.sum() + alpha * len(POINT_CLASSES))
    return prior.astype(np.float32)


def knn_prior_for_queries(
    db_emb: np.ndarray,
    db_y: np.ndarray,
    query_emb: np.ndarray,
    prefix_prior: np.ndarray,
    k: int,
    tau: float,
    alpha: float,
) -> np.ndarray:
    if len(query_emb) == 0:
        return np.zeros((0, 10), dtype=np.float32)
    sims = torch.from_numpy(query_emb).float() @ torch.from_numpy(db_emb).float().T
    topv, topi = torch.topk(sims, k=min(k, db_emb.shape[0]), dim=1)
    weights = torch.softmax(topv / tau, dim=1).cpu().numpy()
    idx = topi.cpu().numpy()
    labels = db_y[idx]
    out = np.zeros((len(query_emb), 10), dtype=np.float32)
    for c in POINT_CLASSES:
        out[:, c] = (weights * (labels == c)).sum(axis=1)
    out = (out + alpha * prefix_prior[None, :]) / (1.0 + alpha)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def build_knn_oof(valid_meta, valid_emb, fold_train_records: list[dict[str, object]], args) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    configs = []
    base_knn_by_config: dict[str, np.ndarray] = {}
    for k in [25, 50, 100, 200]:
        for tau in [0.03, 0.05, 0.10]:
            for alpha in [5.0, 20.0]:
                key = f"k{k}_tau{tau}_alpha{alpha:g}"
                configs.append((key, k, tau, alpha))
                base_knn_by_config[key] = np.zeros((len(valid_meta), 10), dtype=np.float32)
    for rec in fold_train_records:
        fold = int(rec["fold"])
        train_meta = rec["train_meta"]
        train_emb = rec["train_emb"]
        idx_valid_fold = valid_meta.index[valid_meta["fold"].eq(fold)].to_numpy()
        for prefix_len in [1, 2]:
            query_idx = idx_valid_fold[valid_meta.iloc[idx_valid_fold]["prefix_len"].eq(prefix_len).to_numpy()]
            db_idx = train_meta.index[train_meta["prefix_len"].eq(prefix_len)].to_numpy()
            if len(query_idx) == 0 or len(db_idx) == 0:
                continue
            q_emb = valid_emb[query_idx]
            db_emb = train_emb[db_idx]
            db_y = train_meta.iloc[db_idx]["next_pointId"].to_numpy(dtype=int)
            prefix_prior = prior_by_prefix(train_meta, prefix_len)
            for key, k, tau, alpha in configs:
                base_knn_by_config[key][query_idx] = knn_prior_for_queries(db_emb, db_y, q_emb, prefix_prior, k, tau, alpha)
    return pd.DataFrame([{"config": key, "k": k, "tau": tau, "alpha": alpha} for key, k, tau, alpha in configs]), base_knn_by_config


def select_knn_config(
    valid_meta: pd.DataFrame,
    v3_point: np.ndarray,
    base_pred: np.ndarray,
    v3_tuning: V3Tuning,
    config_df: pd.DataFrame,
    knn_probs: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    y = valid_meta["next_pointId"].to_numpy(dtype=int)
    base_point = f1_score(y, base_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    short_mask = valid_meta["prefix_len"].le(2).to_numpy()
    fold_base = {}
    for fold in sorted(valid_meta["fold"].unique()):
        idx = valid_meta["fold"].eq(fold).to_numpy()
        fold_base[int(fold)] = f1_score(y[idx], base_pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0)
    rows = []
    best = None
    for _, cfg in config_df.iterrows():
        p_knn = knn_probs[str(cfg["config"])]
        for lam1 in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
            for lam2 in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
                mixed = v3_point.copy()
                mask1 = valid_meta["prefix_len"].eq(1).to_numpy()
                mask2 = valid_meta["prefix_len"].eq(2).to_numpy()
                if lam1 > 0:
                    mixed[mask1] = (1.0 - lam1) * v3_point[mask1] + lam1 * p_knn[mask1]
                if lam2 > 0:
                    mixed[mask2] = (1.0 - lam2) * v3_point[mask2] + lam2 * p_knn[mask2]
                mixed = mixed / mixed.sum(axis=1, keepdims=True)
                pred = apply_segmented_multipliers(valid_meta, mixed, v3_tuning.point_multipliers, POINT_CLASSES, v3_tuning.bins_mode)
                point = f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)
                short_point = f1_score(y[short_mask], pred[short_mask], average="macro", labels=POINT_CLASSES, zero_division=0)
                base_short = f1_score(y[short_mask], base_pred[short_mask], average="macro", labels=POINT_CLASSES, zero_division=0)
                churn = float((pred != base_pred).mean())
                short_churn = float((pred[short_mask] != base_pred[short_mask]).mean())
                folds_improved = 0
                for fold in sorted(valid_meta["fold"].unique()):
                    idx = valid_meta["fold"].eq(fold).to_numpy()
                    f = f1_score(y[idx], pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0)
                    folds_improved += int(f > fold_base[int(fold)] + 1e-12)
                row = {
                    "config": cfg["config"],
                    "k": int(cfg["k"]),
                    "tau": float(cfg["tau"]),
                    "alpha": float(cfg["alpha"]),
                    "lambda_len1": lam1,
                    "lambda_len2": lam2,
                    "point_macro_f1": float(point),
                    "gain_vs_v3": float(point - base_point),
                    "short_point_macro_f1": float(short_point),
                    "short_gain_vs_v3": float(short_point - base_short),
                    "churn_vs_v3": churn,
                    "short_churn_vs_v3": short_churn,
                    "folds_improved": folds_improved,
                }
                rows.append(row)
                eligible = (
                    row["gain_vs_v3"] >= 0.003
                    and row["short_gain_vs_v3"] >= 0.005
                    and row["churn_vs_v3"] <= 0.03
                    and row["short_churn_vs_v3"] <= 0.08
                    and row["folds_improved"] >= 4
                    and (lam1 > 0 or lam2 > 0)
                )
                score = row["point_macro_f1"] - 0.02 * row["churn_vs_v3"]
                if eligible and (best is None or score > best["score"]):
                    best = {"score": score, "row": row, "pred": pred, "prob": mixed}
    search = pd.DataFrame(rows).sort_values(["gain_vs_v3", "short_gain_vs_v3"], ascending=[False, False])
    if best is None:
        selected = {
            "config": "base_v3",
            "lambda_len1": 0.0,
            "lambda_len2": 0.0,
            "point_macro_f1": float(base_point),
            "gain_vs_v3": 0.0,
            "reason": "No eligible V14 kNN config met stopping criteria.",
        }
        return search, v3_point, selected
    return search, best["prob"], best["row"]


def prefix_report(meta, action_prob, point_prob, server_prob, action_mult, point_mult, mode) -> pd.DataFrame:
    rows = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
        ("le2", meta["prefix_len"].le(2).to_numpy()),
        ("ge3", meta["prefix_len"].ge(3).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        metrics = evaluate(meta.iloc[idx].reset_index(drop=True), action_prob[idx], point_prob[idx], server_prob[idx], action_mult, point_mult, mode)
        metrics.update({"prefix_len_bin": label, "count": int(len(idx))})
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_meta = build_train_meta(train)
    test_meta = build_test_meta(test)
    test_ssl_meta = build_test_internal_meta(test)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards, _, class_sizes = cat_cardinalities_with_mask(train, test)
    device = torch.device(args.device)
    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    test_lengths = test_meta["prefix_len"].to_numpy(dtype=int)
    test_ssl_arrays = (
        build_sequence_arrays(test, test_ssl_meta, args.max_len, num_mean, num_std)
        if len(test_ssl_meta)
        else None
    )
    print(f"public test SSL rows: {len(test_ssl_meta):,}")
    valid_parts, action_parts, point_parts, server_parts, emb_parts = [], [], [], [], []
    fold_train_records = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        if fold > args.fold_limit:
            break
        tr_ids = set(rally_meta.iloc[tr_idx]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_idx]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        va_idx_sample = sample_validation_prefixes(va_pool, test_lengths, args.seed + fold)
        va_meta = va_pool.loc[va_idx_sample].copy().reset_index(drop=True)
        train_arrays = build_sequence_arrays(train, tr_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        train_ssl_arrays = build_sequence_arrays(train, mark_train_ssl_meta(tr_meta), args.max_len, num_mean, num_std)
        pretrain_arrays = (
            concat_sequence_arrays([train_ssl_arrays, test_ssl_arrays])
            if test_ssl_arrays is not None
            else train_ssl_arrays
        )
        action_w = class_weights(tr_meta["next_actionId"], ACTION_CLASSES)
        point_w = class_weights(tr_meta[tr_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
        print(
            f"fold {fold}: train={len(tr_meta):,} ssl={len(pretrain_arrays.meta):,} "
            f"valid={len(va_meta):,} device={device}"
        )
        model = train_model(
            train_arrays,
            valid_arrays,
            cat_cards,
            class_sizes,
            args,
            args.seed + fold,
            action_w,
            point_w,
            pretrain_arrays=pretrain_arrays,
        )
        a, p, s, e = predict_model(model, valid_arrays, args.batch_size, device)
        _, _, _, e_train = predict_model(model, train_arrays, args.batch_size, device)
        metrics = evaluate(va_meta, a, p, s)
        metrics.update({"fold": fold, "train_rows": len(tr_meta), "valid_rows": len(va_meta)})
        fold_rows.append(metrics)
        va_meta = va_meta.copy()
        va_meta["fold"] = fold
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)
        emb_parts.append(e)
        fold_train_records.append({"fold": fold, "train_meta": tr_meta.reset_index(drop=True), "train_emb": e_train})

    valid_meta = pd.concat(valid_parts, ignore_index=True)
    v12_action = np.vstack(action_parts)
    v12_point = np.vstack(point_parts)
    v12_server = np.concatenate(server_parts)
    valid_emb = np.vstack(emb_parts)
    v3_action, v3_point, v3_server, v3_tuning = load_v3_subset(args.v3_oof, valid_meta)
    base_pred = apply_segmented_multipliers(valid_meta, v3_point, v3_tuning.point_multipliers, POINT_CLASSES, v3_tuning.bins_mode)
    config_df, knn_probs = build_knn_oof(valid_meta, valid_emb, fold_train_records, args)
    search, selected_point, selected = select_knn_config(valid_meta, v3_point, base_pred, v3_tuning, config_df, knn_probs)
    search.to_csv(args.search_report, index=False)

    v12_single = evaluate(valid_meta, v12_action, v12_point, v12_server)
    selected_metrics = evaluate(valid_meta, v3_action, selected_point, v3_server, v3_tuning.action_multipliers, v3_tuning.point_multipliers, v3_tuning.bins_mode)
    prefix_report(valid_meta, v3_action, selected_point, v3_server, v3_tuning.action_multipliers, v3_tuning.point_multipliers, v3_tuning.bins_mode).to_csv(args.prefix_report, index=False)
    pd.DataFrame([{**{f"v14_single_{k}": v for k, v in v12_single.items()}, **{f"selected_{k}": v for k, v in selected_metrics.items()}, **selected}]).to_csv(args.cv_report, index=False)
    with open(args.oof_proba, "wb") as f:
        pickle.dump(
            {
                "valid_meta": valid_meta,
                "v12_action": v12_action,
                "v12_point": v12_point,
                "v12_server": v12_server,
                "v12_embedding": valid_emb,
                "v13_point": selected_point,
                "selected": selected,
                "fold_report": pd.DataFrame(fold_rows),
            },
            f,
        )

    metadata = {
        "args": vars(args),
        "mask_field_rates": MASK_FIELD_RATES,
        "public_test_ssl_rows": int(len(test_ssl_meta)),
        "v14_single": v12_single,
        "selected": selected,
        "selected_metrics": selected_metrics,
    }
    wrote_submission = False
    if not args.skip_full_train and selected.get("config") != "base_v3":
        print("training full V14 for test retrieval...")
        train_arrays = build_sequence_arrays(train, prefix_meta, args.max_len, num_mean, num_std)
        test_arrays = build_sequence_arrays(test, test_meta, args.max_len, num_mean, num_std)
        train_ssl_arrays = build_sequence_arrays(train, mark_train_ssl_meta(prefix_meta), args.max_len, num_mean, num_std)
        full_pretrain_arrays = (
            concat_sequence_arrays([train_ssl_arrays, test_ssl_arrays])
            if test_ssl_arrays is not None
            else train_ssl_arrays
        )
        action_w = class_weights(prefix_meta["next_actionId"], ACTION_CLASSES)
        point_w = class_weights(prefix_meta[prefix_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
        model = train_model(
            train_arrays,
            test_arrays,
            cat_cards,
            class_sizes,
            args,
            args.seed + 999,
            action_w,
            point_w,
            pretrain_arrays=full_pretrain_arrays,
        )
        _, _, _, train_emb = predict_model(model, train_arrays, args.batch_size, device)
        _, _, _, test_emb = predict_model(model, test_arrays, args.batch_size, device)
        test_prefix, _, test_v3_point, test_v3_server = compose_v3_full(train, test, v3_tuning)
        if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
            raise ValueError("Test rows not aligned.")
        test_knn = np.zeros_like(test_v3_point)
        cfg = selected
        for prefix_len in [1, 2]:
            q_idx = test_meta.index[test_meta["prefix_len"].eq(prefix_len)].to_numpy()
            db_idx = prefix_meta.index[prefix_meta["prefix_len"].eq(prefix_len)].to_numpy()
            prior = prior_by_prefix(prefix_meta, prefix_len)
            test_knn[q_idx] = knn_prior_for_queries(
                train_emb[db_idx],
                prefix_meta.iloc[db_idx]["next_pointId"].to_numpy(dtype=int),
                test_emb[q_idx],
                prior,
                int(cfg["k"]),
                float(cfg["tau"]),
                float(cfg["alpha"]),
            )
        final_point = test_v3_point.copy()
        for prefix_len, lam_key in [(1, "lambda_len1"), (2, "lambda_len2")]:
            idx = test_meta["prefix_len"].eq(prefix_len).to_numpy()
            lam = float(cfg[lam_key])
            final_point[idx] = (1.0 - lam) * test_v3_point[idx] + lam * test_knn[idx]
        final_point = final_point / final_point.sum(axis=1, keepdims=True)
        selected_v10 = json.loads(Path(args.v10b_r1_selected).read_text(encoding="utf-8"))
        with open(args.r1_sequence_proba, "rb") as f:
            r1_full = pickle.load(f)
        with open(args.v10b_full_proba, "rb") as f:
            v10b_full = pickle.load(f)
        r1_action = 0.4 * r1_full["gru_action"] + 0.6 * r1_full["tr_action"]
        r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
        full_action = blend_probs(r1_action, v10b_full["v10_action"], float(selected_v10["action_v10_weight"]))
        r1_server = 0.8 * test_v3_server + 0.1 * r1_full["gru_server"] + 0.1 * r1_full["tr_server"]
        full_server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(selected_v10["server_v10_weight"]) * v10b_full["v10_server"]
        action_pred = apply_segmented_multipliers(test_meta, full_action, selected_v10["action_multipliers"], ACTION_CLASSES, "two")
        point_pred = apply_segmented_multipliers(test_meta, final_point, v3_tuning.point_multipliers, POINT_CLASSES, v3_tuning.bins_mode)
        sub = pd.DataFrame(
            {
                "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
                "actionId": action_pred.astype(int),
                "pointId": point_pred.astype(int),
                "serverGetPoint": np.round(np.clip(full_server, 1e-6, 1.0 - 1e-6), 8),
            }
        )
        sub.to_csv(args.submission, index=False, float_format="%.8f")
        wrote_submission = True
        metadata["submission_rows"] = int(len(sub))
    elif selected.get("config") == "base_v3":
        metadata["submission_note"] = "No V14 kNN config selected; submission not regenerated because point branch remains V3."

    metadata["wrote_submission"] = wrote_submission
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("V14 single:")
    print(json.dumps(v12_single, indent=2))
    print("V14 selected:")
    print(json.dumps(selected, indent=2))
    print("selected metrics:")
    print(json.dumps(selected_metrics, indent=2))
    print(f"wrote {args.cv_report}, {args.search_report}, {args.oof_proba}, {args.feature_report}")


if __name__ == "__main__":
    main()
