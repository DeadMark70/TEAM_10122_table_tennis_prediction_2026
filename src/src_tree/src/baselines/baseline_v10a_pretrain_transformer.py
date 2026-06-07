"""V10A: fold-safe masked-field + causal next-stroke pretraining smoke.

This is a research script, not a submission generator. It pretrains a compact
stroke-event Transformer on fold-train prefixes only, fine-tunes on the same
fold-train supervised labels, and evaluates on the sampled validation prefixes.

Default budget is intentionally small: one fold, two pretraining epochs, four
fine-tuning epochs. The goal is to verify whether pretraining adds useful
signal before scaling to full 5-fold V10B.
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
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import (
    CAT_FIELDS,
    NUM_FIELDS,
    SequenceArrays,
    build_sequence_arrays,
    build_test_meta,
    build_train_meta,
    fit_numeric_stats,
)


MASK_FIELDS = ["actionId", "pointId", "spinId", "strengthId", "handId"]
POINT_NONTERMINAL_CLASSES = list(range(1, 10))
REMAINING_CLASSES = list(range(1, 8))


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
    parser = argparse.ArgumentParser(description="Run V10A pretraining smoke.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--cv-report", default="cv_report_v10a.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v10a.csv")
    parser.add_argument("--feature-report", default="feature_report_v10a.json")
    parser.add_argument("--oof-proba", default="oof_proba_v10a.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr-pretrain", type=float, default=7e-4)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-loss-weight", type=float, default=0.5)
    parser.add_argument("--causal-loss-weight", type=float, default=0.5)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def cat_cardinalities_with_mask(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    cards: list[int] = []
    mask_ids: list[int] = []
    class_sizes: list[int] = []
    for field in CAT_FIELDS:
        max_value = int(max(train[field].max(), test[field].max()))
        class_size = max_value + 1
        card = max_value + 3  # pad 0, raw+1, mask max+2
        cards.append(card)
        mask_ids.append(card - 1)
        class_sizes.append(class_size)
    return cards, mask_ids, class_sizes


def class_weights(values: pd.Series, classes: list[int], beta: float = 0.25) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    weights = np.array([float(counts.get(cls, 1)) ** (-beta) for cls in classes], dtype=np.float32)
    return torch.from_numpy(weights / weights.mean())


class StrokePretrainDataset(Dataset):
    def __init__(
        self,
        arrays: SequenceArrays,
        mask_ids: list[int],
        mask_prob: float,
        seed: int,
    ) -> None:
        self.cat = arrays.cat
        self.num = arrays.num
        self.lengths = arrays.lengths
        self.meta = arrays.meta.reset_index(drop=True)
        self.mask_ids = np.asarray(mask_ids, dtype=np.int64)
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)
        self.mask_field_indices = [CAT_FIELDS.index(f) for f in MASK_FIELDS]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cat = self.cat[idx].copy()
        num = self.num[idx].copy()
        length = int(self.lengths[idx])
        row = self.meta.iloc[idx]
        rng = np.random.default_rng(self.seed + idx)
        mask_targets = np.full((cat.shape[0], cat.shape[1]), -100, dtype=np.int64)
        for field_idx in self.mask_field_indices:
            if length <= 0:
                continue
            mask = rng.random(length) < self.mask_prob
            if not mask.any():
                mask[int(rng.integers(0, length))] = True
            original = cat[:length, field_idx].copy()
            mask_targets[:length, field_idx][mask] = original[mask] - 1
            cat[:length, field_idx][mask] = self.mask_ids[field_idx]
        next_point = int(row.get("next_pointId", 0))
        return {
            "cat": torch.from_numpy(cat).long(),
            "num": torch.from_numpy(num).float(),
            "lengths": torch.tensor(length, dtype=torch.long),
            "mask_targets": torch.from_numpy(mask_targets).long(),
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point_nonterminal": torch.tensor(next_point - 1 if next_point > 0 else 0, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
        }


class StrokeEvalDataset(Dataset):
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
        return {
            "cat": self.cat[idx],
            "num": self.num[idx],
            "lengths": self.lengths[idx],
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point_nonterminal": torch.tensor(next_point - 1 if next_point > 0 else 0, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
        }


class StrokeTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        cat_class_sizes: list[int],
        mask_field_indices: list[int],
        num_dim: int,
        max_len: int,
        d_model: int,
        emb_dim: int,
        numeric_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.mask_field_indices = mask_field_indices
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
        self.action_head = nn.Linear(d_model * 2, 19)
        self.terminal_head = nn.Linear(d_model * 2, 1)
        self.point_head = nn.Linear(d_model * 2, 9)
        self.server_head = nn.Linear(d_model * 2, 1)
        self.parity_head = nn.Linear(d_model * 2, 1)
        self.remaining_head = nn.Linear(d_model * 2, 7)
        self.mask_heads = nn.ModuleDict(
            {CAT_FIELDS[i]: nn.Linear(d_model, cat_class_sizes[i]) for i in self.mask_field_indices}
        )

    def encode(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        return x, h

    def forward(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        token_h, h = self.encode(cat, num, lengths)
        mask_logits = {field: self.mask_heads[field](token_h) for field in self.mask_heads}
        return {
            "token_h": token_h,
            "mask": mask_logits,
            "action": self.action_head(h),
            "terminal": self.terminal_head(h).squeeze(-1),
            "point": self.point_head(h),
            "server": self.server_head(h).squeeze(-1),
            "parity": self.parity_head(h).squeeze(-1),
            "remaining": self.remaining_head(h),
        }


def supervised_loss(outputs, batch, action_w, point_w, device) -> torch.Tensor:
    action_loss = F.cross_entropy(outputs["action"], batch["action"], weight=action_w.to(device))
    terminal_loss = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"])
    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        point_loss = F.cross_entropy(outputs["point"][point_mask], batch["point_nonterminal"][point_mask], weight=point_w.to(device))
    else:
        point_loss = outputs["point"].sum() * 0.0
    server_loss_raw = F.binary_cross_entropy_with_logits(outputs["server"], batch["server"], reduction="none")
    server_loss = (server_loss_raw * batch["server_weight"]).mean()
    parity_loss = F.binary_cross_entropy_with_logits(outputs["parity"], batch["parity"])
    remaining_loss = F.cross_entropy(outputs["remaining"], batch["remaining"])
    return (
        0.35 * action_loss
        + 0.15 * terminal_loss
        + 0.25 * point_loss
        + 0.15 * server_loss
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


def predict_model(model, arrays, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(StrokeEvalDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    action_parts, point_parts, server_parts = [], [], []
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
            action_parts.append(action)
            point_parts.append(point)
            server_parts.append(server)
    return np.vstack(action_parts), np.vstack(point_parts), np.concatenate(server_parts)


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


def train_epoch(model, loader, opt, action_w, point_w, args, device, pretrain: bool) -> dict[str, float]:
    model.train()
    losses, mask_losses, sup_losses = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["cat"], batch["num"], batch["lengths"])
        sup = supervised_loss(out, batch, action_w, point_w, device)
        if pretrain:
            m = mask_loss(out, batch, model.mask_field_indices)
            loss = args.mask_loss_weight * m + args.causal_loss_weight * sup
            mask_losses.append(float(m.detach().cpu()))
        else:
            m = out["action"].sum() * 0.0
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
    }


def load_v3_subset(path: str, valid_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, V3Tuning]:
    with open(path, "rb") as f:
        oof = pickle.load(f)
    tuning = oof["tuning"]
    meta = oof["valid_meta"].reset_index(drop=True)
    key_cols = ["rally_uid", "prefix_len"]
    merged = valid_meta[key_cols].reset_index().merge(meta[key_cols].reset_index(), on=key_cols, how="left", suffixes=("", "_v3"))
    if merged["index_v3"].isna().any():
        raise ValueError("Could not align V10A valid rows to V3 OOF.")
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


def prefix_report(meta, action_prob, point_prob, server_prob, action_mult, point_mult, mode) -> pd.DataFrame:
    rows = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        metrics = evaluate(
            meta.iloc[idx].reset_index(drop=True),
            action_prob[idx],
            point_prob[idx],
            server_prob[idx],
            action_mult,
            point_mult,
            mode,
        )
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
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards, mask_ids, class_sizes = cat_cardinalities_with_mask(train, test)
    mask_field_indices = [CAT_FIELDS.index(f) for f in MASK_FIELDS]
    device = torch.device(args.device)

    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    fold_rows = []
    valid_parts, action_parts, point_parts, server_parts = [], [], [], []
    test_lengths = test_meta["prefix_len"].to_numpy(dtype=int)

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
        action_w = class_weights(tr_meta["next_actionId"], ACTION_CLASSES)
        point_w = class_weights(tr_meta[tr_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))

        model = StrokeTransformer(
            cat_cards,
            class_sizes,
            mask_field_indices,
            len(NUM_FIELDS),
            args.max_len,
            args.d_model,
            args.emb_dim,
            args.numeric_dim,
            args.num_layers,
            args.num_heads,
            args.dropout,
        ).to(device)
        print(f"fold {fold}: pretrain rows={len(tr_meta)} valid={len(va_meta)} device={device}")
        pre_ds = StrokePretrainDataset(train_arrays, mask_ids, args.mask_prob, args.seed + fold * 1000)
        pre_loader = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed + fold))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        for epoch in range(1, args.pretrain_epochs + 1):
            losses = train_epoch(model, pre_loader, opt, action_w, point_w, args, device, pretrain=True)
            print(f"  pretrain epoch {epoch:02d}: loss={losses['loss']:.5f} mask={losses['mask_loss']:.5f} causal={losses['supervised_loss']:.5f}")

        ft_ds = StrokeEvalDataset(train_arrays)
        ft_loader = DataLoader(ft_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed + fold + 77))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
        best_state, best_metrics = None, {"overall": -1.0}
        bad = 0
        for epoch in range(1, args.finetune_epochs + 1):
            losses = train_epoch(model, ft_loader, opt, action_w, point_w, args, device, pretrain=False)
            a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
            metrics = evaluate(va_meta, a, p, s)
            print(
                f"  finetune epoch {epoch:02d}: loss={losses['loss']:.5f} overall={metrics['overall']:.6f} "
                f"action={metrics['action_macro_f1']:.6f} point={metrics['point_macro_f1']:.6f} server={metrics['server_auc']:.6f}"
            )
            if metrics["overall"] > best_metrics["overall"] + 1e-6:
                best_metrics = metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
        fold_metrics = evaluate(va_meta, a, p, s)
        fold_metrics.update({"fold": fold, "train_rows": len(tr_meta), "valid_rows": len(va_meta)})
        fold_rows.append(fold_metrics)
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)

    valid_meta = pd.concat(valid_parts, ignore_index=True)
    v10_action = np.vstack(action_parts)
    v10_point = np.vstack(point_parts)
    v10_server = np.concatenate(server_parts)
    v3_action, v3_point, v3_server, v3_tuning = load_v3_subset(args.v3_oof, valid_meta)

    action_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    best = None
    for aw in action_weights:
        action_prob = blend_probs(v3_action, v10_action, aw)
        action_mult = tune_segmented_multipliers(valid_meta, action_prob, ACTION_CLASSES, "action", args.multiplier_bins)
        point_prob = v3_point
        point_mult = v3_tuning.point_multipliers
        for sw in [0.0, 0.1, 0.2, 0.3]:
            server_prob = (1.0 - sw) * v3_server + sw * v10_server
            metrics = evaluate(valid_meta, action_prob, point_prob, server_prob, action_mult, point_mult, args.multiplier_bins)
            candidate = {"action_weight": aw, "server_weight": sw, "action_mult": action_mult, "metrics": metrics}
            if best is None or metrics["overall"] > best["metrics"]["overall"]:
                best = candidate
    final_action = blend_probs(v3_action, v10_action, best["action_weight"])
    final_point = v3_point
    final_server = (1.0 - best["server_weight"]) * v3_server + best["server_weight"] * v10_server
    final_metrics = evaluate(
        valid_meta,
        final_action,
        final_point,
        final_server,
        best["action_mult"],
        v3_tuning.point_multipliers,
        args.multiplier_bins,
    )

    report = pd.DataFrame(fold_rows)
    single_mean = {f"v10_single_{k}": float(report[k].mean()) for k in ["action_macro_f1", "point_macro_f1", "server_auc", "overall"]}
    cv_report = pd.DataFrame([{**single_mean, **{f"selected_{k}": v for k, v in final_metrics.items()}, "action_weight": best["action_weight"], "server_weight": best["server_weight"], "folds_run": len(fold_rows)}])
    cv_report.to_csv(args.cv_report, index=False)
    prefix_report(valid_meta, final_action, final_point, final_server, best["action_mult"], v3_tuning.point_multipliers, args.multiplier_bins).to_csv(args.prefix_len_report, index=False)
    with open(args.oof_proba, "wb") as f:
        pickle.dump({"valid_meta": valid_meta, "v10_action": v10_action, "v10_point": v10_point, "v10_server": v10_server, "selected": best}, f)
    metadata = {
        "args": vars(args),
        "cat_fields": CAT_FIELDS,
        "mask_fields": MASK_FIELDS,
        "cat_cardinalities": cat_cards,
        "mask_ids": mask_ids,
        "class_sizes": class_sizes,
        "v10_single_mean": single_mean,
        "selected": {
            "action_weight": best["action_weight"],
            "server_weight": best["server_weight"],
            "metrics": final_metrics,
        },
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("selected:")
    print(json.dumps(metadata["selected"], indent=2))
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
