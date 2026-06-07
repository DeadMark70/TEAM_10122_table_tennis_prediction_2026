"""V37/V38 Macro-F1 aligned sequence fine-tuning experiments.

V37 keeps the V36 data pipeline, but changes fine-tuning to use a
Macro-F1-aligned head objective:
- class-balanced focal loss for action / point
- small differentiable soft Macro-F1 surrogate
- head warm-up and gradual unfreeze

V38 adds a small prefix-state consistency regularizer on top of V37:
T(h(prefix k)) ~= stopgrad(h(prefix k+1)).

It does not use hidden test targets, old-test server labels, or future
scoreboard features.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

import baseline_v10a_pretrain_transformer as v10
from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, sample_validation_prefixes, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import build_sequence_arrays, build_test_meta, build_train_meta, fit_numeric_stats
from generate_r1_submission import compose_v3_full


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


MASK_FIELDS = ["actionId", "pointId", "spinId", "strengthId", "handId"]
POINT_NONTERMINAL_CLASSES = list(range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V37/V38 Macro-F1 aligned sequence fine-tuning.")
    parser.add_argument("--variant", choices=["v37", "v38"], default="v37")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--r7-full-proba", default="r7_full_lgbm_proba.pkl")
    parser.add_argument("--cv-report", default=None)
    parser.add_argument("--prefix-len-report", default=None)
    parser.add_argument("--ensemble-summary", default=None)
    parser.add_argument("--feature-report", default=None)
    parser.add_argument("--oof-proba", default=None)
    parser.add_argument("--full-proba", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--submission-safe-point", default=None)
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--point-epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr-pretrain", type=float, default=7e-4)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--lr-point", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.2)
    parser.add_argument("--mask-loss-weight", type=float, default=0.4)
    parser.add_argument("--causal-loss-weight", type=float, default=0.6)
    parser.add_argument("--internal-weight", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--soft-f1-weight", type=float, default=0.15)
    parser.add_argument("--focal-weight", type=float, default=0.50)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--state-weight", type=float, default=0.05)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--reuse-full-proba", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    prefix = args.variant
    args.cv_report = args.cv_report or f"cv_report_{prefix}.csv"
    args.prefix_len_report = args.prefix_len_report or f"prefix_len_report_{prefix}.csv"
    args.ensemble_summary = args.ensemble_summary or f"{prefix}_r33_ensemble_summary.csv"
    args.feature_report = args.feature_report or f"feature_report_{prefix}.json"
    args.oof_proba = args.oof_proba or f"oof_proba_{prefix}.pkl"
    args.full_proba = args.full_proba or f"{prefix}_full_sequence_proba.pkl"
    args.submission = args.submission or f"submission_{prefix}_r33_ensemble.csv"
    args.submission_safe_point = args.submission_safe_point or f"submission_{prefix}_r33_safe_point.csv"
    if args.variant == "v37":
        args.state_weight = 0.0
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


class V36Dataset(Dataset):
    def __init__(
        self,
        arrays,
        mask_ids: list[int] | None = None,
        mask_prob: float = 0.0,
        seed: int = 42,
        pair_indices: np.ndarray | None = None,
    ) -> None:
        self.cat = arrays.cat
        self.num = arrays.num
        self.lengths = arrays.lengths
        self.meta = arrays.meta.reset_index(drop=True)
        self.pair_indices = pair_indices
        self.mask_ids = np.asarray(mask_ids, dtype=np.int64) if mask_ids is not None else None
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)
        self.mask_field_indices = [v10.CAT_FIELDS.index(f) for f in MASK_FIELDS]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cat = self.cat[idx].copy()
        num = self.num[idx].copy()
        length = int(self.lengths[idx])
        row = self.meta.iloc[idx]
        mask_targets = np.full((cat.shape[0], cat.shape[1]), -100, dtype=np.int64)
        if self.mask_ids is not None and self.mask_prob > 0:
            rng = np.random.default_rng(self.seed + idx)
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
        sample_weight = float(row.get("sample_weight", 1.0))
        aux_weight = float(row.get("aux_weight", 1.0))
        item = {
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
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "aux_weight": torch.tensor(aux_weight, dtype=torch.float32),
        }
        if self.pair_indices is not None:
            pair_idx = int(self.pair_indices[idx])
            if pair_idx >= 0:
                item["pair_cat"] = torch.from_numpy(self.cat[pair_idx]).long()
                item["pair_num"] = torch.from_numpy(self.num[pair_idx]).float()
                item["pair_lengths"] = torch.tensor(int(self.lengths[pair_idx]), dtype=torch.long)
                item["pair_mask"] = torch.tensor(1.0, dtype=torch.float32)
            else:
                item["pair_cat"] = torch.from_numpy(self.cat[idx]).long()
                item["pair_num"] = torch.from_numpy(self.num[idx]).float()
                item["pair_lengths"] = torch.tensor(length, dtype=torch.long)
                item["pair_mask"] = torch.tensor(0.0, dtype=torch.float32)
        return item


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.sum().clamp(min=1e-6)
    return (loss * weight).sum() / denom


def focal_ce_loss(logits: torch.Tensor, target: torch.Tensor, class_weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
    pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
    return ((1.0 - pt) ** gamma) * ce


def soft_macro_f1_loss(probs: torch.Tensor, target: torch.Tensor, num_classes: int, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if probs.numel() == 0:
        return probs.sum() * 0.0
    y = F.one_hot(target, num_classes=num_classes).float()
    if sample_weight is not None:
        w = sample_weight.reshape(-1, 1).float()
        probs = probs * w
        y = y * w
    tp = (probs * y).sum(dim=0)
    fp = (probs * (1.0 - y)).sum(dim=0)
    fn = ((1.0 - probs) * y).sum(dim=0)
    f1 = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
    return 1.0 - f1.mean()


def v36_supervised_loss(outputs, batch, action_w, point_w, args, device, phase: str) -> torch.Tensor:
    sw = batch["sample_weight"]
    aux = batch["aux_weight"]
    server_w = batch["server_weight"] * aux
    action_weight = action_w.to(device)
    point_weight = point_w.to(device)

    action_ce = F.cross_entropy(outputs["action"], batch["action"], weight=action_weight, reduction="none")
    action_focal = focal_ce_loss(outputs["action"], batch["action"], action_weight, gamma=args.focal_gamma)
    action_f1 = soft_macro_f1_loss(F.softmax(outputs["action"], dim=-1), batch["action"], 19, sw)
    action_loss = (
        weighted_mean(action_ce, sw)
        + args.focal_weight * weighted_mean(action_focal, sw)
        + args.soft_f1_weight * action_f1
    )

    terminal_raw = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"], reduction="none")
    terminal_loss = weighted_mean(terminal_raw, sw)

    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        point_ce = F.cross_entropy(
            outputs["point"][point_mask],
            batch["point_nonterminal"][point_mask],
            weight=point_weight,
            reduction="none",
        )
        point_focal = focal_ce_loss(outputs["point"][point_mask], batch["point_nonterminal"][point_mask], point_weight, gamma=args.focal_gamma)
        point_f1 = soft_macro_f1_loss(
            F.softmax(outputs["point"][point_mask], dim=-1),
            batch["point_nonterminal"][point_mask],
            9,
            sw[point_mask],
        )
        point_loss = (
            weighted_mean(point_ce, sw[point_mask])
            + args.focal_weight * weighted_mean(point_focal, sw[point_mask])
            + args.soft_f1_weight * point_f1
        )
    else:
        point_loss = outputs["point"].sum() * 0.0

    server_raw = F.binary_cross_entropy_with_logits(outputs["server"], batch["server"], reduction="none")
    server_loss = weighted_mean(server_raw, server_w)
    parity_raw = F.binary_cross_entropy_with_logits(outputs["parity"], batch["parity"], reduction="none")
    parity_loss = weighted_mean(parity_raw, aux)
    remaining_raw = F.cross_entropy(outputs["remaining"], batch["remaining"], reduction="none")
    remaining_loss = weighted_mean(remaining_raw, aux)

    if phase == "point":
        weights = {"action": 0.05, "terminal": 0.25, "point": 0.50, "server": 0.10, "parity": 0.05, "remaining": 0.05}
    else:
        weights = {"action": 0.35, "terminal": 0.15, "point": 0.25, "server": 0.15, "parity": 0.05, "remaining": 0.05}
    return (
        weights["action"] * action_loss
        + weights["terminal"] * terminal_loss
        + weights["point"] * point_loss
        + weights["server"] * server_loss
        + weights["parity"] * parity_loss
        + weights["remaining"] * remaining_loss
    )


def mask_loss(outputs, batch, mask_field_indices: list[int]) -> torch.Tensor:
    losses = []
    targets = batch["mask_targets"]
    for field_idx in mask_field_indices:
        field = v10.CAT_FIELDS[field_idx]
        target = targets[:, :, field_idx]
        if target.ge(0).any():
            logits = outputs["mask"][field]
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100))
    if not losses:
        return outputs["action"].sum() * 0.0
    return torch.stack(losses).mean()


def pooled_state(model, outputs: dict[str, torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
    token_h = outputs["token_h"]
    bsz, seq_len, _ = token_h.shape
    pad_mask = torch.arange(seq_len, device=token_h.device).unsqueeze(0) >= lengths.unsqueeze(1)
    last_idx = (lengths - 1).clamp(min=0)
    h_last = token_h[torch.arange(bsz, device=token_h.device), last_idx]
    gate = model.pool_gate(token_h).squeeze(-1).masked_fill(pad_mask, -1e9)
    attn = torch.softmax(gate, dim=1)
    h_pool = (token_h * attn.unsqueeze(-1)).sum(dim=1)
    return model.head(torch.cat([h_last, h_pool], dim=-1))


def state_regularization_loss(model, outputs, batch, state_weight: float) -> torch.Tensor:
    if state_weight <= 0 or "pair_cat" not in batch:
        return outputs["action"].sum() * 0.0
    pair_mask = batch["pair_mask"]
    if pair_mask.sum() <= 0:
        return outputs["action"].sum() * 0.0
    h = pooled_state(model, outputs, batch["lengths"])
    with torch.no_grad():
        pair_out = model(batch["pair_cat"], batch["pair_num"], batch["pair_lengths"])
        h_next = pooled_state(model, pair_out, batch["pair_lengths"])
    h_pred = model.state_proj(h) if hasattr(model, "state_proj") else h
    mse = ((h_pred - h_next.detach()) ** 2).mean(dim=1)
    return state_weight * weighted_mean(mse, pair_mask)


def train_epoch(model, loader, opt, action_w, point_w, args, device, phase: str, pretrain: bool) -> dict[str, float]:
    model.train()
    losses, mask_losses, sup_losses, state_losses = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["cat"], batch["num"], batch["lengths"])
        sup = v36_supervised_loss(out, batch, action_w, point_w, args, device, phase)
        state = state_regularization_loss(model, out, batch, args.state_weight if not pretrain else 0.0)
        if pretrain:
            m = mask_loss(out, batch, model.mask_field_indices)
            loss = args.mask_loss_weight * m + args.causal_loss_weight * sup
            mask_losses.append(float(m.detach().cpu()))
        else:
            m = out["action"].sum() * 0.0
            loss = sup + state
            state_losses.append(float(state.detach().cpu()))
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        sup_losses.append(float(sup.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "mask_loss": float(np.mean(mask_losses)) if mask_losses else 0.0,
        "supervised_loss": float(np.mean(sup_losses)),
        "state_loss": float(np.mean(state_losses)) if state_losses else 0.0,
    }


def evaluate(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, server_prob: np.ndarray) -> dict[str, float]:
    action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def predict_model(model, arrays, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(V36Dataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
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


def cat_cardinalities_with_mask(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    cards, mask_ids, class_sizes = [], [], []
    for field in v10.CAT_FIELDS:
        max_value = int(max(train[field].max(), test[field].max()))
        class_size = max_value + 1
        card = max_value + 3
        cards.append(card)
        mask_ids.append(card - 1)
        class_sizes.append(class_size)
    return cards, mask_ids, class_sizes


def prepare_meta(meta: pd.DataFrame, sample_weight: float, aux_weight: float) -> pd.DataFrame:
    out = meta.copy()
    out["sample_weight"] = float(sample_weight)
    out["aux_weight"] = float(aux_weight)
    if "server_weight" not in out:
        out["server_weight"] = 1.0
    return out


def make_internal_meta_from_public(test_df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    work = test_df.copy()
    if "serverGetPoint" not in work:
        work["serverGetPoint"] = 0
    meta = build_train_meta(work)
    if len(meta) == 0:
        return meta
    meta["next_is_terminal"] = 0
    meta["serverGetPoint"] = 0
    meta["server_weight"] = 0.0
    meta["final_parity_even"] = 0
    meta["remaining_len"] = 1
    meta["remaining_len_bucket"] = 1
    return meta


def make_internal_meta_from_valid(valid_pool: pd.DataFrame, fold_valid: pd.DataFrame) -> pd.DataFrame:
    limits = fold_valid[["rally_uid", "prefix_len"]].rename(columns={"prefix_len": "observed_prefix_len"})
    internal = valid_pool.merge(limits, on="rally_uid", how="inner")
    internal = internal[internal["prefix_len"].lt(internal["observed_prefix_len"])].copy()
    internal = internal.drop(columns=["observed_prefix_len"])
    if len(internal) == 0:
        return internal
    internal["next_is_terminal"] = 0
    internal["serverGetPoint"] = 0
    internal["server_weight"] = 0.0
    internal["final_parity_even"] = 0
    internal["remaining_len"] = 1
    internal["remaining_len_bucket"] = 1
    return internal


def make_model(cat_cards, class_sizes, mask_field_indices, args, device):
    model = v10.StrokeTransformer(
        cat_cards,
        class_sizes,
        mask_field_indices,
        len(v10.NUM_FIELDS),
        args.max_len,
        args.d_model,
        args.emb_dim,
        args.numeric_dim,
        args.num_layers,
        args.num_heads,
        args.dropout,
    ).to(device)
    model.state_proj = torch.nn.Sequential(
        torch.nn.LayerNorm(args.d_model * 2),
        torch.nn.Linear(args.d_model * 2, args.d_model * 2),
    ).to(device)
    return model


def build_pair_indices(meta: pd.DataFrame) -> np.ndarray:
    key_to_index = {
        (int(row.rally_uid), int(row.prefix_index)): idx
        for idx, row in enumerate(meta.itertuples(index=False))
    }
    pairs = np.full(len(meta), -1, dtype=np.int64)
    for idx, row in enumerate(meta.itertuples(index=False)):
        pairs[idx] = key_to_index.get((int(row.rally_uid), int(row.prefix_index) + 1), -1)
    return pairs


def set_backbone_trainable(model, trainable: bool) -> None:
    head_prefixes = ("action_head", "terminal_head", "point_head", "server_head", "parity_head", "remaining_head", "mask_heads", "state_proj")
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith(head_prefixes)


def train_v36_model(train_raw, arrays_train, arrays_valid, action_w, point_w, cat_cards, mask_ids, class_sizes, args, fold_seed, device):
    del train_raw
    mask_field_indices = [v10.CAT_FIELDS.index(f) for f in MASK_FIELDS]
    model = make_model(cat_cards, class_sizes, mask_field_indices, args, device)
    pair_indices = build_pair_indices(arrays_train.meta) if args.state_weight > 0 else None

    pre_loader = DataLoader(
        V36Dataset(arrays_train, mask_ids, args.mask_prob, fold_seed + 9000),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
    for epoch in range(1, args.pretrain_epochs + 1):
        losses = train_epoch(model, pre_loader, opt, action_w, point_w, args, device, phase="joint", pretrain=True)
        print(f"  pretrain {epoch:02d}: loss={losses['loss']:.5f} mask={losses['mask_loss']:.5f} sup={losses['supervised_loss']:.5f}")

    set_backbone_trainable(model, False)
    ft_loader = DataLoader(
        V36Dataset(arrays_train, pair_indices=pair_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(fold_seed + 77),
    )
    best_state, best_metrics = None, {"overall": -1.0}
    stages = []
    if args.head_warmup_epochs > 0:
        stages.append(("warmup", args.head_warmup_epochs, args.lr_finetune))
    set_backbone_trainable(model, True)
    stages.extend([
        ("joint", args.finetune_epochs, args.lr_finetune),
        ("point", args.point_epochs, args.lr_point),
        ("joint", args.final_epochs, args.lr_point * 0.5),
    ])
    for phase, epochs, lr in stages:
        set_backbone_trainable(model, phase != "warmup")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.weight_decay)
        bad = 0
        for epoch in range(1, epochs + 1):
            losses = train_epoch(model, ft_loader, opt, action_w, point_w, args, device, phase=phase, pretrain=False)
            if arrays_valid is not None:
                a, p, s = predict_model(model, arrays_valid, args.batch_size, device)
                metrics = evaluate(arrays_valid.meta, a, p, s)
                print(f"  {phase} {epoch:02d}: loss={losses['loss']:.5f} state={losses['state_loss']:.5f} overall={metrics['overall']:.6f} action={metrics['action_macro_f1']:.6f} point={metrics['point_macro_f1']:.6f} server={metrics['server_auc']:.6f}")
                if metrics["overall"] > best_metrics["overall"] + 1e-6:
                    best_metrics = metrics
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            else:
                print(f"  {phase} {epoch:02d}: loss={losses['loss']:.5f} state={losses['state_loss']:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def search_r33_v36_ensemble(meta, v36_action, v36_point, v36_server, v3, v5, v7, v10_oof, r7_oof, args):
    v3_action, v3_point, v3_server = compose_v3(v3)
    r7_action, _, r7_server = compose_v3(r7_oof)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    r1_server = np.clip(0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"], 1e-6, 1.0 - 1e-6)
    r33_action = normalize_rows(0.85 * r1_action + 0.05 * r7_action + 0.10 * v5["gru_action"])
    r33_server = np.clip(0.70 * r1_server + 0.15 * v10_oof["v10_server"] + 0.15 * r7_server, 1e-6, 1.0 - 1e-6)

    rows, best = [], None
    for aw in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        action_prob = blend_probs(r33_action, v36_action, aw)
        action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", args.multiplier_bins)
        action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, args.multiplier_bins)
        action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        for pw in [0.0, 0.02, 0.05, 0.1, 0.2]:
            point_prob = blend_probs(v3_point, v36_point, pw)
            point_mult = tune_segmented_multipliers(meta, point_prob, POINT_CLASSES, "point", args.multiplier_bins)
            point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, args.multiplier_bins)
            point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
            for sw in [0.0, 0.05, 0.1, 0.2, 0.3]:
                server_prob = (1.0 - sw) * r33_server + sw * v36_server
                server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
                overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
                row = {
                    "action_v36_weight": aw,
                    "point_v36_weight": pw,
                    "server_v36_weight": sw,
                    "action_macro_f1": float(action_f1),
                    "point_macro_f1": float(point_f1),
                    "server_auc": float(server_auc),
                    "overall": float(overall),
                }
                rows.append(row)
                if best is None or overall > best["overall"]:
                    best = {**row, "action_mult": action_mult, "point_mult": point_mult}
    return pd.DataFrame(rows).sort_values("overall", ascending=False), best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards, mask_ids, class_sizes = cat_cardinalities_with_mask(train, test)
    test_meta = build_test_meta(test)
    test_lengths = test_meta["prefix_len"].to_numpy(dtype=int)
    prefix_meta = build_train_meta(train)

    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    valid_parts, action_parts, point_parts, server_parts, fold_rows = [], [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        if fold > args.fold_limit:
            break
        tr_ids = set(rally_meta.iloc[tr_idx]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_idx]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        sampled = sample_validation_prefixes(va_pool, test_lengths, args.seed + fold)
        va_meta = va_pool.loc[sampled].copy().reset_index(drop=True)
        internal = make_internal_meta_from_valid(va_pool, va_meta)
        tr_meta = prepare_meta(tr_meta, 1.0, 1.0)
        internal = prepare_meta(internal, args.internal_weight, 0.0) if len(internal) else internal
        combined_meta = pd.concat([tr_meta, internal], ignore_index=True) if len(internal) else tr_meta
        print(f"fold {fold}: train={len(tr_meta):,} internal={len(internal):,} valid={len(va_meta):,} device={device}")

        train_arrays = build_sequence_arrays(train, combined_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        action_w = v10.class_weights(combined_meta["next_actionId"], ACTION_CLASSES)
        point_w = v10.class_weights(combined_meta[combined_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
        model = train_v36_model(train, train_arrays, valid_arrays, action_w, point_w, cat_cards, mask_ids, class_sizes, args, args.seed + fold * 101, device)
        a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(va_meta, a, p, s)
        metrics.update({"fold": fold, "train_rows": len(tr_meta), "internal_rows": len(internal), "valid_rows": len(va_meta)})
        print(f"fold {fold} selected: {metrics}")
        fold_rows.append(metrics)
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)

    valid_meta = pd.concat(valid_parts, ignore_index=True)
    v36_action = np.vstack(action_parts)
    v36_point = np.vstack(point_parts)
    v36_server = np.concatenate(server_parts)
    oof = {"valid_meta": valid_meta, "v36_action": v36_action, "v36_point": v36_point, "v36_server": v36_server, "fold_report": pd.DataFrame(fold_rows)}
    with open(args.oof_proba, "wb") as f:
        pickle.dump(oof, f)
    pd.DataFrame(fold_rows).to_csv(args.cv_report, index=False)

    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10_oof = load_pickle(args.v10b_oof)
    r7_oof = load_pickle(args.r7_oof)
    meta = normalize_meta(valid_meta)
    ensemble_summary, selected = search_r33_v36_ensemble(meta, v36_action, v36_point, v36_server, v3, v5, v7, v10_oof, r7_oof, args)
    ensemble_summary.to_csv(args.ensemble_summary, index=False)
    print("V36 ensemble best:")
    print(json.dumps({k: v for k, v in selected.items() if not k.endswith("mult")}, indent=2))

    if args.reuse_full_proba and Path(args.full_proba).exists():
        full = load_pickle(args.full_proba)
    else:
        public_internal = make_internal_meta_from_public(test, args.max_len)
        train_meta = prepare_meta(prefix_meta.copy(), 1.0, 1.0)
        public_internal = prepare_meta(public_internal, args.internal_weight, 0.0) if len(public_internal) else public_internal
        full_meta = pd.concat([train_meta, public_internal], ignore_index=True) if len(public_internal) else train_meta
        combined_raw = pd.concat([train, test], ignore_index=True)
        full_arrays = build_sequence_arrays(combined_raw, full_meta, args.max_len, num_mean, num_std)
        test_arrays = build_sequence_arrays(test, test_meta, args.max_len, num_mean, num_std)
        action_w = v10.class_weights(full_meta["next_actionId"], ACTION_CLASSES)
        point_w = v10.class_weights(full_meta[full_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
        print(f"full train: train={len(train_meta):,} public_internal={len(public_internal):,} test={len(test_meta):,}")
        model = train_v36_model(combined_raw, full_arrays, None, action_w, point_w, cat_cards, mask_ids, class_sizes, args, args.seed + 999, device)
        fa, fp, fs = predict_model(model, test_arrays, args.batch_size, device)
        full = {"test_meta": test_meta, "v36_action": fa, "v36_point": fp, "v36_server": fs}
        with open(args.full_proba, "wb") as f:
            pickle.dump(full, f)

    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10_full = pickle.load(f)
    if Path(args.r7_full_proba).exists():
        with open(args.r7_full_proba, "rb") as f:
            r7_full = pickle.load(f)
    else:
        raise FileNotFoundError("r7_full_lgbm_proba.pkl is required; run generate_r33_safe_submission.py first.")
    test_prefix, _, v3_point, v3_server = compose_v3_full(train, test, v3["tuning"])
    if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("Full-test rows are not aligned.")
    r1_action = normalize_rows(0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]
    r33_action = normalize_rows(0.85 * r1_action + 0.05 * r7_full["r7_action"] + 0.10 * r1_seq["gru_action"])
    r33_server = 0.70 * r1_server + 0.15 * v10_full["v10_server"] + 0.15 * r7_full["r7_server"]

    action_prob = blend_probs(r33_action, full["v36_action"], float(selected["action_v36_weight"]))
    point_prob = blend_probs(v3_point, full["v36_point"], float(selected["point_v36_weight"]))
    server_prob = (1.0 - float(selected["server_v36_weight"])) * r33_server + float(selected["server_v36_weight"]) * full["v36_server"]
    action_pred = apply_segmented_multipliers(test_meta, action_prob, selected["action_mult"], ACTION_CLASSES, args.multiplier_bins)
    point_pred = apply_segmented_multipliers(test_meta, point_prob, selected["point_mult"], POINT_CLASSES, args.multiplier_bins)
    safe_point_pred = apply_segmented_multipliers(test_meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
    sub = pd.DataFrame({"rally_uid": test_meta["rally_uid"].astype(int), "actionId": action_pred.astype(int), "pointId": point_pred.astype(int), "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8)})
    safe = sub.copy()
    safe["pointId"] = safe_point_pred.astype(int)
    sub.to_csv(args.submission, index=False, float_format="%.8f")
    safe.to_csv(args.submission_safe_point, index=False, float_format="%.8f")

    report = {
        "experiment": "V36 task-aligned pretraining + internal-transition adaptation",
        "cv_report": args.cv_report,
        "ensemble_summary": args.ensemble_summary,
        "selected": {k: v for k, v in selected.items() if not k.endswith("mult")},
        "internal_weight": args.internal_weight,
        "epochs": {"pretrain": args.pretrain_epochs, "finetune": args.finetune_epochs, "point": args.point_epochs, "final": args.final_epochs},
        "submission": args.submission,
        "submission_safe_point": args.submission_safe_point,
    }
    Path(args.feature_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.ensemble_summary}")
    print(f"wrote {args.full_proba}")
    print(f"wrote {args.submission}")
    print(f"wrote {args.submission_safe_point}")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
