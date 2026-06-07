"""V16 local-distribution distillation for pointId.

V16 trains the existing stroke-event Transformer with hard labels plus a
fold-train-only local point distribution target. The local distribution is
computed from backoff tactical conditions with leave-one-out counts for the
training rows, then used as a soft target regularizer for the full 10-class
point probability implied by the terminal gate + nonterminal point head.

This tests whether retrieval/conditional local distributions are more useful
as train-time soft targets than as inference-time kNN/prior blends.
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

from analysis_r7_phase_features import add_phase_features
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import (
    CAT_FIELDS,
    NUM_FIELDS,
    SequenceArrays,
    build_sequence_arrays,
    build_test_meta,
    build_train_meta,
    fit_numeric_stats,
)


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
    parser = argparse.ArgumentParser(description="Run V16 local-distribution distillation.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--variant", choices=["global_local", "phase_local"], default="phase_local")
    parser.add_argument("--cv-report", default="cv_report_v16.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v16.csv")
    parser.add_argument("--point-blend-report", default="point_blend_report_v16.csv")
    parser.add_argument("--class-report-point", default="class_report_v16_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v16.json")
    parser.add_argument("--oof-proba", default="oof_proba_v16.pkl")
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--point-weight-beta", type=float, default=0.25)
    parser.add_argument("--logit-adjust-tau", type=float, default=0.5)
    parser.add_argument("--depth-loss-weight", type=float, default=0.10)
    parser.add_argument("--side-loss-weight", type=float, default=0.10)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.15)
    parser.add_argument("--distill-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def class_weights(values: pd.Series, classes: list[int], beta: float = 0.25) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    weights = np.array([float(counts.get(cls, 1)) ** (-beta) for cls in classes], dtype=np.float32)
    return torch.from_numpy(weights / weights.mean())


def effective_weights(values: pd.Series, classes: list[int], beta: float = 0.99) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    arr = np.array([float(counts.get(cls, 1)) for cls in classes], dtype=np.float32)
    eff = (1.0 - np.power(beta, arr)) / (1.0 - beta)
    weights = 1.0 / np.maximum(eff, 1e-6)
    return torch.from_numpy((weights / weights.mean()).astype(np.float32))


def class_log_prior(values: pd.Series, classes: list[int]) -> torch.Tensor:
    counts = values.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float32)
    prior = (counts + 1.0) / (counts.sum() + len(classes))
    return torch.from_numpy(np.log(prior).astype(np.float32))


def cat_cardinalities(train: pd.DataFrame, test: pd.DataFrame) -> list[int]:
    cards = []
    for field in CAT_FIELDS:
        max_value = int(max(train[field].max(), test[field].max()))
        cards.append(max_value + 2)
    return cards


def point_depth_side(point_id: int) -> tuple[int, int]:
    if point_id <= 0:
        return 0, 0
    zero = point_id - 1
    return zero // 3, zero % 3


POINT_DEPTH = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
}
LOCAL_PRIOR_LEVELS = [
    ["prefix_len", "sex", "lag0_actionId", "lag0_pointId", "lag0_spinId"],
    ["prefix_len", "sex", "lag0_actionId", "lag0_point_depth"],
    ["prefix_len", "sex", "serve_actionId", "receive_actionId", "lag0_actionId"],
    ["prefix_len", "sex", "last2_action_transition"],
    ["prefix_len", "sex", "lag0_actionId"],
    ["prefix_len", "sex"],
    ["prefix_len"],
]


def add_lag0_depth_for_prior(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag0_point_depth"] = df["lag0_pointId"].map(lambda x: POINT_DEPTH.get(int(x), -1)).astype(np.int8)
    return df


def align_phase_rows(phase_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    key = ["rally_uid", "prefix_len"]
    aligned = meta[key].reset_index().merge(phase_df, on=key, how="left")
    if aligned["next_pointId"].isna().any():
        raise ValueError("Could not align local-prior phase rows to sequence meta.")
    return aligned.sort_values("index").drop(columns=["index"]).reset_index(drop=True)


def build_leave_one_out_soft_targets(phase_rows: pd.DataFrame, alpha: float) -> np.ndarray:
    y = phase_rows["next_pointId"].to_numpy(dtype=int)
    global_counts = phase_rows["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=np.float64)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(POINT_CLASSES))
    tables: list[dict[tuple[int, ...], np.ndarray]] = []
    for level in LOCAL_PRIOR_LEVELS:
        table: dict[tuple[int, ...], np.ndarray] = {}
        for key, sub in phase_rows.groupby(level, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            counts = sub["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=np.float64)
            table[tuple(int(v) for v in key)] = counts
        tables.append(table)

    out = np.zeros((len(phase_rows), len(POINT_CLASSES)), dtype=np.float32)
    for i, row in enumerate(phase_rows.itertuples(index=False)):
        row_dict = row._asdict()
        yi = int(y[i])
        global_loo = global_counts.copy()
        global_loo[yi] = max(0.0, global_loo[yi] - 1.0)
        fallback = (global_loo + 1.0) / (global_loo.sum() + len(POINT_CLASSES))
        chosen = fallback
        for level, table in zip(LOCAL_PRIOR_LEVELS, tables):
            key = tuple(int(row_dict[col]) for col in level)
            counts = table[key].copy()
            counts[yi] = max(0.0, counts[yi] - 1.0)
            total = counts.sum()
            if total > 0:
                chosen = (counts + alpha * global_prior) / (total + alpha)
                break
        out[i] = chosen.astype(np.float32)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def attach_soft_targets(meta: pd.DataFrame, phase_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    phase_rows = align_phase_rows(phase_df, meta)
    soft = build_leave_one_out_soft_targets(phase_rows, alpha)
    out = meta.copy().reset_index(drop=True)
    for c in POINT_CLASSES:
        out[f"soft_point_{c}"] = soft[:, c]
    return out


class StrokePointDataset(Dataset):
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
        point_nonterminal = next_point - 1 if next_point > 0 else 0
        depth, side = point_depth_side(next_point)
        if all(f"soft_point_{c}" in row.index for c in POINT_CLASSES):
            soft_point = row[[f"soft_point_{c}" for c in POINT_CLASSES]].to_numpy(dtype=np.float32)
        else:
            soft_point = np.zeros(len(POINT_CLASSES), dtype=np.float32)
            soft_point[next_point] = 1.0
        return {
            "cat": self.cat[idx],
            "num": self.num[idx],
            "lengths": self.lengths[idx],
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point_nonterminal": torch.tensor(point_nonterminal, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "point_depth": torch.tensor(depth, dtype=torch.long),
            "point_side": torch.tensor(side, dtype=torch.long),
            "soft_point": torch.from_numpy(soft_point).float(),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
        }


class StrokePointTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_dim: int,
        max_len: int,
        d_model: int,
        emb_dim: int,
        numeric_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        structured: bool,
    ) -> None:
        super().__init__()
        self.structured = structured
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
        if structured:
            self.depth_head = nn.Linear(d_model * 2, 3)
            self.side_head = nn.Linear(d_model * 2, 3)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
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
        out = {
            "action": self.action_head(h),
            "terminal": self.terminal_head(h).squeeze(-1),
            "point": self.point_head(h),
            "server": self.server_head(h).squeeze(-1),
            "parity": self.parity_head(h).squeeze(-1),
            "remaining": self.remaining_head(h),
        }
        if self.structured:
            out["depth"] = self.depth_head(h)
            out["side"] = self.side_head(h)
        return out


def balanced_softmax_loss(logits: torch.Tensor, target: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    adjusted = logits + torch.log(counts.clamp_min(1.0)).to(logits.device).unsqueeze(0)
    return F.cross_entropy(adjusted, target)


def point_loss(outputs, batch, point_w, point_counts, point_log_prior, args, device) -> torch.Tensor:
    point_mask = batch["point_mask"] > 0.5
    if not point_mask.any():
        return outputs["point"].sum() * 0.0
    logits = outputs["point"][point_mask]
    target = batch["point_nonterminal"][point_mask]
    return F.cross_entropy(logits, target, weight=point_w.to(device))


def soft_point_distill_loss(outputs, batch) -> torch.Tensor:
    soft_target = batch["soft_point"]
    terminal = torch.sigmoid(outputs["terminal"]).clamp(1e-6, 1.0 - 1e-6)
    point_nonterm = F.softmax(outputs["point"], dim=-1).clamp_min(1e-8)
    full_prob = torch.cat([terminal[:, None], (1.0 - terminal[:, None]) * point_nonterm], dim=1)
    full_prob = full_prob / full_prob.sum(dim=1, keepdim=True)
    return -(soft_target * full_prob.clamp_min(1e-8).log()).sum(dim=1).mean()


def structured_losses(outputs, batch, args) -> torch.Tensor:
    if "depth" not in outputs:
        return outputs["point"].sum() * 0.0
    point_mask = batch["point_mask"] > 0.5
    if not point_mask.any():
        return outputs["point"].sum() * 0.0
    depth_logits = outputs["depth"][point_mask]
    side_logits = outputs["side"][point_mask]
    depth_target = batch["point_depth"][point_mask]
    side_target = batch["point_side"][point_mask]
    depth_loss = F.cross_entropy(depth_logits, depth_target)
    side_loss = F.cross_entropy(side_logits, side_target)

    p_direct = F.softmax(outputs["point"][point_mask], dim=-1).reshape(-1, 3, 3)
    depth_marginal = p_direct.sum(dim=2).clamp_min(1e-8)
    side_marginal = p_direct.sum(dim=1).clamp_min(1e-8)
    depth_prob = F.softmax(depth_logits, dim=-1)
    side_prob = F.softmax(side_logits, dim=-1)
    consistency = F.kl_div(depth_prob.clamp_min(1e-8).log(), depth_marginal.detach(), reduction="batchmean")
    consistency = consistency + F.kl_div(side_prob.clamp_min(1e-8).log(), side_marginal.detach(), reduction="batchmean")
    return args.depth_loss_weight * depth_loss + args.side_loss_weight * side_loss + args.consistency_weight * consistency


def supervised_loss(outputs, batch, action_w, point_w, point_counts, point_log_prior, args, device) -> torch.Tensor:
    action_loss = F.cross_entropy(outputs["action"], batch["action"], weight=action_w.to(device))
    terminal_loss = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"])
    p_loss = point_loss(outputs, batch, point_w, point_counts, point_log_prior, args, device)
    distill_loss = soft_point_distill_loss(outputs, batch)
    server_loss_raw = F.binary_cross_entropy_with_logits(outputs["server"], batch["server"], reduction="none")
    server_loss = (server_loss_raw * batch["server_weight"]).mean()
    parity_loss = F.binary_cross_entropy_with_logits(outputs["parity"], batch["parity"])
    remaining_loss = F.cross_entropy(outputs["remaining"], batch["remaining"])
    aux = structured_losses(outputs, batch, args)
    return (
        0.30 * action_loss
        + 0.15 * terminal_loss
        + 0.32 * p_loss
        + args.distill_weight * distill_loss
        + 0.13 * server_loss
        + 0.05 * parity_loss
        + 0.05 * remaining_loss
        + aux
    )


def train_epoch(model, loader, opt, action_w, point_w, point_counts, point_log_prior, args, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["cat"], batch["num"], batch["lengths"])
        loss = supervised_loss(out, batch, action_w, point_w, point_counts, point_log_prior, args, device)
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_model(model, arrays: SequenceArrays, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(StrokePointDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
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


def evaluate(meta, action_prob, point_prob, server_prob, action_mult=None, point_mult=None, mode="global") -> dict[str, float]:
    if action_mult is None:
        action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    else:
        action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, mode)
    if point_mult is None:
        point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]
    else:
        point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, mode)
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
        raise ValueError("Could not align V11 valid rows to V3 OOF.")
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


def point_blend_report(valid_meta, v3_point, v11_point, v3_tuning, args) -> pd.DataFrame:
    rows = []
    for weight in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]:
        prob = blend_probs(v3_point, v11_point, weight)
        base_pred = apply_segmented_multipliers(valid_meta, v3_point, v3_tuning.point_multipliers, POINT_CLASSES, v3_tuning.bins_mode)
        pred_fixed = apply_segmented_multipliers(valid_meta, prob, v3_tuning.point_multipliers, POINT_CLASSES, v3_tuning.bins_mode)
        fixed_f1 = f1_score(valid_meta["next_pointId"], pred_fixed, average="macro", labels=POINT_CLASSES, zero_division=0)
        mult = tune_segmented_multipliers(valid_meta, prob, POINT_CLASSES, "point", args.multiplier_bins)
        pred_tuned = apply_segmented_multipliers(valid_meta, prob, mult, POINT_CLASSES, args.multiplier_bins)
        tuned_f1 = f1_score(valid_meta["next_pointId"], pred_tuned, average="macro", labels=POINT_CLASSES, zero_division=0)
        rows.append(
            {
                "point_v11_weight": weight,
                "fixed_v3_multiplier_f1": float(fixed_f1),
                "fixed_diff_vs_v3": float((pred_fixed != base_pred).mean()),
                "retuned_multiplier_f1": float(tuned_f1),
                "retuned_diff_vs_v3": float((pred_tuned != base_pred).mean()),
            }
        )
    return pd.DataFrame(rows)


def prefix_report(meta, action_prob, point_prob, server_prob) -> pd.DataFrame:
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
        metrics = evaluate(meta.iloc[idx].reset_index(drop=True), action_prob[idx], point_prob[idx], server_prob[idx])
        metrics.update({"prefix_len_bin": label, "count": int(len(idx))})
        rows.append(metrics)
    return pd.DataFrame(rows)


def class_report_point(meta: pd.DataFrame, point_prob: np.ndarray) -> pd.DataFrame:
    y = meta["next_pointId"].to_numpy(dtype=int)
    pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]
    rows = []
    for cls in POINT_CLASSES:
        tp = int(((y == cls) & (pred == cls)).sum())
        pred_count = int((pred == cls).sum())
        support = int((y == cls).sum())
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({"pointId": cls, "support": support, "pred_count": pred_count, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def train_one_fold(train_arrays, valid_arrays, cat_cards, args, fold_seed, action_w, point_w, point_counts, point_log_prior):
    set_seed(fold_seed)
    device = torch.device(args.device)
    structured = args.variant.startswith("structured")
    model = StrokePointTransformer(
        cat_cards,
        len(NUM_FIELDS),
        args.max_len,
        args.d_model,
        args.emb_dim,
        args.numeric_dim,
        args.num_layers,
        args.num_heads,
        args.dropout,
        structured,
    ).to(device)
    train_loader = DataLoader(
        StrokePointDataset(train_arrays),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_point = None, -1.0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, action_w, point_w, point_counts, point_log_prior, args, device)
        _, p, _ = predict_model(model, valid_arrays, args.batch_size, device)
        point_score = f1_score(
            valid_arrays.meta["next_pointId"],
            np.asarray(POINT_CLASSES)[np.argmax(p, axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        )
        print(f"  epoch {epoch:02d}: loss={loss:.5f} point={point_score:.6f}")
        if point_score > best_point + 1e-6:
            best_point = float(point_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_point


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
    prefix_phase = add_lag0_depth_for_prior(
        add_phase_features(add_remaining_bucket(build_train_prefix_table(train, 6)), train)
    )
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards = cat_cardinalities(train, test)
    device = torch.device(args.device)
    print(f"variant={args.variant} device={device} prefix_rows={len(prefix_meta):,}")

    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    test_lengths = test_meta["prefix_len"].to_numpy(dtype=int)
    fold_rows = []
    valid_parts, action_parts, point_parts, server_parts = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        if fold > args.fold_limit:
            break
        tr_ids = set(rally_meta.iloc[tr_idx]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_idx]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        va_sample = sample_validation_prefixes(va_pool, test_lengths, args.seed + fold)
        va_meta = va_pool.loc[va_sample].copy().reset_index(drop=True)
        tr_meta = attach_soft_targets(tr_meta, prefix_phase, args.distill_alpha)
        train_arrays = build_sequence_arrays(train, tr_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        action_w = class_weights(tr_meta["next_actionId"], ACTION_CLASSES)
        point_values = tr_meta[tr_meta["next_pointId"].gt(0)]["next_pointId"] - 1
        if args.variant == "effective_weight":
            point_w = effective_weights(point_values, list(range(9)), beta=0.99)
        else:
            point_w = class_weights(point_values, list(range(9)), beta=args.point_weight_beta)
        counts = point_values.value_counts().reindex(list(range(9)), fill_value=0).to_numpy(dtype=np.float32)
        point_counts = torch.from_numpy(np.maximum(counts, 1.0))
        point_log_prior = class_log_prior(point_values, list(range(9)))
        print(f"fold {fold}: train={len(tr_meta):,} valid={len(va_meta):,}")
        model, best_point = train_one_fold(
            train_arrays,
            valid_arrays,
            cat_cards,
            args,
            args.seed + fold,
            action_w,
            point_w,
            point_counts,
            point_log_prior,
        )
        a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(va_meta, a, p, s)
        metrics.update({"fold": fold, "train_rows": len(tr_meta), "valid_rows": len(va_meta), "best_point_epoch_score": best_point})
        fold_rows.append(metrics)
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)

    valid_meta = pd.concat(valid_parts, ignore_index=True)
    v11_action = np.vstack(action_parts)
    v11_point = np.vstack(point_parts)
    v11_server = np.concatenate(server_parts)
    single = evaluate(valid_meta, v11_action, v11_point, v11_server)
    v3_action, v3_point, v3_server, v3_tuning = load_v3_subset(args.v3_oof, valid_meta)
    blend_df = point_blend_report(valid_meta, v3_point, v11_point, v3_tuning, args)
    blend_df.to_csv(args.point_blend_report, index=False)
    prefix_report(valid_meta, v11_action, v11_point, v11_server).to_csv(args.prefix_len_report, index=False)
    class_report_point(valid_meta, v11_point).to_csv(args.class_report_point, index=False)
    fold_df = pd.DataFrame(fold_rows)
    cv_report = pd.DataFrame(
        [
            {
                "variant": args.variant,
                "folds_run": len(fold_rows),
                **{f"v11_single_{k}": v for k, v in single.items()},
                "best_fixed_blend_point": float(blend_df["fixed_v3_multiplier_f1"].max()),
                "best_retuned_blend_point": float(blend_df["retuned_multiplier_f1"].max()),
            }
        ]
    )
    cv_report.to_csv(args.cv_report, index=False)
    with open(args.oof_proba, "wb") as f:
        pickle.dump(
            {
                "valid_meta": valid_meta,
                "v11_action": v11_action,
                "v11_point": v11_point,
                "v11_server": v11_server,
                "variant": args.variant,
                "fold_report": fold_df,
            },
            f,
        )
    metadata = {
        "args": vars(args),
        "cat_fields": CAT_FIELDS,
        "num_fields": NUM_FIELDS,
        "cat_cardinalities": cat_cards,
        "single": single,
        "point_blend_report": blend_df.to_dict(orient="records"),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(cv_report.iloc[0].to_dict(), indent=2))
    print(f"wrote {args.cv_report}, {args.point_blend_report}, {args.oof_proba}")


if __name__ == "__main__":
    main()
