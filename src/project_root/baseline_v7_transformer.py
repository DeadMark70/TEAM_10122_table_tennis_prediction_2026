"""V7 small Transformer encoder sequence baseline.

This builds on the V5 sequence pipeline, replacing GRU with a compact
Transformer encoder. It evaluates both Transformer single-model probabilities
and V3 + Transformer OOF ensemble probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, sample_validation_prefixes, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import (
    CAT_FIELDS,
    NUM_FIELDS,
    GrUTuning,
    SequenceArrays,
    build_sequence_arrays,
    build_test_meta,
    build_train_meta,
    cat_cardinalities,
    class_weights,
    fit_numeric_stats,
    load_tabular_oof,
)


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
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    metrics: dict[str, float]
    bins_mode: str


class TransformerDataset(Dataset):
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
        return {
            "cat": self.cat[idx],
            "num": self.num[idx],
            "lengths": self.lengths[idx],
            "prefix_len": torch.tensor(int(row.get("prefix_len", self.lengths[idx].item())), dtype=torch.long),
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point_nonterminal": torch.tensor(point_nonterminal, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
        }


class TransformerModel(nn.Module):
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
    ) -> None:
        super().__init__()
        self.max_len = max_len
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
        return {
            "action": self.action_head(h),
            "terminal": self.terminal_head(h).squeeze(-1),
            "point": self.point_head(h),
            "server": self.server_head(h).squeeze(-1),
            "parity": self.parity_head(h).squeeze(-1),
            "remaining": self.remaining_head(h),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V7 Transformer sequence baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--tabular-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--submission", default="submission_v7.csv")
    parser.add_argument("--cv-report", default="cv_report_v7.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v7.csv")
    parser.add_argument("--class-report-action", default="class_report_v7_action.csv")
    parser.add_argument("--class-report-point", default="class_report_v7_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v7.json")
    parser.add_argument("--oof-proba", default="oof_proba_v7.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--short-point-weight", type=float, default=1.5)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--point-beta", type=float, default=0.35)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--skip-full-train", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def weighted_classes(values: pd.Series, classes: list[int], beta: float) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    weights = np.array([float(counts.get(cls, 1)) ** (-beta) for cls in classes], dtype=np.float32)
    return torch.from_numpy(weights / weights.mean())


def compute_loss(outputs, batch, action_weights, point_weights, args, device):
    action_loss = F.cross_entropy(outputs["action"], batch["action"], weight=action_weights.to(device))
    terminal_loss = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"])
    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        point_loss_raw = F.cross_entropy(
            outputs["point"][point_mask],
            batch["point_nonterminal"][point_mask],
            weight=point_weights.to(device),
            reduction="none",
        )
        short_weight = torch.where(batch["prefix_len"][point_mask] <= 2, args.short_point_weight, 1.0)
        point_loss = (point_loss_raw * short_weight).mean()
    else:
        point_loss = outputs["point"].sum() * 0.0
    server_loss_raw = F.binary_cross_entropy_with_logits(outputs["server"], batch["server"], reduction="none")
    server_loss = (server_loss_raw * batch["server_weight"]).mean()
    parity_loss = F.binary_cross_entropy_with_logits(outputs["parity"], batch["parity"])
    remaining_loss = F.cross_entropy(outputs["remaining"], batch["remaining"])
    return 0.35 * action_loss + 0.15 * terminal_loss + 0.25 * point_loss + 0.15 * server_loss + 0.05 * parity_loss + 0.05 * remaining_loss


def predict_model(model, arrays, batch_size, device):
    loader = DataLoader(TransformerDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
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


def evaluate(meta, action_prob, point_prob, server_prob, action_mult=None, point_mult=None, bins_mode="global"):
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


def train_fold(train_arrays, valid_arrays, cat_cards, action_w, point_w, args, seed):
    device = torch.device(args.device)
    model = TransformerModel(cat_cards, len(NUM_FIELDS), args.max_len, args.d_model, args.emb_dim, args.numeric_dim, args.num_layers, args.num_heads, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TransformerDataset(train_arrays), batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    best_state, best_metrics = None, {"overall": -1.0}
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch["cat"], batch["num"], batch["lengths"])
            loss = compute_loss(out, batch, action_w, point_w, args, device)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(valid_arrays.meta, a, p, s)
        print(f"  epoch {epoch:02d}: loss={np.mean(losses):.5f} overall={metrics['overall']:.6f} action={metrics['action_macro_f1']:.6f} point={metrics['point_macro_f1']:.6f} server={metrics['server_auc']:.6f}")
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
    return model, best_metrics


def sample_valid(prefix_meta, test_lengths, seed):
    idx = sample_validation_prefixes(prefix_meta, test_lengths, seed)
    return prefix_meta.loc[idx].copy().reset_index(drop=True)


def run_cv(train, prefix_meta, test_lengths, cat_cards, num_mean, num_std, args):
    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    valid_parts, action_parts, point_parts, server_parts, fold_rows = [], [], [], [], []
    for fold, (tr_r, va_r) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        tr_ids = set(rally_meta.iloc[tr_r]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_r]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        va_meta = sample_valid(va_pool, test_lengths, args.seed + fold)
        train_arrays = build_sequence_arrays(train, tr_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        action_w = weighted_classes(tr_meta["next_actionId"], ACTION_CLASSES, args.action_beta)
        point_w = weighted_classes(tr_meta[tr_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)), args.point_beta)
        print(f"fold {fold}: train={len(tr_meta)} valid={len(va_meta)}")
        model, _ = train_fold(train_arrays, valid_arrays, cat_cards, action_w, point_w, args, args.seed + fold * 10)
        a, p, s = predict_model(model, valid_arrays, args.batch_size, torch.device(args.device))
        m = evaluate(va_meta, a, p, s)
        m.update({"fold": fold, "train_rows": len(tr_meta), "valid_rows": len(va_meta)})
        fold_rows.append(m)
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)
    valid_meta = pd.concat(valid_parts, ignore_index=True)
    report = pd.DataFrame(fold_rows)
    mean = {"fold": 0, "train_rows": 0, "valid_rows": 0}
    for c in ["action_macro_f1", "point_macro_f1", "server_auc", "overall"]:
        mean[c] = float(report[c].mean())
    report = pd.concat([report, pd.DataFrame([mean])], ignore_index=True)
    return {
        "valid_meta": valid_meta,
        "tr_action": np.vstack(action_parts),
        "tr_point": np.vstack(point_parts),
        "tr_server": np.concatenate(server_parts),
        "fold_report": report,
    }


def tune_ensemble(valid_meta, tr_action, tr_point, tr_server, tabular, args):
    if tabular is None:
        action_base, point_base, server_base = tr_action, tr_point, tr_server
        aw = pw = sw = 1.0
    else:
        tab_meta, tab_action, tab_point, tab_server = tabular
        if not valid_meta[["rally_uid", "prefix_len"]].reset_index(drop=True).equals(tab_meta[["rally_uid", "prefix_len"]].reset_index(drop=True)):
            raise ValueError("Transformer and tabular OOF rows do not align.")
        grid_ap = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        grid_s = [0.0, 0.1, 0.2, 0.3]
        aw = max(grid_ap, key=lambda w: f1_score(valid_meta["next_actionId"], np.asarray(ACTION_CLASSES)[np.argmax(blend_probs(tab_action, tr_action, w), axis=1)], average="macro", labels=ACTION_CLASSES, zero_division=0))
        pw = max(grid_ap, key=lambda w: f1_score(valid_meta["next_pointId"], np.asarray(POINT_CLASSES)[np.argmax(blend_probs(tab_point, tr_point, w), axis=1)], average="macro", labels=POINT_CLASSES, zero_division=0))
        sw = max(grid_s, key=lambda w: roc_auc_score(valid_meta["serverGetPoint"], (1.0 - w) * tab_server + w * tr_server))
        action_base = blend_probs(tab_action, tr_action, aw)
        point_base = blend_probs(tab_point, tr_point, pw)
        server_base = (1.0 - sw) * tab_server + sw * tr_server
    action_mult = tune_segmented_multipliers(valid_meta, action_base, ACTION_CLASSES, "action", args.multiplier_bins)
    point_mult = tune_segmented_multipliers(valid_meta, point_base, POINT_CLASSES, "point", args.multiplier_bins)
    metrics = evaluate(valid_meta, action_base, point_base, server_base, action_mult, point_mult, args.multiplier_bins)
    return TransformerTuning(float(aw), float(pw), float(sw), action_mult, point_mult, metrics, args.multiplier_bins), action_base, point_base, server_base


def write_reports(meta, action_prob, point_prob, server_prob, tuning, args):
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    pd.DataFrame(classification_report(meta["next_actionId"], action_pred, labels=ACTION_CLASSES, zero_division=0, output_dict=True)).T.to_csv(args.class_report_action)
    pd.DataFrame(classification_report(meta["next_pointId"], point_pred, labels=POINT_CLASSES, zero_division=0, output_dict=True)).T.to_csv(args.class_report_point)
    rows = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx):
            m = evaluate(meta.iloc[idx].reset_index(drop=True), action_prob[idx], point_prob[idx], server_prob[idx], tuning.action_multipliers, tuning.point_multipliers, tuning.bins_mode)
            m.update({"prefix_len_bin": label, "count": int(len(idx))})
            rows.append(m)
    pd.DataFrame(rows).to_csv(args.prefix_len_report, index=False)


def train_full_transformer(train, prefix_meta, test_arrays, cat_cards, num_mean, num_std, args):
    arrays = build_sequence_arrays(train, prefix_meta, args.max_len, num_mean, num_std)
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(len(prefix_meta), size=min(3000, len(prefix_meta) // 10), replace=False)
    val_arrays = SequenceArrays(arrays.cat[val_idx], arrays.num[val_idx], arrays.lengths[val_idx], prefix_meta.iloc[val_idx].reset_index(drop=True))
    action_w = weighted_classes(prefix_meta["next_actionId"], ACTION_CLASSES, args.action_beta)
    point_w = weighted_classes(prefix_meta[prefix_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)), args.point_beta)
    model, _ = train_fold(arrays, val_arrays, cat_cards, action_w, point_w, args, args.seed)
    return predict_model(model, test_arrays, args.batch_size, torch.device(args.device))


def main():
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
    cat_cards = cat_cardinalities(train, test)
    print(f"device: {args.device}")
    print(f"train prefix rows: {len(prefix_meta):,}")
    oof = run_cv(train, prefix_meta, test_meta["prefix_len"].to_numpy(dtype=int), cat_cards, num_mean, num_std, args)
    tabular = load_tabular_oof(args.tabular_oof)
    tuning, final_action, final_point, final_server = tune_ensemble(oof["valid_meta"], oof["tr_action"], oof["tr_point"], oof["tr_server"], tabular, args)
    single = evaluate(oof["valid_meta"], oof["tr_action"], oof["tr_point"], oof["tr_server"])
    report = oof["fold_report"].copy()
    for k, v in single.items():
        report[f"transformer_single_{k}"] = v
    for k, v in tuning.metrics.items():
        report[f"selected_{k}"] = v
    report["selected_action_transformer_weight"] = tuning.action_weight
    report["selected_point_transformer_weight"] = tuning.point_weight
    report["selected_server_transformer_weight"] = tuning.server_weight
    report.to_csv(args.cv_report, index=False)
    write_reports(oof["valid_meta"], final_action, final_point, final_server, tuning, args)
    with open(args.oof_proba, "wb") as f:
        pickle.dump({**oof, "tuning": tuning}, f)
    print("selected tuning:")
    print(json.dumps({"single": single, "action_transformer_weight": tuning.action_weight, "point_transformer_weight": tuning.point_weight, "server_transformer_weight": tuning.server_weight, **tuning.metrics}, indent=2))
    submission_rows = 0
    if not args.skip_full_train:
        test_arrays = build_sequence_arrays(test, test_meta, args.max_len, num_mean, num_std)
        tr_action_test, tr_point_test, tr_server_test = train_full_transformer(train, prefix_meta, test_arrays, cat_cards, num_mean, num_std, args)
        if tabular is not None:
            # For submission, use V3-equivalent action/server from the currently best full model.
            # We only generate a true V7 submission if the OOF-selected transformer weights are nonzero.
            from baseline_lgbm import build_test_prefix_table, build_train_prefix_table, feature_columns
            from baseline_v3 import add_remaining_bucket, full_predict as v3_full_predict
            from baseline_v6_point_stack import compose_v3_predictions
            with open(args.tabular_oof, "rb") as f:
                tab_oof = pickle.load(f)
            prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
            full_test_prefix = build_test_prefix_table(test, 6)
            full_features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
            full_test_prefix = full_test_prefix[["rally_uid", "match"] + full_features]
            v3_args = type("A", (), {"seeds": [42], "n_estimators": 120, "ngram_alpha": 20.0})()
            tab_pred = v3_full_predict(prefix_df, full_test_prefix, full_features, v3_args)
            tab_action, tab_point, tab_server = compose_v3_predictions(tab_pred, tab_oof["tuning"])
            action_test = blend_probs(tab_action, tr_action_test, tuning.action_weight)
            point_test = blend_probs(tab_point, tr_point_test, tuning.point_weight)
            server_test = (1.0 - tuning.server_weight) * tab_server + tuning.server_weight * tr_server_test
        else:
            action_test, point_test, server_test = tr_action_test, tr_point_test, tr_server_test
        action_pred = apply_segmented_multipliers(test_meta, action_test, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
        point_pred = apply_segmented_multipliers(test_meta, point_test, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
        sub = pd.DataFrame({"rally_uid": test_meta["rally_uid"].astype(int), "actionId": action_pred.astype(int), "pointId": point_pred.astype(int), "serverGetPoint": np.round(np.clip(server_test, 1e-6, 1.0 - 1e-6), 8)})
        sub.to_csv(args.submission, index=False, float_format="%.8f")
        submission_rows = len(sub)
    metadata = {"args": vars(args), "cat_fields": CAT_FIELDS, "num_fields": NUM_FIELDS, "cat_cardinalities": cat_cards, "num_mean": num_mean.tolist(), "num_std": num_std.tolist(), "selected": {"single": single, "action_transformer_weight": tuning.action_weight, "point_transformer_weight": tuning.point_weight, "server_transformer_weight": tuning.server_weight, "metrics": tuning.metrics, "action_multipliers": tuning.action_multipliers, "point_multipliers": tuning.point_multipliers}}
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.class_report_action}")
    print(f"wrote {args.class_report_point}")
    print(f"wrote {args.oof_proba}")
    if args.skip_full_train:
        print("skipped full training/submission")
    else:
        print(f"wrote {args.submission} ({submission_rows:,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
