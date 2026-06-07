"""V5 GRU sequence baseline.

The goal is to build a clean sequence-model pipeline and test whether GRU
probabilities complement the stronger V3 tabular baseline. The script can run
GRU-only CV, optionally load V3 OOF probabilities for ensemble tuning, and
write class/prefix diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_, rnn
from torch.utils.data import DataLoader, Dataset

from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import (
    add_remaining_bucket,
    apply_segmented_multipliers,
    full_predict as v3_full_predict,
    tune_segmented_multipliers,
)


# Compatibility for unpickling oof_proba_v3.pkl, which was written when
# baseline_v3.py ran as __main__.
@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


CAT_FIELDS = [
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "is_server_hitter",
    "sex",
]
NUM_FIELDS = ["serverScore", "receiverScore", "serverScoreDiff", "strikeNumber"]
POINT_NONTERMINAL_CLASSES = list(range(1, 10))
REMAINING_CLASSES = list(range(1, 8))


def compose_v3_predictions(meta: pd.DataFrame, pred: dict[str, np.ndarray], tuning) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del meta
    action_prob = blend_probs(pred["lgbm_action"], pred["ngram_action"], tuning.action_ngram_weight)
    point_prob = blend_probs(pred["lgbm_point"], pred["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server_prob = (
        sw["direct"] * pred["lgbm_server"]
        + sw["ngram"] * pred["ngram_server"]
        + sw["parity"] * pred["parity_server"]
        + sw["remaining"] * pred["remaining_server"]
    )
    return action_prob, point_prob, server_prob


@dataclass
class SequenceArrays:
    cat: np.ndarray
    num: np.ndarray
    lengths: np.ndarray
    meta: pd.DataFrame


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    metrics: dict[str, float]
    bins_mode: str


class StrokeDataset(Dataset):
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
            "action": torch.tensor(int(row.get("next_actionId", 0)), dtype=torch.long),
            "terminal": torch.tensor(float(row.get("next_is_terminal", 0)), dtype=torch.float32),
            "point_nonterminal": torch.tensor(point_nonterminal, dtype=torch.long),
            "point_mask": torch.tensor(float(next_point > 0), dtype=torch.float32),
            "server": torch.tensor(float(row.get("serverGetPoint", 0)), dtype=torch.float32),
            "parity": torch.tensor(float(row.get("final_parity_even", 0)), dtype=torch.float32),
            "remaining": torch.tensor(int(row.get("remaining_len_bucket", 1)) - 1, dtype=torch.long),
            "server_weight": torch.tensor(float(row.get("server_weight", 1.0)), dtype=torch.float32),
        }


class GRUModel(nn.Module):
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
        self.numeric = nn.Sequential(
            nn.Linear(num_dim, numeric_dim),
            nn.LayerNorm(numeric_dim),
            nn.GELU(),
        )
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
        self.action_head = nn.Linear(hidden_dim, 19)
        self.terminal_head = nn.Linear(hidden_dim, 1)
        self.point_head = nn.Linear(hidden_dim, 9)
        self.server_head = nn.Linear(hidden_dim, 1)
        self.parity_head = nn.Linear(hidden_dim, 1)
        self.remaining_head = nn.Linear(hidden_dim, 7)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        embs = [emb(cat[:, :, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [self.numeric(num)], dim=-1)
        x = self.input_proj(x)
        packed = rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        h = self.head(hidden[-1])
        return {
            "action": self.action_head(h),
            "terminal": self.terminal_head(h).squeeze(-1),
            "point": self.point_head(h),
            "server": self.server_head(h).squeeze(-1),
            "parity": self.parity_head(h).squeeze(-1),
            "remaining": self.remaining_head(h),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 GRU sequence baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--tabular-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--submission", default="submission_v5.csv")
    parser.add_argument("--cv-report", default="cv_report_v5.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v5.csv")
    parser.add_argument("--class-report-action", default="class_report_v5_action.csv")
    parser.add_argument("--class-report-point", default="class_report_v5_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v5.json")
    parser.add_argument("--oof-proba", default="oof_proba_v5.pkl")
    parser.add_argument("--test-proba", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--skip-tabular-full", action="store_true")
    parser.add_argument("--skip-full-train", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def class_weights(values: pd.Series, classes: list[int], beta: float = 0.25) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    weights = np.array([float(counts.get(cls, 1)) ** (-beta) for cls in classes], dtype=np.float32)
    weights = weights / weights.mean()
    return torch.from_numpy(weights)


def fit_numeric_stats(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = train[NUM_FIELDS].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def cat_cardinalities(train: pd.DataFrame, test: pd.DataFrame) -> list[int]:
    cards = []
    for field in CAT_FIELDS:
        max_value = int(max(train[field].max(), test[field].max()))
        cards.append(max_value + 2)
    return cards


def build_train_meta(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    for _, group in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        rally_len = len(group)
        if rally_len < 2:
            continue
        server_get_point = int(group.iloc[0]["serverGetPoint"])
        final_parity_even = int(int(group.iloc[-1]["strikeNumber"]) % 2 == 0)
        num_prefixes = rally_len - 1
        for t_idx in range(num_prefixes):
            nxt = group.iloc[t_idx + 1]
            rows.append(
                {
                    "rally_uid": int(group.iloc[0]["rally_uid"]),
                    "match": int(group.iloc[0]["match"]),
                    "prefix_index": int(t_idx),
                    "prefix_len": int(group.iloc[t_idx]["strikeNumber"]),
                    "next_actionId": int(nxt["actionId"]),
                    "next_pointId": int(nxt["pointId"]),
                    "next_is_terminal": int(t_idx + 1 == rally_len - 1),
                    "serverGetPoint": server_get_point,
                    "final_parity_even": final_parity_even,
                    "remaining_len": int(rally_len - (t_idx + 1)),
                    "remaining_len_bucket": int(min(rally_len - (t_idx + 1), 7)),
                    "num_prefixes_in_rally": int(num_prefixes),
                    "server_weight": float(1.0 / num_prefixes),
                }
            )
    meta = pd.DataFrame(rows)
    meta["server_weight"] = meta["server_weight"] / meta["server_weight"].mean()
    return meta


def build_test_meta(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        rows.append(
            {
                "rally_uid": int(group.iloc[0]["rally_uid"]),
                "match": int(group.iloc[0]["match"]),
                "prefix_index": int(len(group) - 1),
                "prefix_len": int(group.iloc[-1]["strikeNumber"]),
            }
        )
    return pd.DataFrame(rows)


def build_sequence_arrays(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    max_len: int,
    num_mean: np.ndarray,
    num_std: np.ndarray,
) -> SequenceArrays:
    groups = {int(rid): group.reset_index(drop=True) for rid, group in df.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False)}
    n = len(meta)
    cat = np.zeros((n, max_len, len(CAT_FIELDS)), dtype=np.int64)
    num = np.zeros((n, max_len, len(NUM_FIELDS)), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    for out_idx, row in enumerate(meta.itertuples(index=False)):
        group = groups[int(row.rally_uid)]
        end = int(row.prefix_index) + 1
        start = max(0, end - max_len)
        seq = group.iloc[start:end]
        seq_len = len(seq)
        lengths[out_idx] = seq_len
        cat_values = seq[CAT_FIELDS].to_numpy(dtype=np.int64) + 1
        num_values = (seq[NUM_FIELDS].to_numpy(dtype=np.float32) - num_mean) / num_std
        cat[out_idx, :seq_len, :] = cat_values
        num[out_idx, :seq_len, :] = num_values
    return SequenceArrays(cat=cat, num=num, lengths=lengths, meta=meta.reset_index(drop=True))


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    action_weights: torch.Tensor,
    point_weights: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    action_loss = F.cross_entropy(outputs["action"], batch["action"], weight=action_weights.to(device))
    terminal_loss = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"])
    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        point_loss = F.cross_entropy(
            outputs["point"][point_mask],
            batch["point_nonterminal"][point_mask],
            weight=point_weights.to(device),
        )
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


def predict_model(model: GRUModel, arrays: SequenceArrays, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(StrokeDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    action_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    server_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            cat = batch["cat"].to(device)
            num = batch["num"].to(device)
            lengths = batch["lengths"].to(device)
            outputs = model(cat, num, lengths)
            action = F.softmax(outputs["action"], dim=-1).cpu().numpy()
            terminal = torch.sigmoid(outputs["terminal"]).cpu().numpy()
            point_nonterm = F.softmax(outputs["point"], dim=-1).cpu().numpy()
            point = np.zeros((len(action), 10), dtype=np.float32)
            point[:, 0] = terminal
            point[:, 1:] = (1.0 - terminal[:, None]) * point_nonterm
            point = point / point.sum(axis=1, keepdims=True)
            server = torch.sigmoid(outputs["server"]).cpu().numpy()
            action_parts.append(action)
            point_parts.append(point)
            server_parts.append(server)
    return np.vstack(action_parts), np.vstack(point_parts), np.concatenate(server_parts)


def evaluate_probs(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    action_multipliers: dict[str, list[float]] | None = None,
    point_multipliers: dict[str, list[float]] | None = None,
    bins_mode: str = "global",
) -> dict[str, float]:
    if action_multipliers is None:
        action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    else:
        action_pred = apply_segmented_multipliers(meta, action_prob, action_multipliers, ACTION_CLASSES, bins_mode)
    if point_multipliers is None:
        point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]
    else:
        point_pred = apply_segmented_multipliers(meta, point_prob, point_multipliers, POINT_CLASSES, bins_mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(overall),
    }


def train_one_fold(
    train_arrays: SequenceArrays,
    valid_arrays: SequenceArrays,
    cat_cards: list[int],
    args: argparse.Namespace,
    fold_seed: int,
    action_weights: torch.Tensor,
    point_weights: torch.Tensor,
) -> tuple[GRUModel, dict[str, float]]:
    device = torch.device(args.device)
    model = GRUModel(
        cat_cards,
        len(NUM_FIELDS),
        args.emb_dim,
        args.numeric_dim,
        args.hidden_dim,
        args.num_layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        StrokeDataset(train_arrays),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    best_state = None
    best_metrics = {"overall": -1.0}
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["cat"], batch["num"], batch["lengths"])
            loss = compute_loss(outputs, batch, action_weights, point_weights, device)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        action_prob, point_prob, server_prob = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate_probs(valid_arrays.meta, action_prob, point_prob, server_prob)
        print(
            f"  epoch {epoch:02d}: loss={np.mean(losses):.5f} overall={metrics['overall']:.6f} "
            f"action={metrics['action_macro_f1']:.6f} point={metrics['point_macro_f1']:.6f} "
            f"server={metrics['server_auc']:.6f}"
        )
        if metrics["overall"] > best_metrics["overall"] + 1e-6:
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def sample_valid_meta(prefix_meta: pd.DataFrame, test_lengths: np.ndarray, seed: int) -> pd.DataFrame:
    sampled_idx = sample_validation_prefixes(prefix_meta, test_lengths, seed)
    return prefix_meta.loc[sampled_idx].copy().reset_index(drop=True)


def run_cv(
    train: pd.DataFrame,
    prefix_meta: pd.DataFrame,
    test_lengths: np.ndarray,
    cat_cards: list[int],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    valid_parts: list[pd.DataFrame] = []
    action_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    server_parts: list[np.ndarray] = []
    fold_rows: list[dict[str, float | int]] = []
    device = torch.device(args.device)
    for fold, (train_rally_idx, valid_rally_idx) in enumerate(
        splitter.split(rally_meta, groups=rally_meta["match"]), start=1
    ):
        train_rallies = set(rally_meta.iloc[train_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_rally_idx]["rally_uid"])
        fold_train_meta = prefix_meta[prefix_meta["rally_uid"].isin(train_rallies)].copy().reset_index(drop=True)
        valid_pool = prefix_meta[prefix_meta["rally_uid"].isin(valid_rallies)].copy()
        fold_valid_meta = sample_valid_meta(valid_pool, test_lengths, args.seed + fold)
        action_w = class_weights(fold_train_meta["next_actionId"], ACTION_CLASSES)
        point_w = class_weights(
            fold_train_meta[fold_train_meta["next_pointId"].gt(0)]["next_pointId"] - 1,
            list(range(9)),
        )
        train_arrays = build_sequence_arrays(train, fold_train_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, fold_valid_meta, args.max_len, num_mean, num_std)
        print(f"fold {fold}: train={len(fold_train_meta)} valid={len(fold_valid_meta)}")
        model, best_metrics = train_one_fold(
            train_arrays,
            valid_arrays,
            cat_cards,
            args,
            args.seed + fold * 10,
            action_w,
            point_w,
        )
        action_prob, point_prob, server_prob = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate_probs(fold_valid_meta, action_prob, point_prob, server_prob)
        metrics.update({"fold": fold, "train_rows": len(fold_train_meta), "valid_rows": len(fold_valid_meta)})
        fold_rows.append(metrics)
        valid_parts.append(fold_valid_meta)
        action_parts.append(action_prob)
        point_parts.append(point_prob)
        server_parts.append(server_prob)
    valid_meta = pd.concat(valid_parts, ignore_index=True)
    fold_report = pd.DataFrame(fold_rows)
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0}
    for col in ["action_macro_f1", "point_macro_f1", "server_auc", "overall"]:
        mean_row[col] = float(fold_report[col].mean())
    fold_report = pd.concat([fold_report, pd.DataFrame([mean_row])], ignore_index=True)
    return {
        "valid_meta": valid_meta,
        "gru_action": np.vstack(action_parts),
        "gru_point": np.vstack(point_parts),
        "gru_server": np.concatenate(server_parts),
        "fold_report": fold_report,
    }


def load_tabular_oof(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray] | None:
    if not path or not Path(path).exists():
        return None
    with open(path, "rb") as f:
        oof = pickle.load(f)
    tuning = oof.get("tuning")
    action_prob, point_prob, server_prob = compose_v3_predictions(oof["valid_meta"], oof, tuning)
    return oof["valid_meta"], action_prob, point_prob, server_prob


def tune_gru_ensemble(
    valid_meta: pd.DataFrame,
    gru_action: np.ndarray,
    gru_point: np.ndarray,
    gru_server: np.ndarray,
    args: argparse.Namespace,
    tabular: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[GrUTuning, np.ndarray, np.ndarray, np.ndarray]:
    if tabular is None:
        action_base, point_base, server_base = gru_action, gru_point, gru_server
        aw = pw = sw = 1.0
    else:
        tab_meta, tab_action, tab_point, tab_server = tabular
        if not valid_meta[["rally_uid", "prefix_len"]].reset_index(drop=True).equals(
            tab_meta[["rally_uid", "prefix_len"]].reset_index(drop=True)
        ):
            raise ValueError("GRU OOF rows do not align with tabular OOF rows.")
        grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        aw = max(
            grid,
            key=lambda w: f1_score(
                valid_meta["next_actionId"],
                np.asarray(ACTION_CLASSES)[np.argmax(blend_probs(tab_action, gru_action, w), axis=1)],
                average="macro",
                labels=ACTION_CLASSES,
                zero_division=0,
            ),
        )
        pw = max(
            grid,
            key=lambda w: f1_score(
                valid_meta["next_pointId"],
                np.asarray(POINT_CLASSES)[np.argmax(blend_probs(tab_point, gru_point, w), axis=1)],
                average="macro",
                labels=POINT_CLASSES,
                zero_division=0,
            ),
        )
        sw = max(grid, key=lambda w: roc_auc_score(valid_meta["serverGetPoint"], (1.0 - w) * tab_server + w * gru_server))
        action_base = blend_probs(tab_action, gru_action, aw)
        point_base = blend_probs(tab_point, gru_point, pw)
        server_base = (1.0 - sw) * tab_server + sw * gru_server
    action_mult = tune_segmented_multipliers(valid_meta, action_base, ACTION_CLASSES, "action", args.multiplier_bins)
    point_mult = tune_segmented_multipliers(valid_meta, point_base, POINT_CLASSES, "point", args.multiplier_bins)
    metrics = evaluate_probs(valid_meta, action_base, point_base, server_base, action_mult, point_mult, args.multiplier_bins)
    tuning = GrUTuning(float(aw), float(pw), float(sw), action_mult, point_mult, metrics, args.multiplier_bins)
    return tuning, action_base, point_base, server_base


def write_reports(
    valid_meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: GrUTuning,
    args: argparse.Namespace,
) -> None:
    action_pred = apply_segmented_multipliers(
        valid_meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode
    )
    point_pred = apply_segmented_multipliers(
        valid_meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode
    )
    action_report = pd.DataFrame(classification_report(
        valid_meta["next_actionId"], action_pred, labels=ACTION_CLASSES, zero_division=0, output_dict=True
    )).T
    point_report = pd.DataFrame(classification_report(
        valid_meta["next_pointId"], point_pred, labels=POINT_CLASSES, zero_division=0, output_dict=True
    )).T
    action_report.to_csv(args.class_report_action)
    point_report.to_csv(args.class_report_point)
    rows = []
    for label, mask in [
        ("1", valid_meta["prefix_len"].eq(1).to_numpy()),
        ("2", valid_meta["prefix_len"].eq(2).to_numpy()),
        ("3", valid_meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", valid_meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", valid_meta["prefix_len"].ge(7).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        metrics = evaluate_probs(
            valid_meta.iloc[idx].reset_index(drop=True),
            action_prob[idx],
            point_prob[idx],
            server_prob[idx],
            tuning.action_multipliers,
            tuning.point_multipliers,
            tuning.bins_mode,
        )
        metrics.update({"prefix_len_bin": label, "count": int(len(idx))})
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(args.prefix_len_report, index=False)


def train_full_gru(
    train: pd.DataFrame,
    prefix_meta: pd.DataFrame,
    test_arrays: SequenceArrays,
    cat_cards: list[int],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_arrays = build_sequence_arrays(train, prefix_meta, args.max_len, num_mean, num_std)
    action_w = class_weights(prefix_meta["next_actionId"], ACTION_CLASSES)
    point_w = class_weights(prefix_meta[prefix_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
    # Use a validation slice for early stopping proxy to avoid training indefinitely on full data.
    val_size = min(3000, len(prefix_meta) // 10)
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(len(prefix_meta), size=val_size, replace=False)
    val_arrays = SequenceArrays(
        cat=train_arrays.cat[val_idx],
        num=train_arrays.num[val_idx],
        lengths=train_arrays.lengths[val_idx],
        meta=prefix_meta.iloc[val_idx].reset_index(drop=True),
    )
    model, _ = train_one_fold(train_arrays, val_arrays, cat_cards, args, args.seed, action_w, point_w)
    return predict_model(model, test_arrays, args.batch_size, torch.device(args.device))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_meta = add_remaining_bucket(build_train_meta(train))
    test_meta = build_test_meta(test)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards = cat_cardinalities(train, test)
    print(f"device: {args.device}")
    print(f"train prefix rows: {len(prefix_meta):,}")
    print(f"test rows: {len(test_meta):,}")

    oof = run_cv(train, prefix_meta, test_meta["prefix_len"].to_numpy(dtype=int), cat_cards, num_mean, num_std, args)
    tabular = load_tabular_oof(args.tabular_oof)
    tuning, final_action_oof, final_point_oof, final_server_oof = tune_gru_ensemble(
        oof["valid_meta"], oof["gru_action"], oof["gru_point"], oof["gru_server"], args, tabular
    )
    gru_single = evaluate_probs(oof["valid_meta"], oof["gru_action"], oof["gru_point"], oof["gru_server"])
    fold_report = oof["fold_report"].copy()
    for key, value in gru_single.items():
        fold_report[f"gru_single_{key}"] = value
    for key, value in tuning.metrics.items():
        fold_report[f"selected_{key}"] = value
    fold_report["selected_action_gru_weight"] = tuning.action_gru_weight
    fold_report["selected_point_gru_weight"] = tuning.point_gru_weight
    fold_report["selected_server_gru_weight"] = tuning.server_gru_weight
    fold_report.to_csv(args.cv_report, index=False)
    write_reports(oof["valid_meta"], final_action_oof, final_point_oof, final_server_oof, tuning, args)
    with open(args.oof_proba, "wb") as f:
        pickle.dump({**oof, "tuning": tuning}, f)

    print("selected tuning:")
    print(json.dumps({
        "gru_single": gru_single,
        "action_gru_weight": tuning.action_gru_weight,
        "point_gru_weight": tuning.point_gru_weight,
        "server_gru_weight": tuning.server_gru_weight,
        **tuning.metrics,
    }, indent=2))

    submission_rows = 0
    if not args.skip_full_train:
        test_arrays = build_sequence_arrays(test, test_meta, args.max_len, num_mean, num_std)
        gru_action_test, gru_point_test, gru_server_test = train_full_gru(
            train, prefix_meta, test_arrays, cat_cards, num_mean, num_std, args
        )
        if tabular is not None and not args.skip_tabular_full:
            tab_prefix = add_remaining_bucket(build_train_prefix_table(train, 6))
            full_test_prefix = build_test_prefix_table(test, 6)
            full_features = [c for c in feature_columns(tab_prefix) if c != "remaining_len_bucket"]
            full_test_prefix = full_test_prefix[["rally_uid", "match"] + full_features]
            v3_args = SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0)
            with open(args.tabular_oof, "rb") as f:
                tab_oof = pickle.load(f)
            tab_pred = v3_full_predict(tab_prefix, full_test_prefix, full_features, v3_args)
            tab_action_test, tab_point_test, tab_server_test = compose_v3_predictions(
                full_test_prefix, tab_pred, tab_oof["tuning"]
            )
            action_test = blend_probs(tab_action_test, gru_action_test, tuning.action_gru_weight)
            point_test = blend_probs(tab_point_test, gru_point_test, tuning.point_gru_weight)
            server_test = (1.0 - tuning.server_gru_weight) * tab_server_test + tuning.server_gru_weight * gru_server_test
        else:
            action_test, point_test, server_test = gru_action_test, gru_point_test, gru_server_test
        action_pred = apply_segmented_multipliers(
            test_meta, action_test, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode
        )
        point_pred = apply_segmented_multipliers(
            test_meta, point_test, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode
        )
        submission = pd.DataFrame({
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_test, 1e-6, 1.0 - 1e-6), 8),
        })
        submission.to_csv(args.submission, index=False, float_format="%.8f")
        submission_rows = len(submission)
        if args.test_proba:
            with open(args.test_proba, "wb") as f:
                pickle.dump(
                    {
                        "test_meta": test_meta.copy(),
                        "gru_action": gru_action_test,
                        "gru_point": gru_point_test,
                        "gru_server": gru_server_test,
                        "action": action_test,
                        "point": point_test,
                        "server": server_test,
                        "tuning": tuning,
                    },
                    f,
                )

    metadata = {
        "cat_fields": CAT_FIELDS,
        "num_fields": NUM_FIELDS,
        "cat_cardinalities": cat_cards,
        "num_mean": num_mean.tolist(),
        "num_std": num_std.tolist(),
        "args": vars(args),
        "selected": {
            "gru_single": gru_single,
            "action_gru_weight": tuning.action_gru_weight,
            "point_gru_weight": tuning.point_gru_weight,
            "server_gru_weight": tuning.server_gru_weight,
            "action_multipliers": tuning.action_multipliers,
            "point_multipliers": tuning.point_multipliers,
            "metrics": tuning.metrics,
        },
    }
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
