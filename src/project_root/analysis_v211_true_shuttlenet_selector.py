"""V211 true-ShuttleNet action selector features.

This is the first staged port of the ShuttleNet-specific parts that V208 did
not implement:

- TPE: split each prefix into target-hitter and receiver subsequences and
  encode them with a shared player GRU.
- PGFN: fuse rally, hitter, receiver, type-area, and static contexts with
  information weights (alpha) and position/static weights (beta).

The raw neural action decoder is diagnostic only.  The exported submissions use
V211 probabilities and PGFN gates as features in the V209 anchor-relative action
selector.  Point remains V188 cap5 and server remains R121.

No ShuttleSet, CoachAI, or TTMATCH rows are read.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from analysis_r166_teacher_distillation_system import load_pickle
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import MATCH_COLS, distribution_match_weights, prepare_data
from analysis_v208_action_ttshuttlenet import action_family_targets, action_loss, weak_action_mask
from analysis_v209_action_selector_reranker import (
    V3Tuning,
    GrUTuning,
    TransformerTuning,
    action_point_compatibility,
    add_probability_features,
    best_non_anchor_by_score,
    build_action_candidate_frame,
    distill_v173_soft_anchor,
    evaluate_candidate,
    load_point_anchor_labels,
    rebuild_r166_best_action,
    rebuild_r184_sources,
    select_capped_action_changes,
    source_probs_for_selector,
    topk_labels,
)
from analysis_v188_point_intent_gru import set_seed
from baseline_lgbm import ACTION_CLASSES
from train_v203_tt_shuttlenet import AREA_IDXS, TYPE_IDXS


OUTDIR = Path("v211_true_shuttlenet_selector")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v211_true_shuttlenet_selector.py")

POINT_ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SERVER_ANCHOR = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"

BATCH_SIZE = 512
EPOCHS = 7
PATIENCE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SELECTOR_CAPS = [0.005, 0.01]
CONTEXT_NAMES = ["rally", "hitter", "receiver", "type_area", "static"]


@dataclass
class TPEGateBatch:
    type_strokes: torch.Tensor
    area_strokes: torch.Tensor
    hitter_type: torch.Tensor
    hitter_area: torch.Tensor
    receiver_type: torch.Tensor
    receiver_area: torch.Tensor
    lengths: torch.Tensor
    hitter_lengths: torch.Tensor
    receiver_lengths: torch.Tensor
    static: torch.Tensor
    action: torch.Tensor
    point: torch.Tensor
    family: torch.Tensor
    terminal: torch.Tensor


def split_player_subsequences(strokes: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split prefix strokes into next-hitter and receiver past subsequences.

    Table-tennis strokes alternate.  If prefix length is L, the next hitter has
    parity L % 2, so their previous strokes are prefix positions with that
    parity.  Empty subsequences are padded with one zero row so GRU packing is
    well-defined.
    """
    arr = np.asarray(strokes)
    lens = np.asarray(lengths, dtype=int)
    max_h = max(1, int(np.ceil(arr.shape[1] / 2)))
    max_r = max_h
    hitter = np.zeros((arr.shape[0], max_h, arr.shape[2]), dtype=arr.dtype)
    receiver = np.zeros((arr.shape[0], max_r, arr.shape[2]), dtype=arr.dtype)
    hitter_len = np.ones(arr.shape[0], dtype=np.int64)
    receiver_len = np.ones(arr.shape[0], dtype=np.int64)
    for i, length in enumerate(lens):
        length = int(max(0, min(length, arr.shape[1])))
        target_parity = length % 2
        h_idx = [j for j in range(length) if j % 2 == target_parity]
        r_idx = [j for j in range(length) if j % 2 != target_parity]
        if h_idx:
            hitter_len[i] = len(h_idx)
            hitter[i, : len(h_idx)] = arr[i, h_idx]
        if r_idx:
            receiver_len[i] = len(r_idx)
            receiver[i, : len(r_idx)] = arr[i, r_idx]
    return hitter, receiver, hitter_len, receiver_len


def combine_pgfn_contexts(contexts: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = alpha * beta.clamp_min(1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    fused = torch.sum(contexts * weights.unsqueeze(-1), dim=1)
    return fused, weights


class TPEGateDataset(Dataset):
    def __init__(self, strokes: np.ndarray, lengths: np.ndarray, static: np.ndarray, action: np.ndarray, point: np.ndarray):
        type_strokes = strokes[:, :, TYPE_IDXS]
        area_strokes = strokes[:, :, AREA_IDXS]
        hitter_type, receiver_type, hitter_len, receiver_len = split_player_subsequences(type_strokes, lengths)
        hitter_area, receiver_area, _, _ = split_player_subsequences(area_strokes, lengths)
        self.type_strokes = torch.as_tensor(type_strokes, dtype=torch.long)
        self.area_strokes = torch.as_tensor(area_strokes, dtype=torch.long)
        self.hitter_type = torch.as_tensor(hitter_type, dtype=torch.long)
        self.hitter_area = torch.as_tensor(hitter_area, dtype=torch.long)
        self.receiver_type = torch.as_tensor(receiver_type, dtype=torch.long)
        self.receiver_area = torch.as_tensor(receiver_area, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.hitter_lengths = torch.as_tensor(hitter_len, dtype=torch.long)
        self.receiver_lengths = torch.as_tensor(receiver_len, dtype=torch.long)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.action = torch.as_tensor(np.asarray(action, dtype=np.int64).copy(), dtype=torch.long)
        self.point = torch.as_tensor(np.asarray(point, dtype=np.int64).copy(), dtype=torch.long)
        self.family = torch.as_tensor(action_family_targets(action), dtype=torch.long)
        self.terminal = torch.as_tensor((np.asarray(point, dtype=int) == 0).astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int) -> TPEGateBatch:
        return TPEGateBatch(
            self.type_strokes[idx],
            self.area_strokes[idx],
            self.hitter_type[idx],
            self.hitter_area[idx],
            self.receiver_type[idx],
            self.receiver_area[idx],
            self.lengths[idx],
            self.hitter_lengths[idx],
            self.receiver_lengths[idx],
            self.static[idx],
            self.action[idx],
            self.point[idx],
            self.family[idx],
            self.terminal[idx],
        )


def collate(batch: list[TPEGateBatch]) -> TPEGateBatch:
    return TPEGateBatch(*(torch.stack([getattr(b, field) for b in batch]) for field in TPEGateBatch.__dataclass_fields__))


class TrueShuttleActionNet(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        type_vocab = [vocab_sizes[i] for i in TYPE_IDXS]
        area_vocab = [vocab_sizes[i] for i in AREA_IDXS]
        self.type_emb = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in type_vocab])
        self.area_emb = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in area_vocab])
        stroke_dim = emb_dim * (len(type_vocab) + len(area_vocab))
        self.rally_gru = nn.GRU(stroke_dim, hidden, batch_first=True)
        self.player_gru = nn.GRU(stroke_dim, hidden, batch_first=True)
        self.type_gru = nn.GRU(emb_dim * len(type_vocab), hidden, batch_first=True)
        self.area_gru = nn.GRU(emb_dim * len(area_vocab), hidden, batch_first=True)
        self.type_area_proj = nn.Linear(hidden * 2, hidden)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, hidden), nn.ReLU())
        self.alpha_net = nn.Sequential(nn.Linear(hidden * 5, 64), nn.ReLU(), nn.Linear(64, 5), nn.Softmax(dim=1))
        self.beta_net = nn.Sequential(nn.Linear(hidden + 1, 32), nn.ReLU(), nn.Linear(32, 5), nn.Sigmoid())
        self.shared = nn.Sequential(nn.Linear(hidden, 96), nn.ReLU(), nn.Dropout(0.15))
        self.action = nn.Linear(96, 19)
        self.family = nn.Linear(96, 5)
        self.point = nn.Linear(96, 10)
        self.terminal = nn.Linear(96, 2)

    def embed_type(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(x[:, :, i]) for i, emb in enumerate(self.type_emb)], dim=2)

    def embed_area(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(x[:, :, i]) for i, emb in enumerate(self.area_emb)], dim=2)

    def embed_stroke(self, type_x: torch.Tensor, area_x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.embed_type(type_x), self.embed_area(area_x)], dim=2)

    @staticmethod
    def encode(gru: nn.GRU, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = gru(packed)
        return h[-1]

    def forward(self, batch: TPEGateBatch) -> dict[str, torch.Tensor]:
        type_strokes = batch.type_strokes.to(DEVICE)
        area_strokes = batch.area_strokes.to(DEVICE)
        lengths = batch.lengths.to(DEVICE)
        hitter_type = batch.hitter_type.to(DEVICE)
        hitter_area = batch.hitter_area.to(DEVICE)
        receiver_type = batch.receiver_type.to(DEVICE)
        receiver_area = batch.receiver_area.to(DEVICE)
        hitter_lengths = batch.hitter_lengths.to(DEVICE)
        receiver_lengths = batch.receiver_lengths.to(DEVICE)
        static = batch.static.to(DEVICE)

        rally_h = self.encode(self.rally_gru, self.embed_stroke(type_strokes, area_strokes), lengths)
        hitter_h = self.encode(self.player_gru, self.embed_stroke(hitter_type, hitter_area), hitter_lengths)
        receiver_h = self.encode(self.player_gru, self.embed_stroke(receiver_type, receiver_area), receiver_lengths)
        type_h = self.encode(self.type_gru, self.embed_type(type_strokes), lengths)
        area_h = self.encode(self.area_gru, self.embed_area(area_strokes), lengths)
        type_area_h = torch.relu(self.type_area_proj(torch.cat([type_h, area_h], dim=1)))
        static_h = self.static_net(static)
        contexts = torch.stack([rally_h, hitter_h, receiver_h, type_area_h, static_h], dim=1)
        alpha = self.alpha_net(torch.cat([rally_h, hitter_h, receiver_h, type_area_h, static_h], dim=1))
        prefix_norm = (lengths.float() / 12.0).clamp(0, 2).unsqueeze(1)
        beta = self.beta_net(torch.cat([static_h, prefix_norm], dim=1))
        fused, weights = combine_pgfn_contexts(contexts, alpha, beta)
        h = self.shared(fused)
        return {
            "action": self.action(h),
            "family": self.family(h),
            "point": self.point(h),
            "terminal": self.terminal(h),
            "alpha": alpha,
            "beta": beta,
            "pgfn_weight": weights,
        }


def make_dataset(data: dict, source: str, idx: np.ndarray | slice) -> TPEGateDataset:
    if source == "oof":
        return TPEGateDataset(data["oof_seq"][idx], data["oof_len"][idx], data["x_oof"][idx], data["rows"]["next_actionId"].to_numpy(dtype=int)[idx], data["y_oof"][idx])
    if source == "full":
        return TPEGateDataset(data["full_seq"][idx], data["full_len"][idx], data["x_full"][idx], data["full_pool"]["next_actionId"].to_numpy(dtype=int)[idx], data["y_full"][idx])
    if source == "test":
        n = len(data["test_seq"][idx])
        return TPEGateDataset(data["test_seq"][idx], data["test_len"][idx], data["x_test_fullstats"][idx], np.zeros(n, dtype=int), np.zeros(n, dtype=int))
    raise ValueError(source)


def model_loss(outputs: dict[str, torch.Tensor], batch: TPEGateBatch) -> torch.Tensor:
    action = batch.action.to(DEVICE)
    family = batch.family.to(DEVICE)
    point = batch.point.to(DEVICE)
    terminal = batch.terminal.to(DEVICE)
    loss = F.cross_entropy(outputs["action"], action)
    loss = loss + 0.25 * F.cross_entropy(outputs["family"], family)
    weak = weak_action_mask(action)
    if weak.any():
        loss = loss + 0.12 * F.cross_entropy(outputs["action"][weak], action[weak])
    loss = loss + 0.08 * F.cross_entropy(outputs["point"], point)
    loss = loss + 0.04 * F.cross_entropy(outputs["terminal"], terminal)
    return loss


def train_model(train_ds: TPEGateDataset, valid_ds: TPEGateDataset, vocab_sizes: list[int], static_dim: int, sample_weights: np.ndarray | None, seed: int) -> tuple[TrueShuttleActionNet, float]:
    set_seed(seed)
    model = TrueShuttleActionNet(vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
    if sample_weights is None:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    else:
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = model_loss(model(batch), batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val_loss = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                loss = model_loss(model(batch), batch)
                val_loss += float(loss.item()) * len(batch.action)
                n += len(batch.action)
        val_loss /= max(n, 1)
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_model(model: TrueShuttleActionNet, dataset: TPEGateDataset) -> tuple[np.ndarray, pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    probs = []
    gate_rows = []
    model.eval()
    with torch.no_grad():
        offset = 0
        for batch in loader:
            out = model(batch)
            p = F.softmax(out["action"], dim=1).cpu().numpy()
            probs.append(p)
            alpha = out["alpha"].cpu().numpy()
            beta = out["beta"].cpu().numpy()
            weight = out["pgfn_weight"].cpu().numpy()
            for i in range(len(p)):
                rec = {"row_id": offset + i}
                for j, name in enumerate(CONTEXT_NAMES):
                    rec[f"v211_alpha_{name}"] = float(alpha[i, j])
                    rec[f"v211_beta_{name}"] = float(beta[i, j])
                    rec[f"v211_pgfn_{name}"] = float(weight[i, j])
                gate_rows.append(rec)
            offset += len(p)
    return normalize_rows_safe(np.vstack(probs)), pd.DataFrame(gate_rows)


def run_v211(data: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, list[dict]]:
    rows = data["rows"]
    train_rows = data["full_pool"]
    weights = distribution_match_weights(train_rows, data["test_rows"], MATCH_COLS)
    test_ds = make_dataset(data, "test", slice(None))
    oof = np.zeros((len(rows), 19), dtype=float)
    oof_gates = []
    test_probs = []
    test_gates = []
    fold_rows = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_mask = ~train_rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_idx = np.where(train_mask)[0]
        train_ds = make_dataset(data, "full", train_idx)
        valid_ds = make_dataset(data, "oof", valid)
        model, val_loss = train_model(train_ds, valid_ds, data["vocab_sizes"], data["x_full"].shape[1], weights[train_idx], 2110 + int(fold))
        valid_prob, valid_gate = predict_model(model, valid_ds)
        oof[valid] = valid_prob
        valid_gate["row_id"] = np.where(valid)[0]
        oof_gates.append(valid_gate)
        test_prob, test_gate = predict_model(model, test_ds)
        test_probs.append(test_prob)
        test_gates.append(test_gate.drop(columns=["row_id"]))
        pred = valid_prob.argmax(axis=1)
        y = rows.loc[valid, "next_actionId"].astype(int).to_numpy()
        fold_rows.append({"fold": int(fold), "val_loss": float(val_loss), "raw_action_macro_f1": float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))})
    test_prob = normalize_rows_safe(np.mean(test_probs, axis=0))
    test_gate_mean = pd.concat(test_gates).groupby(level=0).mean().reset_index(drop=True)
    test_gate_mean.insert(0, "row_id", np.arange(len(test_gate_mean)))
    return normalize_rows_safe(oof), test_prob, pd.concat(oof_gates, ignore_index=True), test_gate_mean, fold_rows


def selector_features_with_gates(frame: pd.DataFrame) -> pd.DataFrame:
    from analysis_v209_action_selector_reranker import selector_features

    x = selector_features(frame)
    gate_cols = [c for c in frame.columns if c.startswith("v211_alpha_") or c.startswith("v211_beta_") or c.startswith("v211_pgfn_")]
    if gate_cols:
        x = pd.concat([x, frame[gate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)], axis=1)
    return x.astype(float)


def align_columns(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[cols].astype(float)


def train_selector(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.20, max_iter=1000, random_state=211)
    clf.fit(x, y)
    return clf


def write_action_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def add_v211_features(frame: pd.DataFrame, probs: dict[str, np.ndarray], point_labels: np.ndarray, compat: np.ndarray | None, gates: pd.DataFrame) -> pd.DataFrame:
    out = add_probability_features(frame, probs, "v173_anchor", "v211", point_labels, compat)
    out = out.merge(gates, on="row_id", how="left", validate="many_to_one")
    gate_cols = [c for c in out.columns if c.startswith("v211_alpha_") or c.startswith("v211_beta_") or c.startswith("v211_pgfn_")]
    out[gate_cols] = out[gate_cols].fillna(0.0)
    return out


def fit_score_frame(train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    y = train_frame["is_correct"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.zeros(len(valid_frame), dtype=float), {"auc": np.nan, "positive_rate": float(y.mean()) if len(y) else 0.0}
    x_train = selector_features_with_gates(train_frame)
    cols = list(x_train.columns)
    clf = train_selector(x_train, y)
    x_valid = align_columns(selector_features_with_gates(valid_frame), cols)
    pred = clf.predict_proba(x_valid)[:, 1]
    y_valid = valid_frame["is_correct"].astype(int).to_numpy() if "is_correct" in valid_frame else None
    return pred, {
        "auc": float(roc_auc_score(y_valid, pred)) if y_valid is not None and len(np.unique(y_valid)) > 1 else np.nan,
        "positive_rate": float(y.mean()),
        "features": len(cols),
    }


def selector_oof_and_test(rows, test_rows, y, sources_oof, sources_test, probs_oof, probs_test, point_oof, point_test, gates_oof, gates_test, use_compat):
    base_frame = build_action_candidate_frame(rows, sources_oof, truth=y, anchor_name="v173")
    test_frame = build_action_candidate_frame(test_rows, sources_test, truth=None, anchor_name="v173")
    oof_best_action = np.zeros(len(rows), dtype=int)
    oof_delta = np.full(len(rows), -np.inf, dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_rows = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows_mask = ~valid_rows
        train_ids = set(np.where(train_rows_mask)[0])
        valid_ids = set(np.where(valid_rows)[0])
        train = base_frame[base_frame["row_id"].isin(train_ids)].copy()
        valid = base_frame[base_frame["row_id"].isin(valid_ids)].copy()
        compat = action_point_compatibility(y[train_rows_mask], point_oof[train_rows_mask], smoothing=1.0) if use_compat else None
        train = add_v211_features(train, probs_oof, point_oof, compat, gates_oof)
        valid = add_v211_features(valid, probs_oof, point_oof, compat, gates_oof)
        score, metric = fit_score_frame(train, valid)
        best_action, delta, _ = best_non_anchor_by_score(valid, score)
        valid_order = valid.drop_duplicates("row_id").sort_values("row_id")["row_id"].astype(int).to_numpy()
        oof_best_action[valid_order] = best_action[valid_order]
        oof_delta[valid_order] = delta[valid_order]
        metric.update({"fold": int(fold), "valid_candidate_rows": int(len(valid))})
        metrics.append(metric)

    compat_full = action_point_compatibility(y, point_oof, smoothing=1.0) if use_compat else None
    full_train = add_v211_features(base_frame.copy(), probs_oof, point_oof, compat_full, gates_oof)
    full_test = add_v211_features(test_frame.copy(), probs_test, point_test, compat_full, gates_test)
    score_test, full_metric = fit_score_frame(full_train, full_test.assign(is_correct=0))
    test_best_action, test_delta, _ = best_non_anchor_by_score(full_test, score_test)
    metrics.append({"fold": "full_test", **full_metric})
    return oof_best_action, oof_delta, test_best_action, test_delta, metrics


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    random.seed(211)
    np.random.seed(211)
    set_seed(211)
    data = prepare_data()
    data["rows"] = add_audit_columns(data["rows"].reset_index(drop=True))
    data["test_rows"] = add_audit_columns(data["test_rows"].reset_index(drop=True))
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()

    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    point_oof, point_test = load_point_anchor_labels(data, point)

    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_soft_oof, v173_soft_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    r166_oof, r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    r184_oof, r184_test = rebuild_r184_sources(state, point)
    v211_oof, v211_test, gates_oof, gates_test, v211_folds = run_v211(data)

    sources_oof = {
        "v173": v173_oof,
        "r166": r166_oof,
        **r184_oof,
        "v211_top1": topk_labels(v211_oof, 1),
        "v211_top2": topk_labels(v211_oof, 2),
    }
    sources_test = {
        "v173": v173_test,
        "r166": r166_test,
        **r184_test,
        "v211_top1": topk_labels(v211_test, 1),
        "v211_top2": topk_labels(v211_test, 2),
    }
    probs_oof = source_probs_for_selector(v173_soft_oof, r166_prob_oof, v211_oof)
    probs_oof["v211"] = probs_oof.pop("v208")
    probs_test = source_probs_for_selector(v173_soft_test, r166_prob_test, v211_test)
    probs_test["v211"] = probs_test.pop("v208")

    records = [evaluate_candidate("v173_anchor", y, v173_oof, v173_oof, {"scheme": "anchor"})]
    pred_store: dict[str, np.ndarray] = {}
    selector_metrics = []
    for tag, use_compat in [("v211_selector", False), ("v211_compat_selector", True)]:
        best_oof, delta_oof, best_test, delta_test, metrics = selector_oof_and_test(
            data["rows"],
            data["test_rows"],
            y,
            sources_oof,
            sources_test,
            probs_oof,
            probs_test,
            point_oof,
            point_test,
            gates_oof,
            gates_test,
            use_compat=use_compat,
        )
        selector_metrics.extend({**m, "selector": tag} for m in metrics)
        for cap in SELECTOR_CAPS:
            pred, changed = select_capped_action_changes(v173_oof, best_oof, delta_oof, cap, min_delta=0.0)
            test_pred, test_changed = select_capped_action_changes(v173_test, best_test, delta_test, cap, min_delta=0.0)
            name = f"{tag}_churn{str(cap).replace('.', 'p')}"
            rec = evaluate_candidate(name, y, pred, v173_oof, {"scheme": tag, "cap": cap})
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            rec["test_changed_rows"] = int(test_changed.sum())
            rec["mean_delta_changed_oof"] = float(delta_oof[changed].mean()) if changed.any() else 0.0
            records.append(rec)
            pred_store[name] = test_pred

    raw_pred = v211_oof.argmax(axis=1).astype(int)
    records.append(evaluate_candidate("v211_raw_diagnostic", y, raw_pred, v173_oof, {"scheme": "raw_diagnostic"}))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v211_action_search.csv", index=False)
    pd.DataFrame(selector_metrics).to_csv(OUTDIR / "v211_selector_fold_metrics.csv", index=False)
    pd.DataFrame(v211_folds).to_csv(OUTDIR / "v211_model_fold_metrics.csv", index=False)
    pd.DataFrame(distill_metrics).to_csv(OUTDIR / "v211_v173_distill_metrics.csv", index=False)
    gates_oof.to_csv(OUTDIR / "v211_oof_pgfn_gates.csv", index=False)
    gates_test.to_csv(OUTDIR / "v211_test_pgfn_gates.csv", index=False)
    np.save(OUTDIR / "v211_action_oof.npy", v211_oof)
    np.save(OUTDIR / "v211_action_test.npy", v211_test)

    generated = []
    eligible = search[
        search["candidate"].str.startswith(("v211_selector", "v211_compat_selector"))
        & search["action_churn_vs_v173_anchor"].gt(0)
        & search["action_churn_vs_v173_anchor"].le(0.012)
    ].copy()
    for _, rec in eligible.head(4).iterrows():
        name = str(rec["candidate"])
        sub_name = f"submission_{name}__pv188cap5__sr121.csv"
        info = write_action_submission(sub_name, pred_store[name], point, server)
        info.update(rec.to_dict())
        generated.append(info)

    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(12).to_dict(orient="records"),
        "notes": [
            "V211 implements true TPE-style hitter/receiver subsequence encoders.",
            "V211 implements PGFN alpha/beta gated context fusion and exports gates as selector features.",
            "The raw V211 decoder is diagnostic only.",
            "Generated submissions change action only; point is V188 cap5 and server is R121.",
            "Multi-step decoder pretraining is deferred to V213.",
            "No ShuttleSet, CoachAI, or TTMATCH rows are read.",
        ],
    }
    (OUTDIR / "v211_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v211_report.md").write_text(
        "# V211 True-ShuttleNet Selector Features\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173 action anchor: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + "\n".join(
            f"- `{g['submission']}` action OOF `{g['action_macro_f1']:.6f}`, delta `{g['delta_vs_v173_anchor']:.6f}`, churn `{g['action_churn_vs_v173_anchor']:.6f}`"
            for g in generated
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v211_true_shuttlenet_selector.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
