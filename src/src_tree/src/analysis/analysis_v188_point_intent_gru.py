"""V188 point-only GRU intent student.

This is a neural point experiment that avoids the old failure mode of training
only a direct 10-class point head.  The model reads the observed stroke prefix,
conditions on V173 action/R119 point/R186 coarse priors, predicts intermediate
intent heads, and is exported only through low-churn residuals on the current
V173 action + R119 point + R121 no-old anchor.

R186 teacher priors supervise only terminal/depth/width/safety heads.  They are
not converted into AI CUP pointId labels.  TTMATCH is not read.
"""

from __future__ import annotations

import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe, point_depth, point_side
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot, point_pred
from analysis_r187_point_intent_student import R187_TEACHER_COLUMNS, add_r186_priors
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, R111_TEST, prepare_prefix_features
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v188_point_intent_gru")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v188_point_intent_gru.py")
R186_DIR = Path("r186_external_coarse_point_teacher")
R186_TRAIN = R186_DIR / "r186_aicup_train_prefix_coarse_priors.csv"
R186_TEST = R186_DIR / "r186_aicup_test_prefix_coarse_priors.csv"

STROKE_COLS = ["actionId", "pointId", "spinId", "strengthId", "handId", "positionId", "strikeId"]
STATIC_BASE_COLS = [
    "sex",
    "numberGame",
    "prefix_len",
    "prefix_len_is_odd",
    "next_hitter_is_server",
    "is_server_hitter",
    "serverScore",
    "receiverScore",
    "serverScoreDiff",
    "scoreTotal",
    "lag0_actionId",
    "lag0_pointId",
    "lag0_spinId",
    "lag0_strengthId",
    "lag0_positionId",
    "remaining_len_bucket",
]
MAX_SEQ_LEN = 16
BATCH_SIZE = 512
EPOCHS = 14
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOSS_SETTINGS = [
    ("base_aux", {"terminal": 0.20, "depth": 0.20, "side": 0.10, "safety": 0.05, "width_teacher": 0.0, "r186": 0.0}),
    ("r186_w002", {"terminal": 0.20, "depth": 0.20, "side": 0.10, "safety": 0.05, "width_teacher": 0.02, "r186": 0.02}),
    ("r186_w005", {"terminal": 0.20, "depth": 0.20, "side": 0.10, "safety": 0.05, "width_teacher": 0.05, "r186": 0.05}),
]
ALPHAS = [0.01, 0.02, 0.03, 0.05]
CHURN_CAPS = [0.01, 0.02, 0.03, 0.05]


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


@dataclass
class V188Batch:
    strokes: torch.Tensor
    lengths: torch.Tensor
    static: torch.Tensor
    point: torch.Tensor
    teacher: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_padded_stroke_tensor(sequences: list[np.ndarray], max_len: int, pad_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((len(sequences), max_len, sequences[0].shape[1]), pad_value, dtype=np.int64)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for i, seq in enumerate(sequences):
        tail = seq[-max_len:]
        lengths[i] = max(1, len(tail))
        out[i, : len(tail)] = tail
    return out, lengths


def point_aux_targets(point: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(point, dtype=np.int64)
    non = y != 0
    depth = np.zeros(len(y), dtype=np.int64)
    side = np.zeros(len(y), dtype=np.int64)
    safety = np.zeros(len(y), dtype=np.int64)
    for i, p in enumerate(y):
        if p == 0:
            continue
        d = point_depth(int(p)) - 1
        s = point_side(int(p)) - 1
        depth[i] = d
        side[i] = s
        if s == 1:
            safety[i] = 0
        elif d == 2:
            safety[i] = 2
        else:
            safety[i] = 1
    return {"terminal": (y == 0).astype(np.int64), "depth": depth, "side": side, "safety": safety, "nonterminal": non}


def soft_kl_loss(logits: torch.Tensor, teacher: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0:
        return logits.sum() * 0.0
    t = torch.clamp(teacher.float(), min=1e-8)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return float(weight) * F.kl_div(F.log_softmax(logits, dim=1), t, reduction="batchmean")


def row_log_blend(base_prob: np.ndarray, residual_prob: np.ndarray, alpha: float) -> np.ndarray:
    base = np.clip(np.asarray(base_prob, dtype=float), 1e-12, 1.0)
    residual = np.clip(np.asarray(residual_prob, dtype=float), 1e-12, 1.0)
    logp = (1.0 - alpha) * np.log(base) + alpha * np.log(residual)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


def capped_residual_labels(base_labels: np.ndarray, prob: np.ndarray, max_churn: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=np.int64)
    pred = np.asarray(prob).argmax(axis=1).astype(np.int64)
    changed = pred != base
    max_rows = int(np.floor(len(base) * float(max_churn)))
    if changed.sum() > max_rows:
        base_score = prob[np.arange(len(prob)), base]
        pred_score = prob[np.arange(len(prob)), pred]
        gain = pred_score - base_score
        cand = np.where(changed)[0]
        keep = cand[np.argsort(gain[cand])[::-1][:max_rows]]
        capped = np.zeros(len(base), dtype=bool)
        capped[keep] = True
        changed = capped
    out = base.copy()
    out[changed] = pred[changed]
    return out, changed


class StrokeDataset(Dataset):
    def __init__(self, strokes: np.ndarray, lengths: np.ndarray, static: np.ndarray, point: np.ndarray, teacher: np.ndarray):
        self.strokes = torch.as_tensor(strokes, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.point = torch.as_tensor(np.asarray(point, dtype=np.int64).copy(), dtype=torch.long)
        self.teacher = torch.as_tensor(teacher, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.point)

    def __getitem__(self, idx: int) -> V188Batch:
        return V188Batch(self.strokes[idx], self.lengths[idx], self.static[idx], self.point[idx], self.teacher[idx])


def collate(batch: list[V188Batch]) -> V188Batch:
    return V188Batch(
        strokes=torch.stack([b.strokes for b in batch]),
        lengths=torch.stack([b.lengths for b in batch]),
        static=torch.stack([b.static for b in batch]),
        point=torch.stack([b.point for b in batch]),
        teacher=torch.stack([b.teacher for b in batch]),
    )


class PointIntentGRU(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in vocab_sizes])
        self.gru = nn.GRU(input_size=emb_dim * len(vocab_sizes), hidden_size=hidden, batch_first=True)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 32), nn.ReLU())
        joined = hidden + 32
        self.shared = nn.Sequential(nn.Linear(joined, 96), nn.ReLU(), nn.Dropout(0.15))
        self.point = nn.Linear(96, 10)
        self.terminal = nn.Linear(96, 2)
        self.depth = nn.Linear(96, 3)
        self.side = nn.Linear(96, 3)
        self.width = nn.Linear(96, 2)
        self.safety = nn.Linear(96, 3)

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        embs = [emb(strokes[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        z = torch.cat([h[-1], self.static_net(static)], dim=1)
        z = self.shared(z)
        return {
            "point": self.point(z),
            "terminal": self.terminal(z),
            "depth": self.depth(z),
            "side": self.side(z),
            "width": self.width(z),
            "safety": self.safety(z),
        }


def batch_loss(outputs: dict[str, torch.Tensor], point: torch.Tensor, teacher: torch.Tensor, weights: dict[str, float]) -> torch.Tensor:
    aux = point_aux_targets(point.detach().cpu().numpy())
    terminal = torch.as_tensor(aux["terminal"], dtype=torch.long, device=point.device)
    depth = torch.as_tensor(aux["depth"], dtype=torch.long, device=point.device)
    side = torch.as_tensor(aux["side"], dtype=torch.long, device=point.device)
    safety = torch.as_tensor(aux["safety"], dtype=torch.long, device=point.device)
    non = torch.as_tensor(aux["nonterminal"], dtype=torch.bool, device=point.device)
    loss = F.cross_entropy(outputs["point"], point)
    loss = loss + weights["terminal"] * F.cross_entropy(outputs["terminal"], terminal)
    if non.any():
        loss = loss + weights["depth"] * F.cross_entropy(outputs["depth"][non], depth[non])
        loss = loss + weights["side"] * F.cross_entropy(outputs["side"][non], side[non])
        loss = loss + weights["safety"] * F.cross_entropy(outputs["safety"][non], safety[non])
    loss = loss + soft_kl_loss(outputs["terminal"], teacher[:, 0:2], weights["r186"])
    loss = loss + soft_kl_loss(outputs["depth"], teacher[:, 2:5], weights["r186"])
    loss = loss + soft_kl_loss(outputs["width"], teacher[:, 5:7], weights["width_teacher"])
    loss = loss + soft_kl_loss(outputs["safety"], teacher[:, 7:10], weights["r186"])
    return loss


def predict_proba(model: PointIntentGRU, dataset: StrokeDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            probs.append(F.softmax(out["point"], dim=1).cpu().numpy())
    return normalize_rows_safe(np.vstack(probs))


def train_model(
    train_ds: StrokeDataset,
    valid_ds: StrokeDataset,
    vocab_sizes: list[int],
    static_dim: int,
    weights: dict[str, float],
    seed: int,
) -> tuple[PointIntentGRU, float]:
    set_seed(seed)
    model = PointIntentGRU(vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
                loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), weights)
                val_loss += float(loss.item()) * len(batch.point)
                n += len(batch.point)
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


def raw_groups(path: str) -> dict[int, pd.DataFrame]:
    raw = pd.read_csv(path).sort_values(["rally_uid", "strikeNumber"])
    return {int(k): g.reset_index(drop=True) for k, g in raw.groupby("rally_uid", sort=False)}


def sequences_for_rows(rows: pd.DataFrame, groups: dict[int, pd.DataFrame], vocab_offsets: bool = True) -> list[np.ndarray]:
    seqs = []
    for r in rows.itertuples(index=False):
        g = groups[int(getattr(r, "rally_uid"))]
        n = int(getattr(r, "prefix_len"))
        arr = g.loc[: n - 1, STROKE_COLS].to_numpy(dtype=np.int64)
        if vocab_offsets:
            arr = arr + 1
        seqs.append(arr)
    return seqs


def static_matrix(rows: pd.DataFrame, action_prob: np.ndarray, base_point: np.ndarray, fit_stats: dict | None = None) -> tuple[np.ndarray, dict]:
    cols = [c for c in STATIC_BASE_COLS if c in rows.columns]
    base = rows[cols].apply(pd.to_numeric, errors="coerce").astype(float).copy()
    for c in cols:
        base[c] = base[c].fillna(base[c].median() if base[c].notna().any() else 0.0)
    for i in range(action_prob.shape[1]):
        base[f"v173_action_p{i}"] = action_prob[:, i]
    for i in range(base_point.shape[1]):
        base[f"base_point_p{i}"] = base_point[:, i]
    for c in R187_TEACHER_COLUMNS:
        base[c] = pd.to_numeric(rows[c], errors="coerce").fillna(0.0)
    if fit_stats is None:
        mean = base.mean(axis=0).to_numpy(dtype=float)
        std = base.std(axis=0).replace(0, 1.0).to_numpy(dtype=float)
        fit_stats = {"columns": list(base.columns), "mean": mean, "std": std}
    for c in fit_stats["columns"]:
        if c not in base.columns:
            base[c] = 0.0
    base = base[fit_stats["columns"]]
    x = (base.to_numpy(dtype=float) - fit_stats["mean"]) / fit_stats["std"]
    return x.astype(np.float32), fit_stats


def teacher_matrix(rows: pd.DataFrame) -> np.ndarray:
    return normalize_teacher_blocks(rows[R187_TEACHER_COLUMNS].to_numpy(dtype=np.float32))


def normalize_teacher_blocks(t: np.ndarray) -> np.ndarray:
    out = np.asarray(t, dtype=np.float32).copy()
    for sl in [slice(0, 2), slice(2, 5), slice(5, 7), slice(7, 10)]:
        block = np.clip(out[:, sl], 1e-8, None)
        out[:, sl] = block / block.sum(axis=1, keepdims=True)
    return out


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, alpha: float, cap: float, setting: str) -> dict:
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    rec = {
        "candidate": name,
        "setting": setting,
        "alpha": float(alpha),
        "churn_cap": float(cap),
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "delta_vs_base": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0) - f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    for k in [0, 1, 3, 4, 7, 8, 9]:
        rec[f"point{k}_f1"] = float(rep[str(k)]["f1-score"])
    return rec


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    set_seed(188)
    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    rows = add_r186_priors(rows, pd.read_csv(R186_TRAIN))
    test_rows = add_r186_priors(test_rows, pd.read_csv(R186_TEST))

    r111_oof = load_pickle(R111_OOF)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    v3_oof = load_pickle("oof_proba_v3.pkl")
    tuning = r111_oof["tuning"]
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    prefix_train = add_r185_columns(prefix, None, pool=True)
    v173_action_oof_prob = one_hot(state["v173_pred_oof"], 19)
    v173_action_test_prob = one_hot(state["v173_pred_test"], 19)
    r119_oof = r119_oof_prior(rows, prefix_train, v173_action_oof_prob)
    r119_test = action_conditioned_point_prior(test_rows, prefix_train, v173_action_test_prob)
    local_base_prob_oof = normalize_rows_safe(0.95 * base_point_oof + 0.05 * r119_oof)
    local_base_prob_test = normalize_rows_safe(0.95 * base_point_test + 0.05 * r119_test)
    local_base_pred_oof = point_pred(rows, local_base_prob_oof, tuning)

    train_groups = raw_groups("train.csv")
    test_groups = raw_groups("test_new.csv")
    train_seq, train_len = build_padded_stroke_tensor(sequences_for_rows(rows, train_groups), MAX_SEQ_LEN, 0)
    test_seq, test_len = build_padded_stroke_tensor(sequences_for_rows(test_rows, test_groups), MAX_SEQ_LEN, 0)
    vocab_sizes = [int(max(train_seq[:, :, i].max(), test_seq[:, :, i].max()) + 1) for i in range(train_seq.shape[2])]
    x_static, stats = static_matrix(rows, v173_action_oof_prob, local_base_prob_oof)
    x_test_static, _ = static_matrix(test_rows, v173_action_test_prob, local_base_prob_test, stats)
    teacher = teacher_matrix(rows)
    teacher_test = teacher_matrix(test_rows)
    y = rows["next_pointId"].astype(int).to_numpy()

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    test_base_point = base_sub["pointId"].astype(int).to_numpy()

    search_records = [eval_candidate("local_v173_r119_base", y, local_base_pred_oof, local_base_pred_oof, 0.0, 0.0, "base")]
    pred_store: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    fold_metrics = []
    for setting_name, weights in LOSS_SETTINGS:
        oof_prob = np.zeros((len(rows), 10), dtype=float)
        for fold in sorted(rows["fold"].unique()):
            valid = rows["fold"].eq(fold).to_numpy()
            train = ~valid
            train_ds = StrokeDataset(train_seq[train], train_len[train], x_static[train], y[train], teacher[train])
            valid_ds = StrokeDataset(train_seq[valid], train_len[valid], x_static[valid], y[valid], teacher[valid])
            model, val_loss = train_model(train_ds, valid_ds, vocab_sizes, x_static.shape[1], weights, 1880 + int(fold))
            oof_prob[valid] = predict_proba(model, valid_ds)
            fold_pred = oof_prob[valid].argmax(axis=1)
            fold_metrics.append(
                {
                    "setting": setting_name,
                    "fold": int(fold),
                    "val_loss": float(val_loss),
                    "raw_point_macro_f1": float(f1_score(y[valid], fold_pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
                }
            )
        full_ds = StrokeDataset(train_seq, train_len, x_static, y, teacher)
        test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
        holdout_ds = StrokeDataset(train_seq[: max(1, len(train_seq) // 10)], train_len[: max(1, len(train_seq) // 10)], x_static[: max(1, len(train_seq) // 10)], y[: max(1, len(train_seq) // 10)], teacher[: max(1, len(train_seq) // 10)])
        full_model, _ = train_model(full_ds, holdout_ds, vocab_sizes, x_static.shape[1], weights, 1988)
        test_prob = predict_proba(full_model, test_ds)

        raw_pred = oof_prob.argmax(axis=1)
        search_records.append(eval_candidate(f"{setting_name}_raw_argmax", y, raw_pred, local_base_pred_oof, 1.0, 1.0, setting_name))
        for alpha in ALPHAS:
            blended = row_log_blend(local_base_prob_oof, oof_prob, alpha)
            blended_test = row_log_blend(local_base_prob_test, test_prob, alpha)
            for cap in CHURN_CAPS:
                pred, changed = capped_residual_labels(local_base_pred_oof, blended, cap)
                test_pred, test_changed = capped_residual_labels(test_base_point, blended_test, cap)
                name = f"v188_{setting_name}_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                rec = eval_candidate(name, y, pred, local_base_pred_oof, alpha, cap, setting_name)
                rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base_point))
                rec["test_changed_rows"] = int(np.sum(test_changed))
                search_records.append(rec)
                pred_store[name] = (pred, test_pred, rec)

    search = pd.DataFrame(search_records)
    search["tier"] = np.select(
        [search["point_churn_vs_base"].le(0.02), search["point_churn_vs_base"].le(0.05)],
        ["clean", "probe"],
        default="high_churn",
    )
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v188_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v188_fold_metrics.csv", index=False)

    generated = []
    positive = search[(search["tier"].eq("clean")) & (search["delta_vs_base"].gt(0)) & (search["candidate"].str.startswith("v188_"))]
    emitted: set[str] = set()
    for setting in [s[0] for s in LOSS_SETTINGS]:
        part = positive[positive["setting"].eq(setting)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        _, test_pred, stored = pred_store[name]
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, base_sub, test_pred)
        info.update(stored)
        info["submission"] = sub_name
        generated.append(info)
        emitted.add(name)
    if not positive.empty:
        rec = positive.iloc[0].to_dict()
        name = str(rec["candidate"])
        if name not in emitted:
            _, test_pred, stored = pred_store[name]
            sub_name = f"submission_{name}__v173action_r121server.csv"
            info = write_submission(sub_name, base_sub, test_pred)
            info.update(stored)
            info["submission"] = sub_name
            generated.append(info)

    probe_positive = search[(search["tier"].eq("probe")) & (search["delta_vs_base"].gt(0)) & (search["candidate"].str.startswith("v188_"))]
    for cap in [0.03, 0.05]:
        part = probe_positive[np.isclose(probe_positive["churn_cap"].astype(float), cap)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        if name in emitted:
            continue
        _, test_pred, stored = pred_store[name]
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, base_sub, test_pred)
        info.update(stored)
        info["submission"] = sub_name
        info["tier"] = "probe"
        generated.append(info)
        emitted.add(name)

    report = {
        "verdict": "CANDIDATES_GENERATED" if generated else "NO_POSITIVE_CLEAN_CANDIDATE",
        "device": DEVICE,
        "base": search[search["candidate"].eq("local_v173_r119_base")].iloc[0].to_dict(),
        "best_clean": search[search["tier"].eq("clean")].head(12).to_dict(orient="records"),
        "best_probe": search[search["tier"].eq("probe")].head(12).to_dict(orient="records"),
        "generated": generated,
        "fold_metrics": fold_metrics,
        "notes": [
            "Point-only GRU uses prefix stroke sequence plus V173 action, R119 point base, R184/R185 context, and R186 coarse priors.",
            "R186 teacher affects only terminal/depth/width/safety auxiliary heads.",
            "Final point submissions are low-churn residuals on V173/R119/R121.",
            "Transformer is intentionally left for a later line after this GRU result is reviewed.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v188_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v188_report.md").write_text(
        "# V188 Point Intent GRU\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Device: `{DEVICE}`\n"
        f"- Base point Macro-F1: `{report['base']['point_macro_f1']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + ("\n".join(f"- `{g['upload_path']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119']:.6f}`" for g in generated) or "- none")
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v188_point_intent_gru.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated_count": len(generated), "search": str(OUTDIR / "v188_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
