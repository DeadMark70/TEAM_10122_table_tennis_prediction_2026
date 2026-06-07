"""V214 ShuttleNet component ablation.

V211 implemented TPE + PGFN but was local-negative.  V214 runs a controlled
component ablation to identify whether TPE, PGFN beta, type-area path, or point
auxiliary loss is hurting the anchor-relative selector.

The output submissions are diagnostics only unless an ablation beats the V173
anchor locally.  Point remains V188 cap5 and server remains R121.  No external
ShuttleSet, CoachAI, or TTMATCH rows are read.
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

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import MATCH_COLS, distribution_match_weights, prepare_data
from analysis_v208_action_ttshuttlenet import action_family_targets, weak_action_mask
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
from analysis_v211_true_shuttlenet_selector import (
    CONTEXT_NAMES,
    POINT_ANCHOR,
    SERVER_ANCHOR,
    SELECTED_DIR,
    TPEGateBatch,
    TPEGateDataset,
    combine_pgfn_contexts,
    collate,
)
from analysis_v188_point_intent_gru import set_seed
from baseline_lgbm import ACTION_CLASSES
from train_v203_tt_shuttlenet import AREA_IDXS, TYPE_IDXS


OUTDIR = Path("v214_shuttlenet_component_ablation")
UPLOAD_DIR = Path("upload_candidates_20260519")
SRC_DEST = Path("src/analysis/analysis_v214_shuttlenet_component_ablation.py")

BATCH_SIZE = 512
EPOCHS = 5
PATIENCE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SELECTOR_CAPS = [0.005]


@dataclass(frozen=True)
class AblationConfig:
    name: str = "full"
    use_tpe: bool = True
    use_pgfn_alpha: bool = True
    use_pgfn_beta: bool = True
    use_type_area: bool = True
    use_point_aux: bool = True


def config_slug(cfg: AblationConfig) -> str:
    off = []
    if not cfg.use_tpe:
        off.append("no_tpe")
    if not cfg.use_pgfn_alpha:
        off.append("no_alpha")
    if not cfg.use_pgfn_beta:
        off.append("no_beta")
    if not cfg.use_type_area:
        off.append("no_taa")
    if not cfg.use_point_aux:
        off.append("no_point_aux")
    return "_".join(off) if off else cfg.name


def should_use_component(cfg: AblationConfig, component: str) -> bool:
    return {
        "tpe": cfg.use_tpe,
        "alpha": cfg.use_pgfn_alpha,
        "beta": cfg.use_pgfn_beta,
        "type_area": cfg.use_type_area,
        "point_aux": cfg.use_point_aux,
    }[component]


def summarize_ablation_winner(rows: list[dict]) -> dict:
    return sorted(rows, key=lambda r: (float(r["best_selector_delta"]), -float(r["best_selector_churn"])), reverse=True)[0]


class AblationActionNet(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, cfg: AblationConfig, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.cfg = cfg
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
        static = batch.static.to(DEVICE)
        rally_h = self.encode(self.rally_gru, self.embed_stroke(type_strokes, area_strokes), lengths)
        if self.cfg.use_tpe:
            hitter_h = self.encode(self.player_gru, self.embed_stroke(batch.hitter_type.to(DEVICE), batch.hitter_area.to(DEVICE)), batch.hitter_lengths.to(DEVICE))
            receiver_h = self.encode(self.player_gru, self.embed_stroke(batch.receiver_type.to(DEVICE), batch.receiver_area.to(DEVICE)), batch.receiver_lengths.to(DEVICE))
        else:
            hitter_h = torch.zeros_like(rally_h)
            receiver_h = torch.zeros_like(rally_h)
        if self.cfg.use_type_area:
            type_h = self.encode(self.type_gru, self.embed_type(type_strokes), lengths)
            area_h = self.encode(self.area_gru, self.embed_area(area_strokes), lengths)
            type_area_h = torch.relu(self.type_area_proj(torch.cat([type_h, area_h], dim=1)))
        else:
            type_area_h = rally_h
        static_h = self.static_net(static)
        contexts = torch.stack([rally_h, hitter_h, receiver_h, type_area_h, static_h], dim=1)
        if self.cfg.use_pgfn_alpha:
            alpha = self.alpha_net(torch.cat([rally_h, hitter_h, receiver_h, type_area_h, static_h], dim=1))
        else:
            alpha = torch.full((len(rally_h), 5), 0.2, dtype=rally_h.dtype, device=rally_h.device)
        if self.cfg.use_pgfn_beta:
            prefix_norm = (lengths.float() / 12.0).clamp(0, 2).unsqueeze(1)
            beta = self.beta_net(torch.cat([static_h, prefix_norm], dim=1))
        else:
            beta = torch.ones((len(rally_h), 5), dtype=rally_h.dtype, device=rally_h.device)
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


def model_loss(outputs: dict[str, torch.Tensor], batch: TPEGateBatch, cfg: AblationConfig) -> torch.Tensor:
    action = batch.action.to(DEVICE)
    family = batch.family.to(DEVICE)
    loss = F.cross_entropy(outputs["action"], action)
    loss = loss + 0.25 * F.cross_entropy(outputs["family"], family)
    weak = weak_action_mask(action)
    if weak.any():
        loss = loss + 0.12 * F.cross_entropy(outputs["action"][weak], action[weak])
    if cfg.use_point_aux:
        loss = loss + 0.08 * F.cross_entropy(outputs["point"], batch.point.to(DEVICE))
        loss = loss + 0.04 * F.cross_entropy(outputs["terminal"], batch.terminal.to(DEVICE))
    return loss


def make_dataset(data: dict, source: str, idx: np.ndarray | slice) -> TPEGateDataset:
    if source == "oof":
        return TPEGateDataset(data["oof_seq"][idx], data["oof_len"][idx], data["x_oof"][idx], data["rows"]["next_actionId"].to_numpy(dtype=int)[idx], data["y_oof"][idx])
    if source == "full":
        return TPEGateDataset(data["full_seq"][idx], data["full_len"][idx], data["x_full"][idx], data["full_pool"]["next_actionId"].to_numpy(dtype=int)[idx], data["y_full"][idx])
    if source == "test":
        n = len(data["test_seq"][idx])
        return TPEGateDataset(data["test_seq"][idx], data["test_len"][idx], data["x_test_fullstats"][idx], np.zeros(n, dtype=int), np.zeros(n, dtype=int))
    raise ValueError(source)


def train_model(train_ds, valid_ds, vocab_sizes, static_dim, weights, seed, cfg):
    set_seed(seed)
    model = AblationActionNet(vocab_sizes, static_dim, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
    if weights is None:
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    else:
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    best_state = None
    best = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = model_loss(model(batch), batch, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                loss = model_loss(model(batch), batch, cfg)
                val += float(loss.item()) * len(batch.action)
                n += len(batch.action)
        val /= max(n, 1)
        if val + 1e-5 < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def predict_model(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    probs = []
    gates = []
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
                    rec[f"v214_alpha_{name}"] = float(alpha[i, j])
                    rec[f"v214_beta_{name}"] = float(beta[i, j])
                    rec[f"v214_pgfn_{name}"] = float(weight[i, j])
                gates.append(rec)
            offset += len(p)
    return normalize_rows_safe(np.vstack(probs)), pd.DataFrame(gates)


def run_ablation(data: dict, cfg: AblationConfig):
    rows = data["rows"]
    train_rows = data["full_pool"]
    weights = distribution_match_weights(train_rows, data["test_rows"], MATCH_COLS)
    test_ds = make_dataset(data, "test", slice(None))
    oof = np.zeros((len(rows), 19), dtype=float)
    oof_gates = []
    test_probs = []
    test_gates = []
    folds = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_idx = np.where(~train_rows["fold"].astype(int).eq(int(fold)).to_numpy())[0]
        train_ds = make_dataset(data, "full", train_idx)
        valid_ds = make_dataset(data, "oof", valid)
        model, val_loss = train_model(train_ds, valid_ds, data["vocab_sizes"], data["x_full"].shape[1], weights[train_idx], 2140 + int(fold), cfg)
        valid_prob, valid_gate = predict_model(model, valid_ds)
        oof[valid] = valid_prob
        valid_gate["row_id"] = np.where(valid)[0]
        oof_gates.append(valid_gate)
        test_prob, test_gate = predict_model(model, test_ds)
        test_probs.append(test_prob)
        test_gates.append(test_gate.drop(columns=["row_id"]))
        y = rows.loc[valid, "next_actionId"].astype(int).to_numpy()
        folds.append({"ablation": config_slug(cfg), "fold": int(fold), "val_loss": float(val_loss), "raw_action_macro_f1": float(f1_score(y, valid_prob.argmax(axis=1), labels=ACTION_CLASSES, average="macro", zero_division=0))})
    test_prob = normalize_rows_safe(np.mean(test_probs, axis=0))
    test_gate = pd.concat(test_gates).groupby(level=0).mean().reset_index(drop=True)
    test_gate.insert(0, "row_id", np.arange(len(test_gate)))
    return normalize_rows_safe(oof), test_prob, pd.concat(oof_gates, ignore_index=True), test_gate, folds


def selector_features_with_gates(frame: pd.DataFrame) -> pd.DataFrame:
    from analysis_v209_action_selector_reranker import selector_features

    x = selector_features(frame)
    gate_cols = [c for c in frame.columns if c.startswith("v214_alpha_") or c.startswith("v214_beta_") or c.startswith("v214_pgfn_")]
    if gate_cols:
        x = pd.concat([x, frame[gate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)], axis=1)
    return x.astype(float)


def align_columns(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[cols].astype(float)


def add_ablation_features(frame, probs, point_labels, compat, gates):
    out = add_probability_features(frame, probs, "v173_anchor", "v214", point_labels, compat)
    out = out.merge(gates, on="row_id", how="left", validate="many_to_one")
    gate_cols = [c for c in out.columns if c.startswith("v214_alpha_") or c.startswith("v214_beta_") or c.startswith("v214_pgfn_")]
    out[gate_cols] = out[gate_cols].fillna(0.0)
    return out


def fit_score_frame(train_frame, valid_frame):
    y = train_frame["is_correct"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.zeros(len(valid_frame), dtype=float), {"auc": np.nan, "positive_rate": float(y.mean()) if len(y) else 0.0}
    x_train = selector_features_with_gates(train_frame)
    cols = list(x_train.columns)
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.20, max_iter=1000, random_state=214)
    clf.fit(x_train, y)
    x_valid = align_columns(selector_features_with_gates(valid_frame), cols)
    pred = clf.predict_proba(x_valid)[:, 1]
    y_valid = valid_frame["is_correct"].astype(int).to_numpy() if "is_correct" in valid_frame else None
    return pred, {"auc": float(roc_auc_score(y_valid, pred)) if y_valid is not None and len(np.unique(y_valid)) > 1 else np.nan, "features": len(cols)}


def selector_oof_and_test(rows, test_rows, y, sources_oof, sources_test, probs_oof, probs_test, point_oof, point_test, gates_oof, gates_test):
    base_frame = build_action_candidate_frame(rows, sources_oof, truth=y, anchor_name="v173")
    test_frame = build_action_candidate_frame(test_rows, sources_test, truth=None, anchor_name="v173")
    oof_best = np.zeros(len(rows), dtype=int)
    oof_delta = np.full(len(rows), -np.inf, dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_rows = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows_mask = ~valid_rows
        train_ids = set(np.where(train_rows_mask)[0])
        valid_ids = set(np.where(valid_rows)[0])
        train = base_frame[base_frame["row_id"].isin(train_ids)].copy()
        valid = base_frame[base_frame["row_id"].isin(valid_ids)].copy()
        compat = action_point_compatibility(y[train_rows_mask], point_oof[train_rows_mask], smoothing=1.0)
        train = add_ablation_features(train, probs_oof, point_oof, compat, gates_oof)
        valid = add_ablation_features(valid, probs_oof, point_oof, compat, gates_oof)
        score, metric = fit_score_frame(train, valid)
        best_action, delta, _ = best_non_anchor_by_score(valid, score)
        valid_order = valid.drop_duplicates("row_id").sort_values("row_id")["row_id"].astype(int).to_numpy()
        oof_best[valid_order] = best_action[valid_order]
        oof_delta[valid_order] = delta[valid_order]
        metric.update({"fold": int(fold), "valid_candidate_rows": int(len(valid))})
        metrics.append(metric)
    compat_full = action_point_compatibility(y, point_oof, smoothing=1.0)
    full_train = add_ablation_features(base_frame.copy(), probs_oof, point_oof, compat_full, gates_oof)
    full_test = add_ablation_features(test_frame.copy(), probs_test, point_test, compat_full, gates_test)
    score_test, metric = fit_score_frame(full_train, full_test.assign(is_correct=0))
    test_best, test_delta, _ = best_non_anchor_by_score(full_test, score_test)
    metrics.append({"fold": "full_test", **metric})
    return oof_best, oof_delta, test_best, test_delta, metrics


def write_submission(name, action, point_src, server_src):
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "rally_uid": point_src["rally_uid"].astype(int),
        "actionId": np.asarray(action, dtype=int),
        "pointId": point_src["pointId"].astype(int),
        "serverGetPoint": server_src["serverGetPoint"].astype(float),
    })
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    random.seed(214)
    np.random.seed(214)
    set_seed(214)
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

    configs = [
        AblationConfig(name="full"),
        AblationConfig(name="no_tpe", use_tpe=False),
        AblationConfig(name="no_beta", use_pgfn_beta=False),
        AblationConfig(name="no_taa_pointaux", use_type_area=False, use_point_aux=False),
    ]
    all_records = [{"candidate": "v173_anchor", "ablation": "anchor", "action_macro_f1": f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0), "delta_vs_v173_anchor": 0.0, "action_churn_vs_v173_anchor": 0.0, "changed_rows": 0}]
    summary = []
    generated = []
    fold_rows = []
    selector_rows = []
    pred_store = {}
    for cfg in configs:
        slug = config_slug(cfg)
        prob_oof, prob_test, gates_oof, gates_test, folds = run_ablation(data, cfg)
        fold_rows.extend(folds)
        raw = prob_oof.argmax(axis=1).astype(int)
        raw_score = f1_score(y, raw, labels=ACTION_CLASSES, average="macro", zero_division=0)
        sources_oof = {"v173": v173_oof, "r166": r166_oof, **r184_oof, f"{slug}_top1": topk_labels(prob_oof, 1), f"{slug}_top2": topk_labels(prob_oof, 2)}
        sources_test = {"v173": v173_test, "r166": r166_test, **r184_test, f"{slug}_top1": topk_labels(prob_test, 1), f"{slug}_top2": topk_labels(prob_test, 2)}
        probs_oof = source_probs_for_selector(v173_soft_oof, r166_prob_oof, prob_oof)
        probs_oof["v214"] = probs_oof.pop("v208")
        probs_test = source_probs_for_selector(v173_soft_test, r166_prob_test, prob_test)
        probs_test["v214"] = probs_test.pop("v208")
        best_oof, delta_oof, best_test, delta_test, metrics = selector_oof_and_test(data["rows"], data["test_rows"], y, sources_oof, sources_test, probs_oof, probs_test, point_oof, point_test, gates_oof, gates_test)
        selector_rows.extend({**m, "ablation": slug} for m in metrics)
        best_delta = -999.0
        best_churn = 0.0
        best_name = ""
        for cap in SELECTOR_CAPS:
            pred, changed = select_capped_action_changes(v173_oof, best_oof, delta_oof, cap, min_delta=0.0)
            test_pred, test_changed = select_capped_action_changes(v173_test, best_test, delta_test, cap, min_delta=0.0)
            name = f"v214_{slug}_selector_churn{str(cap).replace('.', 'p')}"
            score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
            delta = score - f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0)
            rec = {"candidate": name, "ablation": slug, "action_macro_f1": score, "delta_vs_v173_anchor": delta, "action_churn_vs_v173_anchor": float(np.mean(pred != v173_oof)), "changed_rows": int(changed.sum()), "cap": cap, "test_churn_vs_v173": float(np.mean(test_pred != v173_test)), "test_changed_rows": int(test_changed.sum())}
            all_records.append(rec)
            pred_store[name] = test_pred
            if delta > best_delta:
                best_delta, best_churn, best_name = delta, rec["action_churn_vs_v173_anchor"], name
        summary.append({"ablation": slug, "raw_action_macro_f1": raw_score, "raw_delta_vs_v173": raw_score - f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0), "best_selector": best_name, "best_selector_delta": best_delta, "best_selector_churn": best_churn})

    search = pd.DataFrame(all_records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary).sort_values(["best_selector_delta", "best_selector_churn"], ascending=[False, True])
    search.to_csv(OUTDIR / "v214_action_search.csv", index=False)
    summary_df.to_csv(OUTDIR / "v214_ablation_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTDIR / "v214_model_fold_metrics.csv", index=False)
    pd.DataFrame(selector_rows).to_csv(OUTDIR / "v214_selector_fold_metrics.csv", index=False)
    winner = summarize_ablation_winner(summary)
    eligible = search[search["candidate"].str.startswith("v214_") & search["action_churn_vs_v173_anchor"].gt(0) & search["action_churn_vs_v173_anchor"].le(0.012)]
    for _, rec in eligible.head(4).iterrows():
        info = write_submission(f"submission_{rec['candidate']}__pv188cap5__sr121.csv", pred_store[str(rec["candidate"])], point, server)
        info.update(rec.to_dict())
        generated.append(info)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "winner": winner, "generated": generated, "summary": summary_df.to_dict(orient="records"), "notes": ["V214 ablates TPE, PGFN beta, type-area path, and point auxiliary loss.", "Generated submissions change action only; point is V188 cap5 and server is R121.", "No ShuttleSet, CoachAI, or TTMATCH rows are read."]}
    (OUTDIR / "v214_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v214_report.md").write_text(
        "# V214 ShuttleNet Component Ablation\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Winner: `{winner['ablation']}` delta `{winner['best_selector_delta']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v214_shuttlenet_component_ablation.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
