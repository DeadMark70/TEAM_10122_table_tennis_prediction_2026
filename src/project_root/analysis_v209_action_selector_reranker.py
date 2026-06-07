"""V209/V210 anchor-relative action selector/reranker.

V208 showed that TT-ShuttleNet action predictions are weak as a direct decoder.
This script uses them in the more appropriate role: candidate/ranking features
alongside V173, V166, and R184 action sources.  Point stays fixed at the current
public-positive V188 cap5 anchor, and server stays fixed at R121.

No ShuttleSet, CoachAI, or TTMATCH rows are read.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

import analysis_v165_combined_external_pretrain_proxy as v165
from analysis_r179_action_physics_hierarchy import action_family, normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import (
    build_frame,
    compose,
    load_sub,
    mask_by_keys,
    rebuild_v173_best_actions,
    support_keys_oof,
    transition_mask,
)
from analysis_r166_teacher_distillation_system import load_pickle
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v208_action_ttshuttlenet import run_scheme
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v209_action_selector_reranker")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v209_action_selector_reranker.py")

POINT_ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SERVER_ANCHOR = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
V207_OOF_FRAME = Path("v207_anchor_relative_ttselector/v207_oof_anchor_frame.csv")
V207_TEST_FRAME = Path("v207_anchor_relative_ttselector/v207_test_anchor_frame.csv")
R166_TARGETS = Path("r166_teacher_distillation/r166_teacher_targets.npz")

BLEND_WEIGHTS = [0.03, 0.05, 0.075, 0.10]
SELECTOR_CAPS = [0.005, 0.01, 0.02]
WEAK_CLASSES = {3, 4, 7, 8, 9, 11, 12, 14}


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


def geometric_log_blend(anchor_prob: np.ndarray, model_prob: np.ndarray, weight: float, eps: float = 1e-6) -> np.ndarray:
    """Geometric probability blend that can move a soft anchor at low weight."""
    a = np.clip(np.asarray(anchor_prob, dtype=float), eps, 1.0)
    b = np.clip(np.asarray(model_prob, dtype=float), eps, 1.0)
    logp = (1.0 - float(weight)) * np.log(a) + float(weight) * np.log(b)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


def smooth_anchor_probability(raw_prob: np.ndarray, anchor_labels: np.ndarray, peak: float = 0.70) -> np.ndarray:
    """Keep a distilled probability soft while preserving the anchor top label."""
    raw = normalize_rows_safe(raw_prob)
    labels = np.asarray(anchor_labels, dtype=int)
    onehot = np.full_like(raw, (1.0 - peak) / (raw.shape[1] - 1), dtype=float)
    onehot[np.arange(len(labels)), labels] = peak
    out = normalize_rows_safe(0.55 * onehot + 0.45 * raw)
    needs_fix = out.argmax(axis=1) != labels
    if needs_fix.any():
        out[needs_fix] = onehot[needs_fix]
    return normalize_rows_safe(out)


def action_point_compatibility(action_labels: np.ndarray, point_labels: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    table = np.full((19, 10), float(smoothing), dtype=float)
    for a, p in zip(np.asarray(action_labels, dtype=int), np.asarray(point_labels, dtype=int)):
        if 0 <= a < 19 and 0 <= p < 10:
            table[a, p] += 1.0
    return normalize_rows_safe(table)


def action_family_id(action_id: int) -> int:
    fam = action_family(int(action_id))
    return {"Zero": 0, "Attack": 1, "Control": 2, "Defensive": 3, "Serve": 4}.get(fam, 0)


def margin_entropy(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = normalize_rows_safe(prob)
    part = np.partition(p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    entropy = -np.sum(np.clip(p, 1e-9, 1.0) * np.log(np.clip(p, 1e-9, 1.0)), axis=1)
    return margin, entropy


def probability_for_labels(prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    return np.asarray(prob, dtype=float)[np.arange(len(labels)), labels]


def rank_for_labels(prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(prob, dtype=float), axis=1)
    labels = np.asarray(labels, dtype=int)
    ranks = np.empty(len(labels), dtype=float)
    for i, lab in enumerate(labels):
        hit = np.where(order[i] == lab)[0]
        ranks[i] = float(hit[0] + 1) if len(hit) else float(prob.shape[1])
    return ranks


def build_action_candidate_frame(
    rows: pd.DataFrame,
    sources: dict[str, np.ndarray],
    truth: np.ndarray | None,
    anchor_name: str = "v173",
) -> pd.DataFrame:
    """Build row-candidate table from action label sources."""
    n = len(rows)
    source_names = list(sources)
    stack = np.vstack([np.asarray(sources[name], dtype=int) for name in source_names])
    records = []
    for row_id in range(n):
        counts = {int(v): int(np.sum(stack[:, row_id] == v)) for v in np.unique(stack[:, row_id])}
        anchor = int(sources[anchor_name][row_id])
        for source in source_names:
            cand = int(sources[source][row_id])
            rec = {
                "row_id": row_id,
                "source": source,
                "candidate_action": cand,
                "candidate_family": action_family_id(cand),
                "anchor_action": anchor,
                "anchor_family": action_family_id(anchor),
                "differs_anchor": int(cand != anchor),
                "is_anchor": int(source == anchor_name),
                "agreement_count": counts[cand],
            }
            if truth is not None:
                rec["is_correct"] = int(cand == int(truth[row_id]))
            records.append(rec)
    frame = pd.DataFrame(records)
    context_cols = [
        "fold",
        "prefix_len",
        "audit_phase",
        "audit_lag0_action_family",
        "audit_lag0_depth",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
    ]
    available = [c for c in context_cols if c in rows.columns]
    if available:
        context = rows[available].copy().reset_index().rename(columns={"index": "row_id"})
        frame = frame.merge(context, on="row_id", how="left", validate="many_to_one")
    return frame


def add_probability_features(
    frame: pd.DataFrame,
    probs: dict[str, np.ndarray],
    anchor_prob_name: str,
    v208_prob_name: str,
    point_anchor_labels: np.ndarray,
    compat: np.ndarray | None,
) -> pd.DataFrame:
    out = frame.copy()
    n = int(out["row_id"].max()) + 1 if len(out) else 0
    for name, prob in probs.items():
        margin, entropy = margin_entropy(prob)
        labels = out["candidate_action"].astype(int).to_numpy()
        row_id = out["row_id"].astype(int).to_numpy()
        out[f"{name}_p_candidate"] = probability_for_labels(prob[row_id], labels)
        out[f"{name}_rank_candidate"] = rank_for_labels(prob[row_id], labels)
        out[f"{name}_margin"] = margin[row_id]
        out[f"{name}_entropy"] = entropy[row_id]
    anchor_prob = probs[anchor_prob_name]
    v208_prob = probs[v208_prob_name]
    row_id = out["row_id"].astype(int).to_numpy()
    anchor_action = out["anchor_action"].astype(int).to_numpy()
    candidate_action = out["candidate_action"].astype(int).to_numpy()
    out["anchor_prob_on_anchor"] = anchor_prob[row_id, anchor_action]
    out["anchor_prob_on_candidate"] = anchor_prob[row_id, candidate_action]
    out["v208_prob_on_anchor"] = v208_prob[row_id, anchor_action]
    out["v208_prob_on_candidate"] = v208_prob[row_id, candidate_action]
    out["v208_minus_anchor_candidate_prob"] = out["v208_prob_on_candidate"] - out["anchor_prob_on_candidate"]
    if compat is None:
        out["action_point_compat"] = 0.0
    else:
        points = np.asarray(point_anchor_labels, dtype=int)
        out["action_point_compat"] = compat[candidate_action, points[row_id]]
        out["anchor_point_compat"] = compat[anchor_action, points[row_id]]
        out["compat_delta_vs_anchor"] = out["action_point_compat"] - out["anchor_point_compat"]
    if len(out) and int(out["row_id"].max()) >= len(point_anchor_labels):
        raise ValueError("point_anchor_labels length does not cover frame row_id")
    return out


def selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "candidate_action",
        "candidate_family",
        "anchor_action",
        "anchor_family",
        "differs_anchor",
        "is_anchor",
        "agreement_count",
        "prefix_len",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "anchor_prob_on_anchor",
        "anchor_prob_on_candidate",
        "v208_prob_on_anchor",
        "v208_prob_on_candidate",
        "v208_minus_anchor_candidate_prob",
        "action_point_compat",
        "anchor_point_compat",
        "compat_delta_vs_anchor",
    ]
    prob_cols = [c for c in frame.columns if c.endswith("_p_candidate") or c.endswith("_rank_candidate") or c.endswith("_margin") or c.endswith("_entropy")]
    use_numeric = [c for c in numeric + prob_cols if c in frame.columns]
    x = frame[use_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cats = [c for c in ["source", "audit_phase", "audit_lag0_action_family", "audit_lag0_depth"] if c in frame.columns]
    if cats:
        x = pd.concat([x, pd.get_dummies(frame[cats].astype(str), prefix=cats, dtype=float)], axis=1)
    return x.astype(float)


def align_columns(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[cols].astype(float)


def train_selector(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.20, max_iter=1000, random_state=209)
    clf.fit(x, y)
    return clf


def best_non_anchor_by_score(frame: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(frame["row_id"].max()) + 1 if len(frame) else 0
    anchor_score = np.zeros(n, dtype=float)
    best_score = np.full(n, -np.inf, dtype=float)
    best_action = np.zeros(n, dtype=int)
    anchor_action_by_row: dict[int, int] = {}
    for i, row in enumerate(frame.itertuples(index=False)):
        rid = int(row.row_id)
        anchor_action_by_row[rid] = int(row.anchor_action)
        if int(row.is_anchor) == 1:
            anchor_score[rid] = float(score[i])
            if best_score[rid] == -np.inf:
                best_action[rid] = int(row.anchor_action)
        elif int(row.differs_anchor) == 1 and float(score[i]) > best_score[rid]:
            best_score[rid] = float(score[i])
            best_action[rid] = int(row.candidate_action)
    delta = best_score - anchor_score
    no_candidate = ~np.isfinite(best_score)
    for rid, anchor_action in anchor_action_by_row.items():
        if no_candidate[rid]:
            best_action[rid] = anchor_action
            delta[rid] = -np.inf
    return best_action, delta, anchor_score


def select_capped_action_changes(
    anchor_labels: np.ndarray,
    candidate_labels: np.ndarray,
    delta: np.ndarray,
    max_churn: float,
    min_delta: float = 0.0,
    allow_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor_labels, dtype=int)
    cand = np.asarray(candidate_labels, dtype=int)
    delta = np.asarray(delta, dtype=float)
    changed = (cand != anchor) & np.isfinite(delta) & (delta > float(min_delta))
    if allow_mask is not None:
        changed &= np.asarray(allow_mask, dtype=bool)
    max_rows = int(np.floor(len(anchor) * float(max_churn)))
    selected = np.zeros(len(anchor), dtype=bool)
    if max_rows > 0 and changed.any():
        idx = np.where(changed)[0]
        keep = idx[np.argsort(delta[idx])[::-1][:max_rows]]
        selected[keep] = True
    out = anchor.copy()
    out[selected] = cand[selected]
    return out, selected


def f1_action(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def load_pickle_local(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def rebuild_r166_best_action(rows: pd.DataFrame, test_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r111_oof = load_pickle(v165.R111_OOF)
    r111_test = load_pickle(v165.R111_TEST)
    targets = np.load(R166_TARGETS)
    prob_oof = v165.log_blend(r111_oof["gru_action"], targets["teacher_action_oof"], 0.15)
    prob_test = v165.log_blend(r111_test["gru_action"], targets["teacher_action_test"], 0.15)
    pred_oof = v165.action_pred(rows, prob_oof, r111_oof["tuning"]).astype(int)
    pred_test = v165.action_pred(test_rows, prob_test, r111_oof["tuning"]).astype(int)
    return pred_oof, pred_test, prob_oof, prob_test


def rebuild_r184_sources(state: dict, point_anchor: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    submitted_v173 = point_anchor["actionId"].astype(int).to_numpy()
    base_test = submitted_v173.copy()
    oof = build_frame(state["rows"], state["base_pred_oof"], state["v173_pred_oof"], state["rows"]["next_actionId"].astype(int).to_numpy())
    test = build_frame(state["test_rows"], base_test, submitted_v173)
    changed_oof = oof["changed"].to_numpy()
    changed_test = test["changed"].to_numpy()
    pair_keys = support_keys_oof(oof, ["r184_phase", "base_action", "teacher_action"], min_rows=8, min_changed=8, min_delta=0.02)
    specs = {
        "r184_attack_to_control": (
            changed_oof & oof["base_family"].eq("Attack").to_numpy() & oof["teacher_family"].eq("Control").to_numpy(),
            None,
        ),
        "r184_state_pair": (
            changed_oof & mask_by_keys(oof, ["r184_phase", "base_action", "teacher_action"], pair_keys),
            None,
        ),
        "r184_receive_control": (
            changed_oof
            & oof["r184_phase"].eq("receive").to_numpy()
            & oof["r184_lag0_depth"].isin(["short", "half"]).to_numpy()
            & oof["teacher_action"].isin([4, 6, 7, 10, 11, 12]).to_numpy(),
            None,
        ),
    }
    oof_sources: dict[str, np.ndarray] = {}
    test_sources: dict[str, np.ndarray] = {}
    test_files = {
        "r184_attack_to_control": UPLOAD_DIR / "submission_v191_r184_attack_to_control__pv188_r186_w005_cap5__sr121.csv",
        "r184_state_pair": UPLOAD_DIR / "submission_v191_r184_state_pair_supported__pv188_r186_w005_cap5__sr121.csv",
        "r184_receive_control": UPLOAD_DIR / "submission_v191_r184_receive_affordance_control__pv188_r186_w005_cap5__sr121.csv",
    }
    for name, (mask_oof, _) in specs.items():
        oof_sources[name] = compose(state["base_pred_oof"], state["v173_pred_oof"], mask_oof)
        test_sources[name] = load_sub(test_files[name], point_anchor["rally_uid"].astype(int).to_numpy())["actionId"].astype(int).to_numpy()
    return oof_sources, test_sources


def distill_v173_soft_anchor(data: dict, v173_oof: np.ndarray, v173_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = data["rows"]
    oof_prob = np.zeros((len(v173_oof), 19), dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        clf = OneVsRestClassifier(
            LogisticRegression(solver="liblinear", C=0.30, max_iter=1000, random_state=209 + int(fold))
        )
        clf.fit(data["x_oof"][train], v173_oof[train])
        prob = np.zeros((valid.sum(), 19), dtype=float)
        pred_classes = clf.classes_.astype(int)
        prob[:, pred_classes] = clf.predict_proba(data["x_oof"][valid])
        oof_prob[valid] = prob
        metrics.append({"fold": int(fold), "distill_acc": float(np.mean(prob.argmax(axis=1) == v173_oof[valid]))})
    full = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=0.30, max_iter=1000, random_state=2099))
    full.fit(data["x_oof"], v173_oof)
    test_prob = np.zeros((len(v173_test), 19), dtype=float)
    test_prob[:, full.classes_.astype(int)] = full.predict_proba(data["x_test_fullstats"])
    return smooth_anchor_probability(oof_prob, v173_oof), smooth_anchor_probability(test_prob, v173_test), metrics


def source_probs_for_selector(
    v173_prob: np.ndarray,
    r166_prob: np.ndarray,
    v208_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "v173_anchor": v173_prob,
        "r166": normalize_rows_safe(r166_prob),
        "v208": normalize_rows_safe(v208_prob),
    }


def topk_labels(prob: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(prob, dtype=float), axis=1)[:, k - 1].astype(int)


def add_feature_frame(
    base_frame: pd.DataFrame,
    probs: dict[str, np.ndarray],
    point_labels: np.ndarray,
    compat: np.ndarray | None,
) -> pd.DataFrame:
    return add_probability_features(base_frame, probs, "v173_anchor", "v208", point_labels, compat)


def fit_score_frame(train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    y = train_frame["is_correct"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.zeros(len(valid_frame), dtype=float), {"auc": np.nan, "positive_rate": float(y.mean()) if len(y) else 0.0}
    x_train = selector_features(train_frame)
    cols = list(x_train.columns)
    clf = train_selector(x_train, y)
    x_valid = align_columns(selector_features(valid_frame), cols)
    pred = clf.predict_proba(x_valid)[:, 1]
    y_valid = valid_frame["is_correct"].astype(int).to_numpy() if "is_correct" in valid_frame else None
    return pred, {
        "auc": float(roc_auc_score(y_valid, pred)) if y_valid is not None and len(np.unique(y_valid)) > 1 else np.nan,
        "positive_rate": float(y.mean()),
        "features": len(cols),
    }


def selector_oof_and_test(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    sources_oof: dict[str, np.ndarray],
    sources_test: dict[str, np.ndarray],
    probs_oof: dict[str, np.ndarray],
    probs_test: dict[str, np.ndarray],
    point_oof: np.ndarray,
    point_test: np.ndarray,
    use_compat: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    base_frame = build_action_candidate_frame(rows, sources_oof, truth=y, anchor_name="v173")
    test_frame = build_action_candidate_frame(test_rows, sources_test, truth=None, anchor_name="v173")
    oof_best_action = np.zeros(len(rows), dtype=int)
    oof_delta = np.full(len(rows), -np.inf, dtype=float)
    fold_metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_rows = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows_mask = ~valid_rows
        train_ids = set(np.where(train_rows_mask)[0])
        valid_ids = set(np.where(valid_rows)[0])
        train = base_frame[base_frame["row_id"].isin(train_ids)].copy()
        valid = base_frame[base_frame["row_id"].isin(valid_ids)].copy()
        compat = action_point_compatibility(y[train_rows_mask], point_oof[train_rows_mask], smoothing=1.0) if use_compat else None
        train = add_feature_frame(train, probs_oof, point_oof, compat)
        valid = add_feature_frame(valid, probs_oof, point_oof, compat)
        score, metric = fit_score_frame(train, valid)
        best_action, delta, _ = best_non_anchor_by_score(valid, score)
        valid_order = valid.drop_duplicates("row_id").sort_values("row_id")["row_id"].astype(int).to_numpy()
        oof_best_action[valid_order] = best_action[valid_order]
        oof_delta[valid_order] = delta[valid_order]
        metric.update({"fold": int(fold), "valid_candidate_rows": int(len(valid))})
        fold_metrics.append(metric)

    compat_full = action_point_compatibility(y, point_oof, smoothing=1.0) if use_compat else None
    full_train = add_feature_frame(base_frame.copy(), probs_oof, point_oof, compat_full)
    full_test = add_feature_frame(test_frame.copy(), probs_test, point_test, compat_full)
    score_test, full_metric = fit_score_frame(full_train, full_test.assign(is_correct=0))
    test_best_action, test_delta, _ = best_non_anchor_by_score(full_test, score_test)
    fold_metrics.append({"fold": "full_test", **full_metric})
    return oof_best_action, oof_delta, test_best_action, test_delta, fold_metrics


def evaluate_candidate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, meta: dict) -> dict:
    score = f1_action(y, pred)
    anchor_score = f1_action(y, anchor)
    rec = {
        "candidate": name,
        "action_macro_f1": score,
        "delta_vs_v173_anchor": score - anchor_score,
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
    }
    rec.update(meta)
    return rec


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


def load_point_anchor_labels(data: dict, point_sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if V207_OOF_FRAME.exists() and V207_TEST_FRAME.exists():
        oof = pd.read_csv(V207_OOF_FRAME)["anchor_label"].astype(int).to_numpy()
        test = pd.read_csv(V207_TEST_FRAME)["anchor_label"].astype(int).to_numpy()
        if len(oof) == len(data["rows"]) and len(test) == len(point_sub):
            return oof, test
    return data["rows"]["next_pointId"].astype(int).to_numpy(), point_sub["pointId"].astype(int).to_numpy()


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    data["rows"] = add_audit_columns(data["rows"].reset_index(drop=True))
    data["test_rows"] = add_audit_columns(data["test_rows"].reset_index(drop=True))
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    if not np.array_equal(data["rows"]["rally_uid"].astype(int).to_numpy(), state["rows"]["rally_uid"].astype(int).to_numpy()):
        raise ValueError("V209 row alignment mismatch")

    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    point_oof, point_test = load_point_anchor_labels(data, point)

    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_soft_oof, v173_soft_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    r166_oof, r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    r184_oof, r184_test = rebuild_r184_sources(state, point)
    v208_oof, v208_test, v208_folds = run_scheme(data, aux=True, tag="action_point_aux")

    sources_oof = {
        "v173": v173_oof,
        "r166": r166_oof,
        **r184_oof,
        "v208_top1": topk_labels(v208_oof, 1),
        "v208_top2": topk_labels(v208_oof, 2),
    }
    sources_test = {
        "v173": v173_test,
        "r166": r166_test,
        **r184_test,
        "v208_top1": topk_labels(v208_test, 1),
        "v208_top2": topk_labels(v208_test, 2),
    }
    probs_oof = source_probs_for_selector(v173_soft_oof, r166_prob_oof, v208_oof)
    probs_test = source_probs_for_selector(v173_soft_test, r166_prob_test, v208_test)

    records = [evaluate_candidate("v173_anchor", y, v173_oof, v173_oof, {"scheme": "anchor"})]
    pred_store: dict[str, np.ndarray] = {}

    for w in BLEND_WEIGHTS:
        blended = geometric_log_blend(v173_soft_oof, v208_oof, w)
        blended_test = geometric_log_blend(v173_soft_test, v208_test, w)
        for gate, allowed in [("all", None), ("weak_class_targets", WEAK_CLASSES)]:
            pred = blended.argmax(axis=1).astype(int)
            test_pred = blended_test.argmax(axis=1).astype(int)
            if allowed is not None:
                mask = np.array([int(p) in allowed for p in pred], dtype=bool) & (pred != v173_oof)
                test_mask = np.array([int(p) in allowed for p in test_pred], dtype=bool) & (test_pred != v173_test)
                pred = v173_oof.copy()
                test_out = v173_test.copy()
                pred[mask] = blended.argmax(axis=1)[mask]
                test_out[test_mask] = blended_test.argmax(axis=1)[test_mask]
                test_pred = test_out
            name = f"v208b_softanchor_w{str(w).replace('.', 'p')}_{gate}"
            rec = evaluate_candidate(name, y, pred, v173_oof, {"scheme": "v208b_soft_anchor", "weight": w, "gate": gate})
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            records.append(rec)
            pred_store[name] = test_pred

    selector_metrics = []
    for tag, use_compat in [("v209_selector", False), ("v210_compat_selector", True)]:
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
            use_compat=use_compat,
        )
        selector_metrics.extend({**m, "selector": tag} for m in metrics)
        for cap in SELECTOR_CAPS:
            allow_oof = np.ones(len(v173_oof), dtype=bool)
            allow_test = np.ones(len(v173_test), dtype=bool)
            if use_compat:
                compat_full = action_point_compatibility(y, point_oof, smoothing=1.0)
                allow_oof = compat_full[best_oof, point_oof] >= 0.035
                allow_test = compat_full[best_test, point_test] >= 0.035
            pred, changed = select_capped_action_changes(v173_oof, best_oof, delta_oof, cap, min_delta=0.0, allow_mask=allow_oof)
            test_pred, test_changed = select_capped_action_changes(v173_test, best_test, delta_test, cap, min_delta=0.0, allow_mask=allow_test)
            name = f"{tag}_churn{str(cap).replace('.', 'p')}"
            rec = evaluate_candidate(name, y, pred, v173_oof, {"scheme": tag, "cap": cap})
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            rec["test_changed_rows"] = int(test_changed.sum())
            rec["mean_delta_changed_oof"] = float(delta_oof[changed].mean()) if changed.any() else 0.0
            records.append(rec)
            pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v209_action_search.csv", index=False)
    pd.DataFrame(selector_metrics).to_csv(OUTDIR / "v209_selector_fold_metrics.csv", index=False)
    pd.DataFrame(distill_metrics).to_csv(OUTDIR / "v208b_v173_distill_metrics.csv", index=False)
    pd.DataFrame(v208_folds).to_csv(OUTDIR / "v209_v208_fold_metrics.csv", index=False)
    np.save(OUTDIR / "v209_v208_action_point_aux_oof.npy", v208_oof)
    np.save(OUTDIR / "v209_v208_action_point_aux_test.npy", v208_test)

    generated = []
    eligible = search[
        search["candidate"].str.startswith(("v209_selector", "v210_compat_selector"))
        & search["action_churn_vs_v173_anchor"].gt(0)
        & search["action_churn_vs_v173_anchor"].le(0.025)
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
            "V208B uses a soft V173 distillation probability anchor; one-hot anchor no-op blend is avoided.",
            "V209 trains an anchor-relative action selector over V173/V166/R184/V208 candidates.",
            "V210 adds action-point compatibility against the fixed V188 cap5 point anchor.",
            "Generated submissions change action only; point is V188 cap5 and server is R121.",
            "Raw V208 action is never used as a final decoder.",
            "No ShuttleSet, CoachAI, or TTMATCH rows are read.",
        ],
    }
    (OUTDIR / "v209_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v209_report.md").write_text(
        "# V209/V210 Action Selector Reranker\n\n"
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
    shutil.copy2("analysis_v209_action_selector_reranker.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
