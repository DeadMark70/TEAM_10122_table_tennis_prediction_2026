"""R108/R109/R110 transductive and non-parametric backoff experiments.

R108:
  Transductive label propagation style smoothing. Train hard labels and
  validation/test soft labels are smoothed over a KNN graph in feature/prob
  space. Fold-safe for OOF.

R109:
  Safe duelist proxy. Instead of a risky full hitter/receiver world reversal,
  build receiver-conditioned priors from incoming zone/action/spin context.

R110:
  KNN backoff for high-entropy rows. Use fold-safe nearest train prefixes and
  only blend into low-confidence samples.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r108_r110_r109_transductive")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def one_hot(labels: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(labels), n), dtype=float)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def feature_matrix(rows: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray) -> pd.DataFrame:
    cols = [
        "prefix_len",
        "phase_id",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_positionId",
        "lag0_handId",
        "serverScoreDiff",
        "scoreTotal",
        "next_hitter_is_server",
    ]
    out = rows[[c for c in cols if c in rows.columns]].copy()
    for i in range(action_prob.shape[1]):
        out[f"pa_{i}"] = action_prob[:, i]
    for i in range(point_prob.shape[1]):
        out[f"pp_{i}"] = point_prob[:, i]
    out["action_max"] = action_prob.max(axis=1)
    out["point_max"] = point_prob.max(axis=1)
    out["action_entropy"] = entropy(action_prob)
    out["point_entropy"] = entropy(point_prob)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def knn_prior(
    train_x: pd.DataFrame,
    query_x: pd.DataFrame,
    y_action: np.ndarray,
    y_point: np.ndarray,
    k: int,
    temperature: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train_x)
    xq = scaler.transform(query_x)
    nn = NearestNeighbors(n_neighbors=min(k, len(train_x)), metric="euclidean")
    nn.fit(xtr)
    dist, idx = nn.kneighbors(xq)
    sim = -dist / max(temperature, 1e-6)
    sim -= sim.max(axis=1, keepdims=True)
    w = np.exp(sim)
    w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    a = np.zeros((len(query_x), 19), dtype=float)
    p = np.zeros((len(query_x), 10), dtype=float)
    for j in range(idx.shape[1]):
        a += w[:, j, None] * one_hot(y_action[idx[:, j]], 19)
        p += w[:, j, None] * one_hot(y_point[idx[:, j]], 10)
    return normalize_rows(a), normalize_rows(p)


def tlp_prior(
    train_x: pd.DataFrame,
    query_x: pd.DataFrame,
    y_action: np.ndarray,
    y_point: np.ndarray,
    base_action: np.ndarray,
    base_point: np.ndarray,
    k: int = 50,
    train_weight: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    hard_a, hard_p = knn_prior(train_x, query_x, y_action, y_point, k=k)
    # Soft-label propagation approximation: anchor to model soft labels, then
    # let nearby train labels adjust the manifold posterior.
    return normalize_rows(train_weight * hard_a + (1 - train_weight) * base_action), normalize_rows(train_weight * hard_p + (1 - train_weight) * base_point)


def foldsafe_priors(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    base_action: np.ndarray,
    base_point: np.ndarray,
    mode: str,
    k: int,
    train_weight: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    out_a = np.zeros((len(rows), 19), dtype=float)
    out_p = np.zeros((len(rows), 10), dtype=float)
    x_all = feature_matrix(rows, base_action, base_point)
    # Prefix train features need base probabilities; use one-hot labels to keep
    # fold-safe and avoid needing model predictions for all prefix rows.
    train_action_prob = one_hot(prefix["next_actionId"].to_numpy(dtype=int), 19)
    train_point_prob = one_hot(prefix["next_pointId"].to_numpy(dtype=int), 10)
    x_prefix_all = feature_matrix(prefix, train_action_prob, train_point_prob)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool_mask = ~prefix["match"].isin(valid_matches)
        train_x = x_prefix_all.loc[pool_mask].reset_index(drop=True)
        qx = x_all.loc[idx].reset_index(drop=True)
        ya = prefix.loc[pool_mask, "next_actionId"].to_numpy(dtype=int)
        yp = prefix.loc[pool_mask, "next_pointId"].to_numpy(dtype=int)
        if mode == "tlp":
            pa, pp = tlp_prior(train_x, qx, ya, yp, base_action[idx], base_point[idx], k=k, train_weight=train_weight)
        else:
            pa, pp = knn_prior(train_x, qx, ya, yp, k=k)
        out_a[idx] = pa
        out_p[idx] = pp
    return out_a, out_p


def test_priors(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    base_action: np.ndarray,
    base_point: np.ndarray,
    mode: str,
    k: int,
    train_weight: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    train_action_prob = one_hot(prefix["next_actionId"].to_numpy(dtype=int), 19)
    train_point_prob = one_hot(prefix["next_pointId"].to_numpy(dtype=int), 10)
    train_x = feature_matrix(prefix, train_action_prob, train_point_prob)
    qx = feature_matrix(rows, base_action, base_point)
    ya = prefix["next_actionId"].to_numpy(dtype=int)
    yp = prefix["next_pointId"].to_numpy(dtype=int)
    if mode == "tlp":
        return tlp_prior(train_x, qx, ya, yp, base_action, base_point, k=k, train_weight=train_weight)
    return knn_prior(train_x, qx, ya, yp, k=k)


def receiver_context_prior_oof(rows: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out_a = np.zeros((len(rows), 19), dtype=float)
    out_p = np.zeros((len(rows), 10), dtype=float)
    key_cols = ["next_receiver_id", "lag0_actionId", "lag0_pointId", "lag0_spinId", "phase_id"]
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].copy()
        out_a[idx], out_p[idx] = receiver_context_prior_for_rows(rows.loc[idx], pool, key_cols)
    return out_a, out_p


def receiver_context_prior_for_rows(rows: pd.DataFrame, pool: pd.DataFrame, key_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    global_a = np.bincount(pool["next_actionId"].astype(int), minlength=19).astype(float) + 1.0
    global_a /= global_a.sum()
    global_p = np.bincount(pool["next_pointId"].astype(int), minlength=10).astype(float) + 1.0
    global_p /= global_p.sum()
    grouped = {}
    for key, g in pool.groupby(key_cols, dropna=False):
        a = np.bincount(g["next_actionId"].astype(int), minlength=19).astype(float)
        p = np.bincount(g["next_pointId"].astype(int), minlength=10).astype(float)
        grouped[key] = (normalize_rows((a + 25.0 * global_a)[None, :])[0], normalize_rows((p + 25.0 * global_p)[None, :])[0])
    out_a = np.zeros((len(rows), 19), dtype=float)
    out_p = np.zeros((len(rows), 10), dtype=float)
    for i, (_, row) in enumerate(rows.iterrows()):
        key = tuple(row[c] for c in key_cols)
        pa, pp = grouped.get(key, (global_a, global_p))
        out_a[i] = pa
        out_p[i] = pp
    return out_a, out_p


def eval_branch(meta, action_prob, point_prob, tuning, name, base_action_prob, base_point_prob):
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    base_action_pred = apply_segmented_multipliers(meta, base_action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    base_point_pred = apply_segmented_multipliers(meta, base_point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    return {
        "candidate": name,
        "action_macro_f1": float(f1_score(meta["next_actionId"].astype(int), action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "point_macro_f1": float(f1_score(meta["next_pointId"].astype(int), point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "action_churn": float(np.mean(action_pred != base_action_pred)),
        "point_churn": float(np.mean(point_pred != base_point_pred)),
    }


def write_submission(test_meta, action_prob, point_prob, server_prob, tuning, name, extra=None):
    action_pred = apply_segmented_multipliers(test_meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(test_meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    if extra:
        info.update(extra)
    return info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, _ = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)
    r101_oof = load_pickle("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
    r101_test = load_pickle("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
    r101_meta = normalize_meta(r101_oof["valid_meta"])
    if not r101_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]):
        raise ValueError("R101 OOF does not align.")
    v3_oof = load_pickle("oof_proba_v3.pkl")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])

    current_action_oof = build_current_oof_action()
    teacher_action_oof = normalize_rows(0.80 * current_action_oof + 0.20 * art["experts_oof"]["v47_v64_oof_soft"])
    teacher_action_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    base_action_oof = normalize_rows(0.97 * r101_oof["gru_action"] + 0.03 * teacher_action_oof)
    base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_action_test = normalize_rows(0.97 * r101_test["gru_action"] + 0.03 * teacher_action_test)
    base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    tuning = r101_oof["tuning"]

    priors = {}
    for mode, k, tw in [("knn", 25, 0.70), ("knn", 50, 0.70), ("tlp", 50, 0.50), ("tlp", 100, 0.50)]:
        key = f"{mode}_k{k}_tw{tw}"
        priors[key] = {
            "oof": foldsafe_priors(rows, prefix, base_action_oof, base_point_oof, mode=mode, k=k, train_weight=tw),
            "test": test_priors(test_prefix, prefix, base_action_test, base_point_test, mode=mode, k=k, train_weight=tw),
        }
    r109_a_oof, r109_p_oof = receiver_context_prior_oof(rows, prefix)
    r109_a_test, r109_p_test = receiver_context_prior_for_rows(test_prefix, prefix, ["next_receiver_id", "lag0_actionId", "lag0_pointId", "lag0_spinId", "phase_id"])
    priors["r109_receiver_context"] = {"oof": (r109_a_oof, r109_p_oof), "test": (r109_a_test, r109_p_test)}

    rows_report = []
    best = None
    for name, obj in priors.items():
        pa, pp = obj["oof"]
        for w in [0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
            # R110 backoff: only apply to low confidence rows.
            low_a = base_action_oof.max(axis=1) < 0.40
            high_pe = entropy(base_point_oof) > np.quantile(entropy(base_point_oof), 0.70)
            a = base_action_oof.copy()
            p = base_point_oof.copy()
            a[low_a] = normalize_rows((1 - w) * a[low_a] + w * pa[low_a])
            p[high_pe] = normalize_rows((1 - w) * p[high_pe] + w * pp[high_pe])
            rec = eval_branch(meta, a, p, tuning, f"{name}_w{w}_selective", base_action_oof, base_point_oof)
            rec.update({"prior": name, "weight": w, "mode": "selective"})
            rows_report.append(rec)
            if best is None or (rec["action_macro_f1"] + rec["point_macro_f1"]) > (best["action_macro_f1"] + best["point_macro_f1"]):
                best = rec
            # R108 style: global soft smoothing.
            a2 = normalize_rows((1 - w) * base_action_oof + w * pa)
            p2 = normalize_rows((1 - w) * base_point_oof + w * pp)
            rec2 = eval_branch(meta, a2, p2, tuning, f"{name}_w{w}_global", base_action_oof, base_point_oof)
            rec2.update({"prior": name, "weight": w, "mode": "global"})
            rows_report.append(rec2)
            if (rec2["action_macro_f1"] + rec2["point_macro_f1"]) > (best["action_macro_f1"] + best["point_macro_f1"]):
                best = rec2

    search = pd.DataFrame(rows_report).sort_values(["action_macro_f1", "point_macro_f1"], ascending=False).reset_index(drop=True)
    search.to_csv(OUTDIR / "r108_r110_r109_search.csv", index=False)
    generated = []
    for _, rec in search.head(5).iterrows():
        prior_name = rec["prior"]
        w = float(rec["weight"])
        pa_test, pp_test = priors[prior_name]["test"]
        if rec["mode"] == "selective":
            low_a = base_action_test.max(axis=1) < 0.40
            high_pe = entropy(base_point_test) > np.quantile(entropy(base_point_test), 0.70)
            a = base_action_test.copy()
            p = base_point_test.copy()
            a[low_a] = normalize_rows((1 - w) * a[low_a] + w * pa_test[low_a])
            p[high_pe] = normalize_rows((1 - w) * p[high_pe] + w * pp_test[high_pe])
        else:
            a = normalize_rows((1 - w) * base_action_test + w * pa_test)
            p = normalize_rows((1 - w) * base_point_test + w * pp_test)
        safe = str(rec["candidate"]).replace(".", "p")
        generated.append(
            write_submission(
                r101_test["test_meta"].reset_index(drop=True),
                a,
                p,
                r101_test["gru_server"],
                tuning,
                f"submission_{safe}.csv",
                rec.to_dict(),
            )
        )
    report = {"base_action_f1": eval_branch(meta, base_action_oof, base_point_oof, tuning, "base", base_action_oof, base_point_oof)["action_macro_f1"], "base_point_f1": eval_branch(meta, base_action_oof, base_point_oof, tuning, "base", base_action_oof, base_point_oof)["point_macro_f1"], "best": search.head(15).to_dict(orient="records"), "generated": generated}
    (OUTDIR / "r108_r110_r109_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r108_r110_r109_transductive.py", "src/analysis/analysis_r108_r110_r109_transductive.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
