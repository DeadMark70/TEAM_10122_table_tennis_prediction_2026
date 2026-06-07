"""Common loading, scoring, and packaging utilities for V243-V247."""

from __future__ import annotations

import __main__
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v238_v242_action_model_helpers import normalize_probability_rows
from analysis_v243_v247_action_augmentation_helpers import build_context_key_frame, clip_density_weights
from baseline_lgbm import ACTION_CLASSES


UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
WEAK_ACTIONS = [0, 3, 4, 5, 7, 8, 9, 12, 14]


def ensure_pickle_classes() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning


def load_action_context() -> dict:
    ensure_pickle_classes()
    data = prepare_data()
    state = rebuild_v173_best_actions()
    rows = add_audit_columns(data["rows"].copy())
    test_rows = add_audit_columns(state["test_rows"].copy())
    y = rows["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    return {
        "data": data,
        "state": state,
        "rows": rows,
        "test_rows": test_rows,
        "y": y,
        "v173_oof": v173_oof,
        "v173_test": v173_test,
        "v173_prob_oof": v173_prob_oof,
        "v173_prob_test": v173_prob_test,
        "point": point,
        "server": server,
    }


def feature_columns(rows: pd.DataFrame, drop_keywords: tuple[str, ...] = ()) -> list[str]:
    blocked = {"rally_uid", "match", "next_actionId", "next_pointId", "serverGetPoint", "fold"}
    cols = []
    for col in rows.columns:
        low = col.lower()
        if col in blocked:
            continue
        if drop_keywords and any(k in low for k in drop_keywords):
            continue
        if pd.api.types.is_numeric_dtype(rows[col]):
            cols.append(col)
    return cols


def align_test_columns(rows: pd.DataFrame, test_rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = test_rows.copy()
    for col in cols:
        if col not in out:
            out[col] = 0
    return out


def context_weights(rows: pd.DataFrame, test_rows: pd.DataFrame, low: float = 0.25, high: float = 4.0) -> np.ndarray:
    train_key = build_context_key_frame(rows)
    test_key = build_context_key_frame(test_rows)
    raw = density_ratio_weights(train_key, test_key, ["prefix_bin", "phase", "lag0_family", "lag0_depth"])
    return clip_density_weights(raw, low=low, high=high)


def class_balanced_weights(y: np.ndarray, base_weight: np.ndarray, power: float = 0.35, cap: float = 4.0) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    counts = np.bincount(labels, minlength=19).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    factors = np.power(counts.mean() / counts, float(power))
    weights = np.asarray(base_weight, dtype=float) * factors[labels]
    weights = np.clip(weights, 0.1, float(cap))
    return weights / max(weights.mean(), 1e-12)


def predict_full(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 19), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return normalize_probability_rows(out)


def train_extratrees_oof(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    cols: list[str],
    sample_weight: np.ndarray,
    seed: int,
    n_estimators: int = 140,
    min_samples_leaf: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof = np.zeros((len(rows), 19), dtype=float)
    test_sum = np.zeros((len(test_rows), 19), dtype=float)
    metrics = []
    folds = sorted(rows["fold"].astype(int).unique())
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        model = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(seed) + int(fold),
            n_jobs=1,
        )
        model.fit(rows.loc[train, cols].fillna(0), y[train], sample_weight=sample_weight[train])
        oof[valid] = predict_full(model, rows.loc[valid, cols].fillna(0))
        test_sum += predict_full(model, test_rows.loc[:, cols].fillna(0))
        metrics.append({"fold": int(fold), "valid_rows": int(valid.sum()), "features": int(len(cols))})
    return normalize_probability_rows(oof), normalize_probability_rows(test_sum / max(len(folds), 1)), metrics


def evaluate_action(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    iw = weighted_macro_f1(y, pred, weights)
    base_iw = weighted_macro_f1(y, anchor, weights)
    weak = f1_score(y, pred, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    weak_base = f1_score(y, anchor, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "iw_delta_vs_v173": float(iw - base_iw),
        "weak_delta_vs_v173": float(weak - weak_base),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
    }


def write_submission(outdir: Path, name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    outdir.mkdir(exist_ok=True)
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
    path = outdir / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def finalize_search(records: list[dict], anchor_name: str = "v173_anchor") -> tuple[pd.DataFrame, float, str]:
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"], ascending=[False, False, False])
    best_delta = float(search[search["candidate"].ne(anchor_name)]["delta_vs_v173_anchor"].max()) if (search["candidate"].ne(anchor_name)).any() else 0.0
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    return search, best_delta, verdict
