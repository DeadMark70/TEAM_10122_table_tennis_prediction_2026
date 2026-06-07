"""R187 point intent student.

R186 external data is treated as a low-weight teacher for intermediate intent
heads only: terminal, depth, width, and safety.  The final point head is trained
only on AI CUP exact pointId labels, and submissions are low-churn residuals on
top of the current V173 action + R119 point + R121 server no-old anchor.

No TTMATCH data is read by this script.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score

from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe, point_depth, point_side
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import (
    BASE_V173,
    R121,
    add_r185_columns,
    load_pickle,
    load_sub,
    one_hot,
    point_pred,
)
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, R111_TEST, prepare_prefix_features
from baseline_lgbm import POINT_CLASSES


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


OUTDIR = Path("r187_point_intent_student")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_r187_point_intent_student.py")
R186_DIR = Path("r186_external_coarse_point_teacher")
R186_TRAIN = R186_DIR / "r186_aicup_train_prefix_coarse_priors.csv"
R186_TEST = R186_DIR / "r186_aicup_test_prefix_coarse_priors.csv"

R187_TEACHER_COLUMNS = [
    "T_terminal_nonterminal",
    "T_terminal_terminalish",
    "T_depth_short",
    "T_depth_half",
    "T_depth_long",
    "T_width_center",
    "T_width_wide",
    "T_safety_safe_middle",
    "T_safety_pressure_wide",
    "T_safety_risky_edge",
]

TEACHER_SETTINGS = [
    ("no_teacher", {"terminal": 0.0, "depth": 0.0, "width": 0.0, "safety": 0.0}),
    ("t001_d0005_w00025_s00025", {"terminal": 0.01, "depth": 0.005, "width": 0.0025, "safety": 0.0025}),
    ("t002_d001_w0005_s00025", {"terminal": 0.02, "depth": 0.01, "width": 0.005, "safety": 0.0025}),
    ("t002_d002_w001_s0005", {"terminal": 0.02, "depth": 0.02, "width": 0.01, "safety": 0.005}),
]
ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.05]
CHURN_CAPS = [0.01, 0.02, 0.03, 0.05]

BASE_CONTEXT_COLS = [
    "sex",
    "numberGame",
    "prefix_len",
    "prefix_len_is_odd",
    "next_hitter_is_server",
    "next_strikeId_rule",
    "is_server_hitter",
    "serverScore",
    "receiverScore",
    "serverScoreDiff",
    "scoreTotal",
    "last_action_same_as_prev",
    "last_point_same_as_prev",
    "last_hand_same_as_prev",
]
CATEGORICAL_COLS = [
    "r184_phase",
    "r184_lag0_family",
    "r184_lag0_depth",
    "r184_lag0_side",
    "r184_affordance_state",
    "r184_state_simple",
    "r185_action_family",
]


@dataclass
class HeadPredictions:
    terminal: np.ndarray
    depth: np.ndarray
    width: np.ndarray
    safety: np.ndarray


def normalize_teacher_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return only the low-weight auxiliary-head teacher coefficients."""
    allowed = {"terminal", "depth", "width", "safety"}
    return {k: float(weights.get(k, 0.0)) for k in sorted(allowed)}


def depth_distribution_from_point_prob(prob: np.ndarray) -> np.ndarray:
    """Map 10-class point probabilities to short/half/long masses.

    pointId=0 is terminal and is intentionally not assigned to any depth.
    """
    p = np.asarray(prob, dtype=float)
    out = np.zeros((p.shape[0], 3), dtype=float)
    out[:, 0] = p[:, 1:4].sum(axis=1)
    out[:, 1] = p[:, 4:7].sum(axis=1)
    out[:, 2] = p[:, 7:10].sum(axis=1)
    return out


def row_log_blend(base_prob: np.ndarray, residual_prob: np.ndarray, alpha: float) -> np.ndarray:
    base = np.clip(np.asarray(base_prob, dtype=float), 1e-12, 1.0)
    residual = np.clip(np.asarray(residual_prob, dtype=float), 1e-12, 1.0)
    logp = (1.0 - alpha) * np.log(base) + alpha * np.log(residual)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


def apply_residual_with_churn_cap(
    base_prob: np.ndarray,
    residual_prob: np.ndarray,
    alpha: float,
    max_churn: float,
    base_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend residual probabilities, then keep only the strongest argmax changes."""
    mixed = row_log_blend(base_prob, residual_prob, alpha)
    if base_labels is None:
        base_labels = np.asarray(base_prob).argmax(axis=1).astype(int)
    else:
        base_labels = np.asarray(base_labels, dtype=int)
    new_labels = mixed.argmax(axis=1).astype(int)
    changed = new_labels != base_labels
    max_rows = int(np.floor(len(base_labels) * float(max_churn)))
    if changed.sum() > max_rows:
        base_score = mixed[np.arange(len(mixed)), base_labels]
        new_score = mixed[np.arange(len(mixed)), new_labels]
        gain = new_score - base_score
        candidates = np.where(changed)[0]
        keep = candidates[np.argsort(gain[candidates])[::-1][:max_rows]]
        capped = np.zeros(len(base_labels), dtype=bool)
        capped[keep] = True
        changed = capped
    out = np.asarray(base_prob, dtype=float).copy()
    out[changed] = mixed[changed]
    return normalize_rows_safe(out), changed


def point_depth_id(point_id: int) -> int:
    return max(0, point_depth(int(point_id)) - 1)


def point_width_id(point_id: int) -> int:
    side = point_side(int(point_id))
    return 0 if side == 2 else 1


def point_safety_id(point_id: int) -> int:
    depth = point_depth(int(point_id))
    side = point_side(int(point_id))
    if side == 2:
        return 0
    if depth == 3:
        return 2
    return 1


def labels_depth(y: np.ndarray) -> np.ndarray:
    return np.array([point_depth(v) for v in y], dtype=int)


def labels_width(y: np.ndarray) -> np.ndarray:
    return np.array([0 if v == 0 else point_width_id(v) + 1 for v in y], dtype=int)


def base_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in BASE_CONTEXT_COLS:
        if c in df.columns:
            cols.append(c)
    cols.extend([c for c in df.columns if c.startswith("lag") and c not in cols])
    cols.extend([c for c in df.columns if c.startswith("count_") and c not in cols])
    cols.extend([c for c in df.columns if c.startswith("nunique_") and c not in cols])
    cols.extend([c for c in CATEGORICAL_COLS if c in df.columns and c not in cols])
    return cols


def add_r186_priors(df: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    keys = ["rally_uid", "prefix_len"]
    keep = keys + [c for c in R187_TEACHER_COLUMNS if c in priors.columns]
    out = df.merge(priors[keep], on=keys, how="left", validate="many_to_one")
    missing = [c for c in R187_TEACHER_COLUMNS if c not in out.columns]
    for c in missing:
        out[c] = 0.0
    for c in R187_TEACHER_COLUMNS:
        out[c] = out[c].fillna(out[c].median() if out[c].notna().any() else 0.0)
    return out


def add_base_prob_features(df: pd.DataFrame, prob: np.ndarray, prefix: str) -> pd.DataFrame:
    out = df.copy()
    for k in range(prob.shape[1]):
        out[f"{prefix}{k}"] = prob[:, k]
    return out


def make_xy(train_df: pd.DataFrame, pred_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df[cols].copy()
    pred = pred_df[cols].copy()
    cat_cols = [c for c in cols if c in CATEGORICAL_COLS or train[c].dtype == "object"]
    num_cols = [c for c in cols if c not in cat_cols]
    for c in num_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
        fill = float(train[c].median()) if train[c].notna().any() else 0.0
        train[c] = train[c].fillna(fill)
        pred[c] = pred[c].fillna(fill)
    both = pd.concat([train, pred], axis=0, ignore_index=True)
    both = pd.get_dummies(both, columns=cat_cols, dummy_na=True)
    x_train = both.iloc[: len(train)].reset_index(drop=True)
    x_pred = both.iloc[len(train) :].reset_index(drop=True)
    return x_train, x_pred


def fit_predict_head(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    y: np.ndarray,
    cols: list[str],
    n_classes: int,
    seed: int,
) -> np.ndarray:
    x_train, x_pred = make_xy(train_df, pred_df, cols)
    clf = LGBMClassifier(
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        n_estimators=120,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=45,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.25,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(x_train, y)
    proba = clf.predict_proba(x_pred)
    if n_classes == 2 and proba.shape[1] == 2:
        return normalize_rows_safe(proba)
    if proba.shape[1] != n_classes:
        out = np.zeros((len(x_pred), n_classes), dtype=float)
        for i, cls in enumerate(clf.classes_):
            out[:, int(cls)] = proba[:, i]
        return normalize_rows_safe(out)
    return normalize_rows_safe(proba)


def teacher_arrays(df: pd.DataFrame) -> HeadPredictions:
    return HeadPredictions(
        terminal=df[["T_terminal_nonterminal", "T_terminal_terminalish"]].to_numpy(float),
        depth=df[["T_depth_short", "T_depth_half", "T_depth_long"]].to_numpy(float),
        width=df[["T_width_center", "T_width_wide"]].to_numpy(float),
        safety=df[["T_safety_safe_middle", "T_safety_pressure_wide", "T_safety_risky_edge"]].to_numpy(float),
    )


def blend_heads_with_teacher(heads: HeadPredictions, teacher: HeadPredictions, weights: dict[str, float]) -> HeadPredictions:
    w = normalize_teacher_weights(weights)
    return HeadPredictions(
        terminal=normalize_rows_safe((1.0 - w["terminal"]) * heads.terminal + w["terminal"] * teacher.terminal),
        depth=normalize_rows_safe((1.0 - w["depth"]) * heads.depth + w["depth"] * teacher.depth),
        width=normalize_rows_safe((1.0 - w["width"]) * heads.width + w["width"] * teacher.width),
        safety=normalize_rows_safe((1.0 - w["safety"]) * heads.safety + w["safety"] * teacher.safety),
    )


def adjust_group_mass(base: np.ndarray, target_mass: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    out = np.asarray(base, dtype=float).copy()
    for gi, group in enumerate(groups):
        current = out[:, group].sum(axis=1)
        ratio = np.divide(target_mass[:, gi], current, out=np.ones(len(out)), where=current > 1e-12)
        out[:, group] *= ratio[:, None]
    return normalize_rows_safe(out)


def reconstruct_point_from_heads(base_prob: np.ndarray, heads: HeadPredictions) -> np.ndarray:
    out = np.asarray(base_prob, dtype=float).copy()
    term = np.clip(heads.terminal[:, 1], 0.0, 1.0)
    non = np.maximum(1.0 - term, 1e-12)
    out[:, 0] = term
    old_non = out[:, 1:].sum(axis=1)
    out[:, 1:] *= np.divide(non, old_non, out=np.ones(len(out)), where=old_non > 1e-12)[:, None]
    out = adjust_group_mass(out, heads.depth * non[:, None], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = adjust_group_mass(out, heads.width * non[:, None], [[2, 5, 8], [1, 3, 4, 6, 7, 9]])
    out = adjust_group_mass(out, heads.safety * non[:, None], [[2, 5, 8], [1, 3, 4, 6], [7, 9]])
    out[:, 0] = term
    old_non = out[:, 1:].sum(axis=1)
    out[:, 1:] *= np.divide(non, old_non, out=np.ones(len(out)), where=old_non > 1e-12)[:, None]
    return normalize_rows_safe(out)


def train_aux_oof(rows: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> HeadPredictions:
    out = {
        "terminal": np.zeros((len(rows), 2), dtype=float),
        "depth": np.zeros((len(rows), 3), dtype=float),
        "width": np.zeros((len(rows), 2), dtype=float),
        "safety": np.zeros((len(rows), 3), dtype=float),
    }
    term_y = (y == 0).astype(int)
    non = y != 0
    depth_y = np.array([point_depth_id(v) if v != 0 else -1 for v in y], dtype=int)
    width_y = np.array([point_width_id(v) if v != 0 else -1 for v in y], dtype=int)
    safety_y = np.array([point_safety_id(v) if v != 0 else -1 for v in y], dtype=int)
    for fold in sorted(rows["fold"].unique()):
        valid = rows["fold"].eq(fold).to_numpy()
        train = ~valid
        out["terminal"][valid] = fit_predict_head(rows.loc[train], rows.loc[valid], term_y[train], feature_cols, 2, 1870 + int(fold))
        train_non = train & non
        valid_df = rows.loc[valid]
        out["depth"][valid] = fit_predict_head(rows.loc[train_non], valid_df, depth_y[train_non], feature_cols, 3, 1880 + int(fold))
        out["width"][valid] = fit_predict_head(rows.loc[train_non], valid_df, width_y[train_non], feature_cols, 2, 1890 + int(fold))
        out["safety"][valid] = fit_predict_head(rows.loc[train_non], valid_df, safety_y[train_non], feature_cols, 3, 1900 + int(fold))
    return HeadPredictions(**out)


def train_aux_test(rows: pd.DataFrame, test_rows: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> HeadPredictions:
    term_y = (y == 0).astype(int)
    non = y != 0
    return HeadPredictions(
        terminal=fit_predict_head(rows, test_rows, term_y, feature_cols, 2, 1879),
        depth=fit_predict_head(rows.loc[non], test_rows, np.array([point_depth_id(v) for v in y[non]], dtype=int), feature_cols, 3, 1889),
        width=fit_predict_head(rows.loc[non], test_rows, np.array([point_width_id(v) for v in y[non]], dtype=int), feature_cols, 2, 1899),
        safety=fit_predict_head(rows.loc[non], test_rows, np.array([point_safety_id(v) for v in y[non]], dtype=int), feature_cols, 3, 1909),
    )


def add_head_features(df: pd.DataFrame, heads: HeadPredictions) -> pd.DataFrame:
    out = df.copy()
    for name, arr in [("terminal", heads.terminal), ("depth", heads.depth), ("width", heads.width), ("safety", heads.safety)]:
        for k in range(arr.shape[1]):
            out[f"A_{name}_{k}"] = arr[:, k]
    return out


def train_point_oof(rows: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), 10), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        valid = rows["fold"].eq(fold).to_numpy()
        train = ~valid
        out[valid] = fit_predict_head(rows.loc[train], rows.loc[valid], y[train], feature_cols, 10, 1910 + int(fold))
    return normalize_rows_safe(out)


def train_point_test(rows: pd.DataFrame, test_rows: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> np.ndarray:
    return fit_predict_head(rows, test_rows, y, feature_cols, 10, 1919)


def capped_labels(base_labels: np.ndarray, prob: np.ndarray, changed: np.ndarray) -> np.ndarray:
    out = np.asarray(base_labels, dtype=int).copy()
    pred = np.asarray(prob).argmax(axis=1).astype(int)
    out[changed] = pred[changed]
    return out


def macro(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def per_class(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    return {f"point{k}_f1": float(rep[str(k)]["f1-score"]) for k in [0, 1, 3, 4, 7, 8, 9]}


def eval_candidate(
    name: str,
    source: str,
    y: np.ndarray,
    pred: np.ndarray,
    base: np.ndarray,
    alpha: float,
    cap: float,
    teacher_tag: str,
) -> dict:
    rec = {
        "candidate": name,
        "source": source,
        "teacher_tag": teacher_tag,
        "alpha": float(alpha),
        "churn_cap": float(cap),
        "point_macro_f1": macro(y, pred),
        "delta_vs_base": macro(y, pred) - macro(y, base),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "terminal_f1": float(f1_score((y == 0).astype(int), (pred == 0).astype(int), average="binary", zero_division=0)),
        "depth_macro_f1": float(f1_score(labels_depth(y), labels_depth(pred), labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "width_macro_f1": float(f1_score(labels_width(y), labels_width(pred), labels=[0, 1, 2], average="macro", zero_division=0)),
    }
    rec.update(per_class(y, pred))
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
    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()

    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    train_priors = pd.read_csv(R186_TRAIN)
    test_priors = pd.read_csv(R186_TEST)
    rows = add_r186_priors(rows, train_priors)
    test_rows = add_r186_priors(test_rows, test_priors)

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

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    test_base_point = base_sub["pointId"].astype(int).to_numpy()

    rows = add_base_prob_features(rows, local_base_prob_oof, "B_point_")
    test_rows = add_base_prob_features(test_rows, local_base_prob_test, "B_point_")
    y = rows["next_pointId"].astype(int).to_numpy()

    aux_feature_cols = base_feature_columns(rows) + [f"B_point_{i}" for i in range(10)] + R187_TEACHER_COLUMNS
    aux_oof = train_aux_oof(rows, aux_feature_cols, y)
    aux_test = train_aux_test(rows, test_rows, aux_feature_cols, y)
    rows_point = add_head_features(rows, aux_oof)
    test_point = add_head_features(test_rows, aux_test)
    point_feature_cols = base_feature_columns(rows_point) + [f"B_point_{i}" for i in range(10)]
    point_feature_cols += [c for c in rows_point.columns if c.startswith("A_")]

    direct_oof = train_point_oof(rows_point, point_feature_cols, y)
    direct_test = train_point_test(rows_point, test_point, point_feature_cols, y)

    teacher_oof = teacher_arrays(rows)
    teacher_test = teacher_arrays(test_rows)
    base_f1 = macro(y, local_base_pred_oof)
    search_rows = [
        eval_candidate("local_v173_r119_base", "base", y, local_base_pred_oof, local_base_pred_oof, 0.0, 0.0, "none")
    ]
    pred_store: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    source_probs: list[tuple[str, str, np.ndarray, np.ndarray]] = [("r187a_point_student", "none", direct_oof, direct_test)]
    for tag, weights in TEACHER_SETTINGS:
        blended_oof = blend_heads_with_teacher(aux_oof, teacher_oof, weights)
        blended_test = blend_heads_with_teacher(aux_test, teacher_test, weights)
        aux_prob_oof = reconstruct_point_from_heads(local_base_prob_oof, blended_oof)
        aux_prob_test = reconstruct_point_from_heads(local_base_prob_test, blended_test)
        mixed_oof = normalize_rows_safe(0.75 * direct_oof + 0.25 * aux_prob_oof)
        mixed_test = normalize_rows_safe(0.75 * direct_test + 0.25 * aux_prob_test)
        source_probs.append(("r187b_aux_reconstruct", tag, aux_prob_oof, aux_prob_test))
        source_probs.append(("r187c_direct_aux_mix", tag, mixed_oof, mixed_test))

    for source, tag, prob_oof, prob_test in source_probs:
        for alpha in ALPHAS:
            for cap in CHURN_CAPS:
                capped_oof_prob, changed_oof = apply_residual_with_churn_cap(
                    local_base_prob_oof, prob_oof, alpha, cap, local_base_pred_oof
                )
                pred = capped_labels(local_base_pred_oof, capped_oof_prob, changed_oof)
                name = f"{source}_{tag}_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                rec = eval_candidate(name, source, y, pred, local_base_pred_oof, alpha, cap, tag)
                search_rows.append(rec)
                capped_test_prob, changed_test = apply_residual_with_churn_cap(
                    local_base_prob_test, prob_test, alpha, cap, test_base_point
                )
                test_pred = capped_labels(test_base_point, capped_test_prob, changed_test)
                rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base_point))
                rec["test_changed_rows"] = int(np.sum(test_pred != test_base_point))
                pred_store[name] = (pred, test_pred, rec)

    search = pd.DataFrame(search_rows)
    search["tier"] = np.select(
        [search["point_churn_vs_base"].le(0.02), search["point_churn_vs_base"].le(0.05)],
        ["clean", "probe"],
        default="high_churn",
    )
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r187_search.csv", index=False)

    generated = []
    clean_positive = search[(search["tier"].eq("clean")) & (search["source"].ne("base")) & (search["delta_vs_base"].gt(0))].copy()
    for source in ["r187a_point_student", "r187b_aux_reconstruct", "r187c_direct_aux_mix"]:
        part = clean_positive[clean_positive["source"].eq(source)]
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

    report = {
        "verdict": "CANDIDATES_GENERATED" if generated else "NO_POSITIVE_CLEAN_CANDIDATE",
        "base": search[search["source"].eq("base")].iloc[0].to_dict(),
        "base_point_macro_f1": float(base_f1),
        "best_clean": search[search["tier"].eq("clean")].head(15).to_dict(orient="records"),
        "best_probe": search[search["tier"].eq("probe")].head(15).to_dict(orient="records"),
        "generated": generated,
        "teacher_settings": [{"tag": tag, "weights": normalize_teacher_weights(w)} for tag, w in TEACHER_SETTINGS],
        "notes": [
            "R186 priors are used only by terminal/depth/width/safety intermediate heads.",
            "The final point student is trained only on AI CUP exact pointId labels.",
            "Submissions use V173 action and R121 no-old server, with low-churn residual point changes.",
            "No external teacher is mapped to AI CUP pointId 1..9.",
            "TTMATCH is not read by this script.",
        ],
    }
    (OUTDIR / "r187_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r187_report.md").write_text(
        "# R187 Point Intent Student\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Base point Macro-F1: `{report['base_point_macro_f1']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + ("\n".join(f"- `{g['upload_path']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119']:.6f}`" for g in generated) or "- none")
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r187_point_intent_student.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated_count": len(generated), "search": str(OUTDIR / "r187_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
