"""V272 action-conditioned point residual search.

This branch keeps the current clean public anchor fixed for action/server:

  v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv

Only ``pointId`` is edited.  The point residual is conditioned on fixed anchor
action intent, incoming prefix context, and fold-safe action/point compatibility
tables learned from train only.  Raw model/table full replacements are written
only as diagnostics in the search table; no raw point submission is emitted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis_v263_questionnaire_baseline_helpers import point_depth, point_side
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


OUTDIR = Path("v272_action_conditioned_point_residual")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
MAX_LAG = 6
POINT0_RATE_MIN = 0.24
POINT0_RATE_MAX = 0.31
MAX_POINT0_ADDED_SAFE = 8
RARE_POINT_CLASSES = [1, 3, 4, 7, 8, 9]
CAP_CONFIGS = [
    ("v272_point_actioncond_cap0p005", "submission_v272_point_actioncond_cap0p005__v173action_r121server.csv", 0.005),
    ("v272_point_actioncond_cap0p010", "submission_v272_point_actioncond_cap0p010__v173action_r121server.csv", 0.010),
    ("v272_point_actioncond_cap0p015", "submission_v272_point_actioncond_cap0p015__v173action_r121server.csv", 0.015),
]
TABLE_CONFIG = (
    "v272_point_actioncond_table_cap0p010",
    "submission_v272_point_actioncond_table_cap0p010__v173action_r121server.csv",
    0.010,
)
BLOCKED_FEATURES = {
    "rally_uid",
    "match",
    "server_id",
    "receiver_id",
    "gamePlayerId",
    "gamePlayerOtherId",
    "scoreSelf",
    "scoreOther",
    "next_actionId",
    "next_pointId",
    "next_is_terminal",
    "serverGetPoint",
    "remaining_len",
    "final_parity_even",
    "num_prefixes_in_rally",
    "fold",
}
TABLE_CONTEXT = ["v272_anchor_action_family", "phase", "lag0_point_depth", "lag0_action_family"]


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"bad submission columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"bad submission rows: {len(df)}")


def action_family(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return 0
    if 1 <= action <= 7:
        return 1
    if 8 <= action <= 11:
        return 2
    if 12 <= action <= 14:
        return 3
    if 15 <= action <= 18:
        return 4
    return 0


def prefix_bin(prefix_len: int) -> int:
    val = int(prefix_len)
    if val <= 1:
        return 1
    if val == 2:
        return 2
    if val == 3:
        return 3
    if 4 <= val <= 6:
        return 4
    return 5


def phase_from_prefix(prefix_len: int) -> int:
    val = int(prefix_len)
    if val <= 1:
        return 0
    if val == 2:
        return 1
    if val == 3:
        return 2
    return 3


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def safe_predict_proba(model: ExtraTreesClassifier, frame: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(classes)), dtype=float)
    class_to_pos = {int(cls): i for i, cls in enumerate(classes)}
    for j, cls in enumerate(model.classes_):
        pos = class_to_pos.get(int(cls))
        if pos is not None:
            out[:, pos] = raw[:, j]
    return normalize_rows_safe(out)


def clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].replace([np.inf, -np.inf], 0).fillna(0)


def load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing clean anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_frame(anchor)
    return anchor


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prefix_bin"] = out["prefix_len"].astype(int).map(prefix_bin)
    out["phase"] = out["prefix_len"].astype(int).map(phase_from_prefix)
    for lag in ["lag0", "lag1"]:
        action_col = f"{lag}_actionId"
        point_col = f"{lag}_pointId"
        if action_col in out:
            out[f"{lag}_action_family"] = out[action_col].astype(int).map(action_family)
        if point_col in out:
            out[f"{lag}_point_depth"] = out[point_col].astype(int).map(point_depth)
            out[f"{lag}_point_side"] = out[point_col].astype(int).map(point_side)
            out[f"{lag}_point_is_zero"] = out[point_col].astype(int).eq(0).astype(int)
    return out


def add_test_anchor_columns(test_df: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    out = test_df.merge(anchor[["rally_uid", "actionId", "pointId"]], on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId"]].isna().any().any():
        raise ValueError("V261 anchor did not align one-to-one with test prefix rows.")
    out = out.rename(columns={"actionId": "v272_anchor_action", "pointId": "v272_anchor_point"})
    out["v272_anchor_action"] = out["v272_anchor_action"].astype(int)
    out["v272_anchor_action_family"] = out["v272_anchor_action"].map(action_family)
    out["v272_anchor_point"] = out["v272_anchor_point"].astype(int)
    out["v272_anchor_point_depth"] = out["v272_anchor_point"].map(point_depth)
    out["v272_anchor_point_side"] = out["v272_anchor_point"].map(point_side)
    out["v272_anchor_point_is_zero"] = out["v272_anchor_point"].eq(0).astype(int)
    return out


def numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame, *, include_anchor: bool) -> list[str]:
    blocked = set(BLOCKED_FEATURES)
    if not include_anchor:
        blocked.update(
            {
                "v272_anchor_action",
                "v272_anchor_action_family",
                "v272_anchor_point",
                "v272_anchor_point_depth",
                "v272_anchor_point_side",
                "v272_anchor_point_is_zero",
            }
        )
        blocked.update({f"v272_table_p{cls}" for cls in POINT_CLASSES})
        blocked.update({"v272_table_top_point", "v272_table_top_prob", "v272_table_anchor_prob", "v272_table_margin"})
    features: list[str] = []
    for col in train_df.columns:
        if col in blocked or col not in test_df:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            features.append(col)
    leaked = [c for c in features if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player leakage features detected: {leaked}")
    return features


def assign_folds(train_df: pd.DataFrame) -> pd.Series:
    folds = pd.Series(-1, index=train_df.index, dtype=int)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, valid_idx) in enumerate(splitter.split(train_df, groups=train_df["match"].astype(int))):
        folds.iloc[valid_idx] = fold
    if folds.lt(0).any():
        raise RuntimeError("fold assignment failed")
    return folds


def fit_action_model(fold: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=160,
        min_samples_leaf=5,
        class_weight="balanced",
        max_features="sqrt",
        random_state=27200 + fold,
        n_jobs=1,
    )


def fit_point_model(fold: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=260,
        min_samples_leaf=4,
        class_weight="balanced",
        max_features="sqrt",
        random_state=27300 + fold,
        n_jobs=1,
    )


def add_action_anchor_proxy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    base_features = numeric_features(train_df, test_df, include_anchor=False)
    y = train_df["next_actionId"].astype(int).to_numpy()
    oof = np.zeros((len(train_df), len(ACTION_CLASSES)), dtype=float)
    folds: list[dict[str, int]] = []
    x_test = clean_matrix(test_df, base_features)
    test_sum = np.zeros((len(test_df), len(ACTION_CLASSES)), dtype=float)
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(fold).to_numpy()
        fit = ~valid
        model = fit_action_model(fold)
        model.fit(clean_matrix(train_df.loc[fit], base_features), y[fit])
        oof[valid] = safe_predict_proba(model, clean_matrix(train_df.loc[valid], base_features), ACTION_CLASSES)
        test_sum += safe_predict_proba(model, x_test, ACTION_CLASSES)
        folds.append({"stage": "foldsafe_action_anchor_proxy", "fold": int(fold), "train_rows": int(fit.sum()), "valid_rows": int(valid.sum())})
        print(f"action proxy fold {fold}: train={int(fit.sum())} valid={int(valid.sum())}")

    train_out = train_df.copy()
    train_out["v272_anchor_action"] = np.asarray(ACTION_CLASSES, dtype=int)[oof.argmax(axis=1)]
    train_out["v272_anchor_action_family"] = train_out["v272_anchor_action"].astype(int).map(action_family)
    train_out["v272_anchor_point"] = train_out["lag0_pointId"].astype(int)
    train_out["v272_anchor_point_depth"] = train_out["v272_anchor_point"].map(point_depth)
    train_out["v272_anchor_point_side"] = train_out["v272_anchor_point"].map(point_side)
    train_out["v272_anchor_point_is_zero"] = train_out["v272_anchor_point"].eq(0).astype(int)
    test_out = test_df.copy()
    test_out["v272_action_proxy_top"] = np.asarray(ACTION_CLASSES, dtype=int)[normalize_rows_safe(test_sum / len(folds)).argmax(axis=1)]
    return train_out, test_out, folds


def conditional_point_table(fit_df: pd.DataFrame, pred_df: pd.DataFrame, *, alpha: float = 18.0) -> np.ndarray:
    counts = (
        fit_df.groupby(TABLE_CONTEXT + ["next_pointId"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=POINT_CLASSES, fill_value=0)
    )
    global_counts = np.bincount(fit_df["next_pointId"].astype(int).to_numpy(), minlength=len(POINT_CLASSES)).astype(float)
    global_prior = global_counts / max(global_counts.sum(), 1.0)
    pred_index = pd.MultiIndex.from_frame(pred_df[TABLE_CONTEXT].astype(int))
    aligned = counts.reindex(pred_index).fillna(0.0).to_numpy(dtype=float)
    smoothed = aligned + alpha * global_prior[None, :]
    denom = smoothed.sum(axis=1, keepdims=True)
    return normalize_rows_safe(smoothed / np.maximum(denom, 1e-12))


def add_table_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, int]]]:
    table_oof = np.zeros((len(train_df), len(POINT_CLASSES)), dtype=float)
    folds: list[dict[str, int]] = []
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(fold).to_numpy()
        fit = ~valid
        table_oof[valid] = conditional_point_table(train_df.loc[fit], train_df.loc[valid])
        folds.append({"stage": "foldsafe_action_point_table", "fold": int(fold), "train_rows": int(fit.sum()), "valid_rows": int(valid.sum())})
        print(f"compat table fold {fold}: train={int(fit.sum())} valid={int(valid.sum())}")
    table_test = conditional_point_table(train_df, test_df)

    train_out = train_df.copy()
    test_out = test_df.copy()
    for pos, cls in enumerate(POINT_CLASSES):
        train_out[f"v272_table_p{cls}"] = table_oof[:, pos]
        test_out[f"v272_table_p{cls}"] = table_test[:, pos]

    classes = np.asarray(POINT_CLASSES, dtype=int)
    for frame, prob in [(train_out, table_oof), (test_out, table_test)]:
        top_pos = prob.argmax(axis=1)
        base = frame["v272_anchor_point"].astype(int).clip(0, len(POINT_CLASSES) - 1).to_numpy()
        frame["v272_table_top_point"] = classes[top_pos]
        frame["v272_table_top_prob"] = prob[np.arange(len(frame)), top_pos]
        frame["v272_table_anchor_prob"] = prob[np.arange(len(frame)), base]
        frame["v272_table_margin"] = frame["v272_table_top_prob"] - frame["v272_table_anchor_prob"]
    return train_out, test_out, normalize_rows_safe(table_oof), normalize_rows_safe(table_test), folds


def train_point_oof_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    y = train_df["next_pointId"].astype(int).to_numpy()
    oof = np.zeros((len(train_df), len(POINT_CLASSES)), dtype=float)
    test_sum = np.zeros((len(test_df), len(POINT_CLASSES)), dtype=float)
    x_test = clean_matrix(test_df, features)
    folds: list[dict[str, int]] = []
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(fold).to_numpy()
        fit = ~valid
        model = fit_point_model(fold)
        model.fit(clean_matrix(train_df.loc[fit], features), y[fit])
        oof[valid] = safe_predict_proba(model, clean_matrix(train_df.loc[valid], features), POINT_CLASSES)
        test_sum += safe_predict_proba(model, x_test, POINT_CLASSES)
        folds.append({"stage": "point_model", "fold": int(fold), "train_rows": int(fit.sum()), "valid_rows": int(valid.sum())})
        print(f"point model fold {fold}: train={int(fit.sum())} valid={int(valid.sum())}")
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / len(folds)), folds


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = add_context_columns(build_train_prefix_table(train_raw, MAX_LAG))
    test_df = add_context_columns(build_test_prefix_table(test_raw, MAX_LAG))
    anchor = load_anchor_submission()
    if len(test_df) != EXPECTED_ROWS:
        raise ValueError(f"test prefix rows={len(test_df)}, expected {EXPECTED_ROWS}")
    test_df = add_test_anchor_columns(test_df, anchor)
    train_df["fold"] = assign_folds(train_df)
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]) and col not in {"fold"}:
            test_df[col] = 0
    train_df, test_df, action_folds = add_action_anchor_proxy(train_df, test_df)
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]) and col not in {"fold"}:
            test_df[col] = 0
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), anchor, action_folds


def residual_labels(
    base_labels: np.ndarray,
    prob: np.ndarray,
    cap: float | None,
    *,
    enforce_point0: bool,
    max_point0_added: int = MAX_POINT0_ADDED_SAFE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    p = normalize_rows_safe(prob)
    classes = np.asarray(POINT_CLASSES, dtype=int)
    top_pos = p.argmax(axis=1)
    top = classes[top_pos]
    base_pos = np.array([POINT_CLASSES.index(int(label)) if int(label) in POINT_CLASSES else 0 for label in base], dtype=int)
    gain = p[np.arange(len(p)), top_pos] - p[np.arange(len(p)), base_pos]
    eligible_score = np.where((top != base) & np.isfinite(gain) & (gain > 0), gain, -np.inf)
    budget = int(np.floor(len(base) * float(cap))) if cap is not None else len(base)
    changed = np.zeros(len(base), dtype=bool)
    out = base.copy()
    if budget <= 0:
        return out, changed, gain

    point0_count = int(np.sum(out == 0))
    point0_added = 0
    order = np.argsort(-eligible_score, kind="mergesort")
    accepted = 0
    for idx in order:
        if not np.isfinite(eligible_score[idx]) or accepted >= budget:
            break
        old = int(out[idx])
        new = int(top[idx])
        next_point0_count = point0_count + int(old != 0 and new == 0) - int(old == 0 and new != 0)
        next_point0_added = point0_added + int(old != 0 and new == 0)
        next_rate = next_point0_count / len(out)
        if enforce_point0:
            if next_point0_added > max_point0_added:
                continue
            if not (POINT0_RATE_MIN <= next_rate <= POINT0_RATE_MAX):
                continue
        out[idx] = new
        changed[idx] = True
        point0_count = next_point0_count
        point0_added = next_point0_added
        accepted += 1
    return out, changed, gain


def class_f1(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    return float(f1_score(np.asarray(y_true, dtype=int) == int(cls), np.asarray(y_pred, dtype=int) == int(cls), zero_division=0))


def rare_point_mean_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([class_f1(y_true, y_pred, cls) for cls in RARE_POINT_CLASSES]))


def distribution_json(labels: np.ndarray) -> str:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(POINT_CLASSES))
    return pd.Series({str(i): int(v) for i, v in enumerate(counts) if v > 0}).to_json()


def candidate_verdict(
    *,
    diagnostic: bool,
    ordinary_delta_vs_base: float,
    point_churn: float,
    point0_rate_test: float,
    point0_added_rows: int,
) -> str:
    if diagnostic:
        return "DIAGNOSTIC_ONLY"
    if not (POINT0_RATE_MIN <= point0_rate_test <= POINT0_RATE_MAX):
        return "REJECT_POINT0_RATE"
    if point0_added_rows > MAX_POINT0_ADDED_SAFE:
        return "REJECT_POINT0_ADDED_ROWS"
    if point_churn > 0.015 + (1.0 / EXPECTED_ROWS):
        return "REJECT_CHURN"
    if ordinary_delta_vs_base >= 0.0015:
        return "CANDIDATE_FOR_PUBLIC_PROBE"
    if ordinary_delta_vs_base > 0.0:
        return "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def write_submission(path: Path, anchor: pd.DataFrame, point_pred: np.ndarray) -> None:
    out = anchor.copy()
    out["pointId"] = np.asarray(point_pred, dtype=int)
    out = out[EXPECTED_COLUMNS]
    validate_submission_frame(out)
    if not out["actionId"].equals(anchor["actionId"]):
        raise ValueError("actionId changed; V272 must keep anchor action fixed")
    if not np.allclose(out["serverGetPoint"].astype(float), anchor["serverGetPoint"].astype(float)):
        raise ValueError("serverGetPoint changed; V272 must keep anchor server fixed")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.8f")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, UPLOAD_DIR / path.name)


def add_candidate_record(
    *,
    records: list[dict[str, object]],
    submissions: list[str],
    candidate: str,
    path: Path | None,
    anchor: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    base_macro_f1: float,
    base_rare_f1: float,
    oof_pred: np.ndarray,
    test_pred: np.ndarray,
    test_changed: np.ndarray,
    cap: float | None,
    diagnostic: bool,
    source: str,
) -> None:
    test_base = anchor["pointId"].astype(int).to_numpy()
    ordinary_point_macro_f1 = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    rare_f1 = rare_point_mean_f1(y, oof_pred)
    point_churn = float(np.mean(test_changed))
    point0_rate_test = float(np.mean(np.asarray(test_pred, dtype=int) == 0))
    point0_added_rows = int(np.sum((test_base != 0) & (np.asarray(test_pred, dtype=int) == 0)))
    point0_removed_rows = int(np.sum((test_base == 0) & (np.asarray(test_pred, dtype=int) != 0)))
    verdict = candidate_verdict(
        diagnostic=diagnostic,
        ordinary_delta_vs_base=ordinary_point_macro_f1 - base_macro_f1,
        point_churn=point_churn,
        point0_rate_test=point0_rate_test,
        point0_added_rows=point0_added_rows,
    )
    if path is not None:
        write_submission(path, anchor, test_pred)
        submissions.append(path.name)
    records.append(
        {
            "candidate": candidate,
            "path": "" if path is None else str(path),
            "ordinary_point_macro_f1": ordinary_point_macro_f1,
            "ordinary_delta_vs_base": ordinary_point_macro_f1 - base_macro_f1,
            "point_churn": point_churn,
            "point0_rate_test": point0_rate_test,
            "point0_added_rows": point0_added_rows,
            "rare_point_mean_f1": rare_f1,
            "verdict": verdict,
            "source": source,
            "cap": np.nan if cap is None else float(cap),
            "changed_rows": int(np.sum(test_changed)),
            "point0_removed_rows": point0_removed_rows,
            "base_rare_point_mean_f1": base_rare_f1,
            "test_point_distribution": distribution_json(test_pred),
        }
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, anchor, action_folds = build_frames()
    train_df, test_df, table_oof, table_test, table_folds = add_table_features(train_df, test_df)
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]) and col not in {"fold"}:
            test_df[col] = 0

    features = numeric_features(train_df, test_df, include_anchor=True)
    y = train_df["next_pointId"].astype(int).to_numpy()
    print(f"train rows={len(train_df)} test rows={len(test_df)} features={len(features)}")
    model_oof, model_test, point_folds = train_point_oof_test(train_df, test_df, features)

    base_oof = train_df["lag0_pointId"].astype(int).clip(0, len(POINT_CLASSES) - 1).to_numpy()
    base_macro_f1 = float(f1_score(y, base_oof, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_rare_f1 = rare_point_mean_f1(y, base_oof)
    test_base = anchor["pointId"].astype(int).to_numpy()

    records: list[dict[str, object]] = []
    submissions: list[str] = []
    for candidate, filename, cap in CAP_CONFIGS:
        oof_pred, _, _ = residual_labels(base_oof, model_oof, cap, enforce_point0=False)
        test_pred, test_changed, _ = residual_labels(test_base, model_test, cap, enforce_point0=True)
        add_candidate_record(
            records=records,
            submissions=submissions,
            candidate=candidate,
            path=OUTDIR / filename,
            anchor=anchor,
            y=y,
            base_oof=base_oof,
            base_macro_f1=base_macro_f1,
            base_rare_f1=base_rare_f1,
            oof_pred=oof_pred,
            test_pred=test_pred,
            test_changed=test_changed,
            cap=cap,
            diagnostic=False,
            source="model_residual",
        )

    table_candidate, table_filename, table_cap = TABLE_CONFIG
    table_oof_pred, _, _ = residual_labels(base_oof, table_oof, table_cap, enforce_point0=False)
    table_test_pred, table_test_changed, _ = residual_labels(test_base, table_test, table_cap, enforce_point0=True)
    add_candidate_record(
        records=records,
        submissions=submissions,
        candidate=table_candidate,
        path=OUTDIR / table_filename,
        anchor=anchor,
        y=y,
        base_oof=base_oof,
        base_macro_f1=base_macro_f1,
        base_rare_f1=base_rare_f1,
        oof_pred=table_oof_pred,
        test_pred=table_test_pred,
        test_changed=table_test_changed,
        cap=table_cap,
        diagnostic=False,
        source="conditional_table_residual",
    )

    classes = np.asarray(POINT_CLASSES, dtype=int)
    raw_model_oof = classes[model_oof.argmax(axis=1)]
    raw_model_test = classes[model_test.argmax(axis=1)]
    add_candidate_record(
        records=records,
        submissions=submissions,
        candidate="v272_point_actioncond_raw_model_diagnostic",
        path=None,
        anchor=anchor,
        y=y,
        base_oof=base_oof,
        base_macro_f1=base_macro_f1,
        base_rare_f1=base_rare_f1,
        oof_pred=raw_model_oof,
        test_pred=raw_model_test,
        test_changed=raw_model_test != test_base,
        cap=None,
        diagnostic=True,
        source="raw_model_no_submission",
    )
    raw_table_oof = classes[table_oof.argmax(axis=1)]
    raw_table_test = classes[table_test.argmax(axis=1)]
    add_candidate_record(
        records=records,
        submissions=submissions,
        candidate="v272_point_actioncond_raw_table_diagnostic",
        path=None,
        anchor=anchor,
        y=y,
        base_oof=base_oof,
        base_macro_f1=base_macro_f1,
        base_rare_f1=base_rare_f1,
        oof_pred=raw_table_oof,
        test_pred=raw_table_test,
        test_changed=raw_table_test != test_base,
        cap=None,
        diagnostic=True,
        source="raw_table_no_submission",
    )

    search = pd.DataFrame(records)
    ordered_cols = [
        "candidate",
        "path",
        "ordinary_point_macro_f1",
        "ordinary_delta_vs_base",
        "point_churn",
        "point0_rate_test",
        "point0_added_rows",
        "rare_point_mean_f1",
        "verdict",
    ]
    search = search[ordered_cols + [c for c in search.columns if c not in ordered_cols]]
    search.to_csv(OUTDIR / "v272_point_search.csv", index=False)

    uploadable = search[search["verdict"].eq("CANDIDATE_FOR_PUBLIC_PROBE")]
    if uploadable.empty:
        best = search[~search["candidate"].str.contains("diagnostic")].sort_values(
            ["ordinary_delta_vs_base", "point0_added_rows", "point_churn"], ascending=[False, True, True]
        ).iloc[0]
    else:
        best = uploadable.sort_values(["ordinary_delta_vs_base", "point_churn"], ascending=[False, True]).iloc[0]

    report_lines = [
        "# V272 Action-Conditioned Point Residual",
        "",
        "Fixed clean anchor:",
        "",
        "```text",
        str(ANCHOR_PATH),
        "action = fixed V173 action anchor",
        "server = fixed R121 server anchor",
        "changed field = pointId only",
        "```",
        "",
        "## Summary",
        "",
        f"- Train prefix rows: `{len(train_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Numeric feature count: `{len(features)}`",
        f"- Base lag0 point Macro-F1 proxy: `{base_macro_f1:.6f}`",
        f"- Base rare point mean F1: `{base_rare_f1:.6f}`",
        f"- Anchor point0 rate: `{float(np.mean(test_base == 0)):.6f}`",
        f"- Best non-diagnostic row: `{best['candidate']}` / verdict `{best['verdict']}`",
        f"- Submissions copied to `{UPLOAD_DIR}`: `{len(submissions)}`",
        "",
        "## Candidates",
        "",
    ]
    for row in search.to_dict("records"):
        report_lines.append(
            f"- `{row['candidate']}`: OOF={float(row['ordinary_point_macro_f1']):.6f}, "
            f"delta={float(row['ordinary_delta_vs_base']):.6f}, "
            f"churn={float(row['point_churn']):.6f}, "
            f"point0_rate={float(row['point0_rate_test']):.6f}, "
            f"point0_added={int(row['point0_added_rows'])}, verdict=`{row['verdict']}`"
        )
    report_lines.extend(
        [
            "",
            "## Policy Checks",
            "",
            "- No TTMATCH inputs are read.",
            "- No old-server or old-test labels are read.",
            "- No raw point model/table submission is written; raw rows are diagnostics only.",
            f"- Submission caps are limited to `0.5%`, `1.0%`, and `1.5%`.",
            f"- Test point0 rate gate is `[{POINT0_RATE_MIN:.2f}, {POINT0_RATE_MAX:.2f}]`; point0 additions are capped at `{MAX_POINT0_ADDED_SAFE}` rows.",
            "- All emitted submissions preserve anchor `actionId` and `serverGetPoint`.",
        ]
    )
    (OUTDIR / "v272_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTDIR / "v272_run_summary.json").write_text(
        json.dumps(
            {
                "branch": "v272_action_conditioned_point_residual",
                "outdir": str(OUTDIR),
                "anchor": str(ANCHOR_PATH),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "feature_count": int(len(features)),
                "base_point_macro_f1": base_macro_f1,
                "best_candidate": best.to_dict(),
                "generated_submissions": submissions,
                "copied_to_upload_candidates": True,
                "folds": action_folds + table_folds + point_folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "generated_submissions": submissions,
                "best_candidate": str(best["candidate"]),
                "best_verdict": str(best["verdict"]),
                "best_delta": float(best["ordinary_delta_vs_base"]),
                "best_point0_rate_test": float(best["point0_rate_test"]),
                "best_point_churn": float(best["point_churn"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
