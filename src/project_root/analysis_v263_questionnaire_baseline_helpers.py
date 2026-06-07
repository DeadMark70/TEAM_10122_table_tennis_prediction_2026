"""Shared helpers for V263 questionnaire-style baseline experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


OUTDIR = Path("v263_questionnaire_baseline")
CURRENT_ANCHOR = Path("upload_candidates_20260519/submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv")
V261_CAP1 = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero, :] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def cap_by_score(scores: np.ndarray, cap: float) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    budget = int(np.floor(len(arr) * float(cap)))
    mask = np.zeros(len(arr), dtype=bool)
    if budget <= 0:
        return mask
    clean = np.where(np.isfinite(arr), arr, -np.inf)
    order = np.argsort(-clean, kind="mergesort")[:budget]
    mask[order] = True
    return mask


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 3:
        return 1
    if 4 <= point <= 6:
        return 2
    if 7 <= point <= 9:
        return 3
    return 0


def point_side(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 9:
        return ((point - 1) % 3) + 1
    return 0


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


def class_f1_table(y_true: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray, classes: list[int]) -> pd.DataFrame:
    rows = []
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(base_pred, dtype=int)
    cand = np.asarray(cand_pred, dtype=int)
    for cls in classes:
        base_f1 = f1_score(y == cls, base == cls, zero_division=0)
        cand_f1 = f1_score(y == cls, cand == cls, zero_division=0)
        rows.append({"class_id": int(cls), "base_f1": float(base_f1), "candidate_f1": float(cand_f1), "delta_f1": float(cand_f1 - base_f1)})
    return pd.DataFrame(rows)


def log_loss_safe_prob(base_prob: np.ndarray, model_prob: np.ndarray, weight: float) -> np.ndarray:
    base = normalize_rows(base_prob)
    model = normalize_rows(model_prob)
    w = float(weight)
    logits = (1.0 - w) * np.log(np.clip(base, 1e-12, 1.0)) + w * np.log(np.clip(model, 1e-12, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_rows(np.exp(logits))


def safe_predict_proba(model, frame: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(classes)), dtype=float)
    class_to_pos = {int(cls): i for i, cls in enumerate(classes)}
    for j, cls in enumerate(model.classes_):
        pos = class_to_pos.get(int(cls))
        if pos is not None:
            out[:, pos] = raw[:, j]
    return normalize_rows(out)


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} has columns {list(sub.columns)}")
    if len(sub) != 1845:
        raise ValueError(f"{path} has {len(sub)} rows, expected 1845")
    return sub


def load_current_anchor_submission() -> pd.DataFrame:
    return load_submission(CURRENT_ANCHOR)


def load_v261_cap1_anchor() -> pd.DataFrame:
    """Return public-confirmed clean anchor: V173 action, V261 cap1 point, R121 server."""
    if V261_CAP1.exists():
        return load_submission(V261_CAP1)
    return load_current_anchor_submission()


def write_local_submission(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[EXPECTED_COLUMNS].copy()
    if len(out) != 1845:
        raise ValueError(f"{path} has {len(out)} rows, expected 1845")
    out.to_csv(path, index=False, float_format="%.8f")


def distribution_json(labels: np.ndarray, classes: int) -> str:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=classes)
    return pd.Series({str(i): int(v) for i, v in enumerate(counts) if v > 0}).to_json()


def add_questionnaire_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in ["lag0", "lag1"]:
        action_col = f"{lag}_actionId"
        point_col = f"{lag}_pointId"
        if action_col in out:
            out[f"{lag}_action_family"] = out[action_col].astype(int).map(action_family)
        if point_col in out:
            out[f"{lag}_point_depth"] = out[point_col].astype(int).map(point_depth)
            out[f"{lag}_point_side"] = out[point_col].astype(int).map(point_side)
    if "prefix_len" in out:
        prefix = pd.to_numeric(out["prefix_len"], errors="coerce").fillna(0)
        out["prefix_bin"] = np.select(
            [prefix.eq(1), prefix.eq(2), prefix.eq(3), prefix.between(4, 6), prefix.ge(7)],
            [1, 2, 3, 4, 5],
            default=0,
        )
    return out


def numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame, blocked: set[str]) -> list[str]:
    features = []
    for col in train_df.columns:
        if col in blocked or col not in test_df:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            features.append(col)
    return features
