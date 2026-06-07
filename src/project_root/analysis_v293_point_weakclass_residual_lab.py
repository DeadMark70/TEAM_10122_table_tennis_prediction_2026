from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


POINT_GROUPS = {
    "rare134": [1, 3, 4],
    "long789": [7, 8, 9],
    "point0": [0],
}

LOCAL_MOVES = {
    "long789": {7: [8, 9], 8: [7, 9], 9: [7, 8]},
    "rare134": {1: [3, 4], 3: [1, 4], 4: [1, 3, 7]},
    "point0": {0: [0]},
}

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v293_point_weakclass_residual_lab"
ANCHOR_PATH = ROOT / "upload_candidates_20260519" / (
    "submission_v261_cap0p01__v173action_r121server.csv"
)
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
POINT_CLASSES = list(range(10))
CAPS = [0.0025, 0.005, 0.010]
TARGET_CLASSES = [0, 1, 3, 4, 7, 8, 9]
TERMINAL_ACTIONS = {0, 3}


def point_depth(point_id: int) -> str:
    point_id = int(point_id)
    if point_id == 0:
        return "zero"
    if point_id in {1, 2, 3}:
        return "short"
    if point_id in {4, 5, 6}:
        return "half"
    if point_id in {7, 8, 9}:
        return "long"
    return "zero"


def point_side(point_id: int) -> str:
    point_id = int(point_id)
    if point_id in {1, 4, 7}:
        return "forehand"
    if point_id in {2, 5, 8}:
        return "middle"
    if point_id in {3, 6, 9}:
        return "backhand"
    return "none"


def normalize_score01(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, 0.0, 1.0)


def preserve_long_identity(base_point: int, candidate_point: int) -> bool:
    return point_depth(base_point) == "long" and point_depth(candidate_point) == "long"


def point0_addition_allowed(
    base_point: int, p0_score: float, phase: str, terminal_proxy: float
) -> bool:
    if int(base_point) == 0:
        return False
    if float(p0_score) < 0.90:
        return False
    if float(terminal_proxy) < 0.75:
        return False
    return str(phase) in {"third", "fourth", "rally"}


def apply_point_caps(
    base: np.ndarray, candidates: pd.DataFrame, max_churn: float
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(base, dtype=int).copy()
    selected = np.zeros(len(pred), dtype=bool)
    max_rows = int(np.floor(len(pred) * float(max_churn)))
    if max_rows <= 0 or candidates.empty:
        return pred, selected
    ranked = candidates.sort_values(["score"], ascending=False).head(max_rows)
    row_ids = ranked["row_id"].astype(int).to_numpy()
    selected[row_ids] = True
    pred[row_ids] = ranked["candidate_point"].astype(int).to_numpy()
    return pred, selected


def validate_submission_frame(df: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"bad submission columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"bad submission rows: {len(df)}")
    if not df["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of [0, 9]")


def depth_code(point_id: int) -> int:
    return {"zero": 0, "short": 1, "half": 2, "long": 3}[point_depth(point_id)]


def side_code(point_id: int) -> int:
    return {"none": 0, "forehand": 1, "middle": 2, "backhand": 3}[point_side(point_id)]


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


def phase_label(prefix_len: int) -> str:
    prefix = int(prefix_len)
    if prefix <= 1:
        return "receive"
    if prefix == 2:
        return "third"
    if prefix == 3:
        return "fourth"
    return "rally"


def phase_code(prefix_len: int) -> int:
    return {"receive": 0, "third": 1, "fourth": 2, "rally": 3}[phase_label(prefix_len)]


def point_distribution(point: np.ndarray) -> str:
    counts = np.bincount(np.asarray(point, dtype=int), minlength=10)
    return json.dumps({str(i): int(v) for i, v in enumerate(counts) if v > 0}, sort_keys=True)


def class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] = POINT_CLASSES) -> dict[int, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        int(label): float(f1_score(y_true == int(label), y_pred == int(label), zero_division=0))
        for label in labels
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] = POINT_CLASSES) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def weighted_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    scores: list[float] = []
    for label in POINT_CLASSES:
        yt = y_true == label
        yp = y_pred == label
        tp = float(weights[yt & yp].sum())
        fp = float(weights[~yt & yp].sum())
        fn = float(weights[yt & ~yp].sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].replace([np.inf, -np.inf], 0).fillna(0.0)


def numeric_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> list[str]:
    blocked = {
        "rally_uid",
        "rally_id",
        "match",
        "server_id",
        "receiver_id",
        "gamePlayerId",
        "gamePlayerOtherId",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "fold",
    }
    features: list[str] = []
    for col in train_frame.columns:
        if col in blocked or col not in test_frame:
            continue
        if pd.api.types.is_numeric_dtype(train_frame[col]):
            features.append(col)
    return features


def assign_folds(train_frame: pd.DataFrame) -> pd.Series:
    folds = pd.Series(-1, index=train_frame.index, dtype=int)
    groups = train_frame["match"].astype(str) if "match" in train_frame else train_frame["rally_uid"].astype(str)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, valid_idx) in enumerate(splitter.split(train_frame, groups=groups)):
        folds.iloc[valid_idx] = fold
    if folds.lt(0).any():
        raise RuntimeError("fold assignment failed")
    return folds


def load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"missing V261 anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_frame(anchor)
    return anchor


def add_context_columns(frame: pd.DataFrame, *, base_point: np.ndarray, anchor_action: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["base_point"] = np.asarray(base_point, dtype=int)
    out["base_point_depth"] = out["base_point"].map(depth_code)
    out["base_point_side"] = out["base_point"].map(side_code)
    out["anchor_action"] = np.asarray(anchor_action, dtype=int)
    out["anchor_action_family"] = out["anchor_action"].map(action_family)
    out["phase_code"] = out["prefix_len"].astype(int).map(phase_code)
    out["is_receive"] = out["prefix_len"].astype(int).le(1).astype(int)
    out["is_third"] = out["prefix_len"].astype(int).eq(2).astype(int)
    out["is_fourth"] = out["prefix_len"].astype(int).eq(3).astype(int)
    out["is_rally"] = out["prefix_len"].astype(int).ge(4).astype(int)
    if "serverScore" in out and "receiverScore" in out:
        out["score_pressure"] = (out["serverScore"].astype(float) + out["receiverScore"].astype(float)).clip(0, 20)
    elif "scoreTotal" in out:
        out["score_pressure"] = out["scoreTotal"].astype(float).clip(0, 20)
    else:
        out["score_pressure"] = 0.0
    out["terminal_proxy"] = (
        out["base_point"].eq(0) | out["anchor_action"].astype(int).isin(TERMINAL_ACTIONS)
    ).astype(float)
    for lag in range(2):
        point_col = f"lag{lag}_pointId"
        action_col = f"lag{lag}_actionId"
        if point_col in out:
            out[f"lag{lag}_point_depth_code"] = out[point_col].astype(int).map(depth_code)
            out[f"lag{lag}_point_side_code"] = out[point_col].astype(int).map(side_code)
        if action_col in out:
            out[f"lag{lag}_action_family"] = out[action_col].astype(int).map(action_family)
    return out


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    from baseline_lgbm import (
        add_role_and_score_features,
        build_test_prefix_table,
        build_train_prefix_table,
        validate_raw_data,
    )

    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_rows = build_train_prefix_table(train_raw, 6).reset_index(drop=True)
    test_rows = build_test_prefix_table(test_raw, 6).reset_index(drop=True)
    if "next_pointId" not in train_rows:
        raise RuntimeError("cannot rebuild V293: train prefix rows lack next_pointId")
    if "lag0_pointId" not in train_rows:
        raise RuntimeError("cannot rebuild V293: train prefix rows lack lag0_pointId base point")
    anchor = load_anchor_submission().reset_index(drop=True)
    if len(test_rows) != len(anchor):
        raise RuntimeError(f"test rows {len(test_rows)} do not align with V261 anchor rows {len(anchor)}")

    y_true_point = train_rows["next_pointId"].astype(int).to_numpy()
    base_point_oof = train_rows["lag0_pointId"].astype(int).clip(0, 9).to_numpy()
    base_point_test = anchor["pointId"].astype(int).to_numpy()
    train_action_proxy = train_rows["lag0_actionId"].astype(int).clip(0, 18).to_numpy()
    test_action_anchor = anchor["actionId"].astype(int).to_numpy()
    train_rows["fold"] = assign_folds(train_rows)
    train_rows = add_context_columns(train_rows, base_point=base_point_oof, anchor_action=train_action_proxy)
    test_rows = add_context_columns(test_rows, base_point=base_point_test, anchor_action=test_action_anchor)
    for col in train_rows.columns:
        if col not in test_rows and pd.api.types.is_numeric_dtype(train_rows[col]) and col != "fold":
            test_rows[col] = 0
    for col in test_rows.columns:
        if col not in train_rows and pd.api.types.is_numeric_dtype(test_rows[col]):
            train_rows[col] = 0
    return train_rows, test_rows, anchor, y_true_point, base_point_oof


def _fit_extra_trees(seed: int, *, binary: bool = False) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=120 if binary else 160,
        min_samples_leaf=10 if binary else 8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def _predict_class_proba(model: ExtraTreesClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=float)
    pos = {int(cls): i for i, cls in enumerate(classes)}
    for j, cls in enumerate(model.classes_):
        if int(cls) in pos:
            out[:, pos[int(cls)]] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        out[zero, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def _predict_binary_proba(model: ExtraTreesClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    if 1 in model.classes_:
        return normalize_score01(raw[:, list(model.classes_).index(1)])
    return np.zeros(len(x), dtype=float)


def train_long789_specialist(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    classes = POINT_GROUPS["long789"]
    oof = np.zeros((len(train_rows), len(classes)), dtype=float)
    test_sum = np.zeros((len(test_rows), len(classes)), dtype=float)
    fitted = 0
    y_local = np.where(np.isin(y_true, classes), y_true, base_oof).astype(int)
    eligible_train = np.isin(y_true, classes) | np.isin(base_oof, classes)
    for fold in sorted(train_rows["fold"].astype(int).unique()):
        valid = train_rows["fold"].astype(int).eq(fold).to_numpy()
        fit = (~valid) & eligible_train
        if len(np.unique(y_local[fit])) < 2:
            continue
        model = _fit_extra_trees(29300 + int(fold))
        model.fit(clean_matrix(train_rows.loc[fit], features), y_local[fit])
        oof[valid] = _predict_class_proba(model, clean_matrix(train_rows.loc[valid], features), classes)
        test_sum += _predict_class_proba(model, clean_matrix(test_rows, features), classes)
        fitted += 1
    if fitted == 0:
        return (
            pd.DataFrame(columns=["row_id", "candidate_point", "score", "specialist"]),
            pd.DataFrame(columns=["row_id", "candidate_point", "score", "specialist"]),
            {"specialist": "long789", "oof_macro_or_auc": 0.0, "positive_rows": int(eligible_train.sum()), "train_rows": 0, "test_candidate_rows": 0},
        )
    test_prob = test_sum / fitted
    train_candidates = long789_candidates(base_oof, oof)
    test_candidates = long789_candidates(test_rows["base_point"].astype(int).to_numpy(), test_prob)
    metric_rows = np.isin(y_true, classes)
    pred = np.asarray(classes, dtype=int)[oof.argmax(axis=1)]
    metric = macro_f1(y_true[metric_rows], pred[metric_rows], classes) if metric_rows.any() else 0.0
    report = {
        "specialist": "long789",
        "oof_macro_or_auc": float(metric),
        "positive_rows": int(metric_rows.sum()),
        "train_rows": int(eligible_train.sum()),
        "test_candidate_rows": int(len(test_candidates)),
    }
    return train_candidates, test_candidates, report


def long789_candidates(base: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    classes = POINT_GROUPS["long789"]
    class_arr = np.asarray(classes, dtype=int)
    base = np.asarray(base, dtype=int)
    rows: list[dict[str, Any]] = []
    for i, old in enumerate(base):
        if int(old) not in classes:
            continue
        best = int(class_arr[int(np.argmax(prob[i]))])
        if best == int(old) or not preserve_long_identity(int(old), best):
            continue
        old_pos = classes.index(int(old))
        best_pos = classes.index(best)
        score = float(prob[i, best_pos] - prob[i, old_pos])
        if score > 0.0:
            rows.append({"row_id": i, "candidate_point": best, "score": score, "specialist": "long789"})
    return pd.DataFrame(rows)


def train_rare134_specialist(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rare = POINT_GROUPS["rare134"]
    train_score = pd.DataFrame(index=train_rows.index)
    test_score = pd.DataFrame(index=test_rows.index)
    eligible_train = np.isin(y_true, rare) | np.isin(base_oof, rare + [7])
    metrics: list[float] = []
    positive_rows = 0
    for cls in rare:
        target = (y_true == int(cls)).astype(int)
        positive_rows += int(target.sum())
        oof = np.zeros(len(train_rows), dtype=float)
        test_sum = np.zeros(len(test_rows), dtype=float)
        fitted = 0
        for fold in sorted(train_rows["fold"].astype(int).unique()):
            valid = train_rows["fold"].astype(int).eq(fold).to_numpy()
            fit = (~valid) & eligible_train
            if len(np.unique(target[fit])) < 2:
                continue
            model = _fit_extra_trees(29400 + int(cls) * 17 + int(fold), binary=True)
            model.fit(clean_matrix(train_rows.loc[fit], features), target[fit])
            oof[valid] = _predict_binary_proba(model, clean_matrix(train_rows.loc[valid], features))
            test_sum += _predict_binary_proba(model, clean_matrix(test_rows, features))
            fitted += 1
        if fitted:
            test_sum /= fitted
        train_score[f"p{cls}"] = normalize_score01(oof)
        test_score[f"p{cls}"] = normalize_score01(test_sum)
        if len(np.unique(target[eligible_train])) > 1:
            metrics.append(float(average_precision_score(target[eligible_train], oof[eligible_train])))
    train_candidates = rare134_candidates(base_oof, train_score)
    test_candidates = rare134_candidates(test_rows["base_point"].astype(int).to_numpy(), test_score)
    report = {
        "specialist": "rare134",
        "oof_macro_or_auc": float(np.mean(metrics)) if metrics else 0.0,
        "positive_rows": int(positive_rows),
        "train_rows": int(eligible_train.sum()),
        "test_candidate_rows": int(len(test_candidates)),
    }
    return train_candidates, test_candidates, report


def rare134_candidates(base: np.ndarray, scores: pd.DataFrame) -> pd.DataFrame:
    rare = POINT_GROUPS["rare134"]
    base = np.asarray(base, dtype=int)
    rows: list[dict[str, Any]] = []
    score_arr = scores[[f"p{cls}" for cls in rare]].to_numpy(dtype=float)
    for i, old in enumerate(base):
        if int(old) in {0, 8, 9}:
            continue
        base_proxy = 0.50
        if int(old) in rare:
            base_proxy = float(scores.iloc[i][f"p{int(old)}"])
        best_pos = int(np.argmax(score_arr[i]))
        best = int(rare[best_pos])
        if best == int(old):
            continue
        score = float(score_arr[i, best_pos] - base_proxy)
        if score > 0.0:
            rows.append({"row_id": i, "candidate_point": best, "score": score, "specialist": "rare134"})
    return pd.DataFrame(rows)


def train_point0_specialist(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = (y_true == 0).astype(int)
    oof = np.zeros(len(train_rows), dtype=float)
    test_sum = np.zeros(len(test_rows), dtype=float)
    fitted = 0
    for fold in sorted(train_rows["fold"].astype(int).unique()):
        valid = train_rows["fold"].astype(int).eq(fold).to_numpy()
        fit = ~valid
        if len(np.unique(target[fit])) < 2:
            continue
        model = _fit_extra_trees(29500 + int(fold), binary=True)
        model.fit(clean_matrix(train_rows.loc[fit], features), target[fit])
        oof[valid] = _predict_binary_proba(model, clean_matrix(train_rows.loc[valid], features))
        test_sum += _predict_binary_proba(model, clean_matrix(test_rows, features))
        fitted += 1
    test_prob = test_sum / fitted if fitted else test_sum
    train_candidates = point0_candidates(base_oof, oof, train_rows)
    test_candidates = point0_candidates(test_rows["base_point"].astype(int).to_numpy(), test_prob, test_rows)
    metric = float(roc_auc_score(target, oof)) if len(np.unique(target)) > 1 else 0.5
    report = {
        "specialist": "point0",
        "oof_macro_or_auc": metric,
        "positive_rows": int(target.sum()),
        "train_rows": int(len(train_rows)),
        "test_candidate_rows": int(len(test_candidates)),
    }
    return train_candidates, test_candidates, report


def point0_candidates(base: np.ndarray, p0_score: np.ndarray, rows: pd.DataFrame) -> pd.DataFrame:
    base = np.asarray(base, dtype=int)
    scores = normalize_score01(p0_score)
    out: list[dict[str, Any]] = []
    phases = rows["prefix_len"].astype(int).map(phase_label).to_numpy()
    terminal = rows["terminal_proxy"].astype(float).to_numpy()
    for i, old in enumerate(base):
        if point0_addition_allowed(int(old), float(scores[i]), str(phases[i]), float(terminal[i])):
            out.append({"row_id": i, "candidate_point": 0, "score": float(scores[i] - 0.90), "specialist": "point0"})
    return pd.DataFrame(out)


def best_per_row(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ranked = candidates.sort_values(["row_id", "score"], ascending=[True, False])
    return ranked.groupby("row_id", as_index=False, sort=False).head(1).reset_index(drop=True)


def apply_candidates(base: np.ndarray, candidates: pd.DataFrame, cap: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if candidates.empty or "row_id" not in candidates:
        pred = np.asarray(base, dtype=int).copy()
        selected = np.zeros(len(pred), dtype=bool)
        return pred, selected, pd.DataFrame(columns=["row_id", "candidate_point", "score", "specialist"])
    unique = best_per_row(candidates)
    pred, selected = apply_point_caps(base, unique, cap)
    selected_rows = unique[unique["row_id"].astype(int).isin(np.where(selected)[0])].copy()
    return pred, selected, selected_rows


def evaluate_variant(
    name: str,
    group: str,
    cap: float,
    train_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    public_weights: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    pred_oof, selected_oof, _selected_train = apply_candidates(base_oof, train_candidates, cap)
    pred_test, selected_test, selected_test_rows = apply_candidates(base_test, test_candidates, cap)
    base_macro = macro_f1(y_true, base_oof)
    pred_macro = macro_f1(y_true, pred_oof)
    base_public = weighted_macro_f1(y_true, base_oof, public_weights)
    pred_public = weighted_macro_f1(y_true, pred_oof, public_weights)
    base_class = class_f1(y_true, base_oof)
    pred_class = class_f1(y_true, pred_oof)
    class_delta = {str(cls): float(pred_class[cls] - base_class[cls]) for cls in TARGET_CLASSES}
    point0_rate_delta = float(np.mean(pred_test == 0) - np.mean(base_test == 0))
    rec = {
        "candidate": name,
        "specialist_group": group,
        "cap": float(cap),
        "point_macro_f1": pred_macro,
        "delta_vs_v261": pred_macro - base_macro,
        "public_like_delta": pred_public - base_public,
        "point0_f1_delta": class_delta["0"],
        "rare134_mean_delta": float(np.mean([class_delta[str(cls)] for cls in [1, 3, 4]])),
        "long789_mean_delta": float(np.mean([class_delta[str(cls)] for cls in [7, 8, 9]])),
        "point_churn": float(np.mean(selected_oof)),
        "oof_changed_rows": int(selected_oof.sum()),
        "test_changed_rows": int(selected_test.sum()),
        "test_point0_rate_delta": point0_rate_delta,
        "class_delta_json": json.dumps(class_delta, sort_keys=True),
        "test_point_distribution": point_distribution(pred_test),
    }
    rec["upload_recommendation"] = (
        "REVIEW_UPLOAD"
        if rec["delta_vs_v261"] >= 0.0015
        and rec["public_like_delta"] >= 0.0008
        and rec["point_churn"] <= 0.010
        and 3 <= rec["test_changed_rows"] <= 20
        and rec["test_point0_rate_delta"] <= 0.005
        else "DO_NOT_UPLOAD"
    )
    return rec, pred_oof, pred_test, selected_test_rows


def write_submission(filename: str, point_pred: np.ndarray, anchor: pd.DataFrame) -> Path:
    out = anchor.copy()
    out["pointId"] = np.asarray(point_pred, dtype=int)
    out = out[EXPECTED_COLUMNS]
    validate_submission_frame(out)
    if not out["actionId"].equals(anchor["actionId"]):
        raise ValueError(f"{filename}: actionId changed")
    if not np.allclose(out["serverGetPoint"].astype(float), anchor["serverGetPoint"].astype(float)):
        raise ValueError(f"{filename}: serverGetPoint changed")
    path = OUT_DIR / filename
    out.to_csv(path, index=False, float_format="%.8f")
    return path


def build_class_report(y_true: np.ndarray, base_oof: np.ndarray, best_pred: np.ndarray) -> pd.DataFrame:
    base_scores = class_f1(y_true, base_oof)
    best_scores = class_f1(y_true, best_pred)
    return pd.DataFrame(
        [
            {
                "point": int(cls),
                "group": "/".join(name for name, vals in POINT_GROUPS.items() if cls in vals),
                "base_f1": float(base_scores[cls]),
                "v293_f1": float(best_scores[cls]),
                "delta": float(best_scores[cls] - base_scores[cls]),
            }
            for cls in POINT_CLASSES
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("submission_v293*.csv"):
        stale.unlink()

    train_rows, test_rows, anchor, y_true, base_oof = build_frames()
    base_test = anchor["pointId"].astype(int).to_numpy()
    if len(y_true) != len(base_oof) or len(test_rows) != len(base_test):
        raise RuntimeError("cannot rebuild V293: y_true/base train/test row lengths do not align")

    features = numeric_features(train_rows, test_rows)
    if not features:
        raise RuntimeError("cannot rebuild V293: no numeric feature columns available")
    public_weights = (
        1.0
        + 0.10 * train_rows["is_rally"].astype(float).to_numpy()
        + 0.10 * train_rows["score_pressure"].astype(float).to_numpy() / 20.0
    )

    long_train, long_test, long_report = train_long789_specialist(train_rows, test_rows, y_true, base_oof, features)
    rare_train, rare_test, rare_report = train_rare134_specialist(train_rows, test_rows, y_true, base_oof, features)
    p0_train, p0_test, p0_report = train_point0_specialist(train_rows, test_rows, y_true, base_oof, features)
    pd.DataFrame([long_report, rare_report, p0_report]).to_csv(OUT_DIR / "v293_specialist_report.csv", index=False)

    variants = [
        (
            "v293_long789_cap0p0025",
            "long789",
            0.0025,
            long_train,
            long_test,
            "submission_v293_long789_cap0p0025__v173action_r121server.csv",
        ),
        (
            "v293_long789_cap0p005",
            "long789",
            0.005,
            long_train,
            long_test,
            "submission_v293_long789_cap0p005__v173action_r121server.csv",
        ),
        (
            "v293_rare134_cap0p0025",
            "rare134",
            0.0025,
            rare_train,
            rare_test,
            "submission_v293_rare134_cap0p0025__v173action_r121server.csv",
        ),
        (
            "v293_rare134_cap0p005",
            "rare134",
            0.005,
            rare_train,
            rare_test,
            "submission_v293_rare134_cap0p005__v173action_r121server.csv",
        ),
        (
            "v293_point0_cap0p0025",
            "point0",
            0.0025,
            p0_train,
            p0_test,
            "submission_v293_point0_cap0p0025__v173action_r121server.csv",
        ),
        (
            "v293_bank_no_point0_cap0p005",
            "bank_no_point0",
            0.005,
            pd.concat([long_train, rare_train], ignore_index=True),
            pd.concat([long_test, rare_test], ignore_index=True),
            "submission_v293_bank_no_point0_cap0p005__v173action_r121server.csv",
        ),
        (
            "v293_bank_no_point0_cap0p010",
            "bank_no_point0",
            0.010,
            pd.concat([long_train, rare_train], ignore_index=True),
            pd.concat([long_test, rare_test], ignore_index=True),
            "submission_v293_bank_no_point0_cap0p010__v173action_r121server.csv",
        ),
        (
            "v293_bank_with_point0_cap0p005",
            "bank_with_point0",
            0.005,
            pd.concat([long_train, rare_train, p0_train], ignore_index=True),
            pd.concat([long_test, rare_test, p0_test], ignore_index=True),
            "submission_v293_bank_with_point0_cap0p005__v173action_r121server.csv",
        ),
    ]

    records: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    generated: list[str] = []
    for name, group, cap, train_candidates, test_candidates, filename in variants:
        rec, pred_oof, pred_test, selected = evaluate_variant(
            name,
            group,
            cap,
            train_candidates,
            test_candidates,
            y_true,
            base_oof,
            base_test,
            public_weights,
        )
        path = write_submission(filename, pred_test, anchor)
        rec["path"] = str(path.relative_to(ROOT))
        records.append(rec)
        predictions[name] = pred_oof
        generated.append(str(path.relative_to(ROOT)))
        if not selected.empty:
            audit = selected.copy()
            audit["candidate"] = name
            audit["rally_uid"] = anchor.iloc[audit["row_id"].astype(int).to_numpy()]["rally_uid"].to_numpy()
            audit["base_point"] = base_test[audit["row_id"].astype(int).to_numpy()]
            audit_rows.append(audit)

    search = pd.DataFrame(records).sort_values(
        ["upload_recommendation", "delta_vs_v261", "public_like_delta", "test_changed_rows"],
        ascending=[False, False, False, True],
    )
    search.to_csv(OUT_DIR / "v293_candidate_search.csv", index=False)
    nonempty = search.copy()
    best_row = nonempty.sort_values(["delta_vs_v261", "public_like_delta", "test_changed_rows"], ascending=[False, False, True]).iloc[0]
    best_name = str(best_row["candidate"])
    build_class_report(y_true, base_oof, predictions[best_name]).to_csv(OUT_DIR / "v293_class_report.csv", index=False)
    if audit_rows:
        changed_audit = pd.concat(audit_rows, ignore_index=True)
    else:
        changed_audit = pd.DataFrame(columns=["candidate", "row_id", "rally_uid", "base_point", "candidate_point", "score", "specialist"])
    changed_audit.to_csv(OUT_DIR / "v293_changed_row_audit.csv", index=False)

    point0_used = bool(
        not changed_audit.empty
        and "specialist" in changed_audit
        and changed_audit["specialist"].astype(str).eq("point0").any()
    )
    report = _json_safe(
        {
            "version": "V293",
            "anchor_submission": str(ANCHOR_PATH.relative_to(ROOT)),
            "current_clean_best_pl": 0.3576720,
            "fixed_output": {
                "actionId": "copied exactly from V261 anchor",
                "serverGetPoint": "copied exactly from V261 anchor",
                "pointId": "capped V293 local residual candidates",
            },
            "no_ttmatch_no_old_server": True,
            "row_level_oof_source": "rebuilt from train prefix rows; base_point_oof is lag0_pointId and test base is V261 anchor pointId",
            "best_candidate": best_row.to_dict(),
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "specialist_reports": [long_report, rare_report, p0_report],
            "point0_used": point0_used,
            "point0_note": "point0 additions require p0>=0.90, terminal_proxy>=0.75, phase in third/fourth/rally; V293 never changes point0 to nonzero.",
            "upload_recommendation": "REVIEW_UPLOAD" if search["upload_recommendation"].eq("REVIEW_UPLOAD").any() else "DO_NOT_UPLOAD",
            "concerns": [
                "No stored V261 row-level OOF point file was present, so local deltas are against a rebuilt lag0-point OOF base rather than literal V261 OOF.",
                "V272/V277 were public-negative; V293 outputs remain local review candidates and are not copied to upload_candidates or submissions/selected.",
            ],
        }
    )
    (OUT_DIR / "v293_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    md_lines = [
        "# V293 point weak-class residual lab",
        "",
        f"Anchor: `{ANCHOR_PATH.relative_to(ROOT)}`",
        "Fixed fields: `actionId` and `serverGetPoint` copied from V261 anchor.",
        "TTMATCH/old-server: not used.",
        f"Generated submissions: `{len(generated)}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best_row['candidate']}`",
        f"Point Macro-F1: `{float(best_row['point_macro_f1']):.6f}`",
        f"Delta vs rebuilt base: `{float(best_row['delta_vs_v261']):.6f}`",
        f"Public-like delta: `{float(best_row['public_like_delta']):.6f}`",
        f"Test changed rows: `{int(best_row['test_changed_rows'])}`",
        f"Upload recommendation: `{report['upload_recommendation']}`",
        "",
        "## Point0",
        "",
        f"Point0 used: `{point0_used}`",
        report["point0_note"],
        "",
        "## Candidates",
        "",
    ]
    for row in search.to_dict("records"):
        md_lines.append(
            f"- `{row['candidate']}`: delta={float(row['delta_vs_v261']):.6f}, "
            f"public_like={float(row['public_like_delta']):.6f}, "
            f"churn={float(row['point_churn']):.6f}, test_changed={int(row['test_changed_rows'])}, "
            f"recommendation=`{row['upload_recommendation']}`"
        )
    md_lines.extend(["", "## Concerns", "", *[f"- {item}" for item in report["concerns"]]])
    (OUT_DIR / "v293_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": str(OUT_DIR.relative_to(ROOT)),
                "best_candidate": best.get("candidate", ""),
                "best_delta_vs_rebuilt_base": best.get("delta_vs_v261", 0.0),
                "best_public_like_delta": best.get("public_like_delta", 0.0),
                "best_test_changed_rows": best.get("test_changed_rows", 0),
                "generated_submissions": report["generated_submission_count"],
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
