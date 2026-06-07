"""V292 weak-class pretraining action teacher.

V292 keeps the V261 point/server anchor fixed and studies whether weak-class
signal is better used as training-time teacher probability than as late hard
row overrides. It deliberately does not use TTMATCH or old-server artifacts.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "v292_weak_class_pretraining_action_teacher"
V286_OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
V286_OOF = V286_OUTDIR / "v286_specialist_oof.csv"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"

N_ACTIONS = 19
ACTION_CLASSES = np.arange(N_ACTIONS, dtype=int)
WEAK_GROUPS = {
    "fast_attack_57": [5, 7],
    "terminal_03": [0, 3],
    "style_control_89": [8, 9],
    "short_control_411": [4, 11],
    "defensive_1214": [12, 14],
}
HARD_NEGATIVES = {
    "fast_attack_57": [1, 2, 4, 6, 10, 13],
    "terminal_03": [1, 2, 5, 10, 13],
    "style_control_89": [10, 11, 13],
    "short_control_411": [1, 7, 10, 13],
    "defensive_1214": [0, 1, 3, 5, 13],
}
PROTECTED_ACTIONS = [1, 10, 12, 13]
SERVE_ACTIONS = [15, 16, 17, 18]
WEAK_CLASS_WEIGHTS = {
    0: 1.50,
    3: 1.75,
    4: 1.35,
    5: 2.25,
    7: 2.25,
    8: 2.00,
    9: 1.70,
    11: 1.35,
    14: 1.80,
}
TEACHER_VARIANTS = ("v292_logreg_balanced", "v292_logreg_weighted_weak", "v292_extratrees_weighted_weak")
BLEND_WEIGHTS = (0.03, 0.05, 0.075, 0.10, 0.15)
EXPORT_WEIGHTS = (0.05, 0.10)
V291_PUBLIC_FAILURE_CONTEXT = (
    "V291 fast57 modelbank had OOF action delta +0.002852 but public PL 0.3559391, "
    "below clean best 0.3576720; V292 therefore treats row edits as diagnostics unless "
    "teacher-level criteria also transfer locally."
)


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if arr.shape[1] == 0:
        raise ValueError("matrix must have at least one column")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    sums = arr.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= 0.0
    if bad.any():
        arr[bad] = 1.0 / arr.shape[1]
        sums = arr.sum(axis=1, keepdims=True)
    return arr / sums


def one_hot(labels: np.ndarray, n_classes: int = N_ACTIONS, smooth: float = 0.0) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if smooth < 0.0 or smooth >= 1.0:
        raise ValueError("smooth must be in [0, 1)")
    out = np.full((len(labels), n_classes), smooth / max(n_classes - 1, 1), dtype=float)
    valid = (labels >= 0) & (labels < n_classes)
    out[np.arange(len(labels))[valid], labels[valid]] = 1.0 - smooth
    return normalize_rows_safe(out)


def sample_weight_for_actions(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    weights = np.ones(len(y), dtype=float)
    for action, weight in WEAK_CLASS_WEIGHTS.items():
        weights[y == int(action)] = float(weight)
    return weights


def blend_with_anchor_probs(anchor_prob: np.ndarray, teacher_prob: np.ndarray, weight: float) -> np.ndarray:
    anchor = normalize_rows_safe(anchor_prob)
    teacher = normalize_rows_safe(teacher_prob)
    if anchor.shape != teacher.shape:
        raise ValueError("anchor_prob and teacher_prob must have the same shape")
    return normalize_rows_safe((1.0 - float(weight)) * anchor + float(weight) * teacher)


def apply_action_caps(
    prob: np.ndarray,
    allowed_override_actions: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    out = normalize_rows_safe(prob)
    out[:, SERVE_ACTIONS] = 0.0
    if allowed_override_actions is not None:
        mask = np.zeros(out.shape[1], dtype=bool)
        mask[list(allowed_override_actions)] = True
        mask[PROTECTED_ACTIONS] = True
        # Probability support stays broad; later row gates decide actual edits.
    return normalize_rows_safe(out)


def hard_negative_mask(y: np.ndarray, group: str) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    positives = np.isin(y, WEAK_GROUPS[group])
    negatives = np.isin(y, HARD_NEGATIVES[group])
    return positives | negatives


def _weak_actions() -> np.ndarray:
    return np.array(sorted({action for actions in WEAK_GROUPS.values() for action in actions}), dtype=int)


def _safe_ap(y_binary: np.ndarray, score: np.ndarray) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    if len(np.unique(y_binary)) < 2:
        return 0.0
    return float(average_precision_score(y_binary, np.asarray(score, dtype=float)))


def _safe_auc(y_binary: np.ndarray, score: np.ndarray) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    if len(np.unique(y_binary)) < 2:
        return 0.5
    return float(roc_auc_score(y_binary, np.asarray(score, dtype=float)))


def _macro(y: np.ndarray, pred: np.ndarray, labels: np.ndarray | list[int] = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def _weighted_macro(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    scores = []
    for label in ACTION_CLASSES:
        yt = y == label
        yp = pred == label
        tp = float(weights[yt & yp].sum())
        fp = float(weights[~yt & yp].sum())
        fn = float(weights[yt & ~yp].sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def _public_weights(frame: pd.DataFrame) -> np.ndarray:
    try:
        return 1.0 + 0.15 * frame["is_receive"].to_numpy(dtype=float) + 0.10 * frame[
            "is_game_point_like"
        ].to_numpy(dtype=float)
    except Exception:
        return np.ones(len(frame), dtype=float)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _feature_columns_from_v291() -> list[str]:
    try:
        from analysis_v291_weak_class_training_upgrade import feature_family_columns

        cols: list[str] = []
        for family_cols in feature_family_columns().values():
            cols.extend(family_cols)
        return list(dict.fromkeys(cols))
    except Exception:
        return []


def numeric_matrix(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred = _feature_columns_from_v291()
    aux_cols = [col for col in set(train_frame.columns) | set(test_frame.columns) if col.startswith("v292_aux_")]
    preferred = list(dict.fromkeys(preferred + sorted(aux_cols)))
    present = [col for col in preferred if col in train_frame.columns or col in test_frame.columns]
    if not present:
        excluded = {
            "rally_uid",
            "next_actionId",
            "actionId",
            "pointId",
            "serverGetPoint",
            "y_true_action",
            "fold",
            "match",
        }
        present = [col for col in sorted(set(train_frame.columns) | set(test_frame.columns)) if col not in excluded]

    train = train_frame.reindex(columns=present)
    test = test_frame.reindex(columns=present)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in combined.columns if col not in numeric_cols]
    parts = []
    if numeric_cols:
        parts.append(combined[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float))
    if categorical_cols:
        cats = combined[categorical_cols].fillna("__missing__").astype(str)
        parts.append(pd.get_dummies(cats, columns=categorical_cols, dtype=float))
    if parts:
        encoded = pd.concat(parts, axis=1)
    else:
        encoded = pd.DataFrame(index=combined.index)
    encoded = encoded.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    x_train = encoded.iloc[: len(train_frame)].reset_index(drop=True)
    x_test = encoded.iloc[len(train_frame) :].reset_index(drop=True)
    x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0.0)
    return x_train.astype(float), x_test.astype(float)


def _splits(rows: pd.DataFrame, y: np.ndarray, seed: int = 292) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    if "fold" in rows:
        folds = pd.to_numeric(rows["fold"], errors="coerce").fillna(-1).astype(int).to_numpy()
        uniq = sorted([fold for fold in np.unique(folds) if fold >= 0])
        pairs = [(np.where(folds != fold)[0], np.where(folds == fold)[0]) for fold in uniq]
        pairs = [(tr, va) for tr, va in pairs if len(tr) and len(va)]
        if len(pairs) >= 2:
            return pairs
    if "match" in rows and rows["match"].nunique(dropna=False) >= 2:
        groups = rows["match"].astype(str).to_numpy()
        n_splits = min(5, rows["match"].nunique(dropna=False))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    class_counts = np.bincount(y, minlength=N_ACTIONS)
    nonzero = class_counts[class_counts > 0]
    n_splits = int(min(5, nonzero.min())) if len(nonzero) else 2
    n_splits = max(2, n_splits)
    if n_splits <= 1 or np.sum(nonzero >= n_splits) < 2:
        idx = np.arange(len(y))
        return [(idx[idx % 2 == 1], idx[idx % 2 == 0]), (idx[idx % 2 == 0], idx[idx % 2 == 1])]
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _binary_model(model_name: str, seed: int) -> Any:
    if model_name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=800, class_weight="balanced", C=0.7, random_state=seed),
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=80,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"unknown auxiliary model {model_name}")


def train_weak_auxiliary_heads(
    train_frame: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_test = numeric_matrix(train_frame, test_frame)
    y = np.asarray(y, dtype=int)
    splits = _splits(rows, y, seed=2921)
    aux_oof = pd.DataFrame(index=pd.RangeIndex(len(train_frame)))
    aux_test = pd.DataFrame(index=pd.RangeIndex(len(test_frame)))
    records: list[dict[str, Any]] = []

    for group, actions in WEAK_GROUPS.items():
        keep = hard_negative_mask(y, group)
        target_full = np.isin(y, actions).astype(int)
        base_rate = float(target_full[keep].mean()) if keep.any() else float(target_full.mean())
        for model_name in ("logistic", "extratrees"):
            oof = np.full(len(train_frame), base_rate, dtype=float)
            test_sum = np.zeros(len(test_frame), dtype=float)
            fitted = 0
            for fold_id, (train_idx, valid_idx) in enumerate(splits):
                train_idx = np.asarray([idx for idx in train_idx if keep[idx]], dtype=int)
                valid_idx = np.asarray(valid_idx, dtype=int)
                if len(train_idx) == 0 or len(valid_idx) == 0:
                    continue
                target = target_full[train_idx]
                if len(np.unique(target)) < 2:
                    continue
                model = _binary_model(model_name, seed=29210 + fold_id * 17 + len(actions))
                try:
                    model.fit(x_train.iloc[train_idx], target)
                    oof[valid_idx] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
                    if len(x_test):
                        test_sum += model.predict_proba(x_test)[:, 1]
                    fitted += 1
                except Exception as exc:
                    warnings.warn(f"auxiliary {group}/{model_name} fold {fold_id} failed: {exc}", RuntimeWarning)
            if fitted:
                test_pred = test_sum / float(fitted) if len(x_test) else np.zeros(0, dtype=float)
            else:
                test_pred = np.full(len(test_frame), base_rate, dtype=float)
            oof = np.clip(np.nan_to_num(oof, nan=base_rate, posinf=1.0, neginf=0.0), 0.0, 1.0)
            test_pred = np.clip(np.nan_to_num(test_pred, nan=base_rate, posinf=1.0, neginf=0.0), 0.0, 1.0)
            col = f"v292_aux_{group}_{model_name}"
            aux_oof[col] = oof
            aux_test[col] = test_pred
            y_eval = target_full[keep]
            s_eval = oof[keep]
            records.append(
                {
                    "group": group,
                    "model": model_name,
                    "ap": _safe_ap(y_eval, s_eval),
                    "auc": _safe_auc(y_eval, s_eval),
                    "positive_rows": int(y_eval.sum()),
                    "hard_negative_rows": int(np.isin(y, HARD_NEGATIVES[group]).sum()),
                    "training_rows": int(keep.sum()),
                    "oof_mean": float(s_eval.mean()) if len(s_eval) else 0.0,
                    "test_mean": float(test_pred.mean()) if len(test_pred) else 0.0,
                    "fitted_folds": int(fitted),
                }
            )
    return aux_oof, aux_test, pd.DataFrame(records)


def load_v292_inputs() -> dict[str, Any]:
    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF file: {V286_OOF}")
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing anchor submission: {ANCHOR_SUBMISSION}")
    from analysis_v290_shortcontrol411_specialist import load_anchor_frames

    oof = pd.read_csv(V286_OOF)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    rows, test_rows, rebuilt_y, rebuilt_anchor = load_anchor_frames()
    rows = rows.reset_index(drop=True)
    test_rows = test_rows.reset_index(drop=True)
    if len(rows) != len(oof):
        raise ValueError(f"V286 OOF length {len(oof)} does not match rebuilt rows {len(rows)}")
    if len(test_rows) != len(anchor_sub):
        raise ValueError(f"test rows length {len(test_rows)} does not match anchor submission {len(anchor_sub)}")
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    if len(rebuilt_y) == len(y) and not np.array_equal(np.asarray(rebuilt_y, dtype=int), y):
        warnings.warn("rebuilt y differs from V286 OOF y_true_action; using V286 OOF labels", RuntimeWarning)
    if len(rebuilt_anchor) == len(anchor_oof) and not np.array_equal(np.asarray(rebuilt_anchor, dtype=int), anchor_oof):
        warnings.warn("rebuilt anchor differs from V286 OOF anchor_action; using V286 OOF anchor", RuntimeWarning)
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    test_rows = test_rows.copy()
    test_rows["anchor_action"] = anchor_test
    return {
        "oof": oof,
        "anchor_sub": anchor_sub,
        "rows": rows,
        "test_rows": test_rows,
        "y": y,
        "anchor_oof": anchor_oof,
        "anchor_test": anchor_test,
    }


def build_v292_feature_frames(inputs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from analysis_v291_weak_class_training_upgrade import build_complete_feature_frame

    rows = inputs["rows"]
    test_rows = inputs["test_rows"]
    y = inputs["y"]
    oof = inputs["oof"]
    train_frame = build_complete_feature_frame(rows, oof, y=y).reset_index(drop=True)
    test_frame = build_complete_feature_frame(test_rows, None, y=y, support_rows=rows).reset_index(drop=True)
    coverage = feature_coverage_report(train_frame, test_frame)
    return train_frame, test_frame, coverage


def feature_coverage_report(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    families: dict[str, list[str]] = {}
    try:
        from analysis_v291_weak_class_training_upgrade import feature_family_columns

        families = feature_family_columns()
    except Exception:
        families = {"fallback": sorted(set(train_frame.columns) | set(test_frame.columns))}
    for family, cols in families.items():
        train_present = [col for col in cols if col in train_frame.columns]
        test_present = [col for col in cols if col in test_frame.columns]
        test_constant = []
        for col in test_present:
            if test_frame[col].nunique(dropna=False) <= 1:
                test_constant.append(col)
        records.append(
            {
                "family": family,
                "required_cols": len(cols),
                "train_present_cols": len(train_present),
                "test_present_cols": len(test_present),
                "train_missing_cols": ",".join(sorted(set(cols) - set(train_present))),
                "test_missing_cols": ",".join(sorted(set(cols) - set(test_present))),
                "test_constant_cols": ",".join(test_constant),
            }
        )
    records.append(
        {
            "family": "teacher_column_warning",
            "required_cols": 0,
            "train_present_cols": 0,
            "test_present_cols": 0,
            "train_missing_cols": "",
            "test_missing_cols": "",
            "test_constant_cols": "test teacher columns are zero-filled; style_response columns may remain diagnostic if constant",
        }
    )
    return pd.DataFrame(records)


def _append_aux(train_frame: pd.DataFrame, test_frame: pd.DataFrame, aux_oof: pd.DataFrame, aux_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_aug = train_frame.reset_index(drop=True).copy()
    test_aug = test_frame.reset_index(drop=True).copy()
    for col in aux_oof.columns:
        train_aug[col] = aux_oof[col].to_numpy(dtype=float)
        test_aug[col] = aux_test[col].to_numpy(dtype=float)
    return train_aug, test_aug


def _teacher_model(name: str, seed: int) -> Any:
    if name == "v292_logreg_balanced":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, class_weight="balanced", C=0.7, random_state=seed),
        )
    if name == "v292_logreg_weighted_weak":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, C=0.5, random_state=seed),
        )
    if name == "v292_extratrees_weighted_weak":
        return ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"unknown teacher variant {name}")


def _fit_model(model: Any, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    if hasattr(model, "steps"):
        final_step = model.steps[-1][0]
        model.fit(x, y, **{f"{final_step}__sample_weight": sample_weight})
        return
    model.fit(x, y, sample_weight=sample_weight)


def _predict_proba_19(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), N_ACTIONS), dtype=float)
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_, dtype=int)
    for j, cls in enumerate(classes):
        if 0 <= int(cls) < N_ACTIONS:
            out[:, int(cls)] = raw[:, j]
    return normalize_rows_safe(out)


def train_action_teachers(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, dict[str, Any]]:
    x_train, x_test = numeric_matrix(train_frame, test_frame)
    y = np.asarray(y, dtype=int)
    splits = _splits(rows, y, seed=2922)
    teachers: dict[str, dict[str, Any]] = {}
    weights = sample_weight_for_actions(y)
    for name in TEACHER_VARIANTS:
        oof = np.zeros((len(x_train), N_ACTIONS), dtype=float)
        test_sum = np.zeros((len(x_test), N_ACTIONS), dtype=float)
        fitted = 0
        for fold_id, (train_idx, valid_idx) in enumerate(splits):
            train_idx = np.asarray(train_idx, dtype=int)
            valid_idx = np.asarray(valid_idx, dtype=int)
            if len(train_idx) == 0 or len(valid_idx) == 0 or len(np.unique(y[train_idx])) < 2:
                continue
            model = _teacher_model(name, seed=29220 + fold_id)
            try:
                if name in {"v292_logreg_weighted_weak", "v292_extratrees_weighted_weak"}:
                    _fit_model(model, x_train.iloc[train_idx], y[train_idx], weights[train_idx])
                else:
                    _fit_model(model, x_train.iloc[train_idx], y[train_idx])
                oof[valid_idx] = _predict_proba_19(model, x_train.iloc[valid_idx])
                test_sum += _predict_proba_19(model, x_test)
                fitted += 1
            except Exception as exc:
                warnings.warn(f"teacher {name} fold {fold_id} failed: {exc}", RuntimeWarning)
        if fitted == 0:
            prior = np.bincount(y, minlength=N_ACTIONS).astype(float)
            prior = prior / max(float(prior.sum()), 1.0)
            oof = np.tile(prior, (len(x_train), 1))
            test_prob = np.tile(prior, (len(x_test), 1))
        else:
            missing = oof.sum(axis=1) <= 0.0
            if missing.any():
                oof[missing] = one_hot(y[missing], smooth=0.02)
            test_prob = test_sum / float(fitted)
        oof = apply_action_caps(oof)
        test_prob = apply_action_caps(test_prob)
        teachers[name] = {"oof_prob": oof, "test_prob": test_prob, "fitted_folds": fitted}
    return teachers


def evaluate_prediction(
    candidate: str,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    pred_oof: np.ndarray,
    anchor_test: np.ndarray,
    pred_test: np.ndarray,
    public_weights: np.ndarray,
    model: str,
    weight: float,
    blend_type: str,
) -> dict[str, Any]:
    weak = _weak_actions()
    protected = np.array(PROTECTED_ACTIONS, dtype=int)
    pred_oof = np.asarray(pred_oof, dtype=int)
    pred_test = np.asarray(pred_test, dtype=int)
    rec = {
        "candidate": candidate,
        "model": model,
        "blend_weight": float(weight),
        "blend_type": blend_type,
        "is_direct_diagnostic": int(blend_type == "direct_teacher"),
        "action_macro_f1": _macro(y, pred_oof),
        "delta_vs_v173": _macro(y, pred_oof) - _macro(y, anchor_oof),
        "public_like_delta_vs_v173": _weighted_macro(y, pred_oof, public_weights)
        - _weighted_macro(y, anchor_oof, public_weights),
        "weak_mean_delta": _macro(y, pred_oof, weak) - _macro(y, anchor_oof, weak),
        "protected_mean_delta": _macro(y, pred_oof, protected) - _macro(y, anchor_oof, protected),
        "test_changed_rows": int(np.sum(pred_test != anchor_test)),
        "changed_rows": int(np.sum(pred_oof != anchor_oof)),
        "test_serve_predictions": int(np.isin(pred_test, SERVE_ACTIONS).sum()),
        "test_action_distribution": json.dumps(
            {str(int(k)): int(v) for k, v in zip(*np.unique(pred_test, return_counts=True))}, sort_keys=True
        ),
    }
    rec["upload_recommendation"] = (
        "REVIEW_CANDIDATE"
        if (
            rec["delta_vs_v173"] >= 0.002
            and rec["public_like_delta_vs_v173"] >= 0.001
            and rec["protected_mean_delta"] >= 0.0
            and 5 <= rec["test_changed_rows"] <= 60
            and rec["test_serve_predictions"] == 0
            and blend_type != "direct_teacher"
        )
        else "DO_NOT_UPLOAD"
    )
    return rec


def _candidate_gate(anchor: np.ndarray, teacher_argmax: np.ndarray, anchor_prob: np.ndarray, teacher_prob: np.ndarray) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=int)
    teacher_argmax = np.asarray(teacher_argmax, dtype=int)
    rows = np.arange(len(anchor))
    weak = _weak_actions()
    teacher_conf = teacher_prob[rows, teacher_argmax]
    anchor_teacher_conf = teacher_prob[rows, anchor]
    anchor_margin_weak = anchor_prob[rows, anchor] < 0.985
    return (
        (teacher_argmax != anchor)
        & np.isin(teacher_argmax, weak)
        & ~np.isin(teacher_argmax, SERVE_ACTIONS)
        & ~np.isin(anchor, PROTECTED_ACTIONS)
        & ((teacher_conf - anchor_teacher_conf) > 0.05)
        & (np.isin(teacher_argmax, weak) | anchor_margin_weak)
    )


def gated_teacher_labels(
    anchor: np.ndarray,
    anchor_prob: np.ndarray,
    teacher_prob: np.ndarray,
    blend_weight: float,
    max_rows: int | None = None,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=int)
    labels = anchor.copy()
    teacher_argmax = teacher_prob.argmax(axis=1).astype(int)
    gate = _candidate_gate(anchor, teacher_argmax, anchor_prob, teacher_prob)
    if not gate.any():
        return labels
    rows = np.where(gate)[0]
    score = teacher_prob[rows, teacher_argmax[rows]] - teacher_prob[rows, anchor[rows]]
    if max_rows is None:
        max_rows = int(np.clip(round(len(anchor) * float(blend_weight) * 0.22), 5, 60))
    selected = rows[np.argsort(-score)[:max_rows]]
    labels[selected] = teacher_argmax[selected]
    return labels


def build_candidate_search(
    teachers: dict[str, dict[str, Any]],
    train_frame: pd.DataFrame,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    anchor_prob_oof = one_hot(anchor_oof, N_ACTIONS, smooth=0.02)
    anchor_prob_test = one_hot(anchor_test, N_ACTIONS, smooth=0.02)
    public_weights = _public_weights(train_frame)
    records: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for model, payload in teachers.items():
        teacher_oof = apply_action_caps(payload["oof_prob"])
        teacher_test = apply_action_caps(payload["test_prob"])
        raw_oof = teacher_oof.argmax(axis=1).astype(int)
        raw_test = teacher_test.argmax(axis=1).astype(int)
        records.append(
            evaluate_prediction(
                f"{model}_raw_teacher_diag",
                y,
                anchor_oof,
                raw_oof,
                anchor_test,
                raw_test,
                public_weights,
                model,
                1.0,
                "direct_teacher",
            )
        )
        for weight in BLEND_WEIGHTS:
            soft_oof = blend_with_anchor_probs(anchor_prob_oof, teacher_oof, weight).argmax(axis=1).astype(int)
            soft_test = blend_with_anchor_probs(anchor_prob_test, teacher_test, weight).argmax(axis=1).astype(int)
            soft_name = f"{model}_w{_weight_token(weight)}_softblend"
            predictions[soft_name] = {"oof": soft_oof, "test": soft_test}
            records.append(
                evaluate_prediction(
                    soft_name,
                    y,
                    anchor_oof,
                    soft_oof,
                    anchor_test,
                    soft_test,
                    public_weights,
                    model,
                    weight,
                    "softblend",
                )
            )
            gated_oof = gated_teacher_labels(anchor_oof, anchor_prob_oof, teacher_oof, weight)
            gated_test = gated_teacher_labels(anchor_test, anchor_prob_test, teacher_test, weight)
            gated_name = f"{model}_w{_weight_token(weight)}_gatedblend"
            predictions[gated_name] = {"oof": gated_oof, "test": gated_test}
            records.append(
                evaluate_prediction(
                    gated_name,
                    y,
                    anchor_oof,
                    gated_oof,
                    anchor_test,
                    gated_test,
                    public_weights,
                    model,
                    weight,
                    "gatedblend",
                )
            )

    search = pd.DataFrame(records).sort_values(
        [
            "upload_recommendation",
            "is_direct_diagnostic",
            "delta_vs_v173",
            "public_like_delta_vs_v173",
            "protected_mean_delta",
            "test_changed_rows",
        ],
        ascending=[False, True, False, False, False, True],
    )
    best_name = str(search.iloc[0]["candidate"]) if len(search) else ""
    best_oof = predictions.get(best_name, {"oof": anchor_oof})["oof"]
    class_report = build_class_report(y, anchor_oof, best_oof)
    return search.reset_index(drop=True), predictions, class_report


def build_class_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    records = []
    for action in ACTION_CLASSES:
        label = [int(action)]
        records.append(
            {
                "action": int(action),
                "group": group_for_action(int(action)),
                "is_protected": int(action in PROTECTED_ACTIONS),
                "is_weak": int(action in _weak_actions()),
                "anchor_f1": _macro(y, anchor, label),
                "v292_f1": _macro(y, pred, label),
                "delta": _macro(y, pred, label) - _macro(y, anchor, label),
            }
        )
    return pd.DataFrame(records)


def group_for_action(action: int) -> str:
    for group, actions in WEAK_GROUPS.items():
        if int(action) in actions:
            return group
    return ""


def _weight_token(weight: float) -> str:
    text = f"{weight:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def write_submission(name: str, action: np.ndarray, anchor_sub: pd.DataFrame) -> Path:
    out = pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return path


def _named_prediction(
    predictions: dict[str, dict[str, np.ndarray]],
    model: str,
    weight: float,
    blend_type: str,
    anchor_test: np.ndarray,
) -> np.ndarray:
    key = f"{model}_w{_weight_token(weight)}_{blend_type}"
    return predictions.get(key, {"test": anchor_test})["test"]


def export_submissions(
    predictions: dict[str, dict[str, np.ndarray]],
    search: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    anchor_test: np.ndarray,
) -> list[str]:
    specs = [
        (
            "submission_v292_logreg_balanced_w0p05__pv261cap1__sr121.csv",
            _named_prediction(predictions, "v292_logreg_balanced", 0.05, "gatedblend", anchor_test),
        ),
        (
            "submission_v292_logreg_balanced_w0p10__pv261cap1__sr121.csv",
            _named_prediction(predictions, "v292_logreg_balanced", 0.10, "gatedblend", anchor_test),
        ),
        (
            "submission_v292_logreg_weighted_weak_w0p05__pv261cap1__sr121.csv",
            _named_prediction(predictions, "v292_logreg_weighted_weak", 0.05, "gatedblend", anchor_test),
        ),
        (
            "submission_v292_logreg_weighted_weak_w0p10__pv261cap1__sr121.csv",
            _named_prediction(predictions, "v292_logreg_weighted_weak", 0.10, "gatedblend", anchor_test),
        ),
        (
            "submission_v292_extratrees_weighted_weak_w0p05__pv261cap1__sr121.csv",
            _named_prediction(predictions, "v292_extratrees_weighted_weak", 0.05, "gatedblend", anchor_test),
        ),
        (
            "submission_v292_extratrees_weighted_weak_w0p10__pv261cap1__sr121.csv",
            _named_prediction(predictions, "v292_extratrees_weighted_weak", 0.10, "gatedblend", anchor_test),
        ),
    ]
    gated = search[search["blend_type"].eq("gatedblend")].copy()
    soft = search[search["blend_type"].eq("softblend")].copy()
    best_gated_name = str(gated.iloc[0]["candidate"]) if len(gated) else ""
    best_soft_name = str(soft.iloc[0]["candidate"]) if len(soft) else ""
    specs.append(
        (
            "submission_v292_best_gatedblend__pv261cap1__sr121.csv",
            predictions.get(best_gated_name, {"test": anchor_test})["test"],
        )
    )
    specs.append(
        (
            "submission_v292_best_softblend_diag__pv261cap1__sr121.csv",
            predictions.get(best_soft_name, {"test": anchor_test})["test"],
        )
    )
    generated = []
    for filename, labels in specs[:8]:
        path = write_submission(filename, labels, anchor_sub)
        generated.append(str(path.relative_to(ROOT)))
    return generated


def changed_row_audit(search: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate",
        "model",
        "blend_weight",
        "blend_type",
        "changed_rows",
        "test_changed_rows",
        "weak_mean_delta",
        "protected_mean_delta",
        "upload_recommendation",
    ]
    return search[[col for col in cols if col in search.columns]].copy()


def _gain_source(best: dict[str, Any]) -> str:
    if not best:
        return "none"
    if best.get("blend_type") == "softblend":
        return "soft_teacher_probability_blend"
    if best.get("blend_type") == "gatedblend":
        return "gated_teacher_row_edits"
    return "direct_teacher_diagnostic"


def write_reports(
    aux_report: pd.DataFrame,
    search: pd.DataFrame,
    class_report: pd.DataFrame,
    coverage: pd.DataFrame,
    generated: list[str],
    anchor_sub: pd.DataFrame,
) -> dict[str, Any]:
    aux_report.to_csv(OUTDIR / "v292_aux_head_report.csv", index=False)
    search.to_csv(OUTDIR / "v292_teacher_search.csv", index=False)
    class_report.to_csv(OUTDIR / "v292_class_report.csv", index=False)
    changed_row_audit(search).to_csv(OUTDIR / "v292_changed_row_audit.csv", index=False)
    coverage.to_csv(OUTDIR / "v292_feature_coverage.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    upload_recommendation = "REVIEW_CANDIDATE" if str(best.get("upload_recommendation", "")) == "REVIEW_CANDIDATE" else "DO_NOT_UPLOAD"
    if _gain_source(best) == "gated_teacher_row_edits" and upload_recommendation == "REVIEW_CANDIDATE":
        upload_recommendation = "REVIEW_WITH_CAUTION"
    report = _json_safe(
        {
            "version": "V292",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "anchor_rows": int(len(anchor_sub)),
            "point_server_fixed_to_v261": True,
            "no_ttmatch_no_old_server": True,
            "v291_public_failure_context": V291_PUBLIC_FAILURE_CONTEXT,
            "test_teacher_column_warning": "test teacher columns are zero-filled; style_response columns remain diagnostic if constant",
            "weak_groups": WEAK_GROUPS,
            "hard_negatives": HARD_NEGATIVES,
            "teacher_variants": list(TEACHER_VARIANTS),
            "blend_weights": list(BLEND_WEIGHTS),
            "best_candidate": best,
            "best_gain_source": _gain_source(best),
            "upload_recommendation": upload_recommendation,
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "copied_to_upload_or_selected": False,
        }
    )
    (OUTDIR / "v292_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    md = [
        "# V292 weak-class pretraining action teacher",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Point/server: fixed from V261 anchor",
        "TTMATCH/old-server: not used",
        f"V291 context: {V291_PUBLIC_FAILURE_CONTEXT}",
        "Test teacher-column warning: test teacher columns are zero-filled; style_response columns may remain diagnostic if constant.",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate', '')}`",
        f"Blend type: `{best.get('blend_type', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Public-like delta vs V173: {float(best.get('public_like_delta_vs_v173', 0.0)):.6f}",
        f"Weak mean delta: {float(best.get('weak_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Best gain source: `{report['best_gain_source']}`",
        f"Upload recommendation: `{upload_recommendation}`",
        "",
        "## Generated submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v292_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v292*.csv"):
        stale.unlink()
    inputs = load_v292_inputs()
    train_frame, test_frame, coverage = build_v292_feature_frames(inputs)
    aux_oof, aux_test, aux_report = train_weak_auxiliary_heads(train_frame, inputs["rows"], inputs["y"], test_frame)
    train_aug, test_aug = _append_aux(train_frame, test_frame, aux_oof, aux_test)
    teachers = train_action_teachers(train_aug, test_aug, inputs["rows"], inputs["y"])
    search, predictions, class_report = build_candidate_search(
        teachers,
        train_aug,
        inputs["y"],
        inputs["anchor_oof"],
        inputs["anchor_test"],
    )
    generated = export_submissions(predictions, search, inputs["anchor_sub"], inputs["anchor_test"])
    return write_reports(aux_report, search, class_report, coverage, generated, inputs["anchor_sub"])


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "best_candidate": best.get("candidate", ""),
                "best_delta_vs_v173": best.get("delta_vs_v173", 0.0),
                "best_public_like_delta_vs_v173": best.get("public_like_delta_vs_v173", 0.0),
                "best_test_changed_rows": best.get("test_changed_rows", 0),
                "upload_recommendation": report.get("upload_recommendation", "DO_NOT_UPLOAD"),
                "generated_submission_count": report.get("generated_submission_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
