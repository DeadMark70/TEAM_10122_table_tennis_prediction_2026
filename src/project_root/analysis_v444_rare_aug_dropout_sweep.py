"""V444 rare augmentation and dropout/masking sweep.

Runs fold-safe rare-class augmentation probes for action and point targets.
This script intentionally writes only score tables and diagnostics; submission
packaging is left to V445.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v416_external_embedding_aicup_finetune import build_test_rows, build_train_transition_rows
from analysis_v433_weak_class_expert_bank import SUBMISSION_COLUMNS, build_feature_matrix


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v444_rare_aug_dropout_sweep"
EXPECTED_TEST_ROWS = 1845

ACTION_WEAK_CLASSES = {0, 3, 4, 5, 7, 8, 9, 12, 14}
POINT_WEAK_CLASSES = {1, 3, 4, 7, 8, 9}


class ConstantProbabilityModel:
    def __init__(self, labels: np.ndarray):
        labels = np.asarray(labels, dtype=int)
        self.classes_ = np.unique(labels) if len(labels) else np.array([0], dtype=int)
        self.probability_ = np.ones(len(self.classes_), dtype=float) / max(1, len(self.classes_))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probability_, (len(x), 1))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def build_sweep_grid() -> list[dict[str, Any]]:
    """Return the fixed V444 variant grid."""

    return [
        {
            "name": "class_weight",
            "use_class_weight": True,
            "resample_multiplier": 1,
            "smote_multiplier": 1,
            "mask_probability": 0.0,
            "description": "balanced multinomial classifier",
        },
        {
            "name": "resample_rare",
            "use_class_weight": False,
            "resample_multiplier": 3,
            "smote_multiplier": 1,
            "mask_probability": 0.0,
            "description": "train-fold-only rare row resampling",
        },
        {
            "name": "smote_rare",
            "use_class_weight": False,
            "resample_multiplier": 1,
            "smote_multiplier": 2,
            "mask_probability": 0.0,
            "description": "train-fold-only SMOTE-like rare interpolation",
        },
        {
            "name": "mask_dropout",
            "use_class_weight": False,
            "resample_multiplier": 1,
            "smote_multiplier": 1,
            "mask_probability": 0.12,
            "description": "feature masking on training folds",
        },
        {
            "name": "class_weight_plus_mask",
            "use_class_weight": True,
            "resample_multiplier": 1,
            "smote_multiplier": 1,
            "mask_probability": 0.10,
            "description": "balanced classifier with training-fold feature masking",
        },
    ]


def apply_train_only_smote_like_augmentation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target_col: str,
    rare_classes: set[int] | tuple[int, ...] | list[int],
    multiplier: int = 2,
    random_state: int = 444,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create SMOTE-like rows from rare training rows only."""

    train_out = train.reset_index(drop=True).copy()
    validation_out = validation.copy()
    labels = {int(v) for v in rare_classes}
    multiplier = max(1, int(multiplier))
    rng = np.random.default_rng(random_state)
    synthetic_rows: list[pd.Series] = []
    per_class: dict[str, int] = {}

    if multiplier > 1 and target_col in train_out.columns:
        numeric_columns = [
            col
            for col in train_out.columns
            if col != target_col and pd.api.types.is_numeric_dtype(pd.to_numeric(train_out[col], errors="coerce"))
        ]
        numeric = train_out[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        target = pd.to_numeric(train_out[target_col], errors="coerce").fillna(-9999).astype(int).to_numpy()
        for cls in sorted(labels):
            class_idx = np.flatnonzero(target == cls)
            n_new = int(len(class_idx) * (multiplier - 1))
            per_class[str(cls)] = n_new
            if len(class_idx) == 0:
                continue
            for _ in range(n_new):
                left = int(rng.choice(class_idx))
                right = int(rng.choice(class_idx))
                lam = float(rng.uniform(0.15, 0.85))
                row = train_out.iloc[left].copy()
                row.loc[numeric_columns] = (lam * numeric.iloc[left]) + ((1.0 - lam) * numeric.iloc[right])
                row.loc[target_col] = cls
                row.loc["_v444_synthetic"] = 1
                synthetic_rows.append(row)

    if synthetic_rows:
        train_out["_v444_synthetic"] = train_out.get("_v444_synthetic", 0)
        synthetic = pd.DataFrame(synthetic_rows, columns=train_out.columns)
        train_out = pd.concat([train_out, synthetic], ignore_index=True)
    elif "_v444_synthetic" not in train_out.columns:
        train_out["_v444_synthetic"] = 0

    report = {
        "train_only": True,
        "target_col": target_col,
        "rare_classes": sorted(labels),
        "multiplier": multiplier,
        "original_train_rows": int(len(train)),
        "synthetic_rows": int(len(train_out) - len(train)),
        "output_train_rows": int(len(train_out)),
        "validation_rows_added": 0,
        "validation_rows_unchanged": int(len(validation_out)),
        "per_class_synthetic_rows": per_class,
    }
    return train_out.reset_index(drop=True), validation_out, report


def _target_column(frame: pd.DataFrame, target: str) -> str:
    candidates = [f"target_{target}Id", f"{target}Id"]
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"frame missing {target} target column; tried {candidates}")


def _splitter(y: np.ndarray, groups: pd.Series | None, *, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    indices = np.arange(len(y), dtype=int)
    if len(indices) < 3:
        return [(indices, indices)]
    if groups is not None and groups.nunique(dropna=True) >= 2:
        splits = int(min(max(2, n_splits), groups.nunique(dropna=True)))
        return list(GroupKFold(n_splits=splits).split(np.zeros(len(y)), y, groups))
    counts = pd.Series(y).value_counts()
    if not counts.empty and counts.min() >= 2:
        splits = int(min(max(2, n_splits), counts.min()))
        return list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state).split(np.zeros(len(y)), y))
    return [(indices, indices)]


def _fit_model(x: pd.DataFrame, y: pd.Series | np.ndarray, *, use_class_weight: bool, random_state: int) -> Any:
    y_array = np.asarray(y, dtype=int)
    if len(np.unique(y_array)) < 2:
        return ConstantProbabilityModel(y_array)
    kwargs: dict[str, Any] = {
        "class_weight": "balanced" if use_class_weight else None,
        "max_iter": 350,
        "random_state": random_state,
    }
    return make_pipeline(StandardScaler(), LogisticRegression(**kwargs)).fit(x, y_array)


def _predict_aligned_proba(model: Any, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    local = model.predict_proba(x)
    model_classes = getattr(model, "classes_", None)
    if model_classes is None and hasattr(model, "named_steps"):
        model_classes = model.named_steps["logisticregression"].classes_
    out = np.zeros((len(x), len(classes)), dtype=float)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    for local_idx, label in enumerate(model_classes):
        if int(label) in class_to_idx:
            out[:, class_to_idx[int(label)]] = local[:, local_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sum, out=out, where=row_sum > 0)
    out[row_sum.ravel() <= 0, :] = 1.0 / max(1, len(classes))
    return out


def _confidence(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prob.shape[1] == 0:
        return np.zeros(len(prob), dtype=float), np.zeros(len(prob), dtype=float)
    ordered = np.sort(prob, axis=1)
    top = ordered[:, -1]
    second = ordered[:, -2] if prob.shape[1] > 1 else np.zeros(len(prob), dtype=float)
    return top, top - second


def _prediction_from_prob(prob: np.ndarray, classes: list[int]) -> np.ndarray:
    return np.array([classes[idx] for idx in prob.argmax(axis=1)], dtype=int)


def _resample_rare_rows(
    train: pd.DataFrame,
    *,
    target_col: str,
    rare_classes: set[int],
    multiplier: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = train.reset_index(drop=True).copy()
    multiplier = max(1, int(multiplier))
    if multiplier <= 1 or target_col not in base.columns:
        return base, {"train_only": True, "added_rows": 0, "output_train_rows": int(len(base))}
    target = pd.to_numeric(base[target_col], errors="coerce").fillna(-9999).astype(int)
    rare = base.loc[target.isin(rare_classes)].copy()
    if rare.empty:
        return base, {"train_only": True, "added_rows": 0, "output_train_rows": int(len(base))}
    sampled = pd.concat([rare] * (multiplier - 1), ignore_index=True)
    sampled["_v444_synthetic"] = 1
    base["_v444_synthetic"] = 0
    out = pd.concat([base, sampled], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out, {
        "train_only": True,
        "rare_classes": sorted(int(v) for v in rare_classes),
        "multiplier": multiplier,
        "original_train_rows": int(len(train)),
        "added_rows": int(len(sampled)),
        "output_train_rows": int(len(out)),
    }


def _apply_mask_dropout(x: pd.DataFrame, *, probability: float, random_state: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    probability = float(np.clip(probability, 0.0, 0.95))
    x_out = x.reset_index(drop=True).copy()
    if probability <= 0.0 or x_out.empty:
        return x_out, {"mask_probability": probability, "masked_cells": 0, "train_only": True}
    rng = np.random.default_rng(random_state)
    mask = rng.random(x_out.shape) < probability
    masked_cells = int(mask.sum())
    values = x_out.to_numpy(dtype=float, copy=True)
    values[mask] = 0.0
    return pd.DataFrame(values, columns=x_out.columns), {
        "mask_probability": probability,
        "masked_cells": masked_cells,
        "train_only": True,
    }


def _variant_train_frame(
    x: pd.DataFrame,
    y: np.ndarray,
    validation_x: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    target_col: str,
    rare_classes: set[int],
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    frame = x.reset_index(drop=True).copy()
    frame[target_col] = np.asarray(y, dtype=int)
    reports: dict[str, Any] = {}

    if int(cfg["resample_multiplier"]) > 1:
        frame, report = _resample_rare_rows(
            frame,
            target_col=target_col,
            rare_classes=rare_classes,
            multiplier=int(cfg["resample_multiplier"]),
            random_state=random_state,
        )
        reports["resample_report"] = report

    if int(cfg["smote_multiplier"]) > 1:
        frame, _validation, report = apply_train_only_smote_like_augmentation(
            frame,
            validation_x.assign(**{target_col: 0}),
            target_col=target_col,
            rare_classes=rare_classes,
            multiplier=int(cfg["smote_multiplier"]),
            random_state=random_state,
        )
        reports["smote_report"] = report

    y_out = pd.to_numeric(frame.pop(target_col), errors="coerce").fillna(-9999).astype(int)
    x_out = frame.drop(columns=["_v444_synthetic"], errors="ignore").reindex(columns=x.columns, fill_value=0.0)
    x_out, mask_report = _apply_mask_dropout(
        x_out,
        probability=float(cfg["mask_probability"]),
        random_state=random_state + 101,
    )
    reports["mask_report"] = mask_report
    return x_out, y_out, reports


def _target_metrics(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray, classes: list[int], weak_classes: set[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
    }
    weak_mask = np.isin(y_true, list(weak_classes))
    metrics["weak_support"] = int(weak_mask.sum())
    metrics["weak_accuracy"] = float(np.mean(pred[weak_mask] == y_true[weak_mask])) if weak_mask.any() else 0.0
    for cls in sorted(weak_classes):
        cls_mask = y_true == cls
        if cls_mask.any():
            precision, recall, f1, support = precision_recall_fscore_support(
                cls_mask.astype(int),
                (pred == cls).astype(int),
                average="binary",
                zero_division=0,
            )
            metrics[f"class_{cls}_precision"] = float(precision)
            metrics[f"class_{cls}_recall"] = float(recall)
            metrics[f"class_{cls}_f1"] = float(f1)
            metrics[f"class_{cls}_support"] = int(cls_mask.sum())
        else:
            metrics[f"class_{cls}_precision"] = 0.0
            metrics[f"class_{cls}_recall"] = 0.0
            metrics[f"class_{cls}_f1"] = 0.0
            metrics[f"class_{cls}_support"] = 0
    weak_indices = [idx for idx, cls in enumerate(classes) if cls in weak_classes]
    metrics["mean_oof_weak_probability"] = float(prob[:, weak_indices].sum(axis=1).mean()) if weak_indices else 0.0
    return metrics


def _run_variant(
    target: str,
    cfg: dict[str, Any],
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    groups: pd.Series | None,
    *,
    weak_classes: set[int],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    classes = sorted(int(v) for v in np.unique(y))
    oof_prob = np.zeros((len(x_train), len(classes)), dtype=float)
    fold_reports: list[dict[str, Any]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(_splitter(y, groups, n_splits=n_splits, random_state=random_state), start=1):
        fold_x, fold_y, prep_report = _variant_train_frame(
            x_train.iloc[train_idx],
            y[train_idx],
            x_train.iloc[valid_idx],
            cfg=cfg,
            target_col="_v444_target",
            rare_classes=weak_classes,
            random_state=random_state + fold_id,
        )
        model = _fit_model(
            fold_x,
            fold_y,
            use_class_weight=bool(cfg["use_class_weight"]),
            random_state=random_state + fold_id,
        )
        oof_prob[valid_idx] = _predict_aligned_proba(model, x_train.iloc[valid_idx], classes)
        fold_reports.append(
            {
                "target": target,
                "variant": cfg["name"],
                "fold": fold_id,
                "train_rows_before": int(len(train_idx)),
                "train_rows_after": int(len(fold_x)),
                "valid_rows": int(len(valid_idx)),
                **prep_report,
            }
        )

    missing = oof_prob.sum(axis=1) <= 0
    if missing.any():
        oof_prob[missing, :] = 1.0 / max(1, len(classes))

    full_x, full_y, full_report = _variant_train_frame(
        x_train,
        y,
        x_test,
        cfg=cfg,
        target_col="_v444_target",
        rare_classes=weak_classes,
        random_state=random_state + 7000,
    )
    model = _fit_model(full_x, full_y, use_class_weight=bool(cfg["use_class_weight"]), random_state=random_state + 8000)
    test_prob = _predict_aligned_proba(model, x_test, classes)
    oof_pred = _prediction_from_prob(oof_prob, classes)
    test_pred = _prediction_from_prob(test_prob, classes)
    metrics = _target_metrics(y, oof_pred, oof_prob, classes, weak_classes)
    test_conf, test_margin = _confidence(test_prob)

    return {
        "target": target,
        "variant": cfg["name"],
        "classes": classes,
        "oof_prob": oof_prob,
        "test_prob": test_prob,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "test_confidence": test_conf,
        "test_margin": test_margin,
        "metrics": metrics,
        "fold_reports": fold_reports,
        "full_train_report": full_report,
    }


def _append_score_columns(score_frame: pd.DataFrame, result: dict[str, Any], *, weak_classes: set[int], target_suffix: str) -> None:
    variant = result["variant"]
    classes = result["classes"]
    prob = result["test_prob"]
    score_frame[f"{variant}_pred_{target_suffix}"] = result["test_pred"].astype(int)
    score_frame[f"{variant}_confidence"] = result["test_confidence"]
    score_frame[f"{variant}_margin"] = result["test_margin"]
    weak_indices = [idx for idx, cls in enumerate(classes) if cls in weak_classes]
    score_frame[f"{variant}_weak_prob_sum"] = prob[:, weak_indices].sum(axis=1) if weak_indices else 0.0
    for cls in sorted(weak_classes):
        if cls in classes:
            score_frame[f"{variant}_prob_{cls}"] = prob[:, classes.index(cls)]
        else:
            score_frame[f"{variant}_prob_{cls}"] = 0.0


def _prepare_real_frames(train_path: Path, test_path: Path, anchor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_raw = pd.read_csv(train_path, low_memory=False)
    test_raw = pd.read_csv(test_path, low_memory=False)
    anchor = pd.read_csv(anchor_path, low_memory=False).loc[:, SUBMISSION_COLUMNS].copy()
    train_rows = build_train_transition_rows(train_raw)
    test_rows = build_test_rows(test_raw, anchor)
    test_rows = test_rows.merge(
        anchor.rename(columns={"actionId": "anchor_actionId", "pointId": "anchor_pointId", "serverGetPoint": "anchor_serverGetPoint"}),
        on="rally_uid",
        how="left",
    )
    train_rows["anchor_actionId"] = train_rows["actionId"]
    train_rows["anchor_pointId"] = train_rows["pointId"]
    if "serverGetPoint" in train_rows.columns:
        train_rows["anchor_serverGetPoint"] = train_rows["serverGetPoint"]
    metadata = {
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows)),
        "test_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "anchor_source": str(anchor_path.resolve()),
    }
    return train_rows, test_rows, metadata


def _quick_training_sample(train_rows: pd.DataFrame, *, max_rows: int = 8000, random_state: int = 444) -> pd.DataFrame:
    if len(train_rows) <= max_rows:
        return train_rows.reset_index(drop=True)
    action_col = _target_column(train_rows, "action")
    point_col = _target_column(train_rows, "point")
    weak_mask = (
        pd.to_numeric(train_rows[action_col], errors="coerce").isin(ACTION_WEAK_CLASSES)
        | pd.to_numeric(train_rows[point_col], errors="coerce").isin(POINT_WEAK_CLASSES)
    )
    weak_rows = train_rows.loc[weak_mask]
    background_rows = train_rows.loc[~weak_mask]
    weak_take = min(len(weak_rows), max_rows // 2)
    background_take = max_rows - weak_take
    sampled = pd.concat(
        [
            weak_rows.sample(n=weak_take, random_state=random_state) if len(weak_rows) > weak_take else weak_rows,
            background_rows.sample(n=min(len(background_rows), background_take), random_state=random_state + 1),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=random_state + 2).reset_index(drop=True)


def write_sweep_outputs(
    action_scores: pd.DataFrame,
    point_scores: pd.DataFrame,
    oof_report: pd.DataFrame,
    summary: dict[str, Any],
    *,
    outdir: Path = OUTDIR,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "action_sweep_scores_test": outdir / "action_sweep_scores_test.csv",
        "point_sweep_scores_test": outdir / "point_sweep_scores_test.csv",
        "oof_sweep_report": outdir / "oof_sweep_report.csv",
        "summary": outdir / "summary.json",
    }
    action_scores.to_csv(paths["action_sweep_scores_test"], index=False)
    point_scores.to_csv(paths["point_sweep_scores_test"], index=False)
    oof_report.to_csv(paths["oof_sweep_report"], index=False)
    write_json(paths["summary"], summary)
    return paths


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
    quick: bool = False,
) -> dict[str, Any]:
    train_rows, test_rows, metadata = _prepare_real_frames(train_path, test_path, anchor_path)
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"test score table rows would be {len(test_rows)}, expected {EXPECTED_TEST_ROWS}")
    if quick:
        original_rows = len(train_rows)
        train_rows = _quick_training_sample(train_rows, max_rows=8000, random_state=444)
        metadata["quick_train_transition_rows_used"] = int(len(train_rows))
        metadata["quick_train_transition_rows_available"] = int(original_rows)

    x_train, x_test, feature_columns = build_feature_matrix(train_rows, test_rows)
    action_y = pd.to_numeric(train_rows[_target_column(train_rows, "action")], errors="coerce").fillna(-9999).astype(int).to_numpy()
    point_y = pd.to_numeric(train_rows[_target_column(train_rows, "point")], errors="coerce").fillna(-9999).astype(int).to_numpy()
    groups = train_rows["match"] if "match" in train_rows.columns else train_rows.get("rally_uid")
    n_splits = 2 if quick else 5
    grid = build_sweep_grid()

    action_scores = pd.DataFrame({"row_id": np.arange(len(test_rows), dtype=int), "rally_uid": test_rows["rally_uid"].to_numpy()})
    point_scores = action_scores.copy()
    report_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for cfg in grid:
        action_result = _run_variant(
            "action",
            cfg,
            x_train,
            action_y,
            x_test,
            groups,
            weak_classes=ACTION_WEAK_CLASSES,
            n_splits=n_splits,
            random_state=444,
        )
        point_result = _run_variant(
            "point",
            cfg,
            x_train,
            point_y,
            x_test,
            groups,
            weak_classes=POINT_WEAK_CLASSES,
            n_splits=n_splits,
            random_state=1444,
        )
        _append_score_columns(action_scores, action_result, weak_classes=ACTION_WEAK_CLASSES, target_suffix="actionId")
        _append_score_columns(point_scores, point_result, weak_classes=POINT_WEAK_CLASSES, target_suffix="pointId")

        for result, weak_classes in ((action_result, ACTION_WEAK_CLASSES), (point_result, POINT_WEAK_CLASSES)):
            metrics = result["metrics"]
            report_rows.append(
                {
                    "target": result["target"],
                    "variant": result["variant"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weak_accuracy": metrics["weak_accuracy"],
                    "weak_support": metrics["weak_support"],
                    "mean_oof_weak_probability": metrics["mean_oof_weak_probability"],
                    "folds": n_splits,
                    "test_rows": int(len(test_rows)),
                }
            )
            for cls in sorted(weak_classes):
                class_rows.append(
                    {
                        "target": result["target"],
                        "variant": result["variant"],
                        "class": int(cls),
                        "support": int(metrics[f"class_{cls}_support"]),
                        "precision": float(metrics[f"class_{cls}_precision"]),
                        "recall": float(metrics[f"class_{cls}_recall"]),
                        "f1": float(metrics[f"class_{cls}_f1"]),
                    }
                )
            for fold_report in result["fold_reports"]:
                fold_rows.append(
                    {
                        "target": fold_report["target"],
                        "variant": fold_report["variant"],
                        "fold": fold_report["fold"],
                        "train_rows_before": fold_report["train_rows_before"],
                        "train_rows_after": fold_report["train_rows_after"],
                        "valid_rows": fold_report["valid_rows"],
                    }
                )

    oof_report = pd.DataFrame(report_rows).sort_values(["target", "macro_f1"], ascending=[True, False]).reset_index(drop=True)
    class_report = pd.DataFrame(class_rows)
    best_action = oof_report.loc[oof_report["target"].eq("action")].sort_values("macro_f1", ascending=False).head(5)
    best_point = oof_report.loc[oof_report["target"].eq("point")].sort_values("macro_f1", ascending=False).head(5)
    best_weak = (
        class_report.sort_values(["target", "f1", "support"], ascending=[True, False, False])
        .groupby(["target", "class"], as_index=False)
        .head(1)
        .sort_values(["target", "f1"], ascending=[True, False])
    )
    summary = {
        "version": "V444",
        "quick": bool(quick),
        "variants": [cfg["name"] for cfg in grid],
        "action_weak_classes": sorted(ACTION_WEAK_CLASSES),
        "point_weak_classes": sorted(POINT_WEAK_CLASSES),
        "metadata": metadata,
        "feature_count": int(len(feature_columns)),
        "action_score_rows": int(len(action_scores)),
        "point_score_rows": int(len(point_scores)),
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "score_tables_row_check": {
            "action": int(len(action_scores) == EXPECTED_TEST_ROWS),
            "point": int(len(point_scores) == EXPECTED_TEST_ROWS),
        },
        "best_action_variants": best_action.to_dict("records"),
        "best_point_variants": best_point.to_dict("records"),
        "top_weak_class_variants": best_weak.to_dict("records"),
        "fold_report_rows": fold_rows,
        "submission_exports": 0,
    }
    paths = write_sweep_outputs(action_scores, point_scores, oof_report, summary, outdir=outdir)
    summary["outputs"] = {key: str(path.resolve()) for key, path in paths.items()}
    write_json(outdir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use a bounded train sample and two folds")
    args = parser.parse_args()
    summary = run_pipeline(quick=args.quick)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
