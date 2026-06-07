"""V433 weak-class expert bank.

Trains fold-safe one-vs-rest weak action and point experts, plus restricted
candidate scorers for the configured weak groups. The script only writes score
tables and diagnostics; submission packaging is intentionally left to later
versions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v416_external_embedding_aicup_finetune import build_test_rows, build_train_transition_rows


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v433_weak_class_expert_bank"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
WEAK_ACTION_CLASSES = {0, 3, 4, 5, 7, 8, 9, 12, 14}
WEAK_POINT_CLASSES = {1, 3, 4, 7, 8, 9}

NUMERIC_FEATURE_COLUMNS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "anchor_actionId",
    "anchor_pointId",
    "anchor_serverGetPoint",
]


@dataclass(frozen=True)
class WeakGroup:
    name: str
    target: str
    labels: tuple[int, ...]
    description: str


class ConstantBinaryModel:
    def __init__(self, positive_probability: float):
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(x), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


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


def weak_group_definitions() -> dict[str, WeakGroup]:
    groups = {
        "action_terminal_zero": WeakGroup("action_terminal_zero", "action", (0,), "weak terminal/unknown action"),
        "action_attack_3_4_5_7": WeakGroup("action_attack_3_4_5_7", "action", (3, 4, 5, 7), "attack and drive weak actions"),
        "action_control_8_9_11": WeakGroup("action_control_8_9_11", "action", (8, 9, 11), "control-style weak actions"),
        "action_defensive_12_14": WeakGroup("action_defensive_12_14", "action", (12, 14), "defensive and late attack actions"),
        "action_long_rally_transition": WeakGroup(
            "action_long_rally_transition",
            "action",
            (3, 4, 5, 7, 12, 14),
            "late-prefix attack/defense transition detector",
        ),
        "point_terminal_zero": WeakGroup("point_terminal_zero", "point", (0,), "terminal point detector"),
        "point_short_side_1_3": WeakGroup("point_short_side_1_3", "point", (1, 3), "short side point detector"),
        "point_half_boundary_4_6": WeakGroup("point_half_boundary_4_6", "point", (4, 6), "half-court boundary detector"),
        "point_long_side_7_8_9": WeakGroup("point_long_side_7_8_9", "point", (7, 8, 9), "long side point detector"),
        "point_nonterminal_swap": WeakGroup("point_nonterminal_swap", "point", (1, 2, 3, 4, 5, 6, 7, 8, 9), "nonzero point swap detector"),
    }
    for label in sorted(WEAK_ACTION_CLASSES):
        groups[f"action_class_{label}"] = WeakGroup(
            f"action_class_{label}",
            "action",
            (int(label),),
            f"one-vs-rest weak action class {label}",
        )
    for label in sorted(WEAK_POINT_CLASSES):
        groups[f"point_class_{label}"] = WeakGroup(
            f"point_class_{label}",
            "point",
            (int(label),),
            f"one-vs-rest weak point class {label}",
        )
    return groups


def _target_column(frame: pd.DataFrame, target: str) -> str:
    candidates = [f"target_{target}Id", f"{target}Id"]
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"frame missing {target} target column; tried {candidates}")


def _anchor_column(frame: pd.DataFrame, target: str) -> str | None:
    for col in (f"anchor_{target}Id", f"v362_{target}Id"):
        if col in frame.columns:
            return col
    return None


def apply_train_only_oversampling(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str,
    positive_labels: set[int] | tuple[int, ...] | list[int],
    multiplier: int = 1,
    random_state: int = 433,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Duplicate positive-label training rows only; validation/test are untouched."""

    train_out = train.reset_index(drop=True).copy()
    validation_out = validation.copy()
    test_out = test.copy()
    labels = {int(v) for v in positive_labels}
    multiplier = max(1, int(multiplier))
    synthetic = pd.DataFrame()
    if multiplier > 1 and target_col in train_out.columns:
        positive = train_out.loc[pd.to_numeric(train_out[target_col], errors="coerce").isin(labels)].copy()
        if not positive.empty:
            synthetic = pd.concat([positive] * (multiplier - 1), ignore_index=True)
            synthetic["_v433_synthetic"] = 1
            train_out["_v433_synthetic"] = 0
            train_out = pd.concat([train_out, synthetic], ignore_index=True)
            train_out = train_out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if "_v433_synthetic" not in train_out.columns:
        train_out["_v433_synthetic"] = 0
    report = {
        "train_only": True,
        "target_col": target_col,
        "positive_labels": sorted(labels),
        "multiplier": multiplier,
        "original_train_rows": int(len(train)),
        "synthetic_rows": int(len(synthetic)),
        "output_train_rows": int(len(train_out)),
        "validation_rows_unchanged": int(len(validation_out)),
        "test_rows_unchanged": int(len(test_out)),
    }
    return train_out, validation_out, test_out, report


def _prefix_len(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _point_depth(point_id: Any) -> int:
    try:
        value = int(point_id)
    except (TypeError, ValueError):
        return 0
    if value in {1, 2, 3}:
        return 1
    if value in {4, 5, 6}:
        return 2
    if value in {7, 8, 9}:
        return 3
    return 0


def _point_side(point_id: Any) -> int:
    try:
        value = int(point_id)
    except (TypeError, ValueError):
        return 0
    if value in {1, 4, 7}:
        return 1
    if value in {2, 5, 8}:
        return 2
    if value in {3, 6, 9}:
        return 3
    return 0


def _augment_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    action_source = out["actionId"] if "actionId" in out.columns else pd.Series(0, index=out.index)
    point_source = out["pointId"] if "pointId" in out.columns else pd.Series(0, index=out.index)
    strike = out["strikeNumber"] if "strikeNumber" in out.columns else pd.Series(0, index=out.index)
    out["prefix_len"] = strike.map(_prefix_len).astype(float)
    out["prefix_ge_4"] = out["prefix_len"].ge(4).astype(float)
    out["current_action_is_weak"] = pd.to_numeric(action_source, errors="coerce").isin(WEAK_ACTION_CLASSES).astype(float)
    out["current_point_is_weak"] = pd.to_numeric(point_source, errors="coerce").isin(WEAK_POINT_CLASSES).astype(float)
    out["current_point_depth"] = point_source.map(_point_depth).astype(float)
    out["current_point_side"] = point_source.map(_point_side).astype(float)
    out["current_point_terminal"] = pd.to_numeric(point_source, errors="coerce").fillna(-1).eq(0).astype(float)
    if "anchor_actionId" in out.columns:
        out["anchor_action_is_weak"] = pd.to_numeric(out["anchor_actionId"], errors="coerce").isin(WEAK_ACTION_CLASSES).astype(float)
    if "anchor_pointId" in out.columns:
        out["anchor_point_is_weak"] = pd.to_numeric(out["anchor_pointId"], errors="coerce").isin(WEAK_POINT_CLASSES).astype(float)
        out["anchor_point_depth"] = out["anchor_pointId"].map(_point_depth).astype(float)
        out["anchor_point_side"] = out["anchor_pointId"].map(_point_side).astype(float)
    return out


def build_feature_matrix(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_aug = _augment_engineered_features(train_rows)
    test_aug = _augment_engineered_features(test_rows)
    numeric_cols: list[str] = []
    for frame in (train_aug, test_aug):
        for col in frame.columns:
            if col.startswith("target_") or col in {"rally_uid", "match"}:
                continue
            if col in NUMERIC_FEATURE_COLUMNS or col.startswith(("v432_", "v420_", "v419_", "anchor_", "prefix_", "current_", "intent_", "point_", "action_")):
                numeric_cols.append(col)
    numeric_cols = sorted(set(numeric_cols))
    train_columns: dict[str, Any] = {}
    test_columns: dict[str, Any] = {}
    for col in numeric_cols:
        train_columns[col] = pd.to_numeric(train_aug[col], errors="coerce") if col in train_aug.columns else pd.Series(0.0, index=train_aug.index)
        test_columns[col] = pd.to_numeric(test_aug[col], errors="coerce") if col in test_aug.columns else pd.Series(0.0, index=test_aug.index)
    train_x = pd.DataFrame(train_columns, index=train_aug.index).fillna(0.0).astype(float)
    test_x = pd.DataFrame(test_columns, index=test_aug.index).fillna(0.0).astype(float)
    return train_x.reset_index(drop=True), test_x.reset_index(drop=True), numeric_cols


def _oversample_feature_rows(
    x: pd.DataFrame,
    y_binary: np.ndarray,
    *,
    multiplier: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    x_out = x.reset_index(drop=True)
    y_out = np.asarray(y_binary, dtype=int)
    multiplier = max(1, int(multiplier))
    if multiplier <= 1:
        return x_out, y_out, 0
    positive_idx = np.flatnonzero(y_out == 1)
    if len(positive_idx) == 0:
        return x_out, y_out, 0
    extra_indices = np.tile(positive_idx, multiplier - 1)
    x_aug = pd.concat([x_out, x_out.iloc[extra_indices].copy()], ignore_index=True)
    y_aug = np.concatenate([y_out, y_out[extra_indices]])
    order = np.random.default_rng(random_state).permutation(len(y_aug))
    return x_aug.iloc[order].reset_index(drop=True), y_aug[order], int(len(extra_indices))


def _splitter(y_binary: np.ndarray, groups: pd.Series | None, *, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y_binary), dtype=int)
    y_binary = np.asarray(y_binary, dtype=int)
    if len(indices) < 3:
        return [(indices, indices)]
    if groups is not None and groups.nunique(dropna=True) >= 2:
        splits = int(min(max(2, n_splits), groups.nunique(dropna=True)))
        return list(GroupKFold(n_splits=splits).split(np.zeros(len(y_binary)), y_binary, groups))
    counts = pd.Series(y_binary).value_counts()
    if not counts.empty and counts.min() >= 2 and len(counts) > 1:
        splits = int(min(max(2, n_splits), counts.min()))
        return list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state).split(np.zeros(len(y_binary)), y_binary))
    return [(indices, indices)]


def _fit_binary_model(x: pd.DataFrame, y_binary: np.ndarray, *, class_weight: str | None = "balanced") -> Any:
    y_binary = np.asarray(y_binary, dtype=int)
    if len(np.unique(y_binary)) < 2:
        return ConstantBinaryModel(float(y_binary.mean()) if len(y_binary) else 0.0)
    kwargs: dict[str, Any] = {
        "max_iter": 500,
        "random_state": 433,
        "class_weight": class_weight,
    }
    return make_pipeline(StandardScaler(), LogisticRegression(**kwargs)).fit(x, y_binary)


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["logisticregression"].classes_
    classes = np.asarray(classes, dtype=int)
    if 1 in classes:
        return prob[:, int(np.flatnonzero(classes == 1)[0])]
    return np.zeros(len(x), dtype=float)


def _binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "mean_score": float(np.mean(score)) if len(score) else 0.0,
    }


def _candidate_from_group(y: pd.Series, score: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    fallback = int(labels[0]) if labels else 0
    if len(labels) == 1:
        return np.full(len(y), fallback, dtype=int)
    label_rates = y[y.isin(labels)].value_counts(normalize=True)
    if label_rates.empty:
        return np.full(len(y), fallback, dtype=int)
    best = int(label_rates.sort_values(ascending=False).index[0])
    return np.full(len(y), best, dtype=int)


def _changed_row_utility(
    y_true: pd.Series,
    candidate: np.ndarray,
    anchor: pd.Series | None,
    score: np.ndarray,
) -> dict[str, Any]:
    if anchor is None:
        return {"feasible": False}
    y = pd.to_numeric(y_true, errors="coerce").fillna(-9999).astype(int).to_numpy()
    anchor_values = pd.to_numeric(anchor, errors="coerce").fillna(-9999).astype(int).to_numpy()
    candidate = np.asarray(candidate, dtype=int)
    changed = candidate != anchor_values
    candidate_correct = candidate == y
    anchor_correct = anchor_values == y
    delta = candidate_correct.astype(int) - anchor_correct.astype(int)
    return {
        "feasible": True,
        "changed_rows": int(changed.sum()),
        "estimated_net_utility": int(delta[changed].sum()) if changed.any() else 0,
        "mean_changed_score": float(np.mean(score[changed])) if changed.any() else 0.0,
    }


def train_expert_bank(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    groups: dict[str, WeakGroup] | None = None,
    n_splits: int = 5,
    oversample_multiplier: int = 1,
    class_weight: str | None = "balanced",
    random_state: int = 433,
) -> dict[str, Any]:
    groups = groups or weak_group_definitions()
    x_train, x_test, feature_columns = build_feature_matrix(train_rows, test_rows)
    group_values = train_rows["match"] if "match" in train_rows.columns else train_rows.get("rally_uid")

    action_oof = pd.DataFrame({"row_id": np.arange(len(train_rows), dtype=int), "rally_uid": train_rows.get("rally_uid", pd.Series(range(len(train_rows))))})
    point_oof = action_oof.copy()
    action_test = pd.DataFrame({"row_id": np.arange(len(test_rows), dtype=int), "rally_uid": test_rows.get("rally_uid", pd.Series(range(len(test_rows))))})
    point_test = action_test.copy()
    reports: list[dict[str, Any]] = []
    oversampling_reports: list[dict[str, Any]] = []

    for name, group in groups.items():
        target_col = _target_column(train_rows, group.target)
        y = pd.to_numeric(train_rows[target_col], errors="coerce").fillna(-9999).astype(int)
        y_binary = y.isin(group.labels).astype(int).to_numpy()
        oof_score = np.zeros(len(train_rows), dtype=float)
        fold_f1: list[float] = []

        for fold_id, (tr_idx, va_idx) in enumerate(_splitter(y_binary, group_values, n_splits=n_splits, random_state=random_state), start=1):
            x_fold, y_fold, synthetic_rows = _oversample_feature_rows(
                x_train.iloc[tr_idx],
                y_binary[tr_idx],
                multiplier=oversample_multiplier,
                random_state=random_state + fold_id,
            )
            oversample_report = {
                "train_only": True,
                "target_col": target_col,
                "positive_labels": sorted(int(v) for v in group.labels),
                "multiplier": max(1, int(oversample_multiplier)),
                "original_train_rows": int(len(tr_idx)),
                "synthetic_rows": int(synthetic_rows),
                "output_train_rows": int(len(x_fold)),
                "validation_rows_unchanged": int(len(va_idx)),
                "test_rows_unchanged": int(len(test_rows)),
            }
            oversample_report["expert"] = name
            oversample_report["fold"] = fold_id
            oversampling_reports.append(oversample_report)
            model = _fit_binary_model(x_fold.reindex(columns=feature_columns, fill_value=0.0), y_fold, class_weight=class_weight)
            fold_score = _positive_probability(model, x_train.iloc[va_idx].reindex(columns=feature_columns, fill_value=0.0))
            oof_score[va_idx] = fold_score
            fold_f1.append(float(f1_score(y_binary[va_idx], fold_score >= 0.5, zero_division=0)))

        missing = oof_score < 0
        if missing.any():
            oof_score[missing] = float(np.mean(y_binary))
        sampled_all_x, sampled_all_y, synthetic_rows = _oversample_feature_rows(
            x_train,
            y_binary,
            multiplier=oversample_multiplier,
            random_state=random_state,
        )
        full_oversample_report = {
            "train_only": True,
            "target_col": target_col,
            "positive_labels": sorted(int(v) for v in group.labels),
            "multiplier": max(1, int(oversample_multiplier)),
            "original_train_rows": int(len(x_train)),
            "synthetic_rows": int(synthetic_rows),
            "output_train_rows": int(len(sampled_all_x)),
            "validation_rows_unchanged": 0,
            "test_rows_unchanged": int(len(test_rows)),
        }
        full_oversample_report["expert"] = name
        full_oversample_report["fold"] = "full_train"
        oversampling_reports.append(full_oversample_report)
        final_model = _fit_binary_model(sampled_all_x.reindex(columns=feature_columns, fill_value=0.0), sampled_all_y, class_weight=class_weight)
        test_score = _positive_probability(final_model, x_test.reindex(columns=feature_columns, fill_value=0.0))

        candidate = _candidate_from_group(y, oof_score, group.labels)
        anchor_col = _anchor_column(train_rows, group.target)
        utility = _changed_row_utility(y, candidate, train_rows[anchor_col] if anchor_col else None, oof_score)
        metrics = _binary_metrics(y_binary, oof_score)
        report = {
            "expert": name,
            "target": group.target,
            "labels": " ".join(str(v) for v in group.labels),
            "description": group.description,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "support": metrics["support"],
            "positive_rate": metrics["positive_rate"],
            "mean_oof_score": metrics["mean_score"],
            "mean_test_score": float(np.mean(test_score)) if len(test_score) else 0.0,
            "fold_f1_mean": float(np.mean(fold_f1)) if fold_f1 else 0.0,
            "changed_rows": utility.get("changed_rows"),
            "estimated_net_utility": utility.get("estimated_net_utility"),
            "utility_feasible": utility["feasible"],
        }
        reports.append(report)

        destination_oof = action_oof if group.target == "action" else point_oof
        destination_test = action_test if group.target == "action" else point_test
        destination_oof[f"{name}_score"] = np.clip(oof_score, 0.0, 1.0)
        destination_oof[f"{name}_candidate"] = candidate.astype(int)
        destination_test[f"{name}_score"] = np.clip(test_score, 0.0, 1.0)
        destination_test[f"{name}_candidate"] = int(group.labels[0]) if group.labels else 0

    expert_reports = pd.DataFrame(reports).sort_values(["target", "f1", "support"], ascending=[True, False, False]).reset_index(drop=True)
    action_groups = [name for name, group in groups.items() if group.target == "action"]
    point_groups = [name for name, group in groups.items() if group.target == "point"]
    combined_oof = action_oof.merge(point_oof.drop(columns=["rally_uid"], errors="ignore"), on="row_id", how="outer", suffixes=("", "_point"))
    combined_test = action_test.merge(point_test.drop(columns=["rally_uid"], errors="ignore"), on="row_id", how="outer", suffixes=("", "_point"))
    return {
        "action_groups": action_groups,
        "point_groups": point_groups,
        "action_expert_scores_oof": action_oof,
        "point_expert_scores_oof": point_oof,
        "expert_oof_scores": combined_oof,
        "action_expert_scores_test": action_test,
        "point_expert_scores_test": point_test,
        "expert_test_scores": combined_test,
        "expert_reports": expert_reports,
        "oversampling_reports": pd.DataFrame(oversampling_reports),
        "feature_columns": feature_columns,
    }


def _merge_optional_predictions(base: pd.DataFrame, prediction_paths: list[Path], *, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy()
    used: list[str] = []
    for path in prediction_paths:
        if not path.exists():
            continue
        pred = pd.read_csv(path, low_memory=False)
        if "rally_uid" not in pred.columns:
            continue
        keep = ["rally_uid"] + [
            col
            for col in pred.columns
            if col.startswith(("action_prob_", "point_prob_", "action_confidence", "point_confidence", "action_margin", "point_margin"))
        ]
        if len(keep) == 1:
            continue
        renamed = pred[keep].copy()
        renamed = renamed.rename(columns={col: f"{prefix}_{path.stem}_{col}" for col in keep if col != "rally_uid"})
        out = out.merge(renamed, on="rally_uid", how="left")
        used.append(str(path.resolve()))
    return out, used


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

    v432_paths = sorted((ROOT / "v432_aicup_finetune_model_zoo").glob("test_predictions_*.csv"))
    v420_paths = sorted((ROOT / "v420_rare_class_augmented_exact_models").glob("test_predictions_*.csv"))
    v419_paths = [ROOT / "v419_intent_first_point_finetune" / "test_predictions.csv"]
    test_rows, used_v432 = _merge_optional_predictions(test_rows, v432_paths, prefix="v432")
    test_rows, used_v420 = _merge_optional_predictions(test_rows, v420_paths, prefix="v420")
    test_rows, used_v419 = _merge_optional_predictions(test_rows, v419_paths, prefix="v419")
    # OOF probabilities are only used when available and aligned to train rallies.
    train_rows, train_v432 = _merge_optional_predictions(train_rows, sorted((ROOT / "v432_aicup_finetune_model_zoo").glob("oof_predictions_*.csv")), prefix="v432")
    train_rows, train_v419 = _merge_optional_predictions(train_rows, [ROOT / "v419_intent_first_point_finetune" / "oof_predictions.csv"], prefix="v419")
    metadata = {
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows)),
        "test_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "v432_probabilities_used": bool(used_v432 or train_v432),
        "fallback_sources_used": {
            "v420": bool(used_v420),
            "v419": bool(used_v419 or train_v419),
            "v362_anchor": True,
        },
        "optional_prediction_files": used_v432 + used_v420 + used_v419 + train_v432 + train_v419,
    }
    return train_rows, test_rows, metadata


def _quick_training_sample(train_rows: pd.DataFrame, *, max_rows: int = 60000, random_state: int = 433) -> pd.DataFrame:
    if len(train_rows) <= max_rows:
        return train_rows.reset_index(drop=True)
    action_col = _target_column(train_rows, "action")
    point_col = _target_column(train_rows, "point")
    weak_mask = (
        pd.to_numeric(train_rows[action_col], errors="coerce").isin(WEAK_ACTION_CLASSES)
        | pd.to_numeric(train_rows[point_col], errors="coerce").isin(WEAK_POINT_CLASSES)
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


def write_expert_outputs(result: dict[str, Any], outdir: Path = OUTDIR) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "action_expert_scores_test": outdir / "action_expert_scores_test.csv",
        "point_expert_scores_test": outdir / "point_expert_scores_test.csv",
        "action_expert_scores_oof": outdir / "action_expert_scores_oof.csv",
        "point_expert_scores_oof": outdir / "point_expert_scores_oof.csv",
        "expert_oof_scores": outdir / "expert_oof_scores.csv",
        "expert_scores_test": outdir / "expert_test_scores.csv",
        "expert_reports": outdir / "expert_reports.csv",
    }
    result["action_expert_scores_test"].to_csv(paths["action_expert_scores_test"], index=False)
    result["point_expert_scores_test"].to_csv(paths["point_expert_scores_test"], index=False)
    result["action_expert_scores_oof"].to_csv(paths["action_expert_scores_oof"], index=False)
    result["point_expert_scores_oof"].to_csv(paths["point_expert_scores_oof"], index=False)
    result["expert_oof_scores"].to_csv(paths["expert_oof_scores"], index=False)
    result["expert_test_scores"].to_csv(paths["expert_scores_test"], index=False)
    result["expert_reports"].to_csv(paths["expert_reports"], index=False)
    if "oversampling_reports" in result:
        result["oversampling_reports"].to_csv(outdir / "oversampling_reports.csv", index=False)
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
    if quick:
        original_rows = len(train_rows)
        train_rows = _quick_training_sample(train_rows, max_rows=60000, random_state=433)
        metadata["quick_train_transition_rows_used"] = int(len(train_rows))
        metadata["quick_train_transition_rows_available"] = int(original_rows)
    result = train_expert_bank(
        train_rows,
        test_rows,
        groups=weak_group_definitions(),
        n_splits=3 if quick else 5,
        oversample_multiplier=2,
        class_weight="balanced",
        random_state=433,
    )
    paths = write_expert_outputs(result, outdir)
    reports = result["expert_reports"]
    best_action = reports.loc[reports["target"].eq("action")].sort_values("f1", ascending=False).head(5)
    best_point = reports.loc[reports["target"].eq("point")].sort_values("f1", ascending=False).head(5)
    summary = {
        "version": "V433",
        "quick": bool(quick),
        "weak_action_classes": sorted(WEAK_ACTION_CLASSES),
        "weak_point_classes": sorted(WEAK_POINT_CLASSES),
        "metadata": metadata,
        "action_experts": result["action_groups"],
        "point_experts": result["point_groups"],
        "feature_count": int(len(result["feature_columns"])),
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "best_action_experts": best_action[["expert", "labels", "precision", "recall", "f1", "support", "estimated_net_utility"]].to_dict("records"),
        "best_point_experts": best_point[["expert", "labels", "precision", "recall", "f1", "support", "estimated_net_utility"]].to_dict("records"),
        "submission_exports": 0,
    }
    write_json(outdir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use fewer folds for smoke verification")
    args = parser.parse_args()
    summary = run_pipeline(quick=args.quick)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
