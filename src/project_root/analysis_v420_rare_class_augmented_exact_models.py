"""V420 rare-class augmentation and SMOTE-safe exact-model probes.

This experiment trains fold-safe action/point variants from AICUP transitions
using the V418 token feature frame when present, falling back to V415.
Synthetic rows are created only from training-fold rows; validation and test
frames are never augmented.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    action_distribution_report,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
    write_json,
)
from analysis_v416_external_embedding_aicup_finetune import (
    build_feature_frame,
    build_test_rows,
    build_train_transition_rows,
)


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
V418_TOKEN_EMBEDDINGS_PATH = ROOT / "v418_clean_external_sequence_pretrain" / "token_embeddings.csv"
V415_TOKEN_EMBEDDINGS_PATH = ROOT / "v415_clean_external_representation" / "token_embeddings.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v420_rare_class_augmented_exact_models"

RARE_ACTION_CLASSES = {0, 3, 4, 5, 7, 8, 9, 12, 14}
RARE_POINT_CLASSES = {1, 3, 4, 7, 8, 9}
VARIANTS = (
    "class_weight_balanced",
    "balanced_sampler_by_rare_action",
    "smote_like_interpolation_action_rare",
    "smote_like_interpolation_point_rare",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


class ConstantProbabilityModel:
    def __init__(self, label: int):
        self.classes_ = np.array([int(label)], dtype=int)
        self.label = int(label)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.ones((len(x), 1), dtype=float)


def _load_token_embeddings() -> tuple[pd.DataFrame, Path, bool]:
    if V418_TOKEN_EMBEDDINGS_PATH.exists():
        return pd.read_csv(V418_TOKEN_EMBEDDINGS_PATH), V418_TOKEN_EMBEDDINGS_PATH, False
    return pd.read_csv(V415_TOKEN_EMBEDDINGS_PATH), V415_TOKEN_EMBEDDINGS_PATH, True


def _splitter(y: np.ndarray, groups: pd.Series | None, *, seed: int = 420) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    if groups is not None and groups.nunique(dropna=True) >= 3:
        n_splits = int(min(3, groups.nunique(dropna=True)))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))

    class_counts = pd.Series(y).value_counts()
    min_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_count >= 2:
        n_splits = int(min(3, min_count))
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    indices = np.arange(len(y), dtype=int)
    return [(indices, indices)]


def _fit_model(x: pd.DataFrame, y: pd.Series | np.ndarray) -> Any:
    y_array = np.asarray(y, dtype=int)
    classes = np.unique(y_array)
    if len(classes) == 1:
        return ConstantProbabilityModel(int(classes[0]))

    kwargs: dict[str, Any] = {
        "class_weight": "balanced",
        "max_iter": 350,
        "random_state": 420,
    }
    signature = inspect.signature(LogisticRegression)
    if "multi_class" in signature.parameters:
        kwargs["multi_class"] = "auto"
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = -1
    return make_pipeline(StandardScaler(), LogisticRegression(**kwargs)).fit(x, y_array)


def _predict_aligned_proba(model: Any, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    local = model.predict_proba(x)
    model_classes = getattr(model, "classes_", None)
    if model_classes is None and hasattr(model, "named_steps"):
        model_classes = model.named_steps["logisticregression"].classes_
    out = np.zeros((len(x), len(classes)), dtype=float)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    for local_idx, label in enumerate(model_classes):
        out[:, class_to_idx[int(label)]] = local[:, local_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sum, out=out, where=row_sum > 0)
    out[row_sum.ravel() <= 0, :] = 1.0 / max(1, len(classes))
    return out


def _pred_from_prob(prob: np.ndarray, classes: list[int]) -> np.ndarray:
    return np.array([classes[idx] for idx in prob.argmax(axis=1)], dtype=int)


def _confidence(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prob.shape[1] == 0:
        return np.zeros(len(prob), dtype=float), np.zeros(len(prob), dtype=float)
    ordered = np.sort(prob, axis=1)
    top = ordered[:, -1]
    second = ordered[:, -2] if prob.shape[1] > 1 else np.zeros(len(prob), dtype=float)
    return top, top - second


def augment_minority_rows(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series | None = None,
    *,
    rare_classes: set[int],
    multiplier: int = 2,
    seed: int = 420,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Create SMOTE-like interpolated rows from rare-class training rows only."""

    x_base = x_train.reset_index(drop=True).copy()
    y_base = pd.Series(y_train, name=getattr(y_train, "name", "target")).reset_index(drop=True).astype(int)
    if groups is None:
        group_base = pd.Series([f"row_{idx}" for idx in range(len(x_base))], name="group")
    else:
        group_base = pd.Series(groups, name=getattr(groups, "name", "group")).reset_index(drop=True)

    numeric = x_base.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    rng = np.random.default_rng(seed)
    synthetic_rows: list[pd.Series] = []
    synthetic_y: list[int] = []
    synthetic_groups: list[str] = []
    per_class: dict[str, int] = {}

    if multiplier > 1:
        for cls in sorted(int(v) for v in rare_classes):
            class_idx = np.flatnonzero(y_base.to_numpy(dtype=int) == cls)
            if len(class_idx) == 0:
                per_class[str(cls)] = 0
                continue
            n_new = int(len(class_idx) * (multiplier - 1))
            per_class[str(cls)] = n_new
            for synth_idx in range(n_new):
                left = int(rng.choice(class_idx))
                right = int(rng.choice(class_idx))
                lam = float(rng.uniform(0.15, 0.85))
                row = (lam * numeric.iloc[left]) + ((1.0 - lam) * numeric.iloc[right])
                synthetic_rows.append(row)
                synthetic_y.append(cls)
                synthetic_groups.append(f"synthetic_{cls}_{synth_idx}")

    if synthetic_rows:
        synth_x = pd.DataFrame(synthetic_rows, columns=x_base.columns)
        aug_x = pd.concat([x_base, synth_x], ignore_index=True)
        aug_y = pd.concat([y_base, pd.Series(synthetic_y, name=y_base.name)], ignore_index=True)
        aug_groups = pd.concat([group_base, pd.Series(synthetic_groups, name=group_base.name)], ignore_index=True)
    else:
        aug_x = x_base
        aug_y = y_base
        aug_groups = group_base

    report = {
        "original_rows": int(len(x_base)),
        "synthetic_rows": int(len(aug_x) - len(x_base)),
        "output_rows": int(len(aug_x)),
        "rare_classes": sorted(int(v) for v in rare_classes),
        "per_class_synthetic_rows": per_class,
        "train_only": True,
    }
    return aug_x, aug_y.astype(int), aug_groups, report


def _oversample_rare_action_rows(
    x_train: pd.DataFrame,
    y_action: np.ndarray,
    y_point: np.ndarray,
    groups: pd.Series,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, dict[str, Any]]:
    rare_mask = np.isin(y_action, list(RARE_ACTION_CLASSES))
    rare_idx = np.flatnonzero(rare_mask)
    nonrare_count = int((~rare_mask).sum())
    if len(rare_idx) == 0 or nonrare_count == 0:
        return x_train, y_action, y_point, groups.reset_index(drop=True), {
            "original_rows": int(len(x_train)),
            "sampled_rows": int(len(x_train)),
            "added_rows": 0,
        }

    target_rare = min(nonrare_count, len(rare_idx) * 3)
    add_count = max(0, target_rare - len(rare_idx))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(rare_idx, size=add_count, replace=True) if add_count else np.array([], dtype=int)
    full_idx = np.concatenate([np.arange(len(x_train), dtype=int), sampled])
    return (
        x_train.iloc[full_idx].reset_index(drop=True),
        y_action[full_idx],
        y_point[full_idx],
        groups.iloc[full_idx].reset_index(drop=True),
        {
            "original_rows": int(len(x_train)),
            "sampled_rows": int(len(full_idx)),
            "added_rows": int(add_count),
            "rare_rows_before": int(len(rare_idx)),
            "rare_rows_after": int(len(rare_idx) + add_count),
        },
    )


def _train_variant(
    variant: str,
    x_train: pd.DataFrame,
    y_action: np.ndarray,
    y_point: np.ndarray,
    x_test: pd.DataFrame,
    groups: pd.Series | None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown V420 variant: {variant}")

    groups_base = groups.reset_index(drop=True) if groups is not None else pd.Series(np.arange(len(x_train)))
    action_classes = sorted(int(v) for v in np.unique(y_action))
    point_classes = sorted(int(v) for v in np.unique(y_point))
    action_oof = np.zeros((len(x_train), len(action_classes)), dtype=float)
    point_oof = np.zeros((len(x_train), len(point_classes)), dtype=float)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(_splitter(y_action, groups_base)):
        fold_x = x_train.iloc[train_idx].reset_index(drop=True)
        fold_action = y_action[train_idx]
        fold_point = y_point[train_idx]
        fold_groups = groups_base.iloc[train_idx].reset_index(drop=True)
        action_x = fold_x
        action_y = pd.Series(fold_action, name="target_actionId")
        point_x = fold_x
        point_y = pd.Series(fold_point, name="target_pointId")
        fold_report: dict[str, Any] = {
            "fold": int(fold_idx),
            "train_rows_before": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "action_synthetic_rows": 0,
            "point_synthetic_rows": 0,
            "sampler_added_rows": 0,
        }

        if variant == "balanced_sampler_by_rare_action":
            sampled_x, sampled_action, sampled_point, _sampled_groups, sampler_report = _oversample_rare_action_rows(
                fold_x,
                fold_action,
                fold_point,
                fold_groups,
                seed=420 + fold_idx,
            )
            action_x = sampled_x
            action_y = pd.Series(sampled_action, name="target_actionId")
            point_x = sampled_x
            point_y = pd.Series(sampled_point, name="target_pointId")
            fold_report["sampler_added_rows"] = int(sampler_report["added_rows"])
            fold_report["sampler_report"] = sampler_report

        if variant == "smote_like_interpolation_action_rare":
            action_x, action_y, _action_groups, action_report = augment_minority_rows(
                fold_x,
                pd.Series(fold_action, name="target_actionId"),
                fold_groups,
                rare_classes=RARE_ACTION_CLASSES,
                multiplier=2,
                seed=420 + fold_idx,
            )
            fold_report["action_synthetic_rows"] = int(action_report["synthetic_rows"])
            fold_report["action_augmentation_report"] = action_report

        if variant == "smote_like_interpolation_point_rare":
            point_x, point_y, _point_groups, point_report = augment_minority_rows(
                fold_x,
                pd.Series(fold_point, name="target_pointId"),
                fold_groups,
                rare_classes=RARE_POINT_CLASSES,
                multiplier=2,
                seed=1420 + fold_idx,
            )
            fold_report["point_synthetic_rows"] = int(point_report["synthetic_rows"])
            fold_report["point_augmentation_report"] = point_report

        action_model = _fit_model(action_x, action_y)
        point_model = _fit_model(point_x, point_y)
        action_oof[valid_idx] = _predict_aligned_proba(action_model, x_train.iloc[valid_idx], action_classes)
        point_oof[valid_idx] = _predict_aligned_proba(point_model, x_train.iloc[valid_idx], point_classes)
        fold_reports.append(fold_report)

    missing_action = action_oof.sum(axis=1) <= 0
    missing_point = point_oof.sum(axis=1) <= 0
    action_oof[missing_action, :] = 1.0 / len(action_classes)
    point_oof[missing_point, :] = 1.0 / len(point_classes)

    full_action_x = x_train.reset_index(drop=True)
    full_action_y = pd.Series(y_action, name="target_actionId")
    full_point_x = x_train.reset_index(drop=True)
    full_point_y = pd.Series(y_point, name="target_pointId")
    full_reports: dict[str, Any] = {}
    if variant == "balanced_sampler_by_rare_action":
        sampled_x, sampled_action, sampled_point, _sampled_groups, sampler_report = _oversample_rare_action_rows(
            full_action_x,
            y_action,
            y_point,
            groups_base,
            seed=4420,
        )
        full_action_x = sampled_x
        full_action_y = pd.Series(sampled_action, name="target_actionId")
        full_point_x = sampled_x
        full_point_y = pd.Series(sampled_point, name="target_pointId")
        full_reports["sampler_report"] = sampler_report
    elif variant == "smote_like_interpolation_action_rare":
        full_action_x, full_action_y, _full_groups, action_report = augment_minority_rows(
            full_action_x,
            full_action_y,
            groups_base,
            rare_classes=RARE_ACTION_CLASSES,
            multiplier=2,
            seed=4520,
        )
        full_reports["action_augmentation_report"] = action_report
    elif variant == "smote_like_interpolation_point_rare":
        full_point_x, full_point_y, _full_groups, point_report = augment_minority_rows(
            full_point_x,
            full_point_y,
            groups_base,
            rare_classes=RARE_POINT_CLASSES,
            multiplier=2,
            seed=4620,
        )
        full_reports["point_augmentation_report"] = point_report

    action_model = _fit_model(full_action_x, full_action_y)
    point_model = _fit_model(full_point_x, full_point_y)
    action_test = _predict_aligned_proba(action_model, x_test, action_classes)
    point_test = _predict_aligned_proba(point_model, x_test, point_classes)
    action_pred = _pred_from_prob(action_oof, action_classes)
    point_pred = _pred_from_prob(point_oof, point_classes)

    return {
        "variant": variant,
        "action_classes": action_classes,
        "point_classes": point_classes,
        "action_oof_prob": action_oof,
        "point_oof_prob": point_oof,
        "action_test_prob": action_test,
        "point_test_prob": point_test,
        "action_oof_pred": action_pred,
        "point_oof_pred": point_pred,
        "action_test_pred": _pred_from_prob(action_test, action_classes),
        "point_test_pred": _pred_from_prob(point_test, point_classes),
        "metrics": {
            "action_accuracy": float(accuracy_score(y_action, action_pred)),
            "point_accuracy": float(accuracy_score(y_point, point_pred)),
            "action_macro_f1": float(f1_score(y_action, action_pred, average="macro", zero_division=0)),
            "point_macro_f1": float(f1_score(y_point, point_pred, average="macro", zero_division=0)),
        },
        "fold_reports": fold_reports,
        "full_train_reports": full_reports,
    }


def _prediction_frame(anchor: pd.DataFrame, result: dict[str, Any]) -> pd.DataFrame:
    action_conf, action_margin = _confidence(result["action_test_prob"])
    point_conf, point_margin = _confidence(result["point_test_prob"])
    frame = pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"].to_numpy(),
            "pred_actionId": np.asarray(result["action_test_pred"], dtype=int),
            "pred_pointId": np.asarray(result["point_test_pred"], dtype=int),
            "action_confidence": action_conf,
            "point_confidence": point_conf,
            "action_margin": action_margin,
            "point_margin": point_margin,
            "joint_confidence": (action_margin + point_margin) / 2.0,
        }
    )
    return frame


def _align_predictions(anchor: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in predictions.columns:
        if len(predictions) != len(anchor):
            raise ValueError("predictions without rally_uid must match anchor row count")
        aligned = predictions.copy()
        aligned.insert(0, "rally_uid", anchor["rally_uid"].to_numpy())
        return aligned.reset_index(drop=True)
    pred_uid = predictions["rally_uid"].astype(str).reset_index(drop=True)
    anchor_uid = anchor["rally_uid"].astype(str).reset_index(drop=True)
    if len(predictions) == len(anchor) and pred_uid.equals(anchor_uid):
        return predictions.reset_index(drop=True).copy()
    reduced = predictions.assign(rally_uid=predictions["rally_uid"].astype(str)).drop_duplicates("rally_uid", keep="last")
    missing = sorted(set(anchor_uid) - set(reduced["rally_uid"].astype(str)))
    if missing:
        raise ValueError(f"predictions cannot align to anchor; missing rally_uid values: {missing[:5]}")
    return reduced.set_index("rally_uid").loc[anchor_uid].reset_index()


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def build_ranked_changes(anchor: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    aligned = _align_predictions(anchor, predictions)
    action_col = _first_existing_column(aligned, ("pred_actionId", "action_pred", "candidate_actionId"))
    point_col = _first_existing_column(aligned, ("pred_pointId", "point_pred", "candidate_pointId"))
    if action_col is None or point_col is None:
        raise ValueError("predictions must include predicted action and point columns")

    rows = pd.DataFrame(
        {
            "row_id": np.arange(len(anchor), dtype=int),
            "rally_uid": anchor["rally_uid"].to_numpy(),
            "anchor_action": anchor["actionId"].astype(int).to_numpy(),
            "pred_action": pd.to_numeric(aligned[action_col], errors="coerce").fillna(anchor["actionId"]).astype(int),
            "anchor_point": anchor["pointId"].astype(int).to_numpy(),
            "pred_point": pd.to_numeric(aligned[point_col], errors="coerce").fillna(anchor["pointId"]).astype(int),
        }
    )
    rows["action_changed"] = rows["pred_action"].ne(rows["anchor_action"])
    rows["point_changed"] = rows["pred_point"].ne(rows["anchor_point"])
    rows["action_eligible"] = (
        rows["action_changed"]
        & ~(rows["pred_action"].isin(SERVE_ACTION_CLASSES) & ~rows["anchor_action"].isin(SERVE_ACTION_CLASSES))
    ).astype(object)
    rows["point_eligible"] = (
        rows["point_changed"] & ~(rows["pred_point"].eq(0) & rows["anchor_point"].ne(0))
    ).astype(object)
    rows["action_confidence"] = pd.to_numeric(
        aligned.get("action_margin", aligned.get("action_confidence", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    rows["point_confidence"] = pd.to_numeric(
        aligned.get("point_margin", aligned.get("point_confidence", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    rows["joint_confidence"] = pd.to_numeric(
        aligned.get("joint_confidence", (rows["action_confidence"] + rows["point_confidence"]) / 2.0),
        errors="coerce",
    ).fillna(0.0)
    return rows


def _selected_changes(changes: pd.DataFrame, *, mode: str, max_changes: int) -> pd.DataFrame:
    if mode == "action":
        mask = changes["action_eligible"].astype(bool)
        confidence = "action_confidence"
    elif mode == "point":
        mask = changes["point_eligible"].astype(bool)
        confidence = "point_confidence"
    elif mode == "joint":
        mask = changes["action_eligible"].astype(bool) | changes["point_eligible"].astype(bool)
        confidence = "joint_confidence"
    else:
        raise ValueError(f"unknown submission mode: {mode}")
    return changes.loc[mask].sort_values([confidence, "row_id"], ascending=[False, True]).head(int(max_changes))


def _candidate_stats(
    *,
    candidate: str,
    path: Path | None,
    anchor: pd.DataFrame,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    changes: pd.DataFrame,
) -> dict[str, Any]:
    action_changed = frame["actionId"].astype(int).ne(anchor["actionId"].astype(int))
    point_changed = frame["pointId"].astype(int).ne(anchor["pointId"].astype(int))
    server_changed = frame["serverGetPoint"].ne(anchor["serverGetPoint"])
    serve_additions = (
        frame["actionId"].astype(int).isin(SERVE_ACTION_CLASSES)
        & ~anchor["actionId"].astype(int).isin(SERVE_ACTION_CLASSES)
    )
    point0_additions = frame["pointId"].astype(int).eq(0) & anchor["pointId"].astype(int).ne(0)
    blocked_serve = (
        changes["action_changed"]
        & changes["pred_action"].isin(SERVE_ACTION_CLASSES)
        & ~changes["anchor_action"].isin(SERVE_ACTION_CLASSES)
    )
    blocked_point0 = changes["point_changed"] & changes["pred_point"].eq(0) & changes["anchor_point"].ne(0)
    return {
        "candidate": candidate,
        "path": str(path.resolve()) if path is not None else "",
        "selected_rows": " ".join(str(int(v)) for v in selected["row_id"].tolist()),
        "selected_row_count": int(len(selected)),
        "action_churn": int(action_changed.sum()),
        "point_churn": int(point_changed.sum()),
        "server_changed": int(server_changed.sum()),
        "serve_15_18_additions": int(serve_additions.sum()),
        "point0_additions": int(point0_additions.sum()),
        "blocked_serve_15_18_additions": int(blocked_serve.sum()),
        "blocked_point0_additions": int(blocked_point0.sum()),
    }


def build_submission(
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    mode: str = "joint",
    max_changes: int = 10,
    expected_rows: int | None = 1845,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_submission = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    changes = build_ranked_changes(anchor_submission, predictions)
    selected = _selected_changes(changes, mode=mode, max_changes=max_changes)

    out = anchor_submission.copy()
    for row in selected.itertuples(index=False):
        row_id = int(row.row_id)
        if mode in {"action", "joint"} and bool(row.action_eligible):
            out.at[row_id, "actionId"] = int(row.pred_action)
        if mode in {"point", "joint"} and bool(row.point_eligible):
            out.at[row_id, "pointId"] = int(row.pred_point)
    out = out.loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(out, expected_rows=expected_rows)
    report = _candidate_stats(
        candidate=f"{mode}_top{max_changes}",
        path=None,
        anchor=anchor_submission,
        frame=out,
        selected=selected,
        changes=changes,
    )
    return out, report


def _write_candidate(
    *,
    candidate: str,
    filename: str,
    mode: str,
    max_changes: int,
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
    outdir: Path,
    expected_rows: int | None,
) -> dict[str, Any]:
    frame, stats = build_submission(anchor, predictions, mode=mode, max_changes=max_changes, expected_rows=expected_rows)
    path = safe_output_path(outdir, filename)
    frame.to_csv(path, index=False)
    changes = build_ranked_changes(anchor, predictions)
    selected = _selected_changes(changes, mode=mode, max_changes=max_changes)
    stats = _candidate_stats(candidate=candidate, path=path, anchor=anchor, frame=frame, selected=selected, changes=changes)
    stats["action_distribution"] = action_distribution_report(anchor["actionId"], frame["actionId"])
    stats["point_distribution"] = point_distribution_report(anchor["pointId"], frame["pointId"])
    return stats


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    train_raw = pd.read_csv(train_path, low_memory=False)
    test_raw = pd.read_csv(test_path, low_memory=False)
    anchor = pd.read_csv(anchor_path).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(anchor, expected_rows=expected_rows)

    token_embeddings, token_path, fallback_used = _load_token_embeddings()
    train_rows = build_train_transition_rows(train_raw)
    test_rows = build_test_rows(test_raw, anchor)
    x_train, _train_meta = build_feature_frame(train_rows, token_embeddings)
    x_test, _test_meta = build_feature_frame(test_rows, token_embeddings)
    y_action = train_rows["target_actionId"].to_numpy(dtype=int)
    y_point = train_rows["target_pointId"].to_numpy(dtype=int)
    groups = train_rows["match"] if "match" in train_rows.columns else train_rows["rally_uid"]

    outdir.mkdir(parents=True, exist_ok=True)
    variant_results: dict[str, dict[str, Any]] = {}
    augmentation_report: dict[str, Any] = {
        "version": "V420",
        "rare_action_classes": sorted(RARE_ACTION_CLASSES),
        "rare_point_classes": sorted(RARE_POINT_CLASSES),
        "train_transition_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "fallback_to_v415_embeddings": bool(fallback_used),
        "token_embeddings_path": str(token_path.resolve()),
        "variants": {},
    }
    for variant in VARIANTS:
        result = _train_variant(variant, x_train, y_action, y_point, x_test, groups)
        variant_results[variant] = result
        augmentation_report["variants"][variant] = {
            "metrics": result["metrics"],
            "fold_reports": result["fold_reports"],
            "full_train_reports": result["full_train_reports"],
        }
        _prediction_frame(anchor, result).to_csv(safe_output_path(outdir, f"test_predictions_{variant}.csv"), index=False)

    def _best_variant(metric: str) -> str:
        return max(VARIANTS, key=lambda name: float(variant_results[name]["metrics"][metric]))

    best_action = _best_variant("action_macro_f1")
    best_point = _best_variant("point_macro_f1")
    best_joint = max(
        VARIANTS,
        key=lambda name: (
            float(variant_results[name]["metrics"]["action_macro_f1"])
            + float(variant_results[name]["metrics"]["point_macro_f1"])
        )
        / 2.0,
    )

    generated = [
        _write_candidate(
            candidate=f"action_rare_top10__{best_action}",
            filename="submission_v420_action_rare_top10__v362anchor.csv",
            mode="action",
            max_changes=10,
            anchor=anchor,
            predictions=_prediction_frame(anchor, variant_results[best_action]),
            outdir=outdir,
            expected_rows=expected_rows,
        ),
        _write_candidate(
            candidate=f"point_rare_top10__{best_point}",
            filename="submission_v420_point_rare_top10__v362anchor.csv",
            mode="point",
            max_changes=10,
            anchor=anchor,
            predictions=_prediction_frame(anchor, variant_results[best_point]),
            outdir=outdir,
            expected_rows=expected_rows,
        ),
        _write_candidate(
            candidate=f"joint_rare_top15__{best_joint}",
            filename="submission_v420_joint_rare_top15__v362anchor.csv",
            mode="joint",
            max_changes=15,
            anchor=anchor,
            predictions=_prediction_frame(anchor, variant_results[best_joint]),
            outdir=outdir,
            expected_rows=expected_rows,
        ),
    ]

    summary = pd.DataFrame(generated)
    summary.to_csv(safe_output_path(outdir, "candidate_summary.csv"), index=False)
    augmentation_report["selected_variants"] = {
        "action_rare_top10": best_action,
        "point_rare_top10": best_point,
        "joint_rare_top15": best_joint,
    }
    augmentation_report["generated_submissions"] = generated
    write_json(safe_output_path(outdir, "augmentation_report.json"), augmentation_report)
    return augmentation_report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
