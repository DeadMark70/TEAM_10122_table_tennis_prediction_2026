"""V467 clean server-only exhaustive sweep.

The pipeline preserves V362 actionId/pointId and changes only serverGetPoint.
It trains bounded clean server models from train.csv labels, scores test_new
prefix rows, aggregates each target rally_uid with several modes, adds masked
specialist blends, optionally trains lightweight server-only sequence models,
and exports MAD-capped server-only submissions.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from analysis_v465_clean_server_line import (
    ANCHOR_RELATIVE,
    EXPECTED_ROWS,
    PROB_MAX,
    PROB_MIN,
    ROOT,
    SUBMISSION_COLUMNS,
    TEST_NEW_RELATIVE,
    TRAIN_RELATIVE,
    ServerSource,
    blend_to_target_mad,
    build_feature_matrices,
    clip_prob,
    load_existing_clean_sources,
    load_submission,
    no_banned_input_guard,
    package_server_only,
    rank_normalize_to_anchor,
)

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:  # pragma: no cover - optional dependency
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - depends on sklearn version
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


OUT_DIR = ROOT / "v467_server_exhaustive_clean_sweep"
TARGET_MADS = (0.0025, 0.0050, 0.0075, 0.0100, 0.0150)
RECOMMENDED_MAD_MAX = 0.0100001
AGGREGATION_MODES = ("mean", "last", "max", "late_weighted")


@dataclass(frozen=True)
class TabularModelConfig:
    name: str
    family: str
    size: str
    seed: int
    factory: Callable[[], Any]
    optional: bool = False


@dataclass(frozen=True)
class ModelSignal:
    name: str
    family: str
    size: str
    seed: int
    oof: np.ndarray
    test: np.ndarray
    auc: float


@dataclass(frozen=True)
class ServerCandidate:
    candidate: str
    server: np.ndarray
    source_names: tuple[str, ...]
    families: tuple[str, ...]
    train_oof_auc: float
    actual_mad: float
    risk: str
    decision: str
    diversity: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _seed_bag(seed: int, count: int) -> list[int]:
    return [seed + 101 * idx for idx in range(count)]


def _add_config(
    configs: list[TabularModelConfig],
    name: str,
    family: str,
    size: str,
    seeds: Iterable[int],
    factory: Callable[[int], Any],
    *,
    optional: bool = False,
) -> None:
    for bag_idx, bag_seed in enumerate(seeds):
        suffix = "" if bag_idx == 0 else f"_s{bag_idx}"
        configs.append(
            TabularModelConfig(
                name=f"{name}{suffix}",
                family=family,
                size=size,
                seed=bag_seed,
                factory=lambda bag_seed=bag_seed, factory=factory: factory(bag_seed),
                optional=optional,
            )
        )


def build_tabular_model_configs(*, seed: int = 467, runtime: str = "fast") -> list[TabularModelConfig]:
    """Build the bounded V467 tabular zoo."""
    if runtime not in {"test", "fast", "full"}:
        raise ValueError(f"unknown runtime: {runtime}")
    cheap_bags = 1 if runtime == "test" else (1 if runtime == "fast" else 3)
    expensive_bags = 1 if runtime in {"test", "fast"} else 2
    configs: list[TabularModelConfig] = []

    _add_config(
        configs,
        "logistic_balanced",
        "linear",
        "small",
        _seed_bag(seed, cheap_bags),
        lambda s: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=s),
        ),
    )
    _add_config(
        configs,
        "sgd_logloss_balanced",
        "linear",
        "small",
        _seed_bag(seed + 7, cheap_bags),
        lambda s: make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=0.0005, class_weight="balanced", max_iter=1000, tol=1e-3, random_state=s),
        ),
    )

    tree_estimators = 24 if runtime == "test" else 140
    _add_config(
        configs,
        "random_forest_medium",
        "tree",
        "medium",
        _seed_bag(seed + 17, expensive_bags),
        lambda s: RandomForestClassifier(
            n_estimators=tree_estimators,
            min_samples_leaf=1 if runtime == "test" else 5,
            class_weight="balanced",
            random_state=s,
            n_jobs=1,
        ),
    )
    _add_config(
        configs,
        "extra_trees_medium",
        "tree",
        "medium",
        _seed_bag(seed + 23, expensive_bags),
        lambda s: ExtraTreesClassifier(
            n_estimators=tree_estimators,
            min_samples_leaf=1 if runtime == "test" else 4,
            class_weight="balanced",
            random_state=s,
            n_jobs=1,
        ),
    )

    _add_config(
        configs,
        "hist_gradient_medium",
        "boosting",
        "medium",
        _seed_bag(seed + 31, expensive_bags),
        lambda s: HistGradientBoostingClassifier(
            max_iter=25 if runtime == "test" else 110,
            learning_rate=0.05,
            max_leaf_nodes=15 if runtime == "test" else 31,
            min_samples_leaf=2 if runtime == "test" else 20,
            random_state=s,
        ),
    )
    if LGBMClassifier is not None:
        _add_config(
            configs,
            "lightgbm_medium",
            "boosting",
            "medium",
            _seed_bag(seed + 37, expensive_bags),
            lambda s: LGBMClassifier(
                n_estimators=35 if runtime == "test" else 160,
                learning_rate=0.04,
                num_leaves=15 if runtime == "test" else 31,
                min_child_samples=2 if runtime == "test" else 20,
                class_weight="balanced",
                random_state=s,
                verbose=-1,
                n_jobs=1,
            ),
            optional=True,
        )
    if XGBClassifier is not None and runtime != "test":
        _add_config(
            configs,
            "xgboost_medium",
            "boosting",
            "medium",
            _seed_bag(seed + 41, expensive_bags),
            lambda s: XGBClassifier(
                n_estimators=120 if runtime == "fast" else 220,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=s,
                n_jobs=1,
                verbosity=0,
            ),
            optional=True,
        )
    if CatBoostClassifier is not None and runtime != "test":
        _add_config(
            configs,
            "catboost_medium",
            "boosting",
            "medium",
            _seed_bag(seed + 43, expensive_bags),
            lambda s: CatBoostClassifier(
                iterations=120 if runtime == "fast" else 220,
                depth=4,
                learning_rate=0.04,
                loss_function="Logloss",
                random_seed=s,
                verbose=False,
                thread_count=1,
                allow_writing_files=False,
            ),
            optional=True,
        )

    mlp_iters = 80 if runtime == "test" else 180
    _add_config(
        configs,
        "mlp_small",
        "mlp",
        "small",
        _seed_bag(seed + 53, expensive_bags),
        lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=0.003, max_iter=mlp_iters, random_state=s)),
    )
    if runtime != "test":
        _add_config(
            configs,
            "mlp_medium",
            "mlp",
            "medium",
            _seed_bag(seed + 59, expensive_bags),
            lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.002, max_iter=200, random_state=s)),
        )
        _add_config(
            configs,
            "mlp_large",
            "mlp",
            "large",
            _seed_bag(seed + 61, expensive_bags),
            lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(48, 24), alpha=0.002, max_iter=220, random_state=s)),
        )
    return configs


def _support_safe_folds(y: np.ndarray) -> int:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    return min(5, int(counts.min()))


def _support_safe_group_folds(y: np.ndarray, groups: np.ndarray) -> int:
    group_frame = pd.DataFrame({"group": groups, "y": y})
    group_labels = group_frame.groupby("group", sort=False)["y"].max()
    counts = np.bincount(group_labels.to_numpy(dtype=int), minlength=2)
    return min(5, int(counts.min()), int(len(group_labels)))


def _split_iter(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, folds: int, seed: int):
    if groups is not None and StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y, groups)
    if groups is not None:
        return GroupKFold(n_splits=folds).split(x, y, groups)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return 1.0 / (1.0 + np.exp(-np.clip(values / scale, -40.0, 40.0)))


def predict_positive(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return clip_prob(proba[:, 1])
        if proba.ndim == 1:
            return clip_prob(proba)
    if hasattr(model, "decision_function"):
        return clip_prob(_sigmoid(model.decision_function(x)))
    return clip_prob(model.predict(x))


def _safe_auc(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        auc = float(roc_auc_score(y, clip_prob(pred)))
    except Exception:
        return float("nan")
    return auc if math.isfinite(auc) else float("nan")


def fit_tabular_zoo(
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    x_test: pd.DataFrame,
    *,
    groups: np.ndarray | pd.Series | None,
    configs: list[TabularModelConfig] | None = None,
    skip_report: dict[str, str] | None = None,
) -> list[ModelSignal]:
    y_arr = np.asarray(y, dtype=int)
    if len(y_arr) != len(x):
        raise ValueError("x and y length mismatch")
    if len(np.unique(y_arr)) != 2:
        raise ValueError("server target is not binary")
    group_arr = None if groups is None else np.asarray(groups)
    if group_arr is not None and len(group_arr) != len(y_arr):
        raise ValueError("groups and y length mismatch")
    folds = _support_safe_group_folds(y_arr, group_arr) if group_arr is not None else _support_safe_folds(y_arr)
    if folds < 2:
        raise ValueError("not enough class support for OOF folds")

    signals: list[ModelSignal] = []
    for config in configs or build_tabular_model_configs(seed=467):
        oof = np.zeros(len(y_arr), dtype=float)
        test_fold_predictions: list[np.ndarray] = []
        try:
            for train_idx, valid_idx in _split_iter(x, y_arr, group_arr, folds, config.seed):
                model = clone(config.factory())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x.iloc[train_idx], y_arr[train_idx])
                oof[valid_idx] = predict_positive(model, x.iloc[valid_idx])
                test_fold_predictions.append(predict_positive(model, x_test))
        except Exception as exc:
            if skip_report is not None:
                skip_report[config.name] = repr(exc)
            if config.optional or config.family == "mlp":
                continue
            raise RuntimeError(f"required model failed: {config.name}") from exc
        if not test_fold_predictions:
            continue
        test_pred = np.mean(np.vstack(test_fold_predictions), axis=0)
        signals.append(
            ModelSignal(
                name=config.name,
                family=config.family,
                size=config.size,
                seed=config.seed,
                oof=clip_prob(oof),
                test=clip_prob(test_pred),
                auc=_safe_auc(y_arr, oof),
            )
        )
    return signals


def aggregate_prefix_predictions(test_new: pd.DataFrame, anchor: pd.DataFrame, pred: np.ndarray | pd.Series, *, mode: str) -> np.ndarray:
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation mode: {mode}")
    if "rally_uid" not in test_new.columns or "rally_uid" not in anchor.columns:
        raise ValueError("test_new and anchor must contain rally_uid")
    pred_arr = clip_prob(pred)
    if len(pred_arr) != len(test_new):
        raise ValueError("prediction length does not match test_new rows")
    work = pd.DataFrame(
        {
            "rally_uid": test_new["rally_uid"].to_numpy(),
            "strikeNumber": pd.to_numeric(test_new.get("strikeNumber", pd.Series(np.arange(len(test_new)))), errors="coerce").fillna(-1).to_numpy(),
            "server": pred_arr,
        }
    )
    if mode == "mean":
        by_uid = work.groupby("rally_uid", sort=False)["server"].mean()
    elif mode == "max":
        by_uid = work.groupby("rally_uid", sort=False)["server"].max()
    elif mode == "last":
        by_uid = work.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").groupby("rally_uid", sort=False)["server"].last()
    else:
        ordered = work.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").copy()
        ordered["_rank"] = ordered.groupby("rally_uid", sort=False).cumcount() + 1
        ordered["_weighted"] = ordered["server"] * ordered["_rank"]
        by_uid = ordered.groupby("rally_uid", sort=False).apply(
            lambda g: float(g["_weighted"].sum() / max(g["_rank"].sum(), 1)),
            include_groups=False,
        )
    anchor_uids = anchor["rally_uid"]
    if not anchor_uids.isin(by_uid.index).all():
        missing = anchor_uids[~anchor_uids.isin(by_uid.index)].head(5).tolist()
        raise ValueError(f"test predictions missing anchor rally_uid values: {missing}")
    return clip_prob(by_uid.reindex(anchor_uids).to_numpy(dtype=float))


def aggregate_row_signals(signals: list[ModelSignal], test_new: pd.DataFrame, anchor: pd.DataFrame) -> list[ModelSignal]:
    out: list[ModelSignal] = []
    for signal in signals:
        for mode in AGGREGATION_MODES:
            out.append(
                ModelSignal(
                    name=f"{signal.name}_uid_{mode}",
                    family=signal.family,
                    size=signal.size,
                    seed=signal.seed,
                    oof=signal.oof,
                    test=aggregate_prefix_predictions(test_new, anchor, signal.test, mode=mode),
                    auc=signal.auc,
                )
            )
    return out


def build_anchor_context(anchor: pd.DataFrame, test_new: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in anchor.columns or "rally_uid" not in test_new.columns:
        raise ValueError("anchor and test_new must contain rally_uid")
    grouped = test_new.groupby("rally_uid", sort=False)
    context = anchor[["rally_uid", "actionId", "pointId", "serverGetPoint"]].copy()
    context["prefix_rows"] = context["rally_uid"].map(grouped.size()).fillna(0).astype(int)
    ordered = test_new.sort_values(["rally_uid", "strikeNumber"], kind="mergesort") if "strikeNumber" in test_new.columns else test_new
    ordered_grouped = ordered.groupby("rally_uid", sort=False)
    for col in ["scoreSelf", "scoreOther", "strikeNumber", "numberGame", "rally_id"]:
        if col not in test_new.columns:
            context[f"{col}_mean"] = -1.0
            context[f"{col}_max"] = -1.0
            context[f"{col}_last"] = -1.0
            continue
        numeric = pd.to_numeric(test_new[col], errors="coerce")
        by_uid = numeric.groupby(test_new["rally_uid"], sort=False).agg(["mean", "max"])
        last_by_uid = pd.to_numeric(ordered[col], errors="coerce").groupby(ordered["rally_uid"], sort=False).last()
        context[f"{col}_mean"] = context["rally_uid"].map(by_uid["mean"]).fillna(-1.0)
        context[f"{col}_max"] = context["rally_uid"].map(by_uid["max"]).fillna(-1.0)
        context[f"{col}_last"] = context["rally_uid"].map(last_by_uid).fillna(-1.0)
    context["score_margin_last"] = context["scoreSelf_last"] - context["scoreOther_last"]
    context["abs_score_margin_last"] = context["score_margin_last"].abs()
    context["point0"] = (context["pointId"].astype(int) == 0).astype(int)
    context["terminal_action"] = (context["actionId"].astype(int) == 0).astype(int)
    return context.replace([np.inf, -np.inf], np.nan).fillna(-1.0)


def build_specialist_masks(context: pd.DataFrame) -> dict[str, np.ndarray]:
    score_self = pd.to_numeric(context.get("scoreSelf_max", -1), errors="coerce").fillna(-1)
    score_other = pd.to_numeric(context.get("scoreOther_max", -1), errors="coerce").fillna(-1)
    strike_max = pd.to_numeric(context.get("strikeNumber_max", -1), errors="coerce").fillna(-1)
    prefix_rows = pd.to_numeric(context.get("prefix_rows", 0), errors="coerce").fillna(0)
    action = pd.to_numeric(context.get("actionId", -1), errors="coerce").fillna(-1).astype(int)
    point = pd.to_numeric(context.get("pointId", -1), errors="coerce").fillna(-1).astype(int)
    close = (score_self - score_other).abs() <= 1
    return {
        "score_pressure": (close | ((score_self >= 10) & (score_other >= 9)) | ((score_other >= 10) & (score_self >= 9))).to_numpy(bool),
        "phase_specialist": ((prefix_rows <= 2) | (strike_max <= 2)).to_numpy(bool),
        "terminal_like": ((point == 0) | (action == 0) | (strike_max >= 7)).to_numpy(bool),
        "action_point_conditioned": ((point.isin([0, 1, 4, 7, 8, 9])) | (action.isin([0, 8, 12, 14, 15]))).to_numpy(bool),
    }


def _mean_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("cannot average an empty target list")
    return clip_prob(np.mean(np.vstack([clip_prob(arr) for arr in arrays]), axis=0))


def _family_target(signals: list[ModelSignal], family: str, anchor_server: np.ndarray) -> np.ndarray | None:
    arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals if signal.family == family]
    return _mean_arrays(arrays) if arrays else None


def _name_target(signals: list[ModelSignal], needle: str, anchor_server: np.ndarray) -> np.ndarray | None:
    arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals if needle in signal.name]
    return _mean_arrays(arrays) if arrays else None


def _aggregation_target(signals: list[ModelSignal], mode: str, anchor_server: np.ndarray) -> np.ndarray | None:
    arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals if signal.name.endswith(f"_uid_{mode}")]
    return _mean_arrays(arrays) if arrays else None


def _blend_under_mask(base: np.ndarray, target: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    out = clip_prob(base).copy()
    mask_arr = np.asarray(mask, dtype=bool)
    if len(mask_arr) != len(out):
        raise ValueError("specialist mask length mismatch")
    out[mask_arr] = clip_prob((1.0 - weight) * out[mask_arr] + weight * clip_prob(target)[mask_arr])
    return clip_prob(out)


def build_model_targets(
    anchor: pd.DataFrame,
    signals: list[ModelSignal],
    *,
    sequence_targets: dict[str, np.ndarray] | None = None,
    clean_sources: list[ServerSource] | None = None,
) -> dict[str, np.ndarray]:
    if not signals:
        raise ValueError("at least one model signal is required")
    anchor_server = clip_prob(anchor["serverGetPoint"])
    targets: dict[str, np.ndarray] = {}
    family_targets: dict[str, np.ndarray] = {}
    for family, target_name in [
        ("tree", "tree_rankmean"),
        ("boosting", "tabular_boosting_rankmean"),
        ("mlp", "mlp_rankmean"),
        ("linear", "linear_rankmean"),
    ]:
        value = _family_target(signals, family, anchor_server)
        if value is not None:
            family_targets[family] = value
            targets[target_name] = value
    targets["tabular_global_rankmean"] = _mean_arrays([rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals])
    cat_xgb = [
        value
        for value in [
            _name_target(signals, "catboost", anchor_server),
            _name_target(signals, "xgboost", anchor_server),
        ]
        if value is not None
    ]
    if cat_xgb:
        targets["catboost_xgboost_rankmean"] = _mean_arrays(cat_xgb)
    for mode in AGGREGATION_MODES:
        value = _aggregation_target(signals, mode, anchor_server)
        if value is not None:
            targets[f"{mode}_aggregation_rankmean"] = value
    if "mean_aggregation_rankmean" in targets:
        targets["mean_aggregation_rankmean"] = targets.pop("mean_aggregation_rankmean")
    if sequence_targets:
        targets.update(sequence_targets)
    if clean_sources:
        targets["clean_source_rankmean"] = _mean_arrays([rank_normalize_to_anchor(source.server, anchor_server) for source in clean_sources])
    else:
        targets["clean_source_rankmean"] = targets["tabular_global_rankmean"]
    ensemble = [targets["tabular_global_rankmean"], targets["clean_source_rankmean"]]
    for name in ["tabular_boosting_rankmean", "tree_rankmean", "mlp_rankmean", "sequence_rankmean"]:
        if name in targets:
            ensemble.append(targets[name])
    targets["full_exhaustive_ensemble"] = _mean_arrays(ensemble)
    return targets


def add_specialist_targets(model_targets: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    targets = dict(model_targets)
    global_target = targets["tabular_global_rankmean"]
    tree = targets.get("tree_rankmean", global_target)
    boosting = targets.get("tabular_boosting_rankmean", tree)
    linear = targets.get("linear_rankmean", global_target)
    mlp = targets.get("mlp_rankmean", global_target)
    full = targets.get("full_exhaustive_ensemble", global_target)
    targets["score_pressure_specialist"] = _blend_under_mask(global_target, _mean_arrays([tree, boosting]), masks["score_pressure"], 0.65)
    targets["phase_specialist"] = _blend_under_mask(global_target, _mean_arrays([linear, tree]), masks["phase_specialist"], 0.55)
    targets["terminal_like_specialist"] = _blend_under_mask(global_target, boosting, masks["terminal_like"], 0.65)
    targets["action_point_specialist"] = _blend_under_mask(global_target, _mean_arrays([full, mlp]), masks["action_point_conditioned"], 0.60)
    targets["full_exhaustive_ensemble"] = _mean_arrays(
        [
            targets["tabular_global_rankmean"],
            targets["score_pressure_specialist"],
            targets["phase_specialist"],
            targets["terminal_like_specialist"],
            targets["action_point_specialist"],
            targets["clean_source_rankmean"],
        ]
    )
    return targets


def train_sequence_models(
    train: pd.DataFrame,
    test_new: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    runtime: str = "fast",
    enabled: bool = True,
    report: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Train tiny sequence models or record a bounded skip reason."""
    info = report if report is not None else {}
    if not enabled:
        info["sequence_status"] = "skipped"
        info["sequence_skip_reason"] = "disabled by caller"
        return {}
    if runtime == "test":
        info["sequence_status"] = "skipped"
        info["sequence_skip_reason"] = "runtime=test keeps sequence models disabled"
        return {}
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - environment dependent
        info["sequence_status"] = "skipped"
        info["sequence_skip_reason"] = f"PyTorch unavailable: {exc!r}"
        return {}

    start = time.time()
    try:  # pragma: no cover - slow optional path
        train_x, test_x = build_feature_matrices(train, test_new, anchor)
        numeric_cols = list(train_x.columns[: min(32, len(train_x.columns))])
        train_frame = train.loc[:, ["rally_uid", "serverGetPoint"]].join(train_x[numeric_cols])
        test_frame = test_new.loc[:, ["rally_uid"]].join(test_x[numeric_cols])

        def build_sequences(frame: pd.DataFrame, labels: pd.Series | None) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
            seqs = []
            ys = []
            uids = []
            for uid, group in frame.groupby("rally_uid", sort=False):
                arr = group[numeric_cols].to_numpy(dtype="float32")
                arr = arr[-12:]
                if len(arr) < 12:
                    pad = np.zeros((12 - len(arr), arr.shape[1]), dtype="float32")
                    arr = np.vstack([pad, arr])
                seqs.append(arr)
                uids.append(uid)
                if labels is not None:
                    ys.append(float(group["serverGetPoint"].max() >= 0.5))
            return np.stack(seqs), (np.asarray(ys, dtype="float32") if labels is not None else None), uids

        x_seq, y_seq, _ = build_sequences(train_frame, train_frame["serverGetPoint"])
        test_seq, _, test_uids = build_sequences(test_frame, None)
        if y_seq is None or len(np.unique(y_seq)) < 2:
            raise ValueError("sequence target lacks both classes")
        x_tensor = torch.tensor(x_seq)
        y_tensor = torch.tensor(y_seq).view(-1, 1)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=128, shuffle=True)
        epochs = 1 if runtime == "fast" else 3
        input_size = x_seq.shape[2]

        class RecurrentHead(nn.Module):
            def __init__(self, kind: str) -> None:
                super().__init__()
                rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
                self.rnn = rnn_cls(input_size, 24, batch_first=True)
                self.out = nn.Linear(24, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.out(out[:, -1, :])

        class TransformerHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = 32
                self.inp = nn.Linear(input_size, width)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=4,
                    dim_feedforward=64,
                    dropout=0.10,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.out = nn.Linear(width, 1)

            def forward(self, x):
                hidden = self.encoder(self.inp(x))
                return self.out(hidden[:, -1, :])

        outputs = []
        model_factories = [
            ("gru", lambda: RecurrentHead("gru")),
            ("lstm", lambda: RecurrentHead("lstm")),
            ("transformer", TransformerHead),
        ]
        for kind, model_factory in model_factories:
            model = model_factory()
            opt = torch.optim.Adam(model.parameters(), lr=0.003)
            loss_fn = nn.BCEWithLogitsLoss()
            for _ in range(epochs):
                for bx, by in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(bx), by)
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                pred = torch.sigmoid(model(torch.tensor(test_seq))).numpy().reshape(-1)
            by_uid = pd.Series(pred, index=test_uids)
            outputs.append(clip_prob(by_uid.reindex(anchor["rally_uid"]).to_numpy(dtype=float)))
            if runtime == "fast" and time.time() - start > 45.0 and kind != "gru":
                break
        if not outputs:
            raise ValueError("no sequence outputs produced")
        info["sequence_status"] = "trained"
        info["sequence_models"] = len(outputs)
        return {"sequence_rankmean": rank_normalize_to_anchor(_mean_arrays(outputs), anchor["serverGetPoint"])}
    except Exception as exc:  # pragma: no cover - environment dependent
        info["sequence_status"] = "skipped"
        info["sequence_skip_reason"] = f"sequence training failed or exceeded bounds: {exc!r}"
        return {}


def _target_families(target_name: str) -> tuple[str, ...]:
    families = []
    for family in ["linear", "tree", "boosting", "mlp", "sequence", "clean"]:
        if family in target_name:
            families.append(family)
    if any(token in target_name for token in ["tabular_global", "full_exhaustive", "specialist", "aggregation"]):
        families.extend(["linear", "tree", "boosting", "mlp"])
    return tuple(sorted(set(families)))


def build_candidate_servers(
    anchor: pd.DataFrame,
    model_targets: dict[str, np.ndarray],
    *,
    masks: dict[str, np.ndarray],
    target_mads: tuple[float, ...] = TARGET_MADS,
) -> list[ServerCandidate]:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    targets = add_specialist_targets(model_targets, masks)
    target_order = [
        "tabular_global_rankmean",
        "tabular_boosting_rankmean",
        "catboost_xgboost_rankmean",
        "tree_rankmean",
        "mlp_rankmean",
        "sequence_rankmean",
        "mean_aggregation_rankmean",
        "last_aggregation_rankmean",
        "max_aggregation_rankmean",
        "late_weighted_aggregation_rankmean",
        "score_pressure_specialist",
        "phase_specialist",
        "terminal_like_specialist",
        "action_point_specialist",
        "clean_source_rankmean",
        "full_exhaustive_ensemble",
    ]
    candidates: list[ServerCandidate] = []
    seen: set[bytes] = set()
    for target_name in target_order:
        if target_name not in targets:
            continue
        normalized = rank_normalize_to_anchor(targets[target_name], anchor_server)
        for target_mad in target_mads:
            server = blend_to_target_mad(anchor_server, normalized, target_mad=target_mad)
            fingerprint = np.round(server, 10).tobytes()
            if fingerprint in seen:
                if target_name != "full_exhaustive_ensemble":
                    continue
                jitter = np.linspace(-5e-9, 5e-9, len(server), dtype=float)
                server = clip_prob(server + jitter)
                fingerprint = np.round(server, 10).tobytes()
                if fingerprint in seen:
                    continue
            seen.add(fingerprint)
            actual_mad = float(np.mean(np.abs(server - anchor_server)))
            mad_key = f"{target_mad:.4f}".replace(".", "p")
            diagnostic = target_mad > RECOMMENDED_MAD_MAX
            risk = "diagnostic" if diagnostic else ("safe" if actual_mad <= 0.0050001 else "exploratory")
            decision = "diagnostic_hold" if diagnostic else "review"
            families = _target_families(target_name)
            candidates.append(
                ServerCandidate(
                    candidate=f"{target_name}_mad{mad_key}",
                    server=server,
                    source_names=(target_name,),
                    families=families,
                    train_oof_auc=float("nan"),
                    actual_mad=actual_mad,
                    risk=risk,
                    decision=decision,
                    diversity=len(families),
                )
            )
    return candidates


def _validate_output_path(path: Path, outdir: Path) -> Path:
    outdir_resolved = Path(outdir).resolve()
    path_resolved = Path(path).resolve()
    if outdir_resolved not in path_resolved.parents and path_resolved != outdir_resolved:
        raise ValueError(f"output path escapes V467 outdir: {path_resolved}")
    no_banned_input_guard([path_resolved])
    return path_resolved


def _candidate_filename(candidate: str) -> str:
    safe = candidate.replace(".", "p").replace("/", "_").replace("\\", "_")
    return f"submission_v467_{safe}__v362action_v362point.csv"


def _selected_rows(anchor: pd.DataFrame, server: np.ndarray, *, candidate: str) -> pd.DataFrame:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["candidate"] = candidate
    out["server_anchor"] = anchor_server
    out["server_candidate"] = clip_prob(server)
    out["server_abs_delta"] = np.abs(out["server_candidate"] - out["server_anchor"])
    return out.loc[out["server_abs_delta"] > 1e-12].sort_values("server_abs_delta", ascending=False)


def _corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _candidate_row(candidate: ServerCandidate, anchor: pd.DataFrame, submission_path: Path, selected_path: Path) -> dict[str, Any]:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    server = clip_prob(candidate.server)
    return {
        "candidate": candidate.candidate,
        "path": str(submission_path.resolve()),
        "selected_path": str(selected_path.resolve()),
        "selected_rows": int(np.sum(np.abs(server - anchor_server) > 1e-12)),
        "action_churn": 0,
        "point_churn": 0,
        "server_changed": int(np.sum(np.abs(server - anchor_server) > 1e-12)),
        "server_mad": float(np.mean(np.abs(server - anchor_server))),
        "server_corr": _corr(server, anchor_server),
        "server_min": float(np.min(server)),
        "server_max": float(np.max(server)),
        "source_names": "|".join(candidate.source_names),
        "families": "|".join(candidate.families),
        "family_diversity": candidate.diversity,
        "risk": candidate.risk,
        "decision": candidate.decision,
    }


def _simple_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    columns = list(frame.columns)
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.to_numpy()]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _write_report_md(path: Path, report: dict[str, Any], board: pd.DataFrame) -> None:
    lines = [
        "# V467 server exhaustive clean sweep",
        "",
        f"Generated candidates: {report['candidate_count']}",
        f"Recommended: {report['recommended_candidate']}",
        f"Runtime mode: {report['runtime']}",
        f"Sequence status: {report.get('sequence_status', 'unknown')}",
        "",
        "Policy: train labels only for supervised server learning; test_new observed fields only; no TTMATCH; no upload_candidates_20260519; no old-server direct labels.",
        "",
        "## Top candidates",
        "",
        _simple_markdown_table(board.head(12)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _board_sort(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    out = board.copy()
    out["_decision_order"] = out["decision"].map({"review": 0, "diagnostic_hold": 1, "hold": 2}).fillna(9)
    out["_risk_order"] = out["risk"].map({"safe": 0, "exploratory": 1, "diagnostic": 2}).fillna(9)
    return out.sort_values(
        ["_decision_order", "_risk_order", "family_diversity", "server_mad", "server_corr"],
        ascending=[True, True, False, True, False],
    ).drop(columns=["_decision_order", "_risk_order"])


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path | None = None,
    expected_rows: int = EXPECTED_ROWS,
    runtime: str = "fast",
    sequence_enabled: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / OUT_DIR.name
    no_banned_input_guard([root / ANCHOR_RELATIVE, root / TRAIN_RELATIVE, root / TEST_NEW_RELATIVE, outdir])
    outdir.mkdir(parents=True, exist_ok=True)

    anchor = load_submission(root / ANCHOR_RELATIVE, expected_rows=expected_rows)
    train = pd.read_csv(root / TRAIN_RELATIVE)
    test_new = pd.read_csv(root / TEST_NEW_RELATIVE)
    if "serverGetPoint" not in train.columns:
        raise ValueError("train.csv must contain serverGetPoint")
    y = pd.to_numeric(train["serverGetPoint"], errors="coerce")
    if y.isna().any():
        raise ValueError("train serverGetPoint contains non-numeric values")
    train_x, test_x = build_feature_matrices(train, test_new, anchor)
    optional_skips: dict[str, str] = {}
    configs = build_tabular_model_configs(seed=467, runtime=runtime)
    row_signals = fit_tabular_zoo(
        train_x,
        (y.to_numpy(dtype=float) >= 0.5).astype(int),
        test_x,
        groups=train["rally_uid"] if "rally_uid" in train.columns else None,
        configs=configs,
        skip_report=optional_skips,
    )
    signals = aggregate_row_signals(row_signals, test_new, anchor)
    context = build_anchor_context(anchor, test_new)
    masks = build_specialist_masks(context)
    sequence_report: dict[str, object] = {}
    sequence_targets = train_sequence_models(train, test_new, anchor, runtime=runtime, enabled=sequence_enabled, report=sequence_report)
    clean_sources = load_existing_clean_sources(root, anchor, expected_rows=expected_rows)
    model_targets = build_model_targets(anchor, signals, sequence_targets=sequence_targets, clean_sources=clean_sources)
    candidates = build_candidate_servers(anchor, model_targets, masks=masks)

    mean_auc = float(np.nanmean([signal.auc for signal in signals])) if signals else float("nan")
    rows = []
    for candidate in candidates:
        candidate = ServerCandidate(
            candidate=candidate.candidate,
            server=candidate.server,
            source_names=candidate.source_names,
            families=candidate.families,
            train_oof_auc=mean_auc,
            actual_mad=candidate.actual_mad,
            risk=candidate.risk,
            decision=candidate.decision,
            diversity=candidate.diversity,
        )
        submission_path = _validate_output_path(outdir / _candidate_filename(candidate.candidate), outdir)
        selected_path = _validate_output_path(outdir / f"selected_rows_{candidate.candidate}.csv", outdir)
        submission = package_server_only(anchor, candidate.server, expected_rows=expected_rows)
        selected = _selected_rows(anchor, candidate.server, candidate=candidate.candidate)
        submission.to_csv(submission_path, index=False)
        selected.to_csv(selected_path, index=False)
        row = _candidate_row(candidate, anchor, submission_path, selected_path)
        row["train_oof_auc"] = candidate.train_oof_auc
        rows.append(row)

    board = _board_sort(pd.DataFrame(rows))
    search_path = _validate_output_path(outdir / "v467_server_search.csv", outdir)
    board.to_csv(search_path, index=False)
    review = board.loc[board["decision"].eq("review")] if not board.empty else board
    recommended = str(review.iloc[0]["candidate"]) if not review.empty else (str(board.iloc[0]["candidate"]) if not board.empty else None)
    report = {
        "pipeline": "v467_server_exhaustive_clean_sweep",
        "runtime": runtime,
        "anchor": str((root / ANCHOR_RELATIVE).resolve()),
        "candidate_count": int(len(board)),
        "recommended_candidate": recommended,
        "trained_row_model_count": int(len(row_signals)),
        "trained_anchor_signal_count": int(len(signals)),
        "clean_source_count": int(len(clean_sources)),
        "trained_models": [
            {"name": signal.name, "family": signal.family, "size": signal.size, "seed": signal.seed, "auc": signal.auc}
            for signal in row_signals
        ],
        "optional_model_skips": optional_skips,
        "aggregation_modes": list(AGGREGATION_MODES),
        "row_test_prediction": "test_new_row_level_then_anchor_rally_uid_aggregation",
        "grouped_oof": "rally_uid",
        "specialist_strategy": "masked blend fallback; tabular grouped OOF base signals",
        **sequence_report,
        "policy": {
            "no_old_server_direct_labels": True,
            "no_ttmatch": True,
            "no_upload_candidates_20260519": True,
            "train_labels_only_for_supervised_server": True,
            "preserve_action_point_from_v362": True,
        },
        "search_path": str(search_path.resolve()),
    }
    report_json_path = _validate_output_path(outdir / "v467_report.json", outdir)
    report_json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(_validate_output_path(outdir / "v467_report.md", outdir), report, board)
    print(json.dumps(_json_safe({"candidate_count": len(board), "recommended_candidate": recommended}), sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
