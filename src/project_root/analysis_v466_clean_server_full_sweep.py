"""V466 clean server-only full sweep.

This module extends the V465 clean server line with a bounded multi-family
model zoo, grouped OOF by rally_uid, row-level test_new prediction followed by
rally aggregation, specialist blends, and MAD-capped server-only candidates on
the V362 anchor.
"""

from __future__ import annotations

import json
import math
import os
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
    aggregate_signals_to_anchor,
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


OUT_DIR = ROOT / "v466_clean_server_full_sweep"
TARGET_MADS = (0.0025, 0.0050, 0.0075, 0.0100)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    size: str
    seed: int
    factory: Callable[[], Any]


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


def build_model_configs(*, seed: int = 466, runtime: str = "fast") -> list[ModelConfig]:
    """Build a bounded V466 model zoo.

    ``runtime='test'`` keeps one seed per core family for tiny tests. ``fast``
    is the default script mode and includes all families with conservative
    sizes. ``full`` adds larger seed bags while staying single-threaded.
    """
    if runtime not in {"test", "fast", "full"}:
        raise ValueError(f"unknown runtime: {runtime}")
    cheap_bags = 1 if runtime == "test" else (3 if runtime == "fast" else 5)
    expensive_bags = 1 if runtime == "test" else (1 if runtime == "fast" else 3)
    configs: list[ModelConfig] = []

    def add(name: str, family: str, size: str, seeds: Iterable[int], factory: Callable[[int], Any]) -> None:
        for bag_idx, bag_seed in enumerate(seeds):
            suffix = "" if bag_idx == 0 else f"_s{bag_idx}"
            configs.append(
                ModelConfig(
                    name=f"{name}{suffix}",
                    family=family,
                    size=size,
                    seed=bag_seed,
                    factory=lambda bag_seed=bag_seed, factory=factory: factory(bag_seed),
                )
            )

    add(
        "logistic_balanced",
        "linear",
        "small",
        _seed_bag(seed, cheap_bags),
        lambda s: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=s),
        ),
    )
    if runtime != "test":
        add(
            "logistic_l2_c05",
            "linear",
            "small",
            _seed_bag(seed + 7, cheap_bags),
            lambda s: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=s,
                ),
            ),
        )
        add(
            "sgd_logloss_balanced",
            "linear",
            "small",
            _seed_bag(seed + 13, cheap_bags),
            lambda s: make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    alpha=0.0005,
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-3,
                    random_state=s,
                ),
            ),
        )

    tree_estimators = 24 if runtime == "test" else 120
    add(
        "extra_trees_small",
        "tree",
        "small",
        _seed_bag(seed + 23, expensive_bags),
        lambda s: ExtraTreesClassifier(
            n_estimators=tree_estimators,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=s,
            n_jobs=1,
        ),
    )
    if runtime != "test":
        add(
            "extra_trees_medium",
            "tree",
            "medium",
            _seed_bag(seed + 29, expensive_bags),
            lambda s: ExtraTreesClassifier(
                n_estimators=220,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=s,
                n_jobs=1,
            ),
        )
        add(
            "random_forest_small",
            "tree",
            "small",
            _seed_bag(seed + 31, expensive_bags),
            lambda s: RandomForestClassifier(
                n_estimators=140,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=s,
                n_jobs=1,
            ),
        )
        add(
            "random_forest_medium",
            "tree",
            "medium",
            _seed_bag(seed + 37, expensive_bags),
            lambda s: RandomForestClassifier(
                n_estimators=240,
                min_samples_leaf=6,
                class_weight="balanced",
                random_state=s,
                n_jobs=1,
            ),
        )

    add(
        "hist_gradient_small",
        "boosting",
        "small",
        _seed_bag(seed + 41, expensive_bags),
        lambda s: HistGradientBoostingClassifier(
            max_iter=30 if runtime == "test" else 90,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=2 if runtime == "test" else 15,
            random_state=s,
        ),
    )
    if runtime != "test":
        add(
            "hist_gradient_medium",
            "boosting",
            "medium",
            _seed_bag(seed + 43, expensive_bags),
            lambda s: HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=s,
            ),
        )
    if LGBMClassifier is not None:
        add(
            "lgbm_small",
            "boosting",
            "small",
            _seed_bag(seed + 47, expensive_bags),
            lambda s: LGBMClassifier(
                n_estimators=40 if runtime == "test" else 140,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=2 if runtime == "test" else 20,
                class_weight="balanced",
                random_state=s,
                verbose=-1,
                n_jobs=1,
            ),
        )
        if runtime != "test":
            add(
                "lgbm_medium",
                "boosting",
                "medium",
                _seed_bag(seed + 53, expensive_bags),
                lambda s: LGBMClassifier(
                    n_estimators=220,
                    learning_rate=0.025,
                    num_leaves=31,
                    min_child_samples=25,
                    class_weight="balanced",
                    random_state=s,
                    verbose=-1,
                    n_jobs=1,
                ),
            )

    add(
        "mlp_small",
        "mlp",
        "small",
        _seed_bag(seed + 59, expensive_bags),
        lambda s: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16,), alpha=0.003, max_iter=180, random_state=s, early_stopping=False),
        ),
    )
    if runtime == "full":
        add(
            "mlp_medium",
            "mlp",
            "medium",
            _seed_bag(seed + 61, expensive_bags),
            lambda s: make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.002, max_iter=220, random_state=s),
            ),
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
    values = np.clip(values / scale, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def predict_positive(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return clip_prob(proba[:, 1])
    if hasattr(model, "decision_function"):
        return clip_prob(_sigmoid(model.decision_function(x)))
    return clip_prob(model.predict(x))


def _safe_auc(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        auc = float(roc_auc_score(y, clip_prob(pred)))
    except Exception:
        return float("nan")
    return auc if math.isfinite(auc) else float("nan")


def fit_model_zoo(
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    x_test: pd.DataFrame,
    *,
    groups: np.ndarray | pd.Series | None,
    configs: list[ModelConfig] | None = None,
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
    for config in configs or build_model_configs(seed=466):
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
            if config.family in {"linear", "tree", "boosting"}:
                raise RuntimeError(f"required model failed: {config.name}") from exc
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


def build_anchor_context(anchor: pd.DataFrame, test_new: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in anchor.columns or "rally_uid" not in test_new.columns:
        raise ValueError("anchor and test_new must contain rally_uid")
    grouped = test_new.groupby("rally_uid", sort=False)
    context = anchor[["rally_uid", "actionId", "pointId", "serverGetPoint"]].copy()
    context["prefix_rows"] = context["rally_uid"].map(grouped.size()).fillna(0).astype(int)
    for col in ["scoreSelf", "scoreOther", "strikeNumber", "numberGame", "rally_id"]:
        numeric = pd.to_numeric(test_new[col], errors="coerce") if col in test_new.columns else pd.Series(dtype=float)
        by_uid = numeric.groupby(test_new["rally_uid"], sort=False).agg(["mean", "max", "last"]) if len(numeric) else None
        if by_uid is None:
            context[f"{col}_mean"] = -1.0
            context[f"{col}_max"] = -1.0
            context[f"{col}_last"] = -1.0
        else:
            context[f"{col}_mean"] = context["rally_uid"].map(by_uid["mean"]).fillna(-1.0)
            context[f"{col}_max"] = context["rally_uid"].map(by_uid["max"]).fillna(-1.0)
            context[f"{col}_last"] = context["rally_uid"].map(by_uid["last"]).fillna(-1.0)
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
        "early_phase": ((prefix_rows <= 2) | (strike_max <= 2)).to_numpy(bool),
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


def aggregate_row_signals(signals: list[ModelSignal], test_new: pd.DataFrame, anchor: pd.DataFrame) -> list[ModelSignal]:
    from analysis_v465_clean_server_line import ServerSignal

    server_signals = [
        ServerSignal(name=signal.name, oof=signal.oof, test=signal.test, auc=signal.auc)
        for signal in signals
    ]
    aggregated = aggregate_signals_to_anchor(server_signals, test_new, anchor)
    by_name = {signal.name: signal for signal in signals}
    out: list[ModelSignal] = []
    for server_signal in aggregated:
        original_name = server_signal.name.removesuffix("_uidmean")
        original = by_name[original_name]
        out.append(
            ModelSignal(
                name=server_signal.name,
                family=original.family,
                size=original.size,
                seed=original.seed,
                oof=server_signal.oof,
                test=server_signal.test,
                auc=server_signal.auc,
            )
        )
    return out


def build_model_targets(
    anchor: pd.DataFrame,
    signals: list[ModelSignal],
    *,
    clean_sources: list[ServerSource] | None = None,
) -> dict[str, np.ndarray]:
    if not signals:
        raise ValueError("at least one model signal is required")
    anchor_server = clip_prob(anchor["serverGetPoint"])
    targets: dict[str, np.ndarray] = {}
    ranked_by_family: dict[str, np.ndarray] = {}
    for family in ["linear", "tree", "boosting", "mlp"]:
        family_value = _family_target(signals, family, anchor_server)
        if family_value is not None:
            ranked_by_family[family] = family_value
            targets[f"{family}_rankmean"] = family_value
    targets["global_all_model_rankmean"] = _mean_arrays(
        [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals]
    )
    if clean_sources:
        targets["clean_source_rankmean"] = _mean_arrays(
            [rank_normalize_to_anchor(source.server, anchor_server) for source in clean_sources]
        )
    else:
        targets["clean_source_rankmean"] = targets["global_all_model_rankmean"]
    ensemble_arrays = list(ranked_by_family.values()) + [targets["clean_source_rankmean"]]
    targets["full_ensemble_rankmean"] = _mean_arrays(ensemble_arrays)
    return targets


def _blend_under_mask(base: np.ndarray, target: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    out = clip_prob(base).copy()
    if len(mask) != len(out):
        raise ValueError("specialist mask length mismatch")
    out[np.asarray(mask, dtype=bool)] = clip_prob((1.0 - weight) * out[np.asarray(mask, dtype=bool)] + weight * target[np.asarray(mask, dtype=bool)])
    return clip_prob(out)


def add_specialist_targets(model_targets: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    targets = dict(model_targets)
    global_target = targets["global_all_model_rankmean"]
    tree = targets.get("tree_rankmean", global_target)
    boosting = targets.get("boosting_rankmean", tree)
    linear = targets.get("linear_rankmean", global_target)
    full = targets.get("full_ensemble_rankmean", global_target)
    targets["score_pressure_specialist"] = _blend_under_mask(global_target, _mean_arrays([tree, boosting]), masks["score_pressure"], 0.65)
    targets["early_phase_specialist"] = _blend_under_mask(global_target, _mean_arrays([linear, tree]), masks["early_phase"], 0.55)
    targets["terminal_like_specialist"] = _blend_under_mask(global_target, boosting, masks["terminal_like"], 0.65)
    targets["action_point_conditioned_specialist"] = _blend_under_mask(global_target, full, masks["action_point_conditioned"], 0.60)
    targets["full_ensemble_rankmean"] = _mean_arrays(
        [
            targets["global_all_model_rankmean"],
            targets["score_pressure_specialist"],
            targets["early_phase_specialist"],
            targets["terminal_like_specialist"],
            targets["action_point_conditioned_specialist"],
            targets["clean_source_rankmean"],
        ]
    )
    return targets


def _target_families(target_name: str) -> tuple[str, ...]:
    families = []
    for family in ["linear", "tree", "boosting", "mlp", "clean"]:
        if family in target_name:
            families.append(family)
    if "global" in target_name or "full_ensemble" in target_name or "specialist" in target_name:
        families.extend(["linear", "tree", "boosting"])
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
    candidates: list[ServerCandidate] = []
    seen: set[bytes] = set()
    for target_name in [
        "global_all_model_rankmean",
        "tree_rankmean",
        "boosting_rankmean",
        "linear_rankmean",
        "mlp_rankmean",
        "score_pressure_specialist",
        "early_phase_specialist",
        "terminal_like_specialist",
        "action_point_conditioned_specialist",
        "clean_source_rankmean",
        "full_ensemble_rankmean",
    ]:
        if target_name not in targets:
            continue
        normalized = rank_normalize_to_anchor(targets[target_name], anchor_server)
        families = _target_families(target_name)
        for target_mad in target_mads:
            server = blend_to_target_mad(anchor_server, normalized, target_mad=target_mad)
            fingerprint = np.round(server, 10).tobytes()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            actual_mad = float(np.mean(np.abs(server - anchor_server)))
            mad_key = f"{target_mad:.4f}".replace(".", "p")
            risk = "safe" if actual_mad <= 0.0050001 else "exploratory"
            decision = "review" if actual_mad <= 0.0100001 else "hold"
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
        raise ValueError(f"output path escapes V466 outdir: {path_resolved}")
    no_banned_input_guard([path_resolved])
    return path_resolved


def _candidate_filename(candidate: str) -> str:
    safe = candidate.replace(".", "p").replace("/", "_").replace("\\", "_")
    return f"submission_v466_{safe}__v362action_v362point.csv"


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
        "# V466 clean server full sweep",
        "",
        f"Generated candidates: {report['candidate_count']}",
        f"Recommended: {report['recommended_candidate']}",
        f"Runtime mode: {report['runtime']}",
        "",
        "Policy: train labels only for supervised server learning; test_new observed fields only; no TTMATCH; no upload_candidates_20260519; no old-server direct labels.",
        "",
        "## Top candidates",
        "",
        _simple_markdown_table(board.head(10)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _board_sort(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    out = board.copy()
    out["_decision_order"] = out["decision"].map({"review": 0, "hold": 1}).fillna(9)
    out["_risk_order"] = out["risk"].map({"safe": 0, "exploratory": 1}).fillna(9)
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
    configs = build_model_configs(seed=466, runtime=runtime)
    row_signals = fit_model_zoo(
        train_x,
        (y.to_numpy(dtype=float) >= 0.5).astype(int),
        test_x,
        groups=train["rally_uid"] if "rally_uid" in train.columns else None,
        configs=configs,
    )
    signals = aggregate_row_signals(row_signals, test_new, anchor)
    context = build_anchor_context(anchor, test_new)
    masks = build_specialist_masks(context)
    clean_sources = load_existing_clean_sources(root, anchor, expected_rows=expected_rows)
    model_targets = build_model_targets(anchor, signals, clean_sources=clean_sources)
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
    search_path = _validate_output_path(outdir / "v466_server_search.csv", outdir)
    board.to_csv(search_path, index=False)
    recommended = str(board.iloc[0]["candidate"]) if not board.empty else None
    report = {
        "pipeline": "v466_clean_server_full_sweep",
        "runtime": runtime,
        "anchor": str((root / ANCHOR_RELATIVE).resolve()),
        "candidate_count": int(len(board)),
        "recommended_candidate": recommended,
        "trained_model_count": int(len(signals)),
        "clean_source_count": int(len(clean_sources)),
        "trained_models": [
            {
                "name": signal.name,
                "family": signal.family,
                "size": signal.size,
                "seed": signal.seed,
                "auc": signal.auc,
            }
            for signal in signals
        ],
        "row_test_prediction": "test_new_row_level_then_rally_uid_mean",
        "grouped_oof": "rally_uid",
        "policy": {
            "no_old_server_direct_labels": True,
            "no_ttmatch": True,
            "no_upload_candidates_20260519": True,
            "train_labels_only_for_supervised_server": True,
            "preserve_action_point_from_v362": True,
        },
        "search_path": str(search_path.resolve()),
    }
    report_json_path = _validate_output_path(outdir / "v466_report.json", outdir)
    report_json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(_validate_output_path(outdir / "v466_report.md", outdir), report, board)
    print(json.dumps(_json_safe({"candidate_count": len(board), "recommended_candidate": recommended}), sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
