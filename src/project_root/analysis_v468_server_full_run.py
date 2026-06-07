"""V468 clean server-only full-run sweep.

This extends V467 with fuller server model families, true specialist
retraining, calibration targets, finer MAD caps, and larger optional sequence
variants. It preserves the V362 actionId/pointId anchor and changes only
serverGetPoint.
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
from analysis_v467_server_exhaustive_clean_sweep import (
    AGGREGATION_MODES,
    ModelSignal,
    TabularModelConfig,
    _safe_auc,
    aggregate_prefix_predictions,
    aggregate_row_signals,
    build_anchor_context,
    fit_tabular_zoo,
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = ROOT / "v468_server_full_run"
FINE_TARGET_MADS = (0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050, 0.0060, 0.0075, 0.0100, 0.0120, 0.0150)
RECOMMENDED_MAD_MAX = 0.0100001


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


def build_full_model_configs(*, runtime: str = "fast", seed: int = 468) -> list[TabularModelConfig]:
    """Return the V468 fuller bounded tabular model zoo."""
    if runtime not in {"test", "fast", "full"}:
        raise ValueError(f"unknown runtime: {runtime}")
    configs: list[TabularModelConfig] = []
    cheap_bags = 1 if runtime != "full" else 2
    full_bags = 1 if runtime in {"test", "fast"} else 2
    n_tree = 20 if runtime == "test" else (220 if runtime == "fast" else 360)
    n_boost = 20 if runtime == "test" else (260 if runtime == "fast" else 520)
    mlp_iter = 60 if runtime == "test" else (260 if runtime == "fast" else 420)

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
            SGDClassifier(loss="log_loss", alpha=0.00035, class_weight="balanced", max_iter=1200, tol=1e-3, random_state=s),
        ),
    )
    _add_config(
        configs,
        "random_forest_large",
        "tree",
        "large",
        _seed_bag(seed + 17, full_bags),
        lambda s: RandomForestClassifier(
            n_estimators=n_tree,
            min_samples_leaf=2 if runtime == "test" else 3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=s,
            n_jobs=1,
        ),
    )
    _add_config(
        configs,
        "extra_trees_large",
        "tree",
        "large",
        _seed_bag(seed + 23, full_bags),
        lambda s: ExtraTreesClassifier(
            n_estimators=n_tree,
            min_samples_leaf=1 if runtime == "test" else 2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=s,
            n_jobs=1,
        ),
    )
    _add_config(
        configs,
        "hist_gradient_full",
        "boosting",
        "large",
        _seed_bag(seed + 29, full_bags),
        lambda s: HistGradientBoostingClassifier(
            max_iter=25 if runtime == "test" else (220 if runtime == "fast" else 420),
            learning_rate=0.035,
            max_leaf_nodes=15 if runtime == "test" else 31,
            min_samples_leaf=2 if runtime == "test" else 20,
            l2_regularization=0.02,
            random_state=s,
        ),
    )
    if LGBMClassifier is not None:
        for depth, leaves in [(3, 15), (5, 31)]:
            _add_config(
                configs,
                f"lightgbm_full_depth{depth}",
                "boosting",
                "large",
                _seed_bag(seed + 37 + depth, full_bags),
                lambda s, depth=depth, leaves=leaves: LGBMClassifier(
                    n_estimators=n_boost,
                    learning_rate=0.025 if runtime == "full" else 0.035,
                    max_depth=depth,
                    num_leaves=leaves,
                    min_child_samples=2 if runtime == "test" else 16,
                    subsample=0.92,
                    colsample_bytree=0.92,
                    class_weight="balanced",
                    random_state=s,
                    verbose=-1,
                    n_jobs=1,
                ),
                optional=True,
            )
    if XGBClassifier is not None and runtime != "test":
        for depth in [3, 5]:
            _add_config(
                configs,
                f"xgboost_full_depth{depth}",
                "boosting",
                "large",
                _seed_bag(seed + 53 + depth, full_bags),
                lambda s, depth=depth: XGBClassifier(
                    n_estimators=n_boost,
                    max_depth=depth,
                    min_child_weight=4 if depth == 3 else 8,
                    learning_rate=0.025 if runtime == "full" else 0.035,
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
        for depth in [4, 6]:
            _add_config(
                configs,
                f"catboost_full_depth{depth}",
                "boosting",
                "large",
                _seed_bag(seed + 67 + depth, full_bags),
                lambda s, depth=depth: CatBoostClassifier(
                    iterations=n_boost,
                    depth=depth,
                    learning_rate=0.025 if runtime == "full" else 0.035,
                    loss_function="Logloss",
                    random_seed=s,
                    verbose=False,
                    thread_count=1,
                    allow_writing_files=False,
                ),
                optional=True,
            )
    _add_config(
        configs,
        "mlp_medium_dropout_like_alpha",
        "mlp",
        "medium",
        _seed_bag(seed + 83, full_bags),
        lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(48, 24), alpha=0.006, max_iter=mlp_iter, random_state=s)),
        optional=True,
    )
    _add_config(
        configs,
        "mlp_large_alpha_low",
        "mlp",
        "large",
        _seed_bag(seed + 89, full_bags),
        lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(96, 48), alpha=0.0015, max_iter=mlp_iter, random_state=s)),
        optional=True,
    )
    _add_config(
        configs,
        "mlp_large_alpha_high",
        "mlp",
        "large",
        _seed_bag(seed + 97, full_bags),
        lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(96, 48), alpha=0.012, max_iter=mlp_iter, random_state=s)),
        optional=True,
    )
    return configs


def build_specialist_train_masks(train: pd.DataFrame) -> dict[str, np.ndarray]:
    score_self = pd.to_numeric(train.get("scoreSelf", -1), errors="coerce").fillna(-1)
    score_other = pd.to_numeric(train.get("scoreOther", -1), errors="coerce").fillna(-1)
    strike = pd.to_numeric(train.get("strikeNumber", -1), errors="coerce").fillna(-1)
    action = pd.to_numeric(train.get("actionId", -1), errors="coerce").fillna(-1).astype(int)
    point = pd.to_numeric(train.get("pointId", -1), errors="coerce").fillna(-1).astype(int)
    return {
        "score_pressure": (((score_self - score_other).abs() <= 1) | (score_self.ge(9) & score_other.ge(9)) | score_self.ge(10) | score_other.ge(10)).to_numpy(bool),
        "phase_early": strike.le(2).to_numpy(bool),
        "terminal_like": ((point == 0) | (action == 0) | strike.ge(7)).to_numpy(bool),
        "action_point_conditioned": (point.isin([0, 1, 4, 7, 8, 9]) | action.isin([0, 8, 12, 14, 15])).to_numpy(bool),
    }


def build_specialist_anchor_masks(anchor: pd.DataFrame, test_new: pd.DataFrame) -> dict[str, np.ndarray]:
    context = build_anchor_context(anchor, test_new)
    score_self = pd.to_numeric(context.get("scoreSelf_max", -1), errors="coerce").fillna(-1)
    score_other = pd.to_numeric(context.get("scoreOther_max", -1), errors="coerce").fillna(-1)
    strike = pd.to_numeric(context.get("strikeNumber_max", -1), errors="coerce").fillna(-1)
    prefix_rows = pd.to_numeric(context.get("prefix_rows", 0), errors="coerce").fillna(0)
    action = pd.to_numeric(context.get("actionId", -1), errors="coerce").fillna(-1).astype(int)
    point = pd.to_numeric(context.get("pointId", -1), errors="coerce").fillna(-1).astype(int)
    return {
        "score_pressure": (((score_self - score_other).abs() <= 1) | (score_self.ge(9) & score_other.ge(9)) | score_self.ge(10) | score_other.ge(10)).to_numpy(bool),
        "phase_early": ((prefix_rows <= 2) | strike.le(2)).to_numpy(bool),
        "terminal_like": ((point == 0) | (action == 0) | strike.ge(7)).to_numpy(bool),
        "action_point_conditioned": (point.isin([0, 1, 4, 7, 8, 9]) | action.isin([0, 8, 12, 14, 15])).to_numpy(bool),
    }


def _mean_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("cannot average empty array list")
    return clip_prob(np.mean(np.vstack([clip_prob(values) for values in arrays]), axis=0))


def _specialist_configs(runtime: str, seed: int) -> list[TabularModelConfig]:
    configs: list[TabularModelConfig] = [
        TabularModelConfig(
            name="specialist_logistic",
            family="linear",
            size="small",
            seed=seed,
            factory=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)),
        ),
        TabularModelConfig(
            name="specialist_extra_trees",
            family="tree",
            size="small",
            seed=seed + 1,
            factory=lambda: ExtraTreesClassifier(n_estimators=20 if runtime == "test" else 100, min_samples_leaf=2, class_weight="balanced", random_state=seed + 1, n_jobs=1),
        ),
        TabularModelConfig(
            name="specialist_hist_gradient",
            family="boosting",
            size="small",
            seed=seed + 2,
            factory=lambda: HistGradientBoostingClassifier(max_iter=20 if runtime == "test" else 100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=2 if runtime == "test" else 15, random_state=seed + 2),
        ),
    ]
    if LGBMClassifier is not None:
        configs.append(
            TabularModelConfig(
                name="specialist_lightgbm",
                family="boosting",
                size="small",
                seed=seed + 3,
                factory=lambda: LGBMClassifier(n_estimators=20 if runtime == "test" else 120, learning_rate=0.04, num_leaves=15, min_child_samples=2 if runtime == "test" else 12, class_weight="balanced", random_state=seed + 3, verbose=-1, n_jobs=1),
                optional=True,
            )
        )
    return configs


def _fallback_specialist_targets(base: np.ndarray, anchor_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for source, target in [
        ("score_pressure", "true_score_pressure_specialist"),
        ("phase_early", "true_phase_early_specialist"),
        ("terminal_like", "true_terminal_like_specialist"),
        ("action_point_conditioned", "true_action_point_specialist"),
    ]:
        arr = clip_prob(base).copy()
        mask = anchor_masks[source]
        if mask.any():
            arr[mask] = rank_normalize_to_anchor(arr, base)[mask]
        out[target] = arr
    return out


def train_true_specialists(
    train: pd.DataFrame,
    test_new: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    base_targets: dict[str, np.ndarray] | None = None,
    base_target: np.ndarray | None = None,
    runtime: str = "fast",
    report: dict[str, object] | None = None,
    min_rows: int | None = None,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Train specialist server models on masked train rows when support allows."""
    info: dict[str, Any] = {}
    train_masks = build_specialist_train_masks(train)
    anchor_masks = build_specialist_anchor_masks(anchor, test_new)
    legacy_tuple_return = base_targets is None and base_target is not None and report is None
    if base_targets is None:
        base_targets = {"global_full_rankmean": base_target if base_target is not None else anchor["serverGetPoint"].to_numpy(dtype=float)}
    base = clip_prob(base_targets.get("global_full_rankmean", anchor["serverGetPoint"].to_numpy(dtype=float)))
    fallback = _fallback_specialist_targets(base, anchor_masks)
    targets: dict[str, np.ndarray] = {}
    name_map = {
        "score_pressure": "true_score_pressure_specialist",
        "phase_early": "true_phase_early_specialist",
        "terminal_like": "true_terminal_like_specialist",
        "action_point_conditioned": "true_action_point_specialist",
    }
    train_x, test_x = build_feature_matrices(train, test_new, anchor)
    y = (pd.to_numeric(train["serverGetPoint"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 0.5).astype(int)
    threshold = int(min_rows) if min_rows is not None else (6 if runtime == "test" else 300)

    for idx, (mask_name, target_name) in enumerate(name_map.items()):
        mask = np.asarray(train_masks[mask_name], dtype=bool)
        y_mask = y[mask]
        record: dict[str, Any] = {
            "train_rows": int(mask.sum()),
            "positive_rate": float(y_mask.mean()) if len(y_mask) else None,
            "status": "fallback",
        }
        if mask.sum() >= threshold and len(np.unique(y_mask)) == 2:
            try:
                signals = fit_tabular_zoo(
                    train_x.loc[mask].reset_index(drop=True),
                    y_mask,
                    test_x,
                    groups=train.loc[mask, "rally_uid"].to_numpy() if "rally_uid" in train.columns else None,
                    configs=_specialist_configs(runtime, 4680 + idx * 17),
                    skip_report={},
                )
                anchor_preds = []
                for signal in signals:
                    mean_pred = aggregate_prefix_predictions(test_new, anchor, signal.test, mode="mean")
                    late_pred = aggregate_prefix_predictions(test_new, anchor, signal.test, mode="late_weighted")
                    anchor_preds.extend([rank_normalize_to_anchor(mean_pred, anchor["serverGetPoint"]), rank_normalize_to_anchor(late_pred, anchor["serverGetPoint"])])
                if anchor_preds:
                    specialized = _mean_arrays(anchor_preds)
                    blended = base.copy()
                    anchor_mask = np.asarray(anchor_masks[mask_name], dtype=bool)
                    blended[anchor_mask] = specialized[anchor_mask]
                    targets[target_name] = clip_prob(blended)
                    record.update(
                        {
                            "status": "trained",
                            "models": [signal.name for signal in signals],
                            "auc_mean": float(np.nanmean([signal.auc for signal in signals])),
                        }
                    )
                else:
                    targets[target_name] = fallback[target_name]
                    record["fallback_reason"] = "no specialist anchor predictions"
            except Exception as exc:
                targets[target_name] = fallback[target_name]
                record["fallback_reason"] = repr(exc)
        else:
            targets[target_name] = fallback[target_name]
            record["fallback_reason"] = "insufficient support or single class"
        info[mask_name] = record
    if report is not None:
        report["specialists"] = info
    if legacy_tuple_return:
        return targets, info
    return targets


def fit_calibrator(kind: str, oof: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    kind = kind.lower()
    oof_arr = clip_prob(oof).reshape(-1)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if len(oof_arr) != len(y_arr):
        raise ValueError("oof and y length mismatch")
    if kind == "identity":
        return lambda values: clip_prob(values)
    if len(np.unique(y_arr)) != 2 or float(np.std(oof_arr)) == 0.0:
        return lambda values: clip_prob(values)
    if kind == "platt":
        model = LogisticRegression(solver="liblinear", random_state=468)
        model.fit(oof_arr.reshape(-1, 1), y_arr)
        return lambda values: clip_prob(model.predict_proba(clip_prob(values).reshape(-1, 1))[:, 1])
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(oof_arr, y_arr)
        return lambda values: clip_prob(model.predict(clip_prob(values)))
    raise ValueError(f"unknown calibrator kind: {kind}")


def _calibrated_targets(
    row_signals: list[ModelSignal],
    test_new: pd.DataFrame,
    anchor: pd.DataFrame,
    y: np.ndarray,
    *,
    family: str | None,
    name: str,
) -> dict[str, np.ndarray]:
    selected = [signal for signal in row_signals if family is None or signal.family == family]
    outputs: dict[str, np.ndarray] = {}
    for kind in ["platt", "isotonic"]:
        arrays = []
        for signal in selected:
            try:
                calibrator = fit_calibrator(kind, signal.oof, y)
                row_cal = calibrator(signal.test)
                arrays.append(rank_normalize_to_anchor(aggregate_prefix_predictions(test_new, anchor, row_cal, mode="mean"), anchor["serverGetPoint"]))
            except Exception:
                continue
        if arrays:
            outputs[f"{name}_{kind}"] = _mean_arrays(arrays)
    if outputs:
        outputs[name] = _mean_arrays(list(outputs.values()))
    return outputs


def _slice_calibrated_targets(row_signals: list[ModelSignal], test_new: pd.DataFrame, anchor: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    if not row_signals:
        return {}
    best = max(row_signals, key=lambda s: -1 if not math.isfinite(s.auc) else s.auc)
    base = rank_normalize_to_anchor(aggregate_prefix_predictions(test_new, anchor, best.test, mode="mean"), anchor["serverGetPoint"])
    masks = build_specialist_anchor_masks(anchor, test_new)
    out = {}
    for train_mask_name, target_name in [("score_pressure", "slice_score_pressure_calibrated"), ("phase_early", "slice_phase_calibrated")]:
        arr = base.copy()
        mask = masks[train_mask_name]
        if mask.any():
            arr[mask] = rank_normalize_to_anchor(aggregate_prefix_predictions(test_new, anchor, best.test, mode="late_weighted"), anchor["serverGetPoint"])[mask]
        out[target_name] = clip_prob(arr)
    return out


def train_sequence_full_models(
    train: pd.DataFrame,
    test_new: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    runtime: str = "fast",
    enabled: bool = True,
    report: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
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
    except Exception as exc:  # pragma: no cover
        info["sequence_status"] = "skipped"
        info["sequence_skip_reason"] = f"PyTorch unavailable: {exc!r}"
        return {}
    start = time.time()
    try:  # pragma: no cover - slow optional path
        torch.manual_seed(468)
        train_x, test_x = build_feature_matrices(train, test_new, anchor)
        numeric_cols = list(train_x.columns[: min(48, len(train_x.columns))])
        train_frame = train.loc[:, ["rally_uid", "serverGetPoint"]].join(train_x[numeric_cols])
        test_frame = test_new.loc[:, ["rally_uid"]].join(test_x[numeric_cols])

        def make_sequences(frame: pd.DataFrame, labels: bool) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
            seqs, ys, uids = [], [], []
            for uid, group in frame.groupby("rally_uid", sort=False):
                arr = group[numeric_cols].to_numpy(dtype="float32")[-14:]
                if len(arr) < 14:
                    arr = np.vstack([np.zeros((14 - len(arr), arr.shape[1]), dtype="float32"), arr])
                seqs.append(arr)
                uids.append(uid)
                if labels:
                    ys.append(float(group["serverGetPoint"].max() >= 0.5))
            return np.stack(seqs), (np.asarray(ys, dtype="float32") if labels else None), uids

        x_seq, y_seq, _ = make_sequences(train_frame, True)
        test_seq, _, test_uids = make_sequences(test_frame, False)
        if y_seq is None or len(np.unique(y_seq)) < 2:
            raise ValueError("sequence target lacks both classes")
        x_tensor = torch.tensor(x_seq)
        y_tensor = torch.tensor(y_seq).view(-1, 1)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=128, shuffle=True)
        input_size = x_seq.shape[2]

        class RNNHead(nn.Module):
            def __init__(self, kind: str, hidden: int) -> None:
                super().__init__()
                cls = nn.GRU if kind == "gru" else nn.LSTM
                self.rnn = cls(input_size, hidden, batch_first=True, dropout=0.0)
                self.out = nn.Linear(hidden, 1)

            def forward(self, x):
                hidden, _ = self.rnn(x)
                return self.out(hidden[:, -1, :])

        class TransformerHead(nn.Module):
            def __init__(self, width: int, layers: int) -> None:
                super().__init__()
                self.inp = nn.Linear(input_size, width)
                block = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.1, batch_first=True, activation="gelu")
                self.encoder = nn.TransformerEncoder(block, num_layers=layers)
                self.out = nn.Linear(width, 1)

            def forward(self, x):
                return self.out(self.encoder(self.inp(x))[:, -1, :])

        if runtime == "fast":
            configs = [
                ("gru_h32_epochs3", lambda: RNNHead("gru", 32), 3),
                ("lstm_h32_epochs3", lambda: RNNHead("lstm", 32), 3),
                ("transformer_d32_l1_epochs3", lambda: TransformerHead(32, 1), 3),
            ]
        else:
            configs = [
                ("gru_h32_epochs3", lambda: RNNHead("gru", 32), 3),
                ("lstm_h32_epochs3", lambda: RNNHead("lstm", 32), 3),
                ("gru_h64_epochs5", lambda: RNNHead("gru", 64), 5),
                ("lstm_h64_epochs5", lambda: RNNHead("lstm", 64), 5),
                ("transformer_d32_l1_epochs3", lambda: TransformerHead(32, 1), 3),
                ("transformer_d64_l2_epochs5", lambda: TransformerHead(64, 2), 5),
            ]

        outputs: dict[str, np.ndarray] = {}
        for name, factory, epochs in configs:
            model = factory()
            opt = torch.optim.Adam(model.parameters(), lr=0.0025)
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
            outputs[name] = rank_normalize_to_anchor(clip_prob(by_uid.reindex(anchor["rally_uid"]).to_numpy(dtype=float)), anchor["serverGetPoint"])
            if runtime == "fast" and time.time() - start > 90.0 and len(outputs) >= 1:
                break
        if not outputs:
            raise ValueError("no sequence outputs produced")
        info["sequence_status"] = "trained"
        info["sequence_models"] = len(outputs)
        out = dict(outputs)
        out["sequence_full_rankmean"] = _mean_arrays(list(outputs.values()))
        return out
    except Exception as exc:  # pragma: no cover
        info["sequence_status"] = "skipped"
        info["sequence_skip_reason"] = f"sequence training failed or exceeded bounds: {exc!r}"
        return {}


def _family_target(signals: list[ModelSignal], family: str, anchor_server: np.ndarray) -> np.ndarray | None:
    arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals if signal.family == family]
    return _mean_arrays(arrays) if arrays else None


def _name_target(signals: list[ModelSignal], needles: tuple[str, ...], anchor_server: np.ndarray) -> np.ndarray | None:
    arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals if any(needle in signal.name for needle in needles)]
    return _mean_arrays(arrays) if arrays else None


def _aggregation_target(signals: list[ModelSignal], mode: str, anchor_server: np.ndarray) -> np.ndarray | None:
    arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals if signal.name.endswith(f"_uid_{mode}")]
    return _mean_arrays(arrays) if arrays else None


def build_model_targets(
    anchor: pd.DataFrame,
    signals: list[ModelSignal],
    *,
    row_signals: list[ModelSignal],
    y: np.ndarray,
    test_new: pd.DataFrame,
    sequence_targets: dict[str, np.ndarray] | None,
    clean_sources: list[ServerSource] | None,
) -> dict[str, np.ndarray]:
    if not signals:
        raise ValueError("at least one model signal is required")
    anchor_server = clip_prob(anchor["serverGetPoint"])
    targets: dict[str, np.ndarray] = {}
    for family, target_name in [
        ("tree", "tree_full_rankmean"),
        ("boosting", "boosting_full_rankmean"),
        ("mlp", "mlp_full_rankmean"),
        ("linear", "linear_rankmean"),
    ]:
        value = _family_target(signals, family, anchor_server)
        if value is not None:
            targets[target_name] = value
    targets["global_full_rankmean"] = _mean_arrays([rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals])
    cat_xgb = _name_target(signals, ("catboost", "xgboost"), anchor_server)
    if cat_xgb is not None:
        targets["catboost_xgboost_full_rankmean"] = cat_xgb
    for mode in AGGREGATION_MODES:
        value = _aggregation_target(signals, mode, anchor_server)
        if value is not None:
            targets[f"{mode}_aggregation_rankmean"] = value
    targets.update(_calibrated_targets(row_signals, test_new, anchor, y, family=None, name="calibrated_global_rankmean"))
    targets.update(_calibrated_targets(row_signals, test_new, anchor, y, family="boosting", name="calibrated_boosting_rankmean"))
    slice_targets = _slice_calibrated_targets(row_signals, test_new, anchor, y)
    targets.update(slice_targets)
    if slice_targets:
        targets["slice_calibrated_rankmean"] = _mean_arrays(list(slice_targets.values()))
    if sequence_targets:
        targets.update(sequence_targets)
    if clean_sources:
        targets["clean_source_rankmean"] = _mean_arrays([rank_normalize_to_anchor(source.server, anchor_server) for source in clean_sources])
    else:
        targets["clean_source_rankmean"] = targets["global_full_rankmean"]
    ensemble_no_sequence = [
        targets["global_full_rankmean"],
        targets.get("boosting_full_rankmean", targets["global_full_rankmean"]),
        targets.get("tree_full_rankmean", targets["global_full_rankmean"]),
        targets.get("mlp_full_rankmean", targets["global_full_rankmean"]),
        targets["clean_source_rankmean"],
    ]
    targets["full_v468_ensemble_no_sequence"] = _mean_arrays(ensemble_no_sequence)
    ensemble = list(ensemble_no_sequence)
    if sequence_targets and "sequence_full_rankmean" in sequence_targets:
        ensemble.append(sequence_targets["sequence_full_rankmean"])
    targets["full_v468_ensemble"] = _mean_arrays(ensemble)
    return targets


def _target_families(target_name: str) -> tuple[str, ...]:
    families = []
    for family in ["linear", "tree", "boosting", "mlp", "sequence", "clean", "calibrated", "specialist"]:
        if family in target_name:
            families.append(family)
    if any(token in target_name for token in ["global", "ensemble", "aggregation"]):
        families.extend(["linear", "tree", "boosting", "mlp"])
    return tuple(sorted(set(families)))


def build_candidate_servers(anchor: pd.DataFrame, model_targets: dict[str, np.ndarray], *, target_mads: tuple[float, ...] = FINE_TARGET_MADS) -> list[ServerCandidate]:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    target_order = [
        "global_full_rankmean",
        "boosting_full_rankmean",
        "catboost_xgboost_full_rankmean",
        "mlp_full_rankmean",
        "tree_full_rankmean",
        "sequence_full_rankmean",
        "true_score_pressure_specialist",
        "true_phase_early_specialist",
        "true_terminal_like_specialist",
        "true_action_point_specialist",
        "calibrated_global_rankmean",
        "calibrated_boosting_rankmean",
        "slice_calibrated_rankmean",
        "mean_aggregation_rankmean",
        "last_aggregation_rankmean",
        "max_aggregation_rankmean",
        "late_weighted_aggregation_rankmean",
        "clean_source_rankmean",
        "full_v468_ensemble",
        "full_v468_ensemble_no_sequence",
    ]
    candidates: list[ServerCandidate] = []
    seen: set[bytes] = set()
    for target_name in target_order:
        if target_name not in model_targets:
            continue
        normalized = rank_normalize_to_anchor(model_targets[target_name], anchor_server)
        for target_mad in target_mads:
            server = blend_to_target_mad(anchor_server, normalized, target_mad=target_mad)
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
        raise ValueError(f"output path escapes V468 outdir: {path_resolved}")
    no_banned_input_guard([path_resolved])
    return path_resolved


def _candidate_filename(candidate: str) -> str:
    safe = candidate.replace(".", "p").replace("/", "_").replace("\\", "_")
    return f"submission_v468_{safe}__v362action_v362point.csv"


def _selected_rows(anchor: pd.DataFrame, server: np.ndarray, *, candidate: str) -> pd.DataFrame:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["candidate"] = candidate
    out["server_anchor"] = anchor_server
    out["server_candidate"] = clip_prob(server)
    out["server_abs_delta"] = np.abs(out["server_candidate"] - out["server_anchor"])
    return out.loc[out["server_abs_delta"] > 1e-12].sort_values("server_abs_delta", ascending=False)


def _corr(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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


def _board_sort(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    out = board.copy()
    out["_decision_order"] = out["decision"].map({"review": 0, "diagnostic_hold": 1}).fillna(9)
    out["_risk_order"] = out["risk"].map({"safe": 0, "exploratory": 1, "diagnostic": 2}).fillna(9)
    return out.sort_values(["_decision_order", "_risk_order", "family_diversity", "server_mad", "server_corr"], ascending=[True, True, False, True, False]).drop(columns=["_decision_order", "_risk_order"])


def _simple_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
            *["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.to_numpy()],
        ]
    )


def _write_report_md(path: Path, report: dict[str, Any], board: pd.DataFrame) -> None:
    lines = [
        "# V468 server full-run clean sweep",
        "",
        f"Generated candidates: {report['candidate_count']}",
        f"Recommended: {report['recommended_candidate']}",
        f"Runtime mode: {report['runtime']}",
        f"Sequence status: {report.get('sequence_status', 'unknown')}",
        "",
        "Policy: train labels only; test_new observed fields only; no old-server labels; no TTMATCH; preserves V362 actionId/pointId.",
        "",
        "## Top candidates",
        "",
        _simple_markdown_table(board.head(12)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    y = (pd.to_numeric(train["serverGetPoint"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 0.5).astype(int)

    train_x, test_x = build_feature_matrices(train, test_new, anchor)
    optional_skips: dict[str, str] = {}
    configs = build_full_model_configs(runtime=runtime)
    row_signals = fit_tabular_zoo(
        train_x,
        y,
        test_x,
        groups=train["rally_uid"] if "rally_uid" in train.columns else None,
        configs=configs,
        skip_report=optional_skips,
    )
    signals = aggregate_row_signals(row_signals, test_new, anchor)
    sequence_report: dict[str, object] = {}
    sequence_targets = train_sequence_full_models(train, test_new, anchor, runtime=runtime, enabled=sequence_enabled, report=sequence_report)
    clean_sources = load_existing_clean_sources(root, anchor, expected_rows=expected_rows)
    model_targets = build_model_targets(
        anchor,
        signals,
        row_signals=row_signals,
        y=y,
        test_new=test_new,
        sequence_targets=sequence_targets,
        clean_sources=clean_sources,
    )
    specialist_report: dict[str, object] = {}
    model_targets.update(train_true_specialists(train, test_new, anchor, base_targets=model_targets, runtime=runtime, report=specialist_report))
    candidates = build_candidate_servers(anchor, model_targets)

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
        package_server_only(anchor, candidate.server, expected_rows=expected_rows).to_csv(submission_path, index=False)
        _selected_rows(anchor, candidate.server, candidate=candidate.candidate).to_csv(selected_path, index=False)
        row = _candidate_row(candidate, anchor, submission_path, selected_path)
        row["train_oof_auc"] = candidate.train_oof_auc
        rows.append(row)

    board = _board_sort(pd.DataFrame(rows))
    search_path = _validate_output_path(outdir / "v468_server_search.csv", outdir)
    board.to_csv(search_path, index=False)
    review = board.loc[board["decision"].eq("review")] if not board.empty else board
    recommended = str(review.iloc[0]["candidate"]) if not review.empty else (str(board.iloc[0]["candidate"]) if not board.empty else None)
    report = {
        "pipeline": "v468_server_full_run",
        "runtime": runtime,
        "anchor": str((root / ANCHOR_RELATIVE).resolve()),
        "candidate_count": int(len(board)),
        "recommended_candidate": recommended,
        "trained_row_model_count": int(len(row_signals)),
        "trained_anchor_signal_count": int(len(signals)),
        "clean_source_count": int(len(clean_sources)),
        "trained_models": [{"name": s.name, "family": s.family, "size": s.size, "seed": s.seed, "auc": s.auc} for s in row_signals],
        "optional_model_skips": optional_skips,
        "aggregation_modes": list(AGGREGATION_MODES),
        "fine_target_mads": list(FINE_TARGET_MADS),
        "row_test_prediction": "test_new_row_level_then_anchor_rally_uid_aggregation",
        "grouped_oof": "rally_uid",
        "specialist_strategy": "true masked retraining when support >= 300 rows and both classes; clean fallback otherwise",
        **sequence_report,
        **specialist_report,
        "policy": {
            "no_old_server_direct_labels": True,
            "no_ttmatch": True,
            "no_upload_candidates_20260519": True,
            "train_labels_only_for_supervised_server": True,
            "preserve_action_point_from_v362": True,
        },
        "search_path": str(search_path.resolve()),
    }
    (_validate_output_path(outdir / "v468_report.json", outdir)).write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(_validate_output_path(outdir / "v468_report.md", outdir), report, board)
    print(json.dumps(_json_safe({"candidate_count": len(board), "recommended_candidate": recommended}), sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
