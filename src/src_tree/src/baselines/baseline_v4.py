"""V4 baseline: V3 LightGBM stack + CatBoost ensemble.

This script keeps the no-leakage prefix validation setup and adds CatBoost
probabilities for action, hierarchical point, and server tasks. OOF
probabilities are used to tune CatBoost blend weights and segmented class
multipliers before writing a submission.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    POINT_NONTERMINAL_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    fit_bundle,
    predict_bundle,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import (
    NGRAM_KEY_LEVELS,
    blend_probs,
    fit_ngram_bundle,
    greedy_multiplier_search,
    predict_ngram_bundle,
)
from baseline_v3 import (
    REMAINING_CLASSES,
    add_remaining_bucket,
    fit_server_aux,
    predict_server_aux,
    search_server_blend,
)


@dataclass
class CatBundle:
    action_model: CatBoostClassifier
    terminal_model: CatBoostClassifier
    point_model: CatBoostClassifier
    server_model: CatBoostClassifier


@dataclass
class V4Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_base_weights: dict[str, float]
    action_cat_weights: dict[str, float]
    point_cat_weights: dict[str, float]
    server_cat_weight: float
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    metrics: dict[str, float]
    bins_mode: str
    ensemble_bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4 CatBoost ensemble baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_v4.csv")
    parser.add_argument("--cv-report", default="cv_report_v4.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v4.csv")
    parser.add_argument("--feature-report", default="feature_report_v4.json")
    parser.add_argument("--oof-proba", default="oof_proba_v4.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--cat-iterations", type=int, default=300)
    parser.add_argument("--cat-depth", type=int, default=6)
    parser.add_argument("--cat-learning-rate", type=float, default=0.05)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--ensemble-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def cat_feature_names(features: list[str]) -> list[str]:
    cats: list[str] = []
    categorical_exact = {
        "sex",
        "numberGame",
        "prefix_len_is_odd",
        "next_hitter_is_server",
        "next_strikeId_rule",
        "is_server_hitter",
        "last_action_same_as_prev",
        "last_point_same_as_prev",
        "last_hand_same_as_prev",
    }
    for col in features:
        if col in categorical_exact:
            cats.append(col)
        elif col.startswith("lag") and (
            col.endswith("_exists")
            or col.endswith("_strikeId")
            or col.endswith("_handId")
            or col.endswith("_strengthId")
            or col.endswith("_spinId")
            or col.endswith("_pointId")
            or col.endswith("_actionId")
            or col.endswith("_positionId")
        ):
            cats.append(col)
    return cats


def make_catboost(loss_function: str, iterations: int, depth: int, learning_rate: float, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function=loss_function,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        bootstrap_type="Bernoulli",
        subsample=0.9,
    )


def server_sample_weight(df: pd.DataFrame) -> np.ndarray:
    weights = 1.0 / df["num_prefixes_in_rally"].to_numpy(dtype=float)
    return weights / np.mean(weights)


def fit_cat_bundle(
    train_df: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    args: argparse.Namespace,
    seed: int,
) -> CatBundle:
    x = train_df[features]

    action_model = make_catboost("MultiClass", args.cat_iterations, args.cat_depth, args.cat_learning_rate, seed)
    action_model.fit(
        x,
        train_df["next_actionId"],
        cat_features=cat_features,
        sample_weight=class_weight_sample(train_df["next_actionId"]),
    )

    terminal_model = make_catboost("Logloss", args.cat_iterations, args.cat_depth, args.cat_learning_rate, seed + 1)
    terminal_model.fit(x, train_df["next_is_terminal"], cat_features=cat_features)

    point_train = train_df[train_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
    point_model = make_catboost("MultiClass", args.cat_iterations, args.cat_depth, args.cat_learning_rate, seed + 2)
    point_model.fit(
        point_train[features],
        point_train["next_pointId"],
        cat_features=cat_features,
        sample_weight=class_weight_sample(point_train["next_pointId"]),
    )

    server_model = make_catboost("Logloss", args.cat_iterations, args.cat_depth, args.cat_learning_rate, seed + 3)
    server_model.fit(
        x,
        train_df["serverGetPoint"],
        cat_features=cat_features,
        sample_weight=server_sample_weight(train_df),
    )
    return CatBundle(action_model, terminal_model, point_model, server_model)


def align_cat_proba(model: CatBoostClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    out = np.zeros((len(x), len(classes)), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    for src_idx, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = proba[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero_rows = row_sum[:, 0] <= 0
    if zero_rows.any():
        out[zero_rows, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def predict_cat_bundle(bundle: CatBundle, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[features]
    action_prob = align_cat_proba(bundle.action_model, x, ACTION_CLASSES)

    terminal_raw = bundle.terminal_model.predict_proba(x)
    terminal_prob = terminal_raw[:, 1] if terminal_raw.ndim == 2 else terminal_raw
    terminal_prob = np.clip(terminal_prob.astype(float), 1e-6, 1.0 - 1e-6)

    point_nonterminal = align_cat_proba(bundle.point_model, x, POINT_NONTERMINAL_CLASSES)
    point_prob = np.zeros((len(df), len(POINT_CLASSES)), dtype=float)
    point_prob[:, 0] = terminal_prob
    point_prob[:, 1:] = (1.0 - terminal_prob[:, None]) * point_nonterminal
    point_prob = point_prob / point_prob.sum(axis=1, keepdims=True)

    server_raw = bundle.server_model.predict_proba(x)
    server_prob = server_raw[:, 1] if server_raw.ndim == 2 else server_raw
    server_prob = np.clip(server_prob.astype(float), 1e-6, 1.0 - 1e-6)
    return action_prob, point_prob, server_prob


def bin_masks(meta: pd.DataFrame, mode: str) -> list[tuple[str, np.ndarray]]:
    prefix = meta["prefix_len"]
    if mode == "global":
        return [("global", np.ones(len(meta), dtype=bool))]
    if mode == "two":
        return [
            ("le2", prefix.le(2).to_numpy()),
            ("ge3", prefix.ge(3).to_numpy()),
        ]
    return [
        ("1", prefix.eq(1).to_numpy()),
        ("2", prefix.eq(2).to_numpy()),
        ("3", prefix.eq(3).to_numpy()),
        ("4-6", prefix.between(4, 6).to_numpy()),
        ("7+", prefix.ge(7).to_numpy()),
    ]


def apply_bin_blend(
    meta: pd.DataFrame,
    base_prob: np.ndarray,
    cat_prob: np.ndarray,
    weights: dict[str, float],
    mode: str,
) -> np.ndarray:
    out = np.zeros_like(base_prob)
    if mode == "global":
        w = float(weights["global"])
        return blend_probs(base_prob, cat_prob, w)
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        w = float(weights[label])
        out[idx] = blend_probs(base_prob[idx], cat_prob[idx], w)
    return out


def tune_bin_blend(
    meta: pd.DataFrame,
    base_prob: np.ndarray,
    cat_prob: np.ndarray,
    classes: list[int],
    target_col: str,
    mode: str,
) -> dict[str, float]:
    candidates = [round(x, 1) for x in np.arange(0.0, 0.8, 0.1)]
    result: dict[str, float] = {}
    if mode == "global":
        y = meta[target_col].to_numpy()
        best = max(
            candidates,
            key=lambda w: f1_score(
                y,
                np.asarray(classes)[np.argmax(blend_probs(base_prob, cat_prob, w), axis=1)],
                average="macro",
                labels=classes,
                zero_division=0,
            ),
        )
        return {"global": float(best)}
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) < 250:
            result[label] = 0.0
            continue
        y = meta.iloc[idx][target_col].to_numpy()
        best = max(
            candidates,
            key=lambda w: f1_score(
                y,
                np.asarray(classes)[np.argmax(blend_probs(base_prob[idx], cat_prob[idx], w), axis=1)],
                average="macro",
                labels=classes,
                zero_division=0,
            ),
        )
        result[label] = float(best)
    return result


def tune_segmented_multipliers(
    meta: pd.DataFrame,
    prob: np.ndarray,
    classes: list[int],
    task: str,
    mode: str,
) -> dict[str, list[float]]:
    target_col = "next_actionId" if task == "action" else "next_pointId"
    result: dict[str, list[float]] = {}
    global_mult = greedy_multiplier_search(meta[target_col].to_numpy(), prob, classes, task)
    if mode == "global":
        return {"global": global_mult.tolist()}
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) < 250:
            result[label] = global_mult.tolist()
        else:
            result[label] = greedy_multiplier_search(
                meta.iloc[idx][target_col].to_numpy(), prob[idx], classes, task
            ).tolist()
    return result


def apply_segmented_multipliers(
    meta: pd.DataFrame,
    prob: np.ndarray,
    multipliers: dict[str, list[float]],
    classes: list[int],
    mode: str,
) -> np.ndarray:
    pred = np.zeros(len(meta), dtype=int)
    if mode == "global":
        mult = np.asarray(multipliers["global"], dtype=float)
        return np.asarray(classes)[np.argmax(prob * mult[None, :], axis=1)]
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        mult = np.asarray(multipliers[label], dtype=float)
        pred[idx] = np.asarray(classes)[np.argmax(prob[idx] * mult[None, :], axis=1)]
    return pred


def evaluate_v4(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    action_multipliers: dict[str, list[float]],
    point_multipliers: dict[str, list[float]],
    bins_mode: str,
) -> dict[str, float]:
    action_pred = apply_segmented_multipliers(meta, action_prob, action_multipliers, ACTION_CLASSES, bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, point_multipliers, POINT_CLASSES, bins_mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(overall),
    }


def tune_v4(oof: dict[str, object], multiplier_bins: str, ensemble_bins: str) -> V4Tuning:
    meta = oof["valid_meta"]
    y_action = meta["next_actionId"].to_numpy()
    y_point = meta["next_pointId"].to_numpy()
    y_server = meta["serverGetPoint"].to_numpy()

    ngram_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    action_ngram_w = max(
        ngram_grid,
        key=lambda w: f1_score(
            y_action,
            np.asarray(ACTION_CLASSES)[np.argmax(blend_probs(oof["lgbm_action"], oof["ngram_action"], w), axis=1)],
            average="macro",
            labels=ACTION_CLASSES,
            zero_division=0,
        ),
    )
    point_ngram_w = max(
        ngram_grid,
        key=lambda w: f1_score(
            y_point,
            np.asarray(POINT_CLASSES)[np.argmax(blend_probs(oof["lgbm_point"], oof["ngram_point"], w), axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        ),
    )
    lgbm_action_base = blend_probs(oof["lgbm_action"], oof["ngram_action"], action_ngram_w)
    lgbm_point_base = blend_probs(oof["lgbm_point"], oof["ngram_point"], point_ngram_w)

    server_base_weights, lgbm_server_base, _ = search_server_blend(
        y_server,
        oof["lgbm_server"],
        oof["ngram_server"],
        oof["parity_server"],
        oof["remaining_server"],
    )

    action_cat_weights = tune_bin_blend(
        meta, lgbm_action_base, oof["cat_action"], ACTION_CLASSES, "next_actionId", ensemble_bins
    )
    point_cat_weights = tune_bin_blend(
        meta, lgbm_point_base, oof["cat_point"], POINT_CLASSES, "next_pointId", ensemble_bins
    )
    action_prob = apply_bin_blend(meta, lgbm_action_base, oof["cat_action"], action_cat_weights, ensemble_bins)
    point_prob = apply_bin_blend(meta, lgbm_point_base, oof["cat_point"], point_cat_weights, ensemble_bins)

    server_cat_grid = [round(x, 1) for x in np.arange(0.0, 0.8, 0.1)]
    server_cat_w = max(
        server_cat_grid,
        key=lambda w: roc_auc_score(y_server, (1.0 - w) * lgbm_server_base + w * oof["cat_server"]),
    )
    server_prob = (1.0 - server_cat_w) * lgbm_server_base + server_cat_w * oof["cat_server"]

    action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", multiplier_bins)
    point_mult = tune_segmented_multipliers(meta, point_prob, POINT_CLASSES, "point", multiplier_bins)
    metrics = evaluate_v4(meta, action_prob, point_prob, server_prob, action_mult, point_mult, multiplier_bins)
    return V4Tuning(
        action_ngram_weight=float(action_ngram_w),
        point_ngram_weight=float(point_ngram_w),
        server_base_weights=server_base_weights,
        action_cat_weights=action_cat_weights,
        point_cat_weights=point_cat_weights,
        server_cat_weight=float(server_cat_w),
        action_multipliers=action_mult,
        point_multipliers=point_mult,
        metrics=metrics,
        bins_mode=multiplier_bins,
        ensemble_bins_mode=ensemble_bins,
    )


def run_cv(prefix_df: pd.DataFrame, test_prefix_lengths: np.ndarray, features: list[str], args: argparse.Namespace) -> dict[str, object]:
    cat_features = cat_feature_names(features)
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    parts: dict[str, list] = {
        "valid_meta": [],
        "lgbm_action": [],
        "lgbm_point": [],
        "lgbm_server": [],
        "ngram_action": [],
        "ngram_point": [],
        "ngram_server": [],
        "parity_server": [],
        "remaining_server": [],
        "cat_action": [],
        "cat_point": [],
        "cat_server": [],
    }
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_rally_idx, valid_rally_idx) in enumerate(
        splitter.split(rally_meta, groups=rally_meta["match"]), start=1
    ):
        train_rallies = set(rally_meta.iloc[train_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_rally_idx]["rally_uid"])
        if set(rally_meta.iloc[train_rally_idx]["match"]) & set(rally_meta.iloc[valid_rally_idx]["match"]):
            raise RuntimeError("GroupKFold leakage: train/valid match overlap.")

        fold_train = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
        valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_prefix_lengths, args.seed + fold)
        fold_valid = valid_pool.loc[sampled_idx].copy()
        fold_seed = args.seed + fold * 10

        lgbm_bundle = fit_bundle(fold_train, features, args.n_estimators, fold_seed)
        lgbm_action, lgbm_point, lgbm_server = predict_bundle(lgbm_bundle, fold_valid, features)
        aux = fit_server_aux(fold_train, features, args.n_estimators, fold_seed)
        parity_server, remaining_server = predict_server_aux(aux, fold_valid, features)

        ngram_bundle = fit_ngram_bundle(fold_train, args.ngram_alpha)
        ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, fold_valid)

        cat_bundle = fit_cat_bundle(fold_train, features, cat_features, args, fold_seed)
        cat_action, cat_point, cat_server = predict_cat_bundle(cat_bundle, fold_valid, features)

        lgbm_action_f1 = f1_score(
            fold_valid["next_actionId"],
            np.asarray(ACTION_CLASSES)[np.argmax(lgbm_action, axis=1)],
            average="macro",
            labels=ACTION_CLASSES,
            zero_division=0,
        )
        lgbm_point_f1 = f1_score(
            fold_valid["next_pointId"],
            np.asarray(POINT_CLASSES)[np.argmax(lgbm_point, axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        )
        lgbm_auc = roc_auc_score(fold_valid["serverGetPoint"], lgbm_server)
        cat_action_f1 = f1_score(
            fold_valid["next_actionId"],
            np.asarray(ACTION_CLASSES)[np.argmax(cat_action, axis=1)],
            average="macro",
            labels=ACTION_CLASSES,
            zero_division=0,
        )
        cat_point_f1 = f1_score(
            fold_valid["next_pointId"],
            np.asarray(POINT_CLASSES)[np.argmax(cat_point, axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        )
        cat_auc = roc_auc_score(fold_valid["serverGetPoint"], cat_server)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "valid_rows": len(fold_valid),
                "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
                "lgbm_action_macro_f1": float(lgbm_action_f1),
                "lgbm_point_macro_f1": float(lgbm_point_f1),
                "lgbm_server_auc": float(lgbm_auc),
                "lgbm_overall": float(0.4 * lgbm_action_f1 + 0.4 * lgbm_point_f1 + 0.2 * lgbm_auc),
                "cat_action_macro_f1": float(cat_action_f1),
                "cat_point_macro_f1": float(cat_point_f1),
                "cat_server_auc": float(cat_auc),
                "cat_overall": float(0.4 * cat_action_f1 + 0.4 * cat_point_f1 + 0.2 * cat_auc),
            }
        )
        print(
            f"fold {fold}: lgbm={fold_rows[-1]['lgbm_overall']:.6f} "
            f"cat={fold_rows[-1]['cat_overall']:.6f}"
        )

        keep_cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
        parts["valid_meta"].append(fold_valid[keep_cols].reset_index(drop=True))
        for key, value in [
            ("lgbm_action", lgbm_action),
            ("lgbm_point", lgbm_point),
            ("lgbm_server", lgbm_server),
            ("ngram_action", ngram_action),
            ("ngram_point", ngram_point),
            ("ngram_server", ngram_server),
            ("parity_server", parity_server),
            ("remaining_server", remaining_server),
            ("cat_action", cat_action),
            ("cat_point", cat_point),
            ("cat_server", cat_server),
        ]:
            parts[key].append(value)

    oof: dict[str, object] = {
        "valid_meta": pd.concat(parts["valid_meta"], ignore_index=True),
        "fold_report": pd.DataFrame(fold_rows),
    }
    for key in ["lgbm_action", "lgbm_point", "ngram_action", "ngram_point", "cat_action", "cat_point"]:
        oof[key] = np.vstack(parts[key])
    for key in ["lgbm_server", "ngram_server", "parity_server", "remaining_server", "cat_server"]:
        oof[key] = np.concatenate(parts[key])

    fold_report = oof["fold_report"]
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0}
    for col in fold_report.columns:
        if col not in mean_row and col != "fold":
            mean_row[col] = float(fold_report[col].mean())
    oof["fold_report"] = pd.concat([fold_report, pd.DataFrame([mean_row])], ignore_index=True)
    return oof


def prefix_len_report(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: V4Tuning,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        metrics = evaluate_v4(
            meta.iloc[idx].reset_index(drop=True),
            action_prob[idx],
            point_prob[idx],
            server_prob[idx],
            tuning.action_multipliers,
            tuning.point_multipliers,
            tuning.bins_mode,
        )
        metrics.update({"prefix_len_bin": label, "count": int(len(idx))})
        rows.append(metrics)
    return pd.DataFrame(rows)


def full_predict(prefix_df: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str], args: argparse.Namespace) -> dict[str, np.ndarray]:
    cat_features = cat_feature_names(features)
    lgbm_bundle = fit_bundle(prefix_df, features, args.n_estimators, args.seed)
    lgbm_action, lgbm_point, lgbm_server = predict_bundle(lgbm_bundle, test_prefix, features)
    aux = fit_server_aux(prefix_df, features, args.n_estimators, args.seed)
    parity_server, remaining_server = predict_server_aux(aux, test_prefix, features)
    ngram_bundle = fit_ngram_bundle(prefix_df, args.ngram_alpha)
    ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, test_prefix)
    cat_bundle = fit_cat_bundle(prefix_df, features, cat_features, args, args.seed)
    cat_action, cat_point, cat_server = predict_cat_bundle(cat_bundle, test_prefix, features)
    return {
        "lgbm_action": lgbm_action,
        "lgbm_point": lgbm_point,
        "lgbm_server": lgbm_server,
        "ngram_action": ngram_action,
        "ngram_point": ngram_point,
        "ngram_server": ngram_server,
        "parity_server": parity_server,
        "remaining_server": remaining_server,
        "cat_action": cat_action,
        "cat_point": cat_point,
        "cat_server": cat_server,
    }


def compose_predictions(meta: pd.DataFrame, pred: dict[str, np.ndarray], tuning: V4Tuning) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lgbm_action_base = blend_probs(pred["lgbm_action"], pred["ngram_action"], tuning.action_ngram_weight)
    lgbm_point_base = blend_probs(pred["lgbm_point"], pred["ngram_point"], tuning.point_ngram_weight)
    action_prob = apply_bin_blend(
        meta, lgbm_action_base, pred["cat_action"], tuning.action_cat_weights, tuning.ensemble_bins_mode
    )
    point_prob = apply_bin_blend(
        meta, lgbm_point_base, pred["cat_point"], tuning.point_cat_weights, tuning.ensemble_bins_mode
    )
    sbw = tuning.server_base_weights
    server_base = (
        sbw["direct"] * pred["lgbm_server"]
        + sbw["ngram"] * pred["ngram_server"]
        + sbw["parity"] * pred["parity_server"]
        + sbw["remaining"] * pred["remaining_server"]
    )
    server_prob = (1.0 - tuning.server_cat_weight) * server_base + tuning.server_cat_weight * pred["cat_server"]
    return action_prob, point_prob, server_prob


def write_submission(test_prefix: pd.DataFrame, pred: dict[str, np.ndarray], tuning: V4Tuning, output: Path) -> pd.DataFrame:
    action_prob, point_prob, server_prob = compose_predictions(test_prefix, pred, tuning)
    action_pred = apply_segmented_multipliers(test_prefix, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(test_prefix, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    sub = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != test_prefix["rally_uid"].nunique():
        raise ValueError("Submission row count does not match test rally count.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    if not sub["actionId"].between(0, 18).all() or not sub["pointId"].between(0, 9).all():
        raise ValueError("Submission contains invalid classes.")
    if not sub["serverGetPoint"].between(0, 1).all():
        raise ValueError("Submission contains invalid server probabilities.")
    sub.to_csv(output, index=False, float_format="%.8f")
    return sub


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building prefix tables...")
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_prefix = build_test_prefix_table(test, args.max_lag)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    print(f"train prefix rows: {len(prefix_df):,}")
    print(f"test prediction rows: {len(test_prefix):,}")
    print(f"feature count: {len(features)}")
    print(f"cat feature count: {len(cat_feature_names(features))}")

    oof = run_cv(prefix_df, test_prefix["prefix_len"].to_numpy(dtype=int), features, args)
    tuning = tune_v4(oof, args.multiplier_bins, args.ensemble_bins)
    action_prob, point_prob, server_prob = compose_predictions(oof["valid_meta"], oof, tuning)
    prefix_report = prefix_len_report(oof["valid_meta"], action_prob, point_prob, server_prob, tuning)

    fold_report = oof["fold_report"].copy()
    fold_report["selected_action_ngram_weight"] = tuning.action_ngram_weight
    fold_report["selected_point_ngram_weight"] = tuning.point_ngram_weight
    fold_report["selected_server_cat_weight"] = tuning.server_cat_weight
    for name, value in tuning.server_base_weights.items():
        fold_report[f"selected_server_base_weight_{name}"] = value
    for name, value in tuning.metrics.items():
        fold_report[f"selected_{name}"] = value
    fold_report.to_csv(args.cv_report, index=False)
    prefix_report.to_csv(args.prefix_len_report, index=False)

    with open(args.oof_proba, "wb") as f:
        pickle.dump({**oof, "tuning": tuning}, f)

    print("selected tuning:")
    print(
        json.dumps(
            {
                "action_ngram_weight": tuning.action_ngram_weight,
                "point_ngram_weight": tuning.point_ngram_weight,
                "server_base_weights": tuning.server_base_weights,
                "action_cat_weights": tuning.action_cat_weights,
                "point_cat_weights": tuning.point_cat_weights,
                "server_cat_weight": tuning.server_cat_weight,
                **tuning.metrics,
            },
            indent=2,
        )
    )

    print("training full-data models...")
    full_pred = full_predict(prefix_df, test_prefix, features, args)
    submission = write_submission(test_prefix, full_pred, tuning, Path(args.submission))

    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_df)),
        "test_prediction_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "cat_feature_count": int(len(cat_feature_names(features))),
        "features": features,
        "cat_features": cat_feature_names(features),
        "n_estimators": args.n_estimators,
        "cat_iterations": args.cat_iterations,
        "cat_depth": args.cat_depth,
        "cat_learning_rate": args.cat_learning_rate,
        "ngram_key_levels": NGRAM_KEY_LEVELS,
        "ngram_alpha": args.ngram_alpha,
        "remaining_classes": REMAINING_CLASSES,
        "selected": {
            "action_ngram_weight": tuning.action_ngram_weight,
            "point_ngram_weight": tuning.point_ngram_weight,
            "server_base_weights": tuning.server_base_weights,
            "action_cat_weights": tuning.action_cat_weights,
            "point_cat_weights": tuning.point_cat_weights,
            "server_cat_weight": tuning.server_cat_weight,
            "action_multipliers": tuning.action_multipliers,
            "point_multipliers": tuning.point_multipliers,
            "multiplier_bins": tuning.bins_mode,
            "ensemble_bins": tuning.ensemble_bins_mode,
            "metrics": tuning.metrics,
        },
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
