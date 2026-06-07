"""V3 baseline: V2 plus server parity/survival and segmented multipliers.

Default run is intentionally conservative:
- one seed by default
- two multiplier bins: prefix_len <= 2 and >= 3
- server probability is selected by OOF blend search over direct/ngram/parity/remaining models
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    aligned_proba,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    fit_bundle,
    make_lgbm,
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


REMAINING_CLASSES = list(range(1, 8))


@dataclass
class ServerAuxBundle:
    parity_model: object
    remaining_model: object
    bucket7_even_rate_by_prefix_odd: dict[int, float]


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict[str, float]
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    metrics: dict[str, float]
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3 baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_v3.csv")
    parser.add_argument("--cv-report", default="cv_report_v3.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v3.csv")
    parser.add_argument("--feature-report", default="feature_report_v3.json")
    parser.add_argument("--oof-proba", default="oof_proba_v3.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def add_remaining_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["remaining_len_bucket"] = np.minimum(df["remaining_len"].astype(int), 7)
    return df


def server_sample_weight(df: pd.DataFrame) -> np.ndarray:
    weights = 1.0 / df["num_prefixes_in_rally"].to_numpy(dtype=float)
    return weights / np.mean(weights)


def fit_server_aux(df: pd.DataFrame, features: list[str], n_estimators: int, seed: int) -> ServerAuxBundle:
    x = df[features]
    weights = server_sample_weight(df)

    parity_model = make_lgbm("binary", n_estimators, seed + 100)
    parity_model.fit(x, df["final_parity_even"], sample_weight=weights)

    remaining_model = make_lgbm("multiclass", n_estimators, seed + 101, num_class=len(REMAINING_CLASSES))
    remaining_model.fit(
        x,
        df["remaining_len_bucket"],
        sample_weight=class_weight_sample(df["remaining_len_bucket"]),
    )

    bucket7 = df[df["remaining_len_bucket"].eq(7)]
    rates: dict[int, float] = {}
    global_rate = float(df["final_parity_even"].mean())
    for prefix_odd in [0, 1]:
        part = bucket7[bucket7["prefix_len_is_odd"].eq(prefix_odd)]
        rates[prefix_odd] = float(part["final_parity_even"].mean()) if len(part) else global_rate
    return ServerAuxBundle(parity_model, remaining_model, rates)


def remaining_to_server_prob(
    remaining_prob: np.ndarray,
    prefix_len: np.ndarray,
    bucket7_even_rate_by_prefix_odd: dict[int, float],
) -> np.ndarray:
    out = np.zeros(len(prefix_len), dtype=float)
    prefix_len = prefix_len.astype(int)
    for col_idx, remaining in enumerate(REMAINING_CLASSES):
        if remaining < 7:
            final_even = ((prefix_len + remaining) % 2 == 0).astype(float)
        else:
            prefix_odd = (prefix_len % 2).astype(int)
            final_even = np.array([bucket7_even_rate_by_prefix_odd[int(v)] for v in prefix_odd], dtype=float)
        out += remaining_prob[:, col_idx] * final_even
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def predict_server_aux(bundle: ServerAuxBundle, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[features]
    parity_raw = bundle.parity_model.predict_proba(x)
    parity_prob = parity_raw[:, 1] if parity_raw.ndim == 2 else parity_raw
    remaining_prob = aligned_proba(bundle.remaining_model, x, REMAINING_CLASSES)
    remaining_server_prob = remaining_to_server_prob(
        remaining_prob,
        df["prefix_len"].to_numpy(dtype=int),
        bundle.bucket7_even_rate_by_prefix_odd,
    )
    return np.clip(parity_prob, 1e-6, 1.0 - 1e-6), remaining_server_prob


def average_lgbm_predictions(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    n_estimators: int,
    seeds: list[int],
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    server_parts: list[np.ndarray] = []
    parity_parts: list[np.ndarray] = []
    remaining_parts: list[np.ndarray] = []
    for seed in seeds:
        fold_seed = seed + fold * 10
        bundle = fit_bundle(train_df, features, n_estimators, fold_seed)
        action_prob, point_prob, server_prob = predict_bundle(bundle, valid_df, features)
        aux = fit_server_aux(train_df, features, n_estimators, fold_seed)
        parity_prob, remaining_server_prob = predict_server_aux(aux, valid_df, features)
        action_parts.append(action_prob)
        point_parts.append(point_prob)
        server_parts.append(server_prob)
        parity_parts.append(parity_prob)
        remaining_parts.append(remaining_server_prob)
    return (
        np.mean(action_parts, axis=0),
        np.mean(point_parts, axis=0),
        np.mean(server_parts, axis=0),
        np.mean(parity_parts, axis=0),
        np.mean(remaining_parts, axis=0),
    )


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
            continue
        result[label] = greedy_multiplier_search(meta.iloc[idx][target_col].to_numpy(), prob[idx], classes, task).tolist()
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


def apply_segmented_multipliers_to_test(
    test_prefix: pd.DataFrame,
    prob: np.ndarray,
    multipliers: dict[str, list[float]],
    classes: list[int],
    mode: str,
) -> np.ndarray:
    meta = test_prefix[["prefix_len"]].copy()
    return apply_segmented_multipliers(meta, prob, multipliers, classes, mode)


def evaluate_v3(
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


def search_server_blend(
    y: np.ndarray,
    direct: np.ndarray,
    ngram: np.ndarray,
    parity: np.ndarray,
    remaining: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, float]:
    candidates: list[tuple[dict[str, float], np.ndarray, float]] = []
    grid = np.linspace(0.0, 1.0, 11)
    for w_direct in grid:
        for w_ngram in np.linspace(0.0, 0.4, 5):
            for w_parity in np.linspace(0.0, 0.8, 9):
                w_remaining = 1.0 - w_direct - w_ngram - w_parity
                if w_remaining < -1e-9 or w_remaining > 0.8:
                    continue
                prob = (
                    w_direct * direct
                    + w_ngram * ngram
                    + w_parity * parity
                    + w_remaining * remaining
                )
                auc = roc_auc_score(y, prob)
                weights = {
                    "direct": float(w_direct),
                    "ngram": float(w_ngram),
                    "parity": float(w_parity),
                    "remaining": float(w_remaining),
                }
                candidates.append((weights, prob, float(auc)))
    return max(candidates, key=lambda item: item[2])


def tune_v3(oof: dict[str, object], bins_mode: str) -> V3Tuning:
    meta = oof["valid_meta"]
    y_action = meta["next_actionId"].to_numpy()
    y_point = meta["next_pointId"].to_numpy()
    y_server = meta["serverGetPoint"].to_numpy()

    blend_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    action_w = max(
        blend_grid,
        key=lambda w: f1_score(
            y_action,
            np.asarray(ACTION_CLASSES)[np.argmax(blend_probs(oof["lgbm_action"], oof["ngram_action"], w), axis=1)],
            average="macro",
            labels=ACTION_CLASSES,
            zero_division=0,
        ),
    )
    point_w = max(
        blend_grid,
        key=lambda w: f1_score(
            y_point,
            np.asarray(POINT_CLASSES)[np.argmax(blend_probs(oof["lgbm_point"], oof["ngram_point"], w), axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        ),
    )
    action_prob = blend_probs(oof["lgbm_action"], oof["ngram_action"], action_w)
    point_prob = blend_probs(oof["lgbm_point"], oof["ngram_point"], point_w)

    server_weights, server_prob, _ = search_server_blend(
        y_server,
        oof["lgbm_server"],
        oof["ngram_server"],
        oof["parity_server"],
        oof["remaining_server"],
    )

    action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", bins_mode)
    point_mult = tune_segmented_multipliers(meta, point_prob, POINT_CLASSES, "point", bins_mode)
    metrics = evaluate_v3(meta, action_prob, point_prob, server_prob, action_mult, point_mult, bins_mode)
    return V3Tuning(
        action_ngram_weight=float(action_w),
        point_ngram_weight=float(point_w),
        server_weights=server_weights,
        action_multipliers=action_mult,
        point_multipliers=point_mult,
        metrics=metrics,
        bins_mode=bins_mode,
    )


def prefix_len_report(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: V3Tuning,
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
        metrics = evaluate_v3(
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


def run_cv(prefix_df: pd.DataFrame, test_prefix_lengths: np.ndarray, features: list[str], args: argparse.Namespace) -> dict[str, object]:
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
        sampled_idx = sample_validation_prefixes(valid_pool, test_prefix_lengths, args.seeds[0] + fold)
        fold_valid = valid_pool.loc[sampled_idx].copy()

        lgbm_action, lgbm_point, lgbm_server, parity_server, remaining_server = average_lgbm_predictions(
            fold_train, fold_valid, features, args.n_estimators, args.seeds, fold
        )
        ngram_bundle = fit_ngram_bundle(fold_train, args.ngram_alpha)
        ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, fold_valid)

        v2_like = 0.4 * f1_score(
            fold_valid["next_actionId"],
            np.asarray(ACTION_CLASSES)[np.argmax(lgbm_action, axis=1)],
            average="macro",
            labels=ACTION_CLASSES,
            zero_division=0,
        ) + 0.4 * f1_score(
            fold_valid["next_pointId"],
            np.asarray(POINT_CLASSES)[np.argmax(lgbm_point, axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        ) + 0.2 * roc_auc_score(fold_valid["serverGetPoint"], lgbm_server)
        parity_auc = roc_auc_score(fold_valid["serverGetPoint"], parity_server)
        remaining_auc = roc_auc_score(fold_valid["serverGetPoint"], remaining_server)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "valid_rows": len(fold_valid),
                "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
                "base_lgbm_overall": float(v2_like),
                "direct_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], lgbm_server)),
                "ngram_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], ngram_server)),
                "parity_server_auc": float(parity_auc),
                "remaining_server_auc": float(remaining_auc),
            }
        )
        print(
            f"fold {fold}: base_lgbm={v2_like:.6f} "
            f"server direct={fold_rows[-1]['direct_server_auc']:.6f} "
            f"parity={parity_auc:.6f} remaining={remaining_auc:.6f}"
        )

        keep_cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
        parts["valid_meta"].append(fold_valid[keep_cols].reset_index(drop=True))
        parts["lgbm_action"].append(lgbm_action)
        parts["lgbm_point"].append(lgbm_point)
        parts["lgbm_server"].append(lgbm_server)
        parts["ngram_action"].append(ngram_action)
        parts["ngram_point"].append(ngram_point)
        parts["ngram_server"].append(ngram_server)
        parts["parity_server"].append(parity_server)
        parts["remaining_server"].append(remaining_server)

    oof: dict[str, object] = {
        "valid_meta": pd.concat(parts["valid_meta"], ignore_index=True),
        "fold_report": pd.DataFrame(fold_rows),
    }
    for key in [
        "lgbm_action",
        "lgbm_point",
        "ngram_action",
        "ngram_point",
    ]:
        oof[key] = np.vstack(parts[key])
    for key in ["lgbm_server", "ngram_server", "parity_server", "remaining_server"]:
        oof[key] = np.concatenate(parts[key])

    fold_report = oof["fold_report"]
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0}
    for col in fold_report.columns:
        if col not in mean_row and col != "fold":
            mean_row[col] = float(fold_report[col].mean())
    oof["fold_report"] = pd.concat([fold_report, pd.DataFrame([mean_row])], ignore_index=True)
    return oof


def full_predict(
    prefix_df: pd.DataFrame,
    test_prefix: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    action_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    server_parts: list[np.ndarray] = []
    parity_parts: list[np.ndarray] = []
    remaining_parts: list[np.ndarray] = []
    for seed in args.seeds:
        bundle = fit_bundle(prefix_df, features, args.n_estimators, seed)
        action_prob, point_prob, server_prob = predict_bundle(bundle, test_prefix, features)
        aux = fit_server_aux(prefix_df, features, args.n_estimators, seed)
        parity_prob, remaining_prob = predict_server_aux(aux, test_prefix, features)
        action_parts.append(action_prob)
        point_parts.append(point_prob)
        server_parts.append(server_prob)
        parity_parts.append(parity_prob)
        remaining_parts.append(remaining_prob)
    ngram_bundle = fit_ngram_bundle(prefix_df, args.ngram_alpha)
    ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, test_prefix)
    return {
        "lgbm_action": np.mean(action_parts, axis=0),
        "lgbm_point": np.mean(point_parts, axis=0),
        "lgbm_server": np.mean(server_parts, axis=0),
        "parity_server": np.mean(parity_parts, axis=0),
        "remaining_server": np.mean(remaining_parts, axis=0),
        "ngram_action": ngram_action,
        "ngram_point": ngram_point,
        "ngram_server": ngram_server,
    }


def write_submission(test_prefix: pd.DataFrame, pred: dict[str, np.ndarray], tuning: V3Tuning, output: Path) -> pd.DataFrame:
    action_prob = blend_probs(pred["lgbm_action"], pred["ngram_action"], tuning.action_ngram_weight)
    point_prob = blend_probs(pred["lgbm_point"], pred["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server_prob = (
        sw["direct"] * pred["lgbm_server"]
        + sw["ngram"] * pred["ngram_server"]
        + sw["parity"] * pred["parity_server"]
        + sw["remaining"] * pred["remaining_server"]
    )
    action_pred = apply_segmented_multipliers_to_test(
        test_prefix, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode
    )
    point_pred = apply_segmented_multipliers_to_test(
        test_prefix, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode
    )
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
    print(f"seeds: {args.seeds}")

    oof = run_cv(prefix_df, test_prefix["prefix_len"].to_numpy(dtype=int), features, args)
    tuning = tune_v3(oof, args.multiplier_bins)

    action_prob = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)
    point_prob = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server_prob = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )
    prefix_report = prefix_len_report(oof["valid_meta"], action_prob, point_prob, server_prob, tuning)

    fold_report = oof["fold_report"].copy()
    fold_report["selected_action_ngram_weight"] = tuning.action_ngram_weight
    fold_report["selected_point_ngram_weight"] = tuning.point_ngram_weight
    for name, value in tuning.server_weights.items():
        fold_report[f"selected_server_weight_{name}"] = value
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
                "server_weights": tuning.server_weights,
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
        "features": features,
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "ngram_key_levels": NGRAM_KEY_LEVELS,
        "ngram_alpha": args.ngram_alpha,
        "remaining_classes": REMAINING_CLASSES,
        "selected": {
            "action_ngram_weight": tuning.action_ngram_weight,
            "point_ngram_weight": tuning.point_ngram_weight,
            "server_weights": tuning.server_weights,
            "action_multipliers": tuning.action_multipliers,
            "point_multipliers": tuning.point_multipliers,
            "bins_mode": tuning.bins_mode,
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
