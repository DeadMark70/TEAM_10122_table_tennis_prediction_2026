"""V9 baseline: V3 plus fold-safe smoothed player profile features.

This version still does not feed raw player IDs to LightGBM. Player IDs are
used only as keys to build smoothed historical profile rates inside each
fold. Validation profiles are computed from fold-train only; train profiles
use leave-one-row-out rates to avoid leaking the row's own target into its
features.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    fit_bundle,
    predict_bundle,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import blend_probs, fit_ngram_bundle, predict_ngram_bundle
from baseline_v3 import (
    REMAINING_CLASSES,
    V3Tuning,
    add_remaining_bucket,
    apply_segmented_multipliers,
    apply_segmented_multipliers_to_test,
    average_lgbm_predictions,
    evaluate_v3,
    fit_server_aux,
    predict_server_aux,
    prefix_len_report,
    search_server_blend,
    tune_v3,
)


PROFILE_SETS = ["none", "hitter", "hitter_target"]
PROFILE_ID_COLS = ["current_hitter_id", "next_hitter_id", "target_receiver_id"]
PROFILE_TARGET_COLS = ["next_spinId", "next_strengthId", "next_point_depth", "next_point_side"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V9 player-profile baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_v9.csv")
    parser.add_argument("--cv-report", default="cv_report_v9.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v9.csv")
    parser.add_argument("--cold-player-report", default="cold_player_report_v9.csv")
    parser.add_argument("--feature-report", default="feature_report_v9.json")
    parser.add_argument("--oof-proba", default="oof_proba_v9.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--profile-k", type=float, default=50.0)
    parser.add_argument("--profile-sets", nargs="+", choices=PROFILE_SETS, default=PROFILE_SETS)
    return parser.parse_args()


def point_depth(point_id: int) -> int:
    if point_id <= 0:
        return -1
    return (point_id - 1) // 3


def point_side(point_id: int) -> int:
    if point_id <= 0:
        return -1
    return (point_id - 1) % 3


def build_train_player_context(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for _, group in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        for t_index in range(len(group) - 1):
            current = group.iloc[t_index]
            nxt = group.iloc[t_index + 1]
            next_point = int(nxt["pointId"])
            rows.append(
                {
                    "rally_uid": int(current["rally_uid"]),
                    "prefix_len": int(current["strikeNumber"]),
                    "current_hitter_id": int(current["gamePlayerId"]),
                    "next_hitter_id": int(nxt["gamePlayerId"]),
                    "target_receiver_id": int(nxt["gamePlayerOtherId"]),
                    "next_spinId": int(nxt["spinId"]),
                    "next_strengthId": int(nxt["strengthId"]),
                    "next_point_depth": point_depth(next_point),
                    "next_point_side": point_side(next_point),
                }
            )
    return pd.DataFrame(rows)


def build_test_player_context(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for _, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        current = group.iloc[-1]
        prefix_len = int(current["strikeNumber"])
        server_id = int(current["server_id"])
        receiver_id = int(current["receiver_id"])
        next_hitter_id = server_id if (prefix_len + 1) % 2 == 1 else receiver_id
        target_receiver_id = receiver_id if next_hitter_id == server_id else server_id
        rows.append(
            {
                "rally_uid": int(current["rally_uid"]),
                "prefix_len": prefix_len,
                "current_hitter_id": int(current["gamePlayerId"]),
                "next_hitter_id": next_hitter_id,
                "target_receiver_id": target_receiver_id,
            }
        )
    return pd.DataFrame(rows)


def attach_player_context(prefix_df: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    out = prefix_df.merge(context, on=["rally_uid", "prefix_len"], how="left", validate="one_to_one")
    missing = [c for c in PROFILE_ID_COLS if out[c].isna().any()]
    if missing:
        raise ValueError(f"Missing player context columns after merge: {missing}")
    for col in PROFILE_ID_COLS + [c for c in PROFILE_TARGET_COLS if c in out.columns]:
        out[col] = out[col].astype(int)
    return out


def _profile_frame(
    source: pd.DataFrame,
    apply: pd.DataFrame,
    player_col: str,
    target_col: str,
    classes: list[int],
    prefix: str,
    k: float,
    loo: bool,
) -> pd.DataFrame:
    global_counts = source[target_col].value_counts().reindex(classes, fill_value=0).astype(float)
    if float(global_counts.sum()) <= 0:
        global_prior = np.full(len(classes), 1.0 / len(classes), dtype=float)
    else:
        global_prior = (global_counts / float(global_counts.sum())).to_numpy(dtype=float)

    ids = apply[player_col].astype(int).to_numpy()
    total_by_player = source.groupby(player_col).size().astype(float)
    total = total_by_player.reindex(ids, fill_value=0).to_numpy(dtype=float)
    counts_by_player = pd.crosstab(source[player_col], source[target_col]).reindex(columns=classes, fill_value=0)
    counts_matrix = counts_by_player.reindex(ids, fill_value=0).to_numpy(dtype=float).copy()

    if loo:
        total = np.maximum(total - 1.0, 0.0)
        class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
        apply_target = apply[target_col].astype(int).to_numpy()
        row_idx = np.arange(len(apply_target))
        for cls, cls_idx in class_to_idx.items():
            mask = apply_target == cls
            if mask.any():
                counts_matrix[row_idx[mask], cls_idx] -= 1.0
        counts_matrix = np.maximum(counts_matrix, 0.0)

    out: dict[str, np.ndarray] = {
        f"{prefix}_count": total,
        f"{prefix}_is_seen": (total > 0).astype(np.int8),
        f"{prefix}_log_count": np.log1p(total),
    }
    denom = total + float(k)
    denom = np.where(denom <= 0, float(k), denom)
    for cls_idx, cls in enumerate(classes):
        counts = counts_matrix[:, cls_idx]
        out[f"{prefix}_rate_{cls}"] = (counts + float(k) * global_prior[cls_idx]) / denom
    return pd.DataFrame(out, index=apply.index)


def add_profile_features(
    train_profile_source: pd.DataFrame,
    apply_df: pd.DataFrame,
    profile_set: str,
    k: float,
    loo: bool,
) -> pd.DataFrame:
    if profile_set == "none":
        return apply_df.copy()

    out = apply_df.copy()
    pieces: list[pd.DataFrame] = []
    if profile_set in {"hitter", "hitter_target"}:
        pieces.extend(
            [
                _profile_frame(
                    train_profile_source,
                    apply_df,
                    "next_hitter_id",
                    "next_actionId",
                    ACTION_CLASSES,
                    "next_hitter_action",
                    k,
                    loo,
                ),
                _profile_frame(
                    train_profile_source,
                    apply_df,
                    "next_hitter_id",
                    "next_pointId",
                    POINT_CLASSES,
                    "next_hitter_point",
                    k,
                    loo,
                ),
                _profile_frame(
                    train_profile_source,
                    apply_df,
                    "next_hitter_id",
                    "next_spinId",
                    list(range(6)),
                    "next_hitter_spin",
                    k,
                    loo,
                ),
                _profile_frame(
                    train_profile_source,
                    apply_df,
                    "next_hitter_id",
                    "next_strengthId",
                    list(range(4)),
                    "next_hitter_strength",
                    k,
                    loo,
                ),
            ]
        )
    if profile_set == "hitter_target":
        nonterminal_source = train_profile_source[train_profile_source["next_pointId"].gt(0)].copy()
        nonterminal_apply = apply_df.copy()
        pieces.extend(
            [
                _profile_frame(
                    train_profile_source,
                    apply_df,
                    "target_receiver_id",
                    "next_pointId",
                    POINT_CLASSES,
                    "target_receiver_point",
                    k,
                    loo,
                ),
                _profile_frame(
                    nonterminal_source,
                    nonterminal_apply,
                    "target_receiver_id",
                    "next_point_depth",
                    [0, 1, 2],
                    "target_receiver_depth",
                    k,
                    loo,
                ),
                _profile_frame(
                    nonterminal_source,
                    nonterminal_apply,
                    "target_receiver_id",
                    "next_point_side",
                    [0, 1, 2],
                    "target_receiver_side",
                    k,
                    loo,
                ),
            ]
        )
    if pieces:
        out = pd.concat([out] + pieces, axis=1)
    return out


def v9_feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden = {
        "rally_uid",
        "match",
        "server_id",
        "receiver_id",
        "gamePlayerId",
        "gamePlayerOtherId",
        "scoreSelf",
        "scoreOther",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "remaining_len",
        "remaining_len_bucket",
        "final_parity_even",
        "num_prefixes_in_rally",
        *PROFILE_ID_COLS,
        *PROFILE_TARGET_COLS,
    }
    cols = [c for c in df.columns if c not in forbidden]
    leaked = [
        c
        for c in cols
        if "PlayerId" in c
        or c in {"server_id", "receiver_id"}
        or c in PROFILE_ID_COLS
        or c.endswith("_player_id")
    ]
    if leaked:
        raise ValueError(f"Raw player feature leakage detected: {leaked}")
    return cols


def selected_probabilities(oof: dict[str, object], tuning: V3Tuning) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_prob = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)
    point_prob = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server_prob = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )
    return action_prob, point_prob, server_prob


def run_cv_variant(
    prefix_df: pd.DataFrame,
    test_prefix_lengths: np.ndarray,
    profile_set: str,
    base_features: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
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
    fold_rows: list[dict[str, float | int | str]] = []
    feature_count = 0
    profile_features: list[str] = []

    for fold, (train_rally_idx, valid_rally_idx) in enumerate(
        splitter.split(rally_meta, groups=rally_meta["match"]), start=1
    ):
        train_rallies = set(rally_meta.iloc[train_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_rally_idx]["rally_uid"])
        if set(rally_meta.iloc[train_rally_idx]["match"]) & set(rally_meta.iloc[valid_rally_idx]["match"]):
            raise RuntimeError("GroupKFold leakage: train/valid match overlap.")

        fold_train_raw = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
        valid_pool_raw = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool_raw, test_prefix_lengths, args.seeds[0] + fold)
        fold_valid_raw = valid_pool_raw.loc[sampled_idx].copy()

        fold_train = add_profile_features(fold_train_raw, fold_train_raw, profile_set, args.profile_k, loo=True)
        fold_valid = add_profile_features(fold_train_raw, fold_valid_raw, profile_set, args.profile_k, loo=False)
        features = v9_feature_columns(fold_train)
        missing = [c for c in features if c not in fold_valid.columns]
        if missing:
            raise ValueError(f"Fold {fold} valid missing profile features: {missing[:5]}")
        fold_valid = fold_valid[fold_train.columns.intersection(fold_valid.columns).tolist()]
        profile_features = [c for c in features if c not in base_features]
        feature_count = len(features)

        lgbm_action, lgbm_point, lgbm_server, parity_server, remaining_server = average_lgbm_predictions(
            fold_train, fold_valid, features, args.n_estimators, args.seeds, fold
        )
        ngram_bundle = fit_ngram_bundle(fold_train_raw, args.ngram_alpha)
        ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, fold_valid_raw)

        base_lgbm_overall = 0.4 * f1_score(
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
        fold_row = {
            "variant": profile_set,
            "fold": fold,
            "train_rows": len(fold_train),
            "valid_rows": len(fold_valid),
            "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
            "feature_count": feature_count,
            "profile_feature_count": len(profile_features),
            "base_lgbm_overall": float(base_lgbm_overall),
            "direct_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], lgbm_server)),
            "ngram_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], ngram_server)),
            "parity_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], parity_server)),
            "remaining_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], remaining_server)),
        }
        fold_rows.append(fold_row)
        print(
            f"{profile_set} fold {fold}: base_lgbm={base_lgbm_overall:.6f} "
            f"features={feature_count} profile_features={len(profile_features)}"
        )

        keep_cols = [
            "rally_uid",
            "match",
            "prefix_len",
            "next_actionId",
            "next_pointId",
            "serverGetPoint",
            "next_hitter_action_count",
            "target_receiver_point_count",
        ]
        for col in keep_cols:
            if col not in fold_valid.columns:
                fold_valid[col] = 0.0
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
        "variant": profile_set,
        "valid_meta": pd.concat(parts["valid_meta"], ignore_index=True),
        "fold_report": pd.DataFrame(fold_rows),
        "profile_features": profile_features,
        "feature_count": feature_count,
    }
    for key in ["lgbm_action", "lgbm_point", "ngram_action", "ngram_point"]:
        oof[key] = np.vstack(parts[key])
    for key in ["lgbm_server", "ngram_server", "parity_server", "remaining_server"]:
        oof[key] = np.concatenate(parts[key])

    fold_report = oof["fold_report"]
    mean_row = {
        "variant": profile_set,
        "fold": 0,
        "train_rows": 0,
        "valid_rows": 0,
        "feature_count": feature_count,
        "profile_feature_count": len(profile_features),
    }
    for col in fold_report.columns:
        if col not in mean_row and col not in {"variant", "fold"}:
            mean_row[col] = float(fold_report[col].mean())
    oof["fold_report"] = pd.concat([fold_report, pd.DataFrame([mean_row])], ignore_index=True)
    return oof


def cold_player_report(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: V3Tuning,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    groups = [
        ("all", np.ones(len(meta), dtype=bool)),
        ("next_hitter_seen", meta["next_hitter_action_count"].gt(0).to_numpy()),
        ("next_hitter_unseen", meta["next_hitter_action_count"].le(0).to_numpy()),
        ("target_receiver_seen", meta["target_receiver_point_count"].gt(0).to_numpy()),
        ("target_receiver_unseen", meta["target_receiver_point_count"].le(0).to_numpy()),
        ("prefix_le2", meta["prefix_len"].le(2).to_numpy()),
        ("prefix_ge3", meta["prefix_len"].ge(3).to_numpy()),
    ]
    for label, mask in groups:
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        try:
            metrics = evaluate_v3(
                meta.iloc[idx].reset_index(drop=True),
                action_prob[idx],
                point_prob[idx],
                server_prob[idx],
                tuning.action_multipliers,
                tuning.point_multipliers,
                tuning.bins_mode,
            )
        except ValueError:
            continue
        metrics.update({"group": label, "count": int(len(idx))})
        rows.append(metrics)
    return pd.DataFrame(rows)


def full_predict_variant(
    prefix_df: pd.DataFrame,
    test_prefix: pd.DataFrame,
    profile_set: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    train_full = add_profile_features(prefix_df, prefix_df, profile_set, args.profile_k, loo=True)
    test_full = add_profile_features(prefix_df, test_prefix, profile_set, args.profile_k, loo=False)
    features = v9_feature_columns(train_full)
    missing = [c for c in features if c not in test_full.columns]
    if missing:
        raise ValueError(f"Test missing profile features: {missing[:5]}")
    action_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    server_parts: list[np.ndarray] = []
    parity_parts: list[np.ndarray] = []
    remaining_parts: list[np.ndarray] = []
    for seed in args.seeds:
        bundle = fit_bundle(train_full, features, args.n_estimators, seed)
        action_prob, point_prob, server_prob = predict_bundle(bundle, test_full, features)
        aux = fit_server_aux(train_full, features, args.n_estimators, seed)
        parity_prob, remaining_prob = predict_server_aux(aux, test_full, features)
        action_parts.append(action_prob)
        point_parts.append(point_prob)
        server_parts.append(server_prob)
        parity_parts.append(parity_prob)
        remaining_parts.append(remaining_prob)
    ngram_bundle = fit_ngram_bundle(prefix_df, args.ngram_alpha)
    ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, test_prefix)
    return (
        {
            "lgbm_action": np.mean(action_parts, axis=0),
            "lgbm_point": np.mean(point_parts, axis=0),
            "lgbm_server": np.mean(server_parts, axis=0),
            "parity_server": np.mean(parity_parts, axis=0),
            "remaining_server": np.mean(remaining_parts, axis=0),
            "ngram_action": ngram_action,
            "ngram_point": ngram_point,
            "ngram_server": ngram_server,
        },
        features,
        [c for c in features if c not in v9_feature_columns(prefix_df)],
    )


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
    prefix_df = attach_player_context(prefix_df, build_train_player_context(train))
    test_prefix = attach_player_context(test_prefix, build_test_player_context(test))
    base_features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    print(f"train prefix rows: {len(prefix_df):,}")
    print(f"test prediction rows: {len(test_prefix):,}")
    print(f"base feature count: {len(base_features)}")
    print(f"profile sets: {args.profile_sets}")

    results: list[dict[str, object]] = []
    oofs: dict[str, dict[str, object]] = {}
    tunings: dict[str, V3Tuning] = {}
    for profile_set in args.profile_sets:
        print(f"\n=== variant: {profile_set} ===")
        oof = run_cv_variant(prefix_df, test_prefix_lengths, profile_set, base_features, args)
        tuning = tune_v3(oof, args.multiplier_bins)
        action_prob, point_prob, server_prob = selected_probabilities(oof, tuning)
        variant_prefix_report = prefix_len_report(oof["valid_meta"], action_prob, point_prob, server_prob, tuning)
        variant_cold_report = cold_player_report(oof["valid_meta"], action_prob, point_prob, server_prob, tuning)

        fold_report = oof["fold_report"].copy()
        fold_report["selected_action_ngram_weight"] = tuning.action_ngram_weight
        fold_report["selected_point_ngram_weight"] = tuning.point_ngram_weight
        for name, value in tuning.server_weights.items():
            fold_report[f"selected_server_weight_{name}"] = value
        for name, value in tuning.metrics.items():
            fold_report[f"selected_{name}"] = value
        fold_report["profile_k"] = args.profile_k
        results.append(
            {
                "variant": profile_set,
                "tuning": tuning,
                "oof": oof,
                "fold_report": fold_report,
                "prefix_report": variant_prefix_report.assign(variant=profile_set),
                "cold_report": variant_cold_report.assign(variant=profile_set),
                **tuning.metrics,
            }
        )
        oofs[profile_set] = oof
        tunings[profile_set] = tuning
        print(
            f"selected {profile_set}: overall={tuning.metrics['overall']:.6f} "
            f"action={tuning.metrics['action_macro_f1']:.6f} "
            f"point={tuning.metrics['point_macro_f1']:.6f} "
            f"server={tuning.metrics['server_auc']:.6f}"
        )

    best = max(results, key=lambda item: float(item["overall"]))
    best_variant = str(best["variant"])
    best_tuning: V3Tuning = best["tuning"]  # type: ignore[assignment]
    print(f"\nbest variant: {best_variant} overall={best_tuning.metrics['overall']:.6f}")

    cv_report = pd.concat([item["fold_report"] for item in results], ignore_index=True)
    prefix_report = pd.concat([item["prefix_report"] for item in results], ignore_index=True)
    cold_report = pd.concat([item["cold_report"] for item in results], ignore_index=True)
    cv_report.to_csv(args.cv_report, index=False)
    prefix_report.to_csv(args.prefix_len_report, index=False)
    cold_report.to_csv(args.cold_player_report, index=False)

    with open(args.oof_proba, "wb") as f:
        pickle.dump({"best_variant": best_variant, "oofs": oofs, "tunings": tunings}, f)

    print("training full-data model for best variant...")
    full_pred, final_features, final_profile_features = full_predict_variant(prefix_df, test_prefix, best_variant, args)
    submission = write_submission(test_prefix, full_pred, best_tuning, Path(args.submission))

    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_df)),
        "test_prediction_rows": int(len(test_prefix)),
        "base_feature_count": int(len(base_features)),
        "final_feature_count": int(len(final_features)),
        "final_profile_feature_count": int(len(final_profile_features)),
        "best_variant": best_variant,
        "profile_sets": args.profile_sets,
        "profile_k": float(args.profile_k),
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "remaining_classes": REMAINING_CLASSES,
        "selected": {
            "action_ngram_weight": best_tuning.action_ngram_weight,
            "point_ngram_weight": best_tuning.point_ngram_weight,
            "server_weights": best_tuning.server_weights,
            "action_multipliers": best_tuning.action_multipliers,
            "point_multipliers": best_tuning.point_multipliers,
            "bins_mode": best_tuning.bins_mode,
            "metrics": best_tuning.metrics,
        },
        "variant_metrics": {
            str(item["variant"]): {
                "action_macro_f1": float(item["action_macro_f1"]),
                "point_macro_f1": float(item["point_macro_f1"]),
                "server_auc": float(item["server_auc"]),
                "overall": float(item["overall"]),
            }
            for item in results
        },
        "raw_player_feature_policy": "IDs used only as fold-safe profile keys; raw IDs excluded from model features.",
        "features": final_features,
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.cold_player_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
