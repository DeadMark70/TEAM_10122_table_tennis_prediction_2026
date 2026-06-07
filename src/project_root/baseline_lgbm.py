"""Leakage-safe LightGBM baseline for the table-tennis rally task.

This script builds prefix samples from complete train rallies, evaluates with
GroupKFold(match), trains full-data models, and writes a submission file.

It intentionally does not use raw player IDs as model features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ACTION_CLASSES = list(range(19))
POINT_CLASSES = list(range(10))
POINT_NONTERMINAL_CLASSES = list(range(1, 10))
LAG_FIELDS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
COUNT_SPECS = {
    "actionId": list(range(19)),
    "pointId": list(range(10)),
    "spinId": list(range(6)),
    "handId": list(range(3)),
    "positionId": list(range(4)),
}


@dataclass(frozen=True)
class ModelBundle:
    action_model: lgb.LGBMClassifier
    terminal_model: lgb.LGBMClassifier
    point_model: lgb.LGBMClassifier
    server_model: lgb.LGBMClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM baseline and create submission.")
    parser.add_argument("--train", default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="test_new.csv", help="Path to test_new.csv")
    parser.add_argument("--submission", default="submission.csv", help="Output submission path")
    parser.add_argument("--cv-report", default="cv_report.csv", help="Output CV report path")
    parser.add_argument("--feature-report", default="feature_report.json", help="Output metadata path")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-lag", type=int, default=6, help="Number of previous strokes used as lag features")
    parser.add_argument("--n-estimators", type=int, default=120, help="LightGBM trees per model")
    parser.add_argument("--skip-cv", action="store_true", help="Train full models only")
    return parser.parse_args()


def validate_raw_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Input contains missing values.")
    if train.duplicated().any() or test.duplicated().any():
        raise ValueError("Input contains duplicated rows.")
    if (train.groupby("rally_uid")["serverGetPoint"].nunique() > 1).any():
        raise ValueError("serverGetPoint is inconsistent within at least one train rally.")
    overlap_match = set(train["match"].unique()) & set(test["match"].unique())
    overlap_rally = set(train["rally_uid"].unique()) & set(test["rally_uid"].unique())
    if overlap_match:
        raise ValueError(f"Train/test match overlap detected: {len(overlap_match)}")
    if overlap_rally:
        raise ValueError(f"Train/test rally_uid overlap detected: {len(overlap_rally)}")


def add_role_and_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["rally_uid", "strikeNumber"]).copy()
    first = (
        df.groupby("rally_uid", sort=False)
        .head(1)[["rally_uid", "gamePlayerId", "gamePlayerOtherId"]]
        .rename(columns={"gamePlayerId": "server_id", "gamePlayerOtherId": "receiver_id"})
    )
    df = df.merge(first, on="rally_uid", how="left")

    is_server_hitter = df["gamePlayerId"].eq(df["server_id"])
    df["is_server_hitter"] = is_server_hitter.astype(np.int8)
    df["serverScore"] = np.where(is_server_hitter, df["scoreSelf"], df["scoreOther"]).astype(np.int16)
    df["receiverScore"] = np.where(is_server_hitter, df["scoreOther"], df["scoreSelf"]).astype(np.int16)
    df["serverScoreDiff"] = (df["serverScore"] - df["receiverScore"]).astype(np.int16)
    df["scoreTotal"] = (df["serverScore"] + df["receiverScore"]).astype(np.int16)
    return df


def _empty_counts() -> dict[str, dict[int, int]]:
    return {field: {value: 0 for value in values} for field, values in COUNT_SPECS.items()}


def _increment_counts(counts: dict[str, dict[int, int]], row: pd.Series) -> None:
    for field, values in COUNT_SPECS.items():
        value = int(row[field])
        if value not in counts[field]:
            # Defensive guard for unexpected categories; known competition IDs
            # are still used as fixed feature columns.
            counts[field][value] = 0
        counts[field][value] += 1


def _base_prefix_features(
    group: pd.DataFrame,
    t_index: int,
    max_lag: int,
    counts: dict[str, dict[int, int]],
    nunique_counts: dict[str, int],
) -> dict[str, int | float]:
    """Build features using rows 0..t_index inclusive only."""
    current = group.iloc[t_index]
    prefix_len = int(current["strikeNumber"])

    feats: dict[str, int | float] = {
        "sex": int(current["sex"]),
        "numberGame": int(current["numberGame"]),
        "rally_id": int(current["rally_id"]),
        "prefix_len": prefix_len,
        "prefix_len_is_odd": int(prefix_len % 2 == 1),
        "next_hitter_is_server": int((prefix_len + 1) % 2 == 1),
        "next_strikeId_rule": 2 if prefix_len == 1 else 4,
        "is_server_hitter": int(current["is_server_hitter"]),
        "serverScore": int(current["serverScore"]),
        "receiverScore": int(current["receiverScore"]),
        "serverScoreDiff": int(current["serverScoreDiff"]),
        "scoreTotal": int(current["scoreTotal"]),
    }

    for lag in range(max_lag):
        idx = t_index - lag
        has_lag = idx >= 0
        feats[f"lag{lag}_exists"] = int(has_lag)
        for field in LAG_FIELDS:
            feats[f"lag{lag}_{field}"] = int(group.iloc[idx][field]) if has_lag else -1

    for field, values in COUNT_SPECS.items():
        for value in values:
            feats[f"count_{field}_{value}"] = int(counts[field].get(value, 0))
        feats[f"nunique_{field}"] = int(nunique_counts[field])

    if prefix_len >= 2:
        last = group.iloc[t_index]
        prev = group.iloc[t_index - 1]
        feats["last_action_same_as_prev"] = int(last["actionId"] == prev["actionId"])
        feats["last_point_same_as_prev"] = int(last["pointId"] == prev["pointId"])
        feats["last_hand_same_as_prev"] = int(last["handId"] == prev["handId"])
    else:
        feats["last_action_same_as_prev"] = 0
        feats["last_point_same_as_prev"] = 0
        feats["last_hand_same_as_prev"] = 0

    return feats


def build_train_prefix_table(train: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    for _, group in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        rally_len = len(group)
        if rally_len < 2:
            continue
        final_strike = int(group.iloc[-1]["strikeNumber"])
        final_parity_even = int(final_strike % 2 == 0)
        server_get_point = int(group.iloc[0]["serverGetPoint"])
        counts = _empty_counts()
        seen: dict[str, set[int]] = {field: set() for field in COUNT_SPECS}
        for t_index in range(rally_len - 1):
            _increment_counts(counts, group.iloc[t_index])
            for field in COUNT_SPECS:
                seen[field].add(int(group.iloc[t_index][field]))
            nxt = group.iloc[t_index + 1]
            feats = _base_prefix_features(
                group,
                t_index,
                max_lag,
                counts,
                {field: len(values) for field, values in seen.items()},
            )
            feats.update(
                {
                    "rally_uid": int(group.iloc[0]["rally_uid"]),
                    "match": int(group.iloc[0]["match"]),
                    "next_actionId": int(nxt["actionId"]),
                    "next_pointId": int(nxt["pointId"]),
                    "next_is_terminal": int(t_index + 1 == rally_len - 1),
                    "serverGetPoint": server_get_point,
                    "remaining_len": int(rally_len - (t_index + 1)),
                    "final_parity_even": final_parity_even,
                    "num_prefixes_in_rally": int(rally_len - 1),
                }
            )
            rows.append(feats)
    return pd.DataFrame(rows)


def build_test_prefix_table(test: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    for _, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        counts = _empty_counts()
        seen: dict[str, set[int]] = {field: set() for field in COUNT_SPECS}
        for row_idx in range(len(group)):
            _increment_counts(counts, group.iloc[row_idx])
            for field in COUNT_SPECS:
                seen[field].add(int(group.iloc[row_idx][field]))
        feats = _base_prefix_features(
            group,
            len(group) - 1,
            max_lag,
            counts,
            {field: len(values) for field, values in seen.items()},
        )
        feats.update({"rally_uid": int(group.iloc[0]["rally_uid"]), "match": int(group.iloc[0]["match"])})
        rows.append(feats)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
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
        "final_parity_even",
        "num_prefixes_in_rally",
    }
    cols = [c for c in df.columns if c not in forbidden]
    leaked = [c for c in cols if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player feature leakage detected: {leaked}")
    return cols


def class_weight_sample(y: pd.Series, beta: float = 0.25) -> np.ndarray:
    counts = y.value_counts().to_dict()
    weights = y.map(lambda cls: float(counts[int(cls)]) ** (-beta)).to_numpy(dtype=float)
    return weights / np.mean(weights)


def binary_balance_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    weights = y.map(lambda cls: 1.0 / float(counts[int(cls)])).to_numpy(dtype=float)
    return weights / np.mean(weights)


def make_lgbm(objective: str, n_estimators: int, seed: int, num_class: int | None = None) -> lgb.LGBMClassifier:
    params: dict[str, int | float | str] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.045,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if num_class is not None:
        params["num_class"] = num_class
    return lgb.LGBMClassifier(**params)


def fit_bundle(train_df: pd.DataFrame, features: list[str], n_estimators: int, seed: int) -> ModelBundle:
    x = train_df[features]

    action_model = make_lgbm("multiclass", n_estimators, seed, num_class=len(ACTION_CLASSES))
    action_model.fit(x, train_df["next_actionId"], sample_weight=class_weight_sample(train_df["next_actionId"]))

    terminal_model = make_lgbm("binary", n_estimators, seed + 1)
    terminal_model.fit(x, train_df["next_is_terminal"])

    point_train = train_df[train_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
    point_model = make_lgbm("multiclass", n_estimators, seed + 2, num_class=len(POINT_NONTERMINAL_CLASSES))
    point_model.fit(
        point_train[features],
        point_train["next_pointId"],
        sample_weight=class_weight_sample(point_train["next_pointId"]),
    )

    server_model = make_lgbm("binary", n_estimators, seed + 3)
    server_weights = 1.0 / train_df["num_prefixes_in_rally"].to_numpy(dtype=float)
    server_weights = server_weights / np.mean(server_weights)
    server_model.fit(x, train_df["serverGetPoint"], sample_weight=server_weights)

    return ModelBundle(action_model, terminal_model, point_model, server_model)


def aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: Iterable[int]) -> np.ndarray:
    classes = list(classes)
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


def predict_bundle(bundle: ModelBundle, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[features]
    action_prob = aligned_proba(bundle.action_model, x, ACTION_CLASSES)

    terminal_raw = bundle.terminal_model.predict_proba(x)
    terminal_prob = terminal_raw[:, 1] if terminal_raw.ndim == 2 else terminal_raw
    terminal_prob = np.clip(terminal_prob.astype(float), 1e-6, 1.0 - 1e-6)

    point_nonterminal = aligned_proba(bundle.point_model, x, POINT_NONTERMINAL_CLASSES)
    point_prob = np.zeros((len(df), len(POINT_CLASSES)), dtype=float)
    point_prob[:, 0] = terminal_prob
    point_prob[:, 1:] = (1.0 - terminal_prob[:, None]) * point_nonterminal
    point_prob = point_prob / point_prob.sum(axis=1, keepdims=True)

    server_raw = bundle.server_model.predict_proba(x)
    server_prob = server_raw[:, 1] if server_raw.ndim == 2 else server_raw
    server_prob = np.clip(server_prob.astype(float), 1e-6, 1.0 - 1e-6)
    return action_prob, point_prob, server_prob


def sample_validation_prefixes(prefix_df: pd.DataFrame, test_prefix_lengths: np.ndarray, seed: int) -> pd.Index:
    rng = np.random.default_rng(seed)
    chosen_indices: list[int] = []
    by_len = {int(k): group.index.to_numpy() for k, group in prefix_df.groupby("prefix_len")}
    legal_by_rally = prefix_df.groupby("rally_uid")["prefix_len"].max().to_dict()

    for rally_uid, max_legal in legal_by_rally.items():
        sampled_len = int(rng.choice(test_prefix_lengths))
        attempts = 0
        while sampled_len > int(max_legal):
            sampled_len = int(rng.choice(test_prefix_lengths))
            attempts += 1
            if attempts > 1000:
                sampled_len = int(rng.integers(1, int(max_legal) + 1))
                break
        candidates = prefix_df[(prefix_df["rally_uid"].eq(rally_uid)) & (prefix_df["prefix_len"].eq(sampled_len))].index
        if len(candidates) != 1:
            raise RuntimeError(f"Unable to sample legal prefix for rally_uid={rally_uid}, len={sampled_len}")
        chosen_indices.append(int(candidates[0]))

    return pd.Index(chosen_indices)


def evaluate_predictions(valid_df: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, server_prob: np.ndarray) -> dict[str, float]:
    action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]

    action_f1 = f1_score(valid_df["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(valid_df["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    try:
        server_auc = roc_auc_score(valid_df["serverGetPoint"], server_prob)
    except ValueError:
        server_auc = float("nan")
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(overall),
    }


def run_cv(prefix_df: pd.DataFrame, test_prefix_lengths: np.ndarray, features: list[str], args: argparse.Namespace) -> pd.DataFrame:
    fold_rows: list[dict[str, float | int]] = []
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)

    for fold, (train_rally_idx, valid_rally_idx) in enumerate(
        splitter.split(rally_meta, groups=rally_meta["match"]), start=1
    ):
        train_rallies = set(rally_meta.iloc[train_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_rally_idx]["rally_uid"])
        train_matches = set(rally_meta.iloc[train_rally_idx]["match"])
        valid_matches = set(rally_meta.iloc[valid_rally_idx]["match"])
        if train_matches & valid_matches:
            raise RuntimeError("GroupKFold leakage: train/valid match overlap.")

        fold_train = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
        valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_prefix_lengths, args.seed + fold)
        fold_valid = valid_pool.loc[sampled_idx].copy()

        bundle = fit_bundle(fold_train, features, args.n_estimators, args.seed + fold * 10)
        action_prob, point_prob, server_prob = predict_bundle(bundle, fold_valid, features)
        metrics = evaluate_predictions(fold_valid, action_prob, point_prob, server_prob)
        metrics.update(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "valid_rows": len(fold_valid),
                "valid_rallies": len(valid_rallies),
                "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
            }
        )
        fold_rows.append(metrics)
        print(
            f"fold {fold}: overall={metrics['overall']:.6f} "
            f"action={metrics['action_macro_f1']:.6f} "
            f"point={metrics['point_macro_f1']:.6f} "
            f"server_auc={metrics['server_auc']:.6f}"
        )

    cv = pd.DataFrame(fold_rows)
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0, "valid_rallies": 0}
    for col in ["action_macro_f1", "point_macro_f1", "server_auc", "overall", "valid_prefix_len_mean"]:
        mean_row[col] = float(cv[col].mean())
    cv = pd.concat([cv, pd.DataFrame([mean_row])], ignore_index=True)
    return cv


def write_submission(bundle: ModelBundle, test_prefix: pd.DataFrame, features: list[str], output_path: Path) -> pd.DataFrame:
    action_prob, point_prob, server_prob = predict_bundle(bundle, test_prefix, features)
    action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)].astype(int)
    point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)].astype(int)
    submission = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred,
            "pointId": point_pred,
            "serverGetPoint": np.round(server_prob, 8),
        }
    )
    if len(submission) != test_prefix["rally_uid"].nunique():
        raise ValueError("Submission row count does not match test rally count.")
    if not submission["actionId"].between(0, 18).all():
        raise ValueError("Invalid actionId in submission.")
    if not submission["pointId"].between(0, 9).all():
        raise ValueError("Invalid pointId in submission.")
    if not submission["serverGetPoint"].between(0, 1).all():
        raise ValueError("Invalid serverGetPoint probability in submission.")
    submission.to_csv(output_path, index=False, float_format="%.8f")
    return submission


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building prefix tables...")
    prefix_df = build_train_prefix_table(train, args.max_lag)
    test_prefix = build_test_prefix_table(test, args.max_lag)
    features = feature_columns(prefix_df)

    missing_in_test = [c for c in features if c not in test_prefix.columns]
    if missing_in_test:
        raise ValueError(f"Features missing in test prefix table: {missing_in_test}")
    test_prefix = test_prefix[["rally_uid", "match"] + features]

    test_prefix_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    print(f"train prefix rows: {len(prefix_df):,}")
    print(f"test prediction rows: {len(test_prefix):,}")
    print(f"feature count: {len(features)}")

    if not args.skip_cv:
        cv = run_cv(prefix_df, test_prefix_lengths, features, args)
        cv.to_csv(args.cv_report, index=False)
        print(f"wrote {args.cv_report}")

    print("training full-data models...")
    bundle = fit_bundle(prefix_df, features, args.n_estimators, args.seed)
    submission = write_submission(bundle, test_prefix, features, Path(args.submission))
    print(f"wrote {args.submission} ({len(submission):,} rows)")

    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_df)),
        "test_prediction_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "features": features,
        "excluded_raw_player_features": ["gamePlayerId", "gamePlayerOtherId", "server_id", "receiver_id"],
        "max_lag": int(args.max_lag),
        "n_estimators": int(args.n_estimators),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
