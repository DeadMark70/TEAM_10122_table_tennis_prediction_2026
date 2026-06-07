"""V2 baseline: LightGBM + n-gram ensemble + OOF tuning.

This version keeps the leakage controls from baseline_lgbm.py and adds:
- out-of-fold probability collection
- backoff n-gram probability models
- CV search for LGBM/n-gram blend weights
- greedy class multiplier tuning for actionId and pointId
- prefix-length diagnostic report
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
    build_test_prefix_table,
    build_train_prefix_table,
    evaluate_predictions,
    feature_columns,
    fit_bundle,
    predict_bundle,
    sample_validation_prefixes,
    validate_raw_data,
)


NGRAM_KEY_LEVELS = [
    ["prefix_len", "lag0_actionId", "lag0_pointId", "lag0_spinId", "sex"],
    ["prefix_len", "lag0_actionId", "lag0_pointId", "sex"],
    ["prefix_len", "lag0_actionId", "sex"],
    ["prefix_len", "sex"],
    ["lag0_actionId", "sex"],
]


@dataclass
class TuningResult:
    action_ngram_weight: float
    point_ngram_weight: float
    server_ngram_weight: float
    action_multipliers: np.ndarray
    point_multipliers: np.ndarray
    metrics: dict[str, float]


class BackoffNgram:
    def __init__(self, classes: list[int], key_levels: list[list[str]], alpha: float = 20.0) -> None:
        self.classes = list(classes)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.key_levels = key_levels
        self.alpha = float(alpha)
        self.global_prior: np.ndarray | None = None
        self.tables: list[dict[tuple[int, ...], np.ndarray]] = []

    def fit(self, df: pd.DataFrame, target: str, sample_weight: np.ndarray | None = None) -> "BackoffNgram":
        work = df.copy()
        if sample_weight is None:
            work["_w"] = 1.0
        else:
            work["_w"] = sample_weight.astype(float)

        global_counts = np.zeros(len(self.classes), dtype=float)
        for cls, total in work.groupby(target)["_w"].sum().items():
            cls = int(cls)
            if cls in self.class_to_index:
                global_counts[self.class_to_index[cls]] += float(total)
        if global_counts.sum() <= 0:
            global_counts[:] = 1.0
        self.global_prior = global_counts / global_counts.sum()

        self.tables = []
        for key_cols in self.key_levels:
            table: dict[tuple[int, ...], np.ndarray] = {}
            grouped = work.groupby(key_cols + [target], dropna=False)["_w"].sum().reset_index()
            for _, row in grouped.iterrows():
                key = tuple(int(row[col]) for col in key_cols)
                cls = int(row[target])
                if cls not in self.class_to_index:
                    continue
                if key not in table:
                    table[key] = np.zeros(len(self.classes), dtype=float)
                table[key][self.class_to_index[cls]] += float(row["_w"])
            self.tables.append(table)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.global_prior is None:
            raise RuntimeError("BackoffNgram must be fit before prediction.")
        out = np.zeros((len(df), len(self.classes)), dtype=float)
        rows = df.reset_index(drop=True)
        for row_idx, row in rows.iterrows():
            counts = None
            for key_cols, table in zip(self.key_levels, self.tables):
                key = tuple(int(row[col]) for col in key_cols)
                if key in table:
                    counts = table[key]
                    break
            if counts is None:
                out[row_idx] = self.global_prior
            else:
                out[row_idx] = (counts + self.alpha * self.global_prior) / (counts.sum() + self.alpha)
        return out


@dataclass
class NgramBundle:
    action_model: BackoffNgram
    point_model: BackoffNgram
    server_model: BackoffNgram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 LightGBM+n-gram baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_v2.csv")
    parser.add_argument("--cv-report", default="cv_report_v2.csv")
    parser.add_argument("--prefix-report", default="prefix_len_report_v2.csv")
    parser.add_argument("--feature-report", default="feature_report_v2.json")
    parser.add_argument("--oof-proba", default="oof_proba_v2.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    return parser.parse_args()


def fit_ngram_bundle(train_df: pd.DataFrame, alpha: float) -> NgramBundle:
    server_weights = 1.0 / train_df["num_prefixes_in_rally"].to_numpy(dtype=float)
    server_weights = server_weights / np.mean(server_weights)
    return NgramBundle(
        action_model=BackoffNgram(ACTION_CLASSES, NGRAM_KEY_LEVELS, alpha).fit(train_df, "next_actionId"),
        point_model=BackoffNgram(POINT_CLASSES, NGRAM_KEY_LEVELS, alpha).fit(train_df, "next_pointId"),
        server_model=BackoffNgram([0, 1], NGRAM_KEY_LEVELS, alpha).fit(
            train_df, "serverGetPoint", sample_weight=server_weights
        ),
    )


def predict_ngram_bundle(bundle: NgramBundle, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_prob = bundle.action_model.predict_proba(df)
    point_prob = bundle.point_model.predict_proba(df)
    server_prob = bundle.server_model.predict_proba(df)[:, 1]
    return action_prob, point_prob, server_prob


def blend_probs(lgbm_prob: np.ndarray, ngram_prob: np.ndarray, ngram_weight: float) -> np.ndarray:
    prob = (1.0 - ngram_weight) * lgbm_prob + ngram_weight * ngram_prob
    return prob / prob.sum(axis=1, keepdims=True)


def score_action(y: np.ndarray, prob: np.ndarray, multipliers: np.ndarray | None = None) -> float:
    adjusted = prob if multipliers is None else prob * multipliers[None, :]
    pred = np.asarray(ACTION_CLASSES)[np.argmax(adjusted, axis=1)]
    return float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0))


def score_point(y: np.ndarray, prob: np.ndarray, multipliers: np.ndarray | None = None) -> float:
    adjusted = prob if multipliers is None else prob * multipliers[None, :]
    pred = np.asarray(POINT_CLASSES)[np.argmax(adjusted, axis=1)]
    return float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0))


def greedy_multiplier_search(
    y: np.ndarray,
    prob: np.ndarray,
    classes: list[int],
    scorer_name: str,
    values: list[float] | None = None,
    passes: int = 2,
) -> np.ndarray:
    if values is None:
        values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    multipliers = np.ones(len(classes), dtype=float)

    def metric(m: np.ndarray) -> float:
        if scorer_name == "action":
            return score_action(y, prob, m)
        if scorer_name == "point":
            return score_point(y, prob, m)
        raise ValueError(scorer_name)

    best = metric(multipliers)
    for _ in range(passes):
        improved = False
        for cls_idx in range(len(classes)):
            local_best = best
            local_value = multipliers[cls_idx]
            old_value = multipliers[cls_idx]
            for value in values:
                multipliers[cls_idx] = value
                current = metric(multipliers)
                if current > local_best + 1e-12:
                    local_best = current
                    local_value = value
            multipliers[cls_idx] = local_value
            if local_best > best + 1e-12:
                best = local_best
                improved = True
            else:
                multipliers[cls_idx] = old_value
        if not improved:
            break
    return multipliers


def evaluate_combined(
    valid_meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    action_multipliers: np.ndarray | None = None,
    point_multipliers: np.ndarray | None = None,
) -> dict[str, float]:
    action_adjusted = action_prob if action_multipliers is None else action_prob * action_multipliers[None, :]
    point_adjusted = point_prob if point_multipliers is None else point_prob * point_multipliers[None, :]
    action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_adjusted, axis=1)]
    point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_adjusted, axis=1)]
    action_f1 = f1_score(
        valid_meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0
    )
    point_f1 = f1_score(
        valid_meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0
    )
    server_auc = roc_auc_score(valid_meta["serverGetPoint"], server_prob)
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(overall),
    }


def tune_oof(
    valid_meta: pd.DataFrame,
    lgbm_action: np.ndarray,
    lgbm_point: np.ndarray,
    lgbm_server: np.ndarray,
    ngram_action: np.ndarray,
    ngram_point: np.ndarray,
    ngram_server: np.ndarray,
) -> TuningResult:
    weights = [0.0, 0.1, 0.2, 0.3, 0.4]
    server_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_action = valid_meta["next_actionId"].to_numpy()
    y_point = valid_meta["next_pointId"].to_numpy()
    y_server = valid_meta["serverGetPoint"].to_numpy()

    best_action_w = max(
        weights, key=lambda w: score_action(y_action, blend_probs(lgbm_action, ngram_action, w))
    )
    best_point_w = max(weights, key=lambda w: score_point(y_point, blend_probs(lgbm_point, ngram_point, w)))
    best_server_w = max(
        server_weights,
        key=lambda w: roc_auc_score(y_server, (1.0 - w) * lgbm_server + w * ngram_server),
    )

    action_prob = blend_probs(lgbm_action, ngram_action, best_action_w)
    point_prob = blend_probs(lgbm_point, ngram_point, best_point_w)
    server_prob = (1.0 - best_server_w) * lgbm_server + best_server_w * ngram_server

    action_multipliers = greedy_multiplier_search(y_action, action_prob, ACTION_CLASSES, "action")
    point_multipliers = greedy_multiplier_search(y_point, point_prob, POINT_CLASSES, "point")
    metrics = evaluate_combined(
        valid_meta, action_prob, point_prob, server_prob, action_multipliers, point_multipliers
    )
    return TuningResult(
        action_ngram_weight=float(best_action_w),
        point_ngram_weight=float(best_point_w),
        server_ngram_weight=float(best_server_w),
        action_multipliers=action_multipliers,
        point_multipliers=point_multipliers,
        metrics=metrics,
    )


def prefix_len_report(
    valid_meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    action_multipliers: np.ndarray,
    point_multipliers: np.ndarray,
) -> pd.DataFrame:
    bins = [
        ("1", valid_meta["prefix_len"].eq(1)),
        ("2", valid_meta["prefix_len"].eq(2)),
        ("3", valid_meta["prefix_len"].eq(3)),
        ("4-6", valid_meta["prefix_len"].between(4, 6)),
        ("7+", valid_meta["prefix_len"].ge(7)),
    ]
    rows: list[dict[str, float | int | str]] = []
    for label, mask in bins:
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        subset = valid_meta.iloc[idx]
        metrics = evaluate_combined(
            subset,
            action_prob[idx],
            point_prob[idx],
            server_prob[idx],
            action_multipliers,
            point_multipliers,
        )
        metrics.update({"prefix_len_bin": label, "count": int(len(idx))})
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_cv_collect_oof(
    prefix_df: pd.DataFrame,
    test_prefix_lengths: np.ndarray,
    features: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    lgbm_action_parts: list[np.ndarray] = []
    lgbm_point_parts: list[np.ndarray] = []
    lgbm_server_parts: list[np.ndarray] = []
    ngram_action_parts: list[np.ndarray] = []
    ngram_point_parts: list[np.ndarray] = []
    ngram_server_parts: list[np.ndarray] = []
    valid_meta_parts: list[pd.DataFrame] = []
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

        lgbm_bundle = fit_bundle(fold_train, features, args.n_estimators, args.seed + fold * 10)
        lgbm_action, lgbm_point, lgbm_server = predict_bundle(lgbm_bundle, fold_valid, features)

        ngram_bundle = fit_ngram_bundle(fold_train, args.ngram_alpha)
        ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, fold_valid)

        lgbm_metrics = evaluate_predictions(fold_valid, lgbm_action, lgbm_point, lgbm_server)
        ngram_metrics = evaluate_combined(fold_valid, ngram_action, ngram_point, ngram_server)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "valid_rows": len(fold_valid),
                "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
                "lgbm_action_macro_f1": lgbm_metrics["action_macro_f1"],
                "lgbm_point_macro_f1": lgbm_metrics["point_macro_f1"],
                "lgbm_server_auc": lgbm_metrics["server_auc"],
                "lgbm_overall": lgbm_metrics["overall"],
                "ngram_action_macro_f1": ngram_metrics["action_macro_f1"],
                "ngram_point_macro_f1": ngram_metrics["point_macro_f1"],
                "ngram_server_auc": ngram_metrics["server_auc"],
                "ngram_overall": ngram_metrics["overall"],
            }
        )
        print(
            f"fold {fold}: lgbm={lgbm_metrics['overall']:.6f} "
            f"ngram={ngram_metrics['overall']:.6f}"
        )

        keep_cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
        valid_meta_parts.append(fold_valid[keep_cols].reset_index(drop=True))
        lgbm_action_parts.append(lgbm_action)
        lgbm_point_parts.append(lgbm_point)
        lgbm_server_parts.append(lgbm_server)
        ngram_action_parts.append(ngram_action)
        ngram_point_parts.append(ngram_point)
        ngram_server_parts.append(ngram_server)

    valid_meta = pd.concat(valid_meta_parts, ignore_index=True)
    fold_report = pd.DataFrame(fold_rows)
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0, "valid_prefix_len_mean": float(valid_meta["prefix_len"].mean())}
    for col in fold_report.columns:
        if col not in mean_row and col != "fold":
            mean_row[col] = float(fold_report[col].mean())
    fold_report = pd.concat([fold_report, pd.DataFrame([mean_row])], ignore_index=True)

    return {
        "valid_meta": valid_meta,
        "fold_report": fold_report,
        "lgbm_action": np.vstack(lgbm_action_parts),
        "lgbm_point": np.vstack(lgbm_point_parts),
        "lgbm_server": np.concatenate(lgbm_server_parts),
        "ngram_action": np.vstack(ngram_action_parts),
        "ngram_point": np.vstack(ngram_point_parts),
        "ngram_server": np.concatenate(ngram_server_parts),
    }


def write_v2_submission(
    test_prefix: pd.DataFrame,
    lgbm_bundle,
    ngram_bundle: NgramBundle,
    features: list[str],
    tuning: TuningResult,
    output_path: Path,
) -> pd.DataFrame:
    lgbm_action, lgbm_point, lgbm_server = predict_bundle(lgbm_bundle, test_prefix, features)
    ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, test_prefix)

    action_prob = blend_probs(lgbm_action, ngram_action, tuning.action_ngram_weight)
    point_prob = blend_probs(lgbm_point, ngram_point, tuning.point_ngram_weight)
    server_prob = (1.0 - tuning.server_ngram_weight) * lgbm_server + tuning.server_ngram_weight * ngram_server

    action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob * tuning.action_multipliers[None, :], axis=1)]
    point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob * tuning.point_multipliers[None, :], axis=1)]
    submission = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
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
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building prefix tables...")
    prefix_df = build_train_prefix_table(train, args.max_lag)
    test_prefix = build_test_prefix_table(test, args.max_lag)
    features = feature_columns(prefix_df)
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    test_prefix_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    print(f"train prefix rows: {len(prefix_df):,}")
    print(f"test prediction rows: {len(test_prefix):,}")
    print(f"feature count: {len(features)}")

    oof = run_cv_collect_oof(prefix_df, test_prefix_lengths, features, args)
    tuning = tune_oof(
        oof["valid_meta"],
        oof["lgbm_action"],
        oof["lgbm_point"],
        oof["lgbm_server"],
        oof["ngram_action"],
        oof["ngram_point"],
        oof["ngram_server"],
    )

    action_prob = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)
    point_prob = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    server_prob = (1.0 - tuning.server_ngram_weight) * oof["lgbm_server"] + tuning.server_ngram_weight * oof["ngram_server"]
    prefix_report = prefix_len_report(
        oof["valid_meta"],
        action_prob,
        point_prob,
        server_prob,
        tuning.action_multipliers,
        tuning.point_multipliers,
    )

    fold_report = oof["fold_report"].copy()
    fold_report["selected_action_ngram_weight"] = tuning.action_ngram_weight
    fold_report["selected_point_ngram_weight"] = tuning.point_ngram_weight
    fold_report["selected_server_ngram_weight"] = tuning.server_ngram_weight
    for key, value in tuning.metrics.items():
        fold_report[f"selected_{key}"] = value
    fold_report.to_csv(args.cv_report, index=False)
    prefix_report.to_csv(args.prefix_report, index=False)

    with open(args.oof_proba, "wb") as f:
        pickle.dump(
            {
                "valid_meta": oof["valid_meta"],
                "lgbm_action": oof["lgbm_action"],
                "lgbm_point": oof["lgbm_point"],
                "lgbm_server": oof["lgbm_server"],
                "ngram_action": oof["ngram_action"],
                "ngram_point": oof["ngram_point"],
                "ngram_server": oof["ngram_server"],
                "tuning": tuning,
            },
            f,
        )

    print("selected tuning:")
    print(json.dumps(
        {
            "action_ngram_weight": tuning.action_ngram_weight,
            "point_ngram_weight": tuning.point_ngram_weight,
            "server_ngram_weight": tuning.server_ngram_weight,
            **tuning.metrics,
        },
        indent=2,
    ))

    print("training full-data models...")
    full_lgbm = fit_bundle(prefix_df, features, args.n_estimators, args.seed)
    full_ngram = fit_ngram_bundle(prefix_df, args.ngram_alpha)
    submission = write_v2_submission(test_prefix, full_lgbm, full_ngram, features, tuning, Path(args.submission))

    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_df)),
        "test_prediction_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "features": features,
        "ngram_key_levels": NGRAM_KEY_LEVELS,
        "ngram_alpha": float(args.ngram_alpha),
        "n_estimators": int(args.n_estimators),
        "selected": {
            "action_ngram_weight": tuning.action_ngram_weight,
            "point_ngram_weight": tuning.point_ngram_weight,
            "server_ngram_weight": tuning.server_ngram_weight,
            "action_multipliers": tuning.action_multipliers.tolist(),
            "point_multipliers": tuning.point_multipliers.tolist(),
            "metrics": tuning.metrics,
        },
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
