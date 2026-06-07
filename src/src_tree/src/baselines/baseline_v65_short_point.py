"""V6.5 short-prefix point specialists.

Targets the identified bottleneck: pointId for prefix_len 1 and 2. The script
trains separate LightGBM point models for short prefixes using leakage-safe OOF
rows and blends them with the V3 point baseline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
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
    make_lgbm,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, full_predict as v3_full_predict


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V6.5 short-prefix point specialists.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--base-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--submission", default="submission_v65.csv")
    parser.add_argument("--cv-report", default="cv_report_v65.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v65.csv")
    parser.add_argument("--class-report-point", default="class_report_v65_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v65.json")
    parser.add_argument("--oof-proba", default="oof_proba_v65.pkl")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=180)
    return parser.parse_args()


def compose_v3_predictions(pred: dict[str, np.ndarray], tuning: V3Tuning) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_prob = blend_probs(pred["lgbm_action"], pred["ngram_action"], tuning.action_ngram_weight)
    point_prob = blend_probs(pred["lgbm_point"], pred["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server_prob = (
        sw["direct"] * pred["lgbm_server"]
        + sw["ngram"] * pred["ngram_server"]
        + sw["parity"] * pred["parity_server"]
        + sw["remaining"] * pred["remaining_server"]
    )
    return action_prob, point_prob, server_prob


def load_base_oof(path: str) -> dict[str, object]:
    with open(path, "rb") as f:
        oof = pickle.load(f)
    action, point, server = compose_v3_predictions(oof, oof["tuning"])
    return {
        "meta": oof["valid_meta"].reset_index(drop=True),
        "action": action,
        "point": point,
        "server": server,
        "tuning": oof["tuning"],
    }


def merge_prefix(prefix_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = meta[["rally_uid", "match", "prefix_len", "next_pointId"]].merge(
        prefix_df, on=["rally_uid", "prefix_len"], how="left", validate="one_to_one"
    )
    if "match_x" in merged.columns:
        merged["match"] = merged["match_x"]
        merged = merged.drop(columns=[c for c in ["match_x", "match_y"] if c in merged.columns])
    if "next_pointId_x" in merged.columns:
        merged["next_pointId"] = merged["next_pointId_x"]
        merged = merged.drop(columns=[c for c in ["next_pointId_x", "next_pointId_y"] if c in merged.columns])
    if merged.isna().any().any():
        raise ValueError("Missing merged prefix features.")
    return merged


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Pair features use lag0 and lag1 where available. They are integer-coded
    # products, not target encodings.
    for a, b in [
        ("actionId", "pointId"),
        ("spinId", "actionId"),
        ("spinId", "pointId"),
        ("handId", "actionId"),
    ]:
        c0 = f"lag0_{a}"
        c1 = f"lag1_{b}"
        if c0 in out.columns and c1 in out.columns:
            out[f"pair_lag0_{a}_lag1_{b}"] = (out[c0].astype(int) + 1) * 100 + (out[c1].astype(int) + 1)
    if "lag0_actionId" in out.columns and "lag1_actionId" in out.columns:
        out["pair_action_0_1"] = (out["lag0_actionId"].astype(int) + 1) * 100 + (out["lag1_actionId"].astype(int) + 1)
    if "lag0_pointId" in out.columns and "lag1_pointId" in out.columns:
        out["pair_point_0_1"] = (out["lag0_pointId"].astype(int) + 1) * 100 + (out["lag1_pointId"].astype(int) + 1)
    return out


def specialist_features(df: pd.DataFrame) -> list[str]:
    forbidden = {
        "rally_uid",
        "match",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "remaining_len",
        "final_parity_even",
        "num_prefixes_in_rally",
        "remaining_len_bucket",
    }
    return [c for c in df.columns if c not in forbidden]


def make_point_model(n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    model = make_lgbm("multiclass", n_estimators, seed, num_class=len(POINT_CLASSES))
    model.set_params(
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.05,
        reg_lambda=1.0,
    )
    return model


def train_specialist_oof(df: pd.DataFrame, prefix_len: int, features: list[str], folds: int, seed: int, n_estimators: int) -> np.ndarray:
    idx_all = np.where(df["prefix_len"].eq(prefix_len).to_numpy())[0]
    part = df.iloc[idx_all].reset_index(drop=True)
    prob_part = np.zeros((len(part), len(POINT_CLASSES)), dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(part, groups=part["match"]), start=1):
        train_part = part.iloc[tr_idx]
        valid_part = part.iloc[va_idx]
        model = make_point_model(n_estimators, seed + prefix_len * 100 + fold)
        model.fit(
            train_part[features],
            train_part["next_pointId"],
            sample_weight=class_weight_sample(train_part["next_pointId"]),
        )
        prob_part[va_idx] = aligned_proba(model, valid_part[features], POINT_CLASSES)
    out = np.zeros((len(df), len(POINT_CLASSES)), dtype=float)
    out[idx_all] = prob_part
    return out


def tune_weights(meta: pd.DataFrame, base_point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    grid = [round(x, 1) for x in np.arange(0.0, 0.9, 0.1)]
    weights = {}
    for length, prob in [(1, p1), (2, p2)]:
        idx = np.where(meta["prefix_len"].eq(length).to_numpy())[0]
        y = meta.iloc[idx]["next_pointId"].to_numpy()
        best = max(
            grid,
            key=lambda w: f1_score(
                y,
                np.asarray(POINT_CLASSES)[np.argmax(blend_probs(base_point[idx], prob[idx], w), axis=1)],
                average="macro",
                labels=POINT_CLASSES,
                zero_division=0,
            ),
        )
        weights[str(length)] = float(best)
    return weights


def compose_point(meta: pd.DataFrame, base_point: np.ndarray, p1: np.ndarray, p2: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    out = base_point.copy()
    for length, prob in [(1, p1), (2, p2)]:
        idx = np.where(meta["prefix_len"].eq(length).to_numpy())[0]
        if len(idx) == 0:
            continue
        out[idx] = blend_probs(base_point[idx], prob[idx], weights[str(length)])
    return out


def metrics(meta: pd.DataFrame, action: np.ndarray, point: np.ndarray, server: np.ndarray, tuning: V3Tuning) -> tuple[dict[str, float], np.ndarray]:
    action_pred = apply_segmented_multipliers(meta, action, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }, point_pred


def fit_full_specialist(df: pd.DataFrame, prefix_len: int, features: list[str], n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    part = df[df["prefix_len"].eq(prefix_len)].copy()
    model = make_point_model(n_estimators, seed + prefix_len * 1000)
    model.fit(part[features], part["next_pointId"], sample_weight=class_weight_sample(part["next_pointId"]))
    return model


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    base = load_base_oof(args.base_oof)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    merged = add_pair_features(merge_prefix(prefix_df, base["meta"]))
    features = specialist_features(merged)

    print(f"short specialist features: {len(features)}")
    p1 = train_specialist_oof(merged, 1, features, args.folds, args.seed, args.n_estimators)
    p2 = train_specialist_oof(merged, 2, features, args.folds, args.seed, args.n_estimators)
    weights = tune_weights(base["meta"], base["point"], p1, p2)
    point = compose_point(base["meta"], base["point"], p1, p2, weights)
    v65_metrics, point_pred = metrics(base["meta"], base["action"], point, base["server"], base["tuning"])
    base_metrics, base_point_pred = metrics(base["meta"], base["action"], base["point"], base["server"], base["tuning"])
    print("base:", json.dumps(base_metrics, indent=2))
    print("v65:", json.dumps({**v65_metrics, "weights": weights}, indent=2))

    pd.DataFrame(
        [
            {"variant": "base_v3", **base_metrics, "w_len1": 0.0, "w_len2": 0.0},
            {"variant": "v65_short_specialist", **v65_metrics, "w_len1": weights["1"], "w_len2": weights["2"]},
        ]
    ).to_csv(args.cv_report, index=False)

    rows = []
    for label, mask in [
        ("1", base["meta"]["prefix_len"].eq(1).to_numpy()),
        ("2", base["meta"]["prefix_len"].eq(2).to_numpy()),
        ("ge3", base["meta"]["prefix_len"].ge(3).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        pred = point_pred[idx]
        y = base["meta"].iloc[idx]["next_pointId"]
        rows.append(
            {
                "prefix_len_bin": label,
                "count": int(len(idx)),
                "point_macro_f1": float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
            }
        )
    pd.DataFrame(rows).to_csv(args.prefix_len_report, index=False)
    pd.DataFrame(
        classification_report(base["meta"]["next_pointId"], point_pred, labels=POINT_CLASSES, zero_division=0, output_dict=True)
    ).T.to_csv(args.class_report_point)

    # Submission generation. If specialists do not help, this writes the base
    # V3-equivalent point predictions and is not a new submission candidate.
    test_prefix = build_test_prefix_table(test, args.max_lag)
    test_merged = add_pair_features(test_prefix.copy())
    v3_args = SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0)
    full_features_for_v3 = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    full_test_prefix = test_prefix[["rally_uid", "match"] + full_features_for_v3]
    tab_pred = v3_full_predict(prefix_df, full_test_prefix, full_features_for_v3, v3_args)
    action_test, point_test, server_test = compose_v3_predictions(tab_pred, base["tuning"])
    if v65_metrics["point_macro_f1"] > base_metrics["point_macro_f1"]:
        for length, prob_name in [(1, "1"), (2, "2")]:
            idx = np.where(test_prefix["prefix_len"].eq(length).to_numpy())[0]
            if len(idx) == 0 or weights[prob_name] <= 0:
                continue
            model = fit_full_specialist(merged, length, features, args.n_estimators, args.seed)
            specialist_prob = aligned_proba(model, test_merged.iloc[idx][features], POINT_CLASSES)
            point_test[idx] = blend_probs(point_test[idx], specialist_prob, weights[prob_name])
    action_pred = apply_segmented_multipliers(test_prefix, action_test, base["tuning"].action_multipliers, ACTION_CLASSES, base["tuning"].bins_mode)
    point_pred_test = apply_segmented_multipliers(test_prefix, point_test, base["tuning"].point_multipliers, POINT_CLASSES, base["tuning"].bins_mode)
    submission = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred_test.astype(int),
            "serverGetPoint": np.round(np.clip(server_test, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    submission.to_csv(args.submission, index=False, float_format="%.8f")
    with open(args.oof_proba, "wb") as f:
        pickle.dump({"base": base, "p1": p1, "p2": p2, "weights": weights, "metrics": v65_metrics}, f)
    metadata = {
        "features": features,
        "weights": weights,
        "base_metrics": base_metrics,
        "v65_metrics": v65_metrics,
        "args": vars(args),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.class_report_point}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
