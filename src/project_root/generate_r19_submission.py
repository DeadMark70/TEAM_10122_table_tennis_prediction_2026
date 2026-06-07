"""Generate full-test submission for R19 observed-style profile ensemble."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r19_observed_style_profiles import (
    add_server_receiver_ids,
    attach_profiles,
    build_profile_tables,
    fit_models,
    predict_models,
)
from analysis_r7_phase_features import add_phase_features
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers
from generate_r1_submission import compose_v3_full


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
    parser = argparse.ArgumentParser(description="Generate R19 submission.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r19-selected", default="r19_selected.json")
    parser.add_argument("--submission", default="submission_r19.csv")
    parser.add_argument("--feature-report", default="feature_report_r19_submission.json")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth-k", type=float, default=50.0)
    return parser.parse_args()


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def build_style_full(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace):
    train_prefix = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_prefix = build_test_prefix_table(test, args.max_lag)
    train_prefix = add_server_receiver_ids(add_phase_features(train_prefix, train), train)
    test_prefix = add_server_receiver_ids(add_phase_features(test_prefix, test), test)

    train_tables = build_profile_tables(train, args.smooth_k)
    test_tables = build_profile_tables(pd.concat([train, test], ignore_index=True), args.smooth_k)
    train_profile = attach_profiles(train_prefix, train_tables)
    test_profile = attach_profiles(test_prefix, test_tables)

    excluded = {
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
    features = [c for c in feature_columns(train_profile) if c in test_profile.columns and c not in excluded]
    features = [c for c in features if "PlayerId" not in c and c not in {"server_id", "receiver_id"}]
    models = fit_models(train_profile, features, args.seed + 1900, args.n_estimators)
    action, point = predict_models(models, test_profile, features)
    return test_profile[["rally_uid", "match", "prefix_len"]].reset_index(drop=True), action, point, features


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    with open(args.v3_oof, "rb") as f:
        v3_oof = pickle.load(f)
    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10_full = pickle.load(f)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    selected_r19 = json.loads(Path(args.r19_selected).read_text(encoding="utf-8"))["selected"]

    test_prefix, _, v3_point, v3_server = compose_v3_full(train, test, v3_oof["tuning"])
    style_meta, style_action, style_point, features = build_style_full(train, test, args)
    if not test_prefix["rally_uid"].reset_index(drop=True).equals(style_meta["rally_uid"].reset_index(drop=True)):
        raise ValueError("R19 style rows and V3 test rows are not aligned.")

    r1_action = normalize_rows(0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]
    server_prob = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10_full["v10_server"]

    action_prob = blend_probs(r1_action, style_action, float(selected_r19["action_blend"]))
    point_prob = blend_probs(v3_point, style_point, float(selected_r19["point_blend"]))
    action_pred = apply_segmented_multipliers(
        test_prefix, action_prob, selected_r19["action_multipliers"], ACTION_CLASSES, "two"
    )
    point_pred = apply_segmented_multipliers(
        test_prefix, point_prob, v3_oof["tuning"].point_multipliers, POINT_CLASSES, v3_oof["tuning"].bins_mode
    )
    sub = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != test["rally_uid"].nunique():
        raise ValueError("Submission row count mismatch.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    sub.to_csv(args.submission, index=False, float_format="%.8f")
    metadata = {
        "source": "R19 observed-style profile",
        "selected": selected_r19,
        "feature_count": len(features),
        "rows": int(len(sub)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.submission} ({len(sub):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
