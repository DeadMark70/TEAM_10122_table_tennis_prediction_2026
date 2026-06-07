"""Generate full-test R17 internal-transition adaptation candidates."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r17_internal_transition_adapt import (
    build_safe_server,
    fit_action_point_models,
    internal_rows_from_public_test,
    predict_action_point,
)
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
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, tune_segmented_multipliers
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


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R17 internal transition adaptation submission.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r17-oof", default="oof_proba_r17.pkl")
    parser.add_argument("--r17-selected", default="r17_selected.json")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--submission", default="submission_r17_internal_adapt.csv")
    parser.add_argument("--feature-report", default="feature_report_r17_submission.json")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r17_oof = load_pickle(args.r17_oof)
    selected_doc = json.loads(Path(args.r17_selected).read_text(encoding="utf-8"))
    selected = selected_doc["selected"]
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10_full = pickle.load(f)

    internal_weight = float(selected["internal_weight"])
    action_blend = float(selected["action_blend"])
    point_blend = float(selected["point_blend"])

    meta = normalize_meta(v3["valid_meta"])
    r1_action_oof = 0.4 * v5["gru_action"] + 0.6 * v7["tr_action"]
    r1_action_oof = r1_action_oof / r1_action_oof.sum(axis=1, keepdims=True)
    action_oof = blend_probs(r1_action_oof, r17_oof[f"action_w{internal_weight}"], action_blend)
    action_mult = tune_segmented_multipliers(meta, action_oof, ACTION_CLASSES, "action", "two")

    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_prefix = build_test_prefix_table(test, args.max_lag)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]

    test_for_internal = test.copy()
    if "serverGetPoint" not in test_for_internal.columns:
        test_for_internal["serverGetPoint"] = 0
    public_internal = add_remaining_bucket(build_train_prefix_table(test_for_internal, args.max_lag))
    public_internal = internal_rows_from_public_test(public_internal)
    for col in features:
        if col not in public_internal.columns:
            public_internal[col] = 0
    public_internal = public_internal[prefix_df.columns.intersection(public_internal.columns).tolist() + [c for c in public_internal.columns if c not in prefix_df.columns]]

    train_adapt = prefix_df.copy()
    train_adapt["is_internal_adapt"] = 0
    public_internal = public_internal.copy()
    public_internal["is_internal_adapt"] = 1
    for col in train_adapt.columns:
        if col not in public_internal.columns:
            public_internal[col] = 0
    for col in public_internal.columns:
        if col not in train_adapt.columns:
            train_adapt[col] = 0
    public_internal = public_internal[train_adapt.columns]
    full_train = pd.concat([train_adapt, public_internal], ignore_index=True)

    models = fit_action_point_models(full_train, features, args.seed + int(internal_weight * 1000), args.n_estimators, internal_weight)
    adapt_action, adapt_point = predict_action_point(models, test_prefix, features)

    test_meta = r1_seq["test_meta"].reset_index(drop=True)
    test_prefix_v3, _, v3_point, v3_server = compose_v3_full(train, test, v3["tuning"])
    if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("R1 sequence and test prefix rows are not aligned.")
    if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix_v3["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 and sequence rows are not aligned.")
    if not test_meta["rally_uid"].reset_index(drop=True).equals(v10_full["test_meta"]["rally_uid"].reset_index(drop=True)):
        raise ValueError("V10B and sequence rows are not aligned.")

    r1_action = 0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"]
    r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
    action_prob = blend_probs(r1_action, adapt_action, action_blend)
    point_prob = blend_probs(v3_point, adapt_point, point_blend)
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]
    server_w = float(selected_v10["server_v10_weight"])
    server_prob = (1.0 - server_w) * r1_server + server_w * v10_full["v10_server"]

    action_pred = apply_segmented_multipliers(test_meta, action_prob, action_mult, ACTION_CLASSES, "two")
    point_pred = apply_segmented_multipliers(
        test_meta, point_prob, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode
    )
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    sub.to_csv(args.submission, index=False, float_format="%.8f")
    report = {
        "source": "R17 internal transition adaptation full-test candidate",
        "internal_weight": internal_weight,
        "action_blend": action_blend,
        "point_blend": point_blend,
        "selected_oof": selected,
        "public_internal_rows": int(len(public_internal)),
        "rows": int(len(sub)),
        "note": "Uses observed test_new internal transitions only; no hidden target labels.",
    }
    Path(args.feature_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.submission} ({len(sub):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
