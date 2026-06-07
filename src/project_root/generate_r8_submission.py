"""Generate submission for R8 action-only ensemble.

R8 uses:
- action = 0.9 * V10B-safe action + 0.1 * R7 phase-feature action
- point = V3 full point probabilities and V3 point multipliers
- server = V10B-safe server
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
from baseline_v3 import (
    add_remaining_bucket,
    apply_segmented_multipliers,
    full_predict as v3_full_predict,
)
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
    parser = argparse.ArgumentParser(description="Generate R8 submission.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--submission", default="submission_r8.csv")
    parser.add_argument("--feature-report", default="feature_report_r8.json")
    return parser.parse_args()


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def build_r7_full_action(train: pd.DataFrame, test: pd.DataFrame, r7_tuning) -> tuple[pd.DataFrame, np.ndarray]:
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    args = type("Args", (), {"seeds": [42], "n_estimators": 120, "ngram_alpha": 20.0})()
    pred = v3_full_predict(prefix_df, test_prefix, features, args)
    action = blend_probs(pred["lgbm_action"], pred["ngram_action"], r7_tuning.action_ngram_weight)
    return test_prefix, normalize_rows(action)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    with open(args.v3_oof, "rb") as f:
        v3_oof = pickle.load(f)
    with open(args.r7_oof, "rb") as f:
        r7_oof = pickle.load(f)
    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10b_full = pickle.load(f)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    selected_r8 = json.loads(Path(args.r8_selected).read_text(encoding="utf-8"))

    test_prefix, _, v3_point, v3_server = compose_v3_full(train, test, v3_oof["tuning"])
    r7_prefix, r7_action = build_r7_full_action(train, test, r7_oof["tuning"])
    if not test_prefix["rally_uid"].reset_index(drop=True).equals(r7_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("R7 and V3 test rows are not aligned.")

    r1_action = normalize_rows(0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]
    safe_action = blend_probs(r1_action, v10b_full["v10_action"], float(selected_v10["action_v10_weight"]))
    safe_server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10b_full["v10_server"]

    r7_weight = float(selected_r8["r7_weight"])
    action_prob = blend_probs(safe_action, r7_action, r7_weight)
    action_pred = apply_segmented_multipliers(
        test_prefix, action_prob, selected_r8["action_multipliers"], ACTION_CLASSES, "two"
    )
    point_pred = apply_segmented_multipliers(
        test_prefix,
        v3_point,
        v3_oof["tuning"].point_multipliers,
        POINT_CLASSES,
        v3_oof["tuning"].bins_mode,
    )
    sub = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(safe_server, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != test["rally_uid"].nunique():
        raise ValueError("Submission row count mismatch.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    sub.to_csv(args.submission, index=False, float_format="%.8f")

    metadata = {
        "source": "R8 action-only",
        "oof_metrics": selected_r8["metrics"],
        "r7_action_weight": r7_weight,
        "point_policy": selected_r8["point_policy"],
        "server_policy": selected_r8["server_policy"],
        "rows": int(len(sub)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.submission} ({len(sub):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
