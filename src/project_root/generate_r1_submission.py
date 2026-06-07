"""Generate full-test submission for the R1 OOF-selected ensemble.

R1 selected:
- action = 0.4 * full GRU + 0.6 * full Transformer
- point = V3 full prediction
- server = 0.8 * V3 + 0.1 * GRU + 0.1 * Transformer

This script trains one full-data GRU and one full-data Transformer using the
same settings as the V5/V7 OOF runs, saves their test probabilities, retunes
the R1 action multiplier from OOF, and writes submission_r1.csv.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import baseline_v5_gru as v5
import baseline_v7_transformer as v7
from analysis_r1_oof_ensemble import compose_v3, load_pickle, normalize_meta
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
    tune_segmented_multipliers,
)


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
    parser = argparse.ArgumentParser(description="Generate R1 full-test submission.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v5-feature-report", default="feature_report_v5.json")
    parser.add_argument("--v7-feature-report", default="feature_report_v7.json")
    parser.add_argument("--sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--submission", default="submission_r1.csv")
    parser.add_argument("--feature-report", default="feature_report_r1.json")
    parser.add_argument("--reuse-sequence-proba", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def args_from_report(path: str, device: str) -> SimpleNamespace:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = report["args"].copy()
    raw["device"] = device
    return SimpleNamespace(**raw)


def compose_v3_full(train: pd.DataFrame, test: pd.DataFrame, tuning: V3Tuning) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    v3_args = SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0)
    pred = v3_full_predict(prefix_df, test_prefix, features, v3_args)
    action = blend_probs(pred["lgbm_action"], pred["ngram_action"], tuning.action_ngram_weight)
    point = blend_probs(pred["lgbm_point"], pred["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server = (
        sw["direct"] * pred["lgbm_server"]
        + sw["ngram"] * pred["ngram_server"]
        + sw["parity"] * pred["parity_server"]
        + sw["remaining"] * pred["remaining_server"]
    )
    return test_prefix, action, point, np.clip(server, 1e-6, 1.0 - 1e-6)


def train_full_sequence_probs(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    v5_args = args_from_report(args.v5_feature_report, args.device)
    v7_args = args_from_report(args.v7_feature_report, args.device)
    # The OOF-selected models used these exact budgets.
    v5_args.skip_full_train = False
    v7_args.skip_full_train = False

    v5.set_seed(int(v5_args.seed))
    prefix_meta = v5.build_train_meta(train)
    test_meta = v5.build_test_meta(test)
    num_mean, num_std = v5.fit_numeric_stats(train)
    cat_cards = v5.cat_cardinalities(train, test)
    test_arrays = v5.build_sequence_arrays(test, test_meta, int(v5_args.max_len), num_mean, num_std)

    print("training full GRU...")
    gru_action, gru_point, gru_server = v5.train_full_gru(
        train, prefix_meta, test_arrays, cat_cards, num_mean, num_std, v5_args
    )

    v7.set_seed(int(v7_args.seed))
    print("training full Transformer...")
    tr_action, tr_point, tr_server = v7.train_full_transformer(
        train, prefix_meta, test_arrays, cat_cards, num_mean, num_std, v7_args
    )

    out = {
        "test_meta": test_meta,
        "gru_action": gru_action,
        "gru_point": gru_point,
        "gru_server": gru_server,
        "tr_action": tr_action,
        "tr_point": tr_point,
        "tr_server": tr_server,
    }
    with open(args.sequence_proba, "wb") as f:
        pickle.dump(out, f)
    return out


def retune_r1_action_multiplier(v3_oof: dict, v5_oof: dict, v7_oof: dict) -> tuple[dict[str, list[float]], dict[str, float]]:
    meta = normalize_meta(v3_oof["valid_meta"])
    v3_action, v3_point, v3_server = compose_v3(v3_oof)
    del v3_point, v3_server
    action_oof = 0.4 * v5_oof["gru_action"] + 0.6 * v7_oof["tr_action"]
    action_oof = action_oof / action_oof.sum(axis=1, keepdims=True)
    mult = tune_segmented_multipliers(meta, action_oof, ACTION_CLASSES, "action", "two")
    pred = apply_segmented_multipliers(meta, action_oof, mult, ACTION_CLASSES, "two")
    from sklearn.metrics import f1_score

    metrics = {
        "action_macro_f1": float(
            f1_score(meta["next_actionId"], pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        )
    }
    return mult, metrics


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    v3_oof = load_pickle(args.v3_oof)
    v5_oof = load_pickle(args.v5_oof)
    v7_oof = load_pickle(args.v7_oof)
    action_mult, action_metrics = retune_r1_action_multiplier(v3_oof, v5_oof, v7_oof)

    if args.reuse_sequence_proba and Path(args.sequence_proba).exists():
        with open(args.sequence_proba, "rb") as f:
            seq = pickle.load(f)
    else:
        seq = train_full_sequence_probs(train, test, args)

    test_meta = seq["test_meta"].reset_index(drop=True)
    test_prefix, v3_action, v3_point, v3_server = compose_v3_full(train, test, v3_oof["tuning"])
    if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("Sequence test rows and V3 test rows are not aligned.")

    action_prob = 0.4 * seq["gru_action"] + 0.6 * seq["tr_action"]
    action_prob = action_prob / action_prob.sum(axis=1, keepdims=True)
    point_prob = v3_point
    server_prob = 0.8 * v3_server + 0.1 * seq["gru_server"] + 0.1 * seq["tr_server"]

    action_pred = apply_segmented_multipliers(test_meta, action_prob, action_mult, ACTION_CLASSES, "two")
    point_pred = apply_segmented_multipliers(
        test_meta, point_prob, v3_oof["tuning"].point_multipliers, POINT_CLASSES, v3_oof["tuning"].bins_mode
    )
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
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
        "r1_oof_overall": 0.31439533985349744,
        "action_weights": {"gru": 0.4, "transformer": 0.6},
        "point_policy": "v3_full",
        "server_weights": {"v3": 0.8, "gru": 0.1, "transformer": 0.1},
        "action_multiplier_metrics": action_metrics,
        "action_multipliers": action_mult,
        "rows": int(len(sub)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.submission} ({len(sub):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
