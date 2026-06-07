"""R7 phase-aware feature audit.

This experiment keeps the V3 training/evaluation protocol and adds only
in-prefix phase/tactical features. It intentionally avoids raw player IDs,
player historical profiles, target encodings, and point decoder changes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_lgbm import (
    ACTION_CLASSES,
    LAG_FIELDS,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    validate_raw_data,
)
from baseline_v3 import (
    add_remaining_bucket,
    blend_probs,
    full_predict,
    prefix_len_report,
    run_cv,
    tune_v3,
    write_submission,
)


POINT_DEPTH = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
}
POINT_SIDE = {
    0: 0,
    1: 1,
    4: 1,
    7: 1,
    2: 2,
    5: 2,
    8: 2,
    3: 3,
    6: 3,
    9: 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R7 phase-aware feature audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_r7.csv")
    parser.add_argument("--cv-report", default="cv_report_r7.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_r7.csv")
    parser.add_argument("--feature-report", default="feature_report_r7.json")
    parser.add_argument("--oof-proba", default="oof_proba_r7.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--v3-feature-report", default="feature_report_v3.json")
    return parser.parse_args()


def encode_pair(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, b_base: int) -> np.ndarray:
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    valid = (a_arr >= 0) & (b_arr >= 0)
    out = np.full(len(a_arr), -1, dtype=np.int32)
    out[valid] = a_arr[valid] * b_base + b_arr[valid]
    return out


def point_depth(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray([POINT_DEPTH.get(int(v), -1) for v in values], dtype=np.int8)


def point_side(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray([POINT_SIDE.get(int(v), -1) for v in values], dtype=np.int8)


def first_two_stroke_features(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for rally_uid, group in raw.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        row: dict[str, int] = {"rally_uid": int(rally_uid)}
        first = group.iloc[0]
        for field in LAG_FIELDS:
            row[f"serve_{field}"] = int(first[field])
        row["serve_point_depth"] = int(POINT_DEPTH.get(int(first["pointId"]), -1))
        row["serve_point_side"] = int(POINT_SIDE.get(int(first["pointId"]), -1))

        if len(group) >= 2:
            second = group.iloc[1]
            for field in LAG_FIELDS:
                row[f"receive_{field}_raw"] = int(second[field])
            row["receive_point_depth_raw"] = int(POINT_DEPTH.get(int(second["pointId"]), -1))
            row["receive_point_side_raw"] = int(POINT_SIDE.get(int(second["pointId"]), -1))
        else:
            for field in LAG_FIELDS:
                row[f"receive_{field}_raw"] = -1
            row["receive_point_depth_raw"] = -1
            row["receive_point_side_raw"] = -1
        rows.append(row)
    return pd.DataFrame(rows)


def add_phase_features(prefix: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = prefix.copy()
    first_two = first_two_stroke_features(raw)
    out = out.merge(first_two, on="rally_uid", how="left")

    prefix_len = out["prefix_len"].astype(int)
    out["phase_id"] = np.select(
        [prefix_len.eq(1), prefix_len.eq(2), prefix_len.eq(3), prefix_len.ge(4)],
        [1, 2, 3, 4],
        default=0,
    ).astype(np.int8)
    out["prefix_len_cap8"] = np.minimum(prefix_len, 8).astype(np.int8)
    out["has_receive_observed"] = prefix_len.ge(2).astype(np.int8)
    out["is_receive_prediction"] = prefix_len.eq(1).astype(np.int8)
    out["is_third_ball_prediction"] = prefix_len.eq(2).astype(np.int8)
    out["is_fourth_ball_prediction"] = prefix_len.eq(3).astype(np.int8)
    out["is_rally_prediction"] = prefix_len.ge(4).astype(np.int8)

    receive_mask = out["has_receive_observed"].astype(bool)
    for field in LAG_FIELDS:
        raw_col = f"receive_{field}_raw"
        out[f"receive_{field}"] = np.where(receive_mask, out[raw_col], -1).astype(np.int16)
        out = out.drop(columns=[raw_col])
    for field in ["receive_point_depth", "receive_point_side"]:
        raw_col = f"{field}_raw"
        out[field] = np.where(receive_mask, out[raw_col], -1).astype(np.int8)
        out = out.drop(columns=[raw_col])

    out["serve_action_point_pair"] = encode_pair(out["serve_actionId"], out["serve_pointId"], 10)
    out["serve_spin_action_pair"] = encode_pair(out["serve_spinId"], out["serve_actionId"], 19)
    out["serve_hand_action_pair"] = encode_pair(out["serve_handId"], out["serve_actionId"], 19)
    out["serve_spin_point_pair"] = encode_pair(out["serve_spinId"], out["serve_pointId"], 10)
    out["serve_depth_side_pair"] = encode_pair(out["serve_point_depth"], out["serve_point_side"], 4)

    out["receive_action_point_pair"] = encode_pair(out["receive_actionId"], out["receive_pointId"], 10)
    out["receive_spin_action_pair"] = encode_pair(out["receive_spinId"], out["receive_actionId"], 19)
    out["receive_depth_side_pair"] = encode_pair(out["receive_point_depth"], out["receive_point_side"], 4)
    out["serve_receive_action_pair"] = encode_pair(out["serve_actionId"], out["receive_actionId"], 19)
    out["serve_receive_point_pair"] = encode_pair(out["serve_pointId"], out["receive_pointId"], 10)
    out["serve_receive_spin_pair"] = encode_pair(out["serve_spinId"], out["receive_spinId"], 6)
    out["serve_spin_receive_action_pair"] = encode_pair(out["serve_spinId"], out["receive_actionId"], 19)
    out["serve_point_receive_action_pair"] = encode_pair(out["serve_pointId"], out["receive_actionId"], 19)
    out["serve_action_receive_point_pair"] = encode_pair(out["serve_actionId"], out["receive_pointId"], 10)

    for lag in range(2):
        point_col = f"lag{lag}_pointId"
        out[f"lag{lag}_point_depth"] = point_depth(out[point_col])
        out[f"lag{lag}_point_side"] = point_side(out[point_col])
        out[f"lag{lag}_action_point_pair"] = encode_pair(out[f"lag{lag}_actionId"], out[point_col], 10)
        out[f"lag{lag}_spin_action_pair"] = encode_pair(out[f"lag{lag}_spinId"], out[f"lag{lag}_actionId"], 19)
        out[f"lag{lag}_spin_point_pair"] = encode_pair(out[f"lag{lag}_spinId"], out[point_col], 10)
        out[f"lag{lag}_hand_action_pair"] = encode_pair(out[f"lag{lag}_handId"], out[f"lag{lag}_actionId"], 19)
        out[f"lag{lag}_position_point_pair"] = encode_pair(out[f"lag{lag}_positionId"], out[point_col], 10)

    out["last2_action_transition"] = encode_pair(out["lag1_actionId"], out["lag0_actionId"], 19)
    out["last2_point_transition"] = encode_pair(out["lag1_pointId"], out["lag0_pointId"], 10)
    out["last2_spin_transition"] = encode_pair(out["lag1_spinId"], out["lag0_spinId"], 6)
    out["last2_hand_transition"] = encode_pair(out["lag1_handId"], out["lag0_handId"], 3)
    out["last2_depth_transition"] = encode_pair(out["lag1_point_depth"], out["lag0_point_depth"], 4)
    out["last2_side_transition"] = encode_pair(out["lag1_point_side"], out["lag0_point_side"], 4)

    short = out[["count_pointId_1", "count_pointId_2", "count_pointId_3"]].sum(axis=1)
    half = out[["count_pointId_4", "count_pointId_5", "count_pointId_6"]].sum(axis=1)
    long = out[["count_pointId_7", "count_pointId_8", "count_pointId_9"]].sum(axis=1)
    forehand = out[["count_pointId_1", "count_pointId_4", "count_pointId_7"]].sum(axis=1)
    middle = out[["count_pointId_2", "count_pointId_5", "count_pointId_8"]].sum(axis=1)
    backhand = out[["count_pointId_3", "count_pointId_6", "count_pointId_9"]].sum(axis=1)
    denom = prefix_len.clip(lower=1).astype(float)
    out["count_point_depth_short"] = short.astype(np.int16)
    out["count_point_depth_half"] = half.astype(np.int16)
    out["count_point_depth_long"] = long.astype(np.int16)
    out["count_point_side_forehand"] = forehand.astype(np.int16)
    out["count_point_side_middle"] = middle.astype(np.int16)
    out["count_point_side_backhand"] = backhand.astype(np.int16)
    out["rate_point_depth_short"] = (short / denom).astype(float)
    out["rate_point_depth_half"] = (half / denom).astype(float)
    out["rate_point_depth_long"] = (long / denom).astype(float)
    out["rate_point_side_forehand"] = (forehand / denom).astype(float)
    out["rate_point_side_middle"] = (middle / denom).astype(float)
    out["rate_point_side_backhand"] = (backhand / denom).astype(float)

    out["score_total_bucket"] = np.minimum(out["scoreTotal"].astype(int), 20).astype(np.int8)
    out["server_lead_bucket"] = np.clip(out["serverScoreDiff"].astype(int), -5, 5).astype(np.int8)
    out["is_deuce_like"] = (out["serverScore"].ge(10) & out["receiverScore"].ge(10)).astype(np.int8)
    out["server_at_game_point_like"] = (
        out["serverScore"].ge(10) & out["serverScore"].sub(out["receiverScore"]).ge(1)
    ).astype(np.int8)
    out["receiver_at_game_point_like"] = (
        out["receiverScore"].ge(10) & out["receiverScore"].sub(out["serverScore"]).ge(1)
    ).astype(np.int8)

    if out.isna().any().any():
        bad_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"R7 feature table contains NaN in {bad_cols[:10]}")
    return out


def load_v3_metrics(path: str) -> dict[str, float] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    selected = data.get("selected", {})
    metrics = selected.get("metrics")
    if isinstance(metrics, dict):
        return {k: float(v) for k, v in metrics.items()}
    return None


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building V3 prefix tables...")
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_prefix = build_test_prefix_table(test, args.max_lag)

    print("adding R7 phase-aware features...")
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]

    print(f"train prefix rows: {len(prefix_df):,}")
    print(f"test prediction rows: {len(test_prefix):,}")
    print(f"feature count: {len(features)}")

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
        pickle.dump({**oof, "tuning": tuning, "features": features}, f)

    print("training full-data R7 models...")
    full_pred = full_predict(prefix_df, test_prefix, features, args)
    submission = write_submission(test_prefix, full_pred, tuning, Path(args.submission))

    v3_metrics = load_v3_metrics(args.v3_feature_report)
    deltas = {}
    if v3_metrics:
        for key, value in tuning.metrics.items():
            if key in v3_metrics:
                deltas[key] = float(value - v3_metrics[key])

    r7_feature_names = [
        c
        for c in features
        if c.startswith(("serve_", "receive_", "phase_", "last2_", "rate_point_", "count_point_depth_", "count_point_side_"))
        or c
        in {
            "has_receive_observed",
            "is_receive_prediction",
            "is_third_ball_prediction",
            "is_fourth_ball_prediction",
            "is_rally_prediction",
            "prefix_len_cap8",
            "score_total_bucket",
            "server_lead_bucket",
            "is_deuce_like",
            "server_at_game_point_like",
            "receiver_at_game_point_like",
        }
        or "_pair" in c
        or "_transition" in c
    ]
    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_df)),
        "test_prediction_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "r7_added_feature_count_estimate": int(len(r7_feature_names)),
        "r7_added_features": r7_feature_names,
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "selected": {
            "action_ngram_weight": tuning.action_ngram_weight,
            "point_ngram_weight": tuning.point_ngram_weight,
            "server_weights": tuning.server_weights,
            "action_multipliers": tuning.action_multipliers,
            "point_multipliers": tuning.point_multipliers,
            "bins_mode": tuning.bins_mode,
            "metrics": tuning.metrics,
            "delta_vs_v3": deltas,
        },
        "submission_rows": int(len(submission)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("selected R7 tuning:")
    print(json.dumps(metadata["selected"], indent=2))
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
