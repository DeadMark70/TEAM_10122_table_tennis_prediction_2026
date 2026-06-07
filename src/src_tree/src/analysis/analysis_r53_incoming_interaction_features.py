"""R53 incoming-ball interaction feature audit.

Adds explicit interaction features suggested by R52:
last observed point/action/spin/strength and coarse incoming depth/side.

This branch is intentionally low-risk:
- no target encoding
- no old-test labels
- no future scoreboard
- no point decoder/reranker changes beyond the existing V3 tuning routine
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_lgbm import (
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


OUTDIR = Path("r53_incoming_interactions")
UPLOAD_DIR = Path("upload_candidates_20260519")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R53 incoming-ball interaction features.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default=str(OUTDIR / "submission_r53_incoming_interactions.csv"))
    parser.add_argument("--cv-report", default=str(OUTDIR / "cv_report_r53.csv"))
    parser.add_argument("--prefix-len-report", default=str(OUTDIR / "prefix_len_report_r53.csv"))
    parser.add_argument("--feature-report", default=str(OUTDIR / "feature_report_r53.json"))
    parser.add_argument("--oof-proba", default=str(OUTDIR / "oof_proba_r53.pkl"))
    parser.add_argument("--recommendation", default=str(OUTDIR / "r53_recommendation.md"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=140)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def point_depth(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) // 3 + 1).astype(np.int16)
    return out


def point_side(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) % 3 + 1).astype(np.int16)
    return out


def action_group(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.full(len(arr), -1, dtype=np.int16)
    out[arr == 0] = 0
    out[(arr >= 1) & (arr <= 7)] = 1
    out[(arr >= 8) & (arr <= 11)] = 2
    out[(arr >= 12) & (arr <= 14)] = 3
    out[(arr >= 15) & (arr <= 18)] = 4
    return out


def encode_combo(*cols: np.ndarray, bases: list[int]) -> np.ndarray:
    if len(cols) != len(bases):
        raise ValueError("cols and bases length mismatch")
    code = np.zeros(len(cols[0]), dtype=np.int64)
    multiplier = 1
    for col, base in zip(cols, bases):
        shifted = np.asarray(col, dtype=np.int64) + 1
        code += shifted * multiplier
        multiplier *= base
    return code.astype(np.int32)


def add_r53_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lag0_point = out["lag0_pointId"].to_numpy(dtype=int)
    lag0_action = out["lag0_actionId"].to_numpy(dtype=int)
    lag0_spin = out["lag0_spinId"].to_numpy(dtype=int)
    lag0_strength = out["lag0_strengthId"].to_numpy(dtype=int)
    lag0_hand = out["lag0_handId"].to_numpy(dtype=int)
    lag0_position = out["lag0_positionId"].to_numpy(dtype=int)
    lag1_point = out["lag1_pointId"].to_numpy(dtype=int)
    lag1_action = out["lag1_actionId"].to_numpy(dtype=int)

    depth0 = point_depth(lag0_point)
    side0 = point_side(lag0_point)
    depth1 = point_depth(lag1_point)
    side1 = point_side(lag1_point)
    group0 = action_group(lag0_action)
    group1 = action_group(lag1_action)
    prefix_bin = np.where(out["prefix_len"].to_numpy(dtype=int) <= 2, 0, 1).astype(np.int16)

    out["r53_incoming_depth"] = depth0
    out["r53_incoming_side"] = side0
    out["r53_prev_depth"] = depth1
    out["r53_prev_side"] = side1
    out["r53_lag0_action_group"] = group0
    out["r53_lag1_action_group"] = group1
    out["r53_prefix_bin_le2"] = prefix_bin

    out["r53_depth_x_action"] = encode_combo(depth0, lag0_action, bases=[5, 21])
    out["r53_side_x_action"] = encode_combo(side0, lag0_action, bases=[5, 21])
    out["r53_depth_x_spin"] = encode_combo(depth0, lag0_spin, bases=[5, 8])
    out["r53_side_x_spin"] = encode_combo(side0, lag0_spin, bases=[5, 8])
    out["r53_depth_side_x_action"] = encode_combo(depth0, side0, lag0_action, bases=[5, 5, 21])
    out["r53_depth_side_x_action_spin"] = encode_combo(depth0, side0, lag0_action, lag0_spin, bases=[5, 5, 21, 8])
    out["r53_depth_group_strength"] = encode_combo(depth0, group0, lag0_strength, bases=[5, 7, 6])
    out["r53_side_group_strength"] = encode_combo(side0, group0, lag0_strength, bases=[5, 7, 6])
    out["r53_action_spin_strength"] = encode_combo(lag0_action, lag0_spin, lag0_strength, bases=[21, 8, 6])
    out["r53_action_spin_hand"] = encode_combo(lag0_action, lag0_spin, lag0_hand, bases=[21, 8, 5])
    out["r53_action_point_position"] = encode_combo(lag0_action, lag0_point, lag0_position, bases=[21, 12, 6])
    out["r53_prefix_depth_action"] = encode_combo(prefix_bin, depth0, lag0_action, bases=[3, 5, 21])
    out["r53_prefix_side_spin"] = encode_combo(prefix_bin, side0, lag0_spin, bases=[3, 5, 8])
    out["r53_lag1_to_lag0_point_transition"] = encode_combo(depth1, side1, depth0, side0, bases=[5, 5, 5, 5])
    out["r53_lag1_to_lag0_action_group_transition"] = encode_combo(group1, group0, bases=[7, 7])

    out["r53_last_is_short"] = (depth0 == 1).astype(np.int8)
    out["r53_last_is_half_long"] = (depth0 == 2).astype(np.int8)
    out["r53_last_is_long"] = (depth0 == 3).astype(np.int8)
    out["r53_last_is_forehand_side"] = (side0 == 1).astype(np.int8)
    out["r53_last_is_middle_side"] = (side0 == 2).astype(np.int8)
    out["r53_last_is_backhand_side"] = (side0 == 3).astype(np.int8)
    out["r53_last_control_or_defense"] = np.isin(group0, [2, 3]).astype(np.int8)
    out["r53_last_attack"] = (group0 == 1).astype(np.int8)
    out["r53_last_zero_burden"] = (
        (lag0_point == 0).astype(int)
        + (lag0_action == 0).astype(int)
        + (lag0_spin == 0).astype(int)
        + (lag0_strength == 0).astype(int)
        + (lag0_hand == 0).astype(int)
    ).astype(np.int8)
    return out


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    prefix_df = build_train_prefix_table(train, args.max_lag)
    test_prefix = build_test_prefix_table(test, args.max_lag)
    prefix_df = add_remaining_bucket(add_r53_features(prefix_df))
    test_prefix = add_r53_features(test_prefix)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]

    print(f"R53 feature count: {len(features)}")
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
    prefix_report.to_csv(args.prefix_len_report, index=False)
    oof["fold_report"].to_csv(args.cv_report, index=False)

    full_pred = full_predict(prefix_df, test_prefix, features, args)
    write_submission(test_prefix, full_pred, tuning, Path(args.submission))
    shutil.copy2(args.submission, UPLOAD_DIR / Path(args.submission).name)

    with open(args.oof_proba, "wb") as f:
        pickle.dump(oof, f)

    metadata = {
        "experiment": "r53_incoming_interactions",
        "feature_count": len(features),
        "new_feature_count": len([c for c in features if c.startswith("r53_")]),
        "new_features": [c for c in features if c.startswith("r53_")],
        "tuning": tuning.__dict__,
        "cv_mean": oof["fold_report"].mean(numeric_only=True).to_dict(),
        "prefix_report": prefix_report.to_dict(orient="records"),
        "submission": str(Path(args.submission)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    base = pd.read_csv("cv_report_v3.csv")
    current_point = float(tuning.metrics.get("point_macro_f1", np.nan))
    base_col = "selected_point_macro_f1" if "selected_point_macro_f1" in base.columns else "point_macro_f1"
    base_point = float(base[base_col].mean()) if base_col in base.columns else float("nan")
    rec = [
        "# R53 incoming interaction feature audit",
        "",
        f"- feature count: `{len(features)}`",
        f"- new R53 feature count: `{metadata['new_feature_count']}`",
        f"- mean CV point: `{current_point:.6f}`",
        f"- V3 mean CV point reference: `{base_point:.6f}`",
        f"- submission: `{Path(args.submission).name}`",
        "",
    ]
    if current_point >= base_point + 0.001:
        rec.append("Recommendation: positive enough for a cautious public probe if action/server do not regress.")
    else:
        rec.append("Recommendation: do not submit unless later ensemble diagnostics show complementary point behavior.")
    Path(args.recommendation).write_text("\n".join(rec), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print("\n".join(rec))


if __name__ == "__main__":
    main()
