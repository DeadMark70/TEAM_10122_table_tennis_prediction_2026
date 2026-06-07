"""R30 low-risk structural feature audit.

This experiment starts from the V3/R7 pipeline and adds only public,
in-prefix structural features. It intentionally avoids old-test labels,
future scoreboard information, target encodings, raw player IDs, and point
decoder changes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r7_phase_features import (
    POINT_DEPTH,
    POINT_SIDE,
    add_phase_features,
    encode_pair,
    point_depth,
    point_side,
)
from baseline_lgbm import (
    LAG_FIELDS,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    validate_raw_data,
)
from baseline_v2 import NGRAM_KEY_LEVELS
from baseline_v3 import (
    REMAINING_CLASSES,
    add_remaining_bucket,
    blend_probs,
    full_predict,
    prefix_len_report,
    run_cv,
    tune_v3,
    write_submission,
)


ZERO_BURDEN_FIELDS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
RUN_FIELDS = ["actionId", "pointId", "handId", "spinId", "strengthId", "positionId"]
STRENGTH_VALUES = [0, 1, 2, 3]
STRIKE_VALUES = [0, 1, 2, 4, 8, 16]
ACTION_GROUP_VALUES = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R30 low-risk structural feature audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_r30.csv")
    parser.add_argument("--cv-report", default="cv_report_r30.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_r30.csv")
    parser.add_argument("--feature-report", default="feature_report_r30.json")
    parser.add_argument("--oof-proba", default="oof_proba_r30.pkl")
    parser.add_argument("--recommendation", default="r30_recommendation.md")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def action_group(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.full(len(arr), -1, dtype=np.int8)
    out[arr == 0] = 0
    out[(arr >= 1) & (arr <= 7)] = 1
    out[(arr >= 8) & (arr <= 11)] = 2
    out[(arr >= 12) & (arr <= 14)] = 3
    out[(arr >= 15) & (arr <= 18)] = 4
    return out


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def build_r30_prefix_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate features for every observed prefix in raw rallies."""
    rows: list[dict[str, int | float]] = []
    point_depth_lut = POINT_DEPTH
    point_side_lut = POINT_SIDE

    for rally_uid, group in raw.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        strength_counts = {v: 0 for v in STRENGTH_VALUES}
        strike_counts = {v: 0 for v in STRIKE_VALUES}
        action_group_counts = {v: 0 for v in ACTION_GROUP_VALUES}
        zero_burden_sum = 0
        strength_seen: set[int] = set()
        strike_seen: set[int] = set()
        action_group_seen: set[int] = set()

        same_receiver_count = 0
        same_receiver_depth_counts = {1: 0, 2: 0, 3: 0}
        same_receiver_side_counts = {1: 0, 2: 0, 3: 0}
        same_receiver_point0_count = 0
        same_receiver_depth_side_counts = {(d, s): 0 for d in [1, 2, 3] for s in [1, 2, 3]}

        for t_index, row in group.iterrows():
            prefix_len = int(row["strikeNumber"])
            strength = int(row["strengthId"])
            strike = int(row["strikeId"])
            ag = int(action_group(np.array([int(row["actionId"])]))[0])

            if strength not in strength_counts:
                strength_counts[strength] = 0
            strength_counts[strength] += 1
            strength_seen.add(strength)

            if strike not in strike_counts:
                strike_counts[strike] = 0
            strike_counts[strike] += 1
            strike_seen.add(strike)

            if ag not in action_group_counts:
                action_group_counts[ag] = 0
            action_group_counts[ag] += 1
            action_group_seen.add(ag)

            zero_burden = sum(int(row[field]) == 0 for field in ZERO_BURDEN_FIELDS)
            zero_burden_sum += zero_burden

            next_receiver_is_server = int((prefix_len + 1) % 2 == 0)
            observed_receiver_is_server = int(prefix_len % 2 == 0)
            if observed_receiver_is_server == next_receiver_is_server:
                same_receiver_count += 1
                point = int(row["pointId"])
                depth = int(point_depth_lut.get(point, 0))
                side = int(point_side_lut.get(point, 0))
                if point == 0:
                    same_receiver_point0_count += 1
                if depth in same_receiver_depth_counts:
                    same_receiver_depth_counts[depth] += 1
                if side in same_receiver_side_counts:
                    same_receiver_side_counts[side] += 1
                if (depth, side) in same_receiver_depth_side_counts:
                    same_receiver_depth_side_counts[(depth, side)] += 1

            denom = max(prefix_len, 1)
            feature_row: dict[str, int | float] = {
                "rally_uid": int(rally_uid),
                "prefix_len": prefix_len,
                "prefix_zero_burden_rate_r30": float(zero_burden_sum) / float(denom * len(ZERO_BURDEN_FIELDS)),
                "nunique_strengthId_prefix_r30": int(len(strength_seen)),
                "nunique_strikeId_prefix_r30": int(len(strike_seen)),
                "nunique_action_group_prefix_r30": int(len(action_group_seen)),
                "same_receiver_count_r30": int(same_receiver_count),
                "same_receiver_point0_rate_r30": _safe_rate(same_receiver_point0_count, same_receiver_count),
            }
            for value in sorted(strength_counts):
                count = int(strength_counts[value])
                feature_row[f"count_strengthId_{value}_r30"] = count
                feature_row[f"rate_strengthId_{value}_r30"] = float(count) / float(denom)
            for value in sorted(strike_counts):
                count = int(strike_counts[value])
                feature_row[f"count_strikeId_{value}_r30"] = count
                feature_row[f"rate_strikeId_{value}_r30"] = float(count) / float(denom)
            for value in sorted(action_group_counts):
                count = int(action_group_counts[value])
                feature_row[f"count_action_group_{value}_r30"] = count
                feature_row[f"rate_action_group_{value}_r30"] = float(count) / float(denom)
            for depth, label in [(1, "short"), (2, "half"), (3, "long")]:
                feature_row[f"same_receiver_depth_{label}_rate_r30"] = _safe_rate(
                    same_receiver_depth_counts[depth], same_receiver_count
                )
            for side, label in [(1, "forehand"), (2, "middle"), (3, "backhand")]:
                feature_row[f"same_receiver_side_{label}_rate_r30"] = _safe_rate(
                    same_receiver_side_counts[side], same_receiver_count
                )
            for depth in [1, 2, 3]:
                for side in [1, 2, 3]:
                    feature_row[f"same_receiver_depth{depth}_side{side}_rate_r30"] = _safe_rate(
                        same_receiver_depth_side_counts[(depth, side)], same_receiver_count
                    )
            rows.append(feature_row)

    return pd.DataFrame(rows)


def add_run_length_features(out: pd.DataFrame) -> pd.DataFrame:
    for field in RUN_FIELDS:
        run = np.ones(len(out), dtype=np.int8)
        for lag in range(1, 6):
            exists = out[f"lag{lag}_exists"].to_numpy(dtype=bool)
            same = out[f"lag{lag}_{field}"].to_numpy() == out["lag0_" + field].to_numpy()
            still_same = exists & same & (run == lag)
            run = np.where(still_same, run + 1, run)
        out[f"last_{field}_run_len_cap6_r30"] = run.astype(np.int8)
    return out


def add_time_since_features(out: pd.DataFrame) -> pd.DataFrame:
    def first_distance(mask_by_lag: list[np.ndarray]) -> np.ndarray:
        result = np.full(len(out), 7, dtype=np.int8)
        for lag, mask in enumerate(mask_by_lag):
            update = mask & (result == 7)
            result[update] = lag
        return result

    action_masks_8 = []
    action_masks_9 = []
    action_masks_89 = []
    point0_masks = []
    zero_burden_masks = []
    for lag in range(6):
        exists = out[f"lag{lag}_exists"].to_numpy(dtype=bool)
        action = out[f"lag{lag}_actionId"].to_numpy(dtype=int)
        point = out[f"lag{lag}_pointId"].to_numpy(dtype=int)
        zero_burden = np.zeros(len(out), dtype=bool)
        for field in ZERO_BURDEN_FIELDS:
            zero_burden |= out[f"lag{lag}_{field}"].to_numpy(dtype=int) == 0
        action_masks_8.append(exists & (action == 8))
        action_masks_9.append(exists & (action == 9))
        action_masks_89.append(exists & ((action == 8) | (action == 9)))
        point0_masks.append(exists & (point == 0))
        zero_burden_masks.append(exists & zero_burden)

    out["time_since_action8_r30"] = first_distance(action_masks_8)
    out["time_since_action9_r30"] = first_distance(action_masks_9)
    out["time_since_action8or9_r30"] = first_distance(action_masks_89)
    out["time_since_point0_r30"] = first_distance(point0_masks)
    out["time_since_any_zero_field_r30"] = first_distance(zero_burden_masks)
    return out


def add_strength_switch_features(out: pd.DataFrame) -> pd.DataFrame:
    switches = np.zeros(len(out), dtype=np.int8)
    for lag in range(3):
        exists_pair = out[f"lag{lag + 1}_exists"].to_numpy(dtype=bool)
        cur = out[f"lag{lag}_strengthId"].to_numpy(dtype=int)
        prev = out[f"lag{lag + 1}_strengthId"].to_numpy(dtype=int)
        switches += (exists_pair & (cur != prev)).astype(np.int8)
    out["strength_switch_count_last4_r30"] = switches
    out["strength_delta_lag0_lag1_r30"] = np.where(
        out["lag1_exists"].astype(bool),
        out["lag0_strengthId"].astype(int) - out["lag1_strengthId"].astype(int),
        0,
    ).astype(np.int8)
    out["strength_delta_lag1_lag2_r30"] = np.where(
        out["lag2_exists"].astype(bool),
        out["lag1_strengthId"].astype(int) - out["lag2_strengthId"].astype(int),
        0,
    ).astype(np.int8)

    last_nonzero = np.zeros(len(out), dtype=np.int8)
    for lag in range(6):
        val = out[f"lag{lag}_strengthId"].to_numpy(dtype=int)
        exists = out[f"lag{lag}_exists"].to_numpy(dtype=bool)
        update = (last_nonzero == 0) & exists & (val > 0)
        last_nonzero[update] = val[update]
    out["last_nonzero_strength_r30"] = last_nonzero
    return out


def add_action_group_features(out: pd.DataFrame) -> pd.DataFrame:
    for lag in range(6):
        out[f"lag{lag}_action_group_r30"] = np.where(
            out[f"lag{lag}_exists"].astype(bool),
            action_group(out[f"lag{lag}_actionId"]),
            -1,
        ).astype(np.int8)

    out["last2_action_group_transition_r30"] = encode_pair(
        out["lag1_action_group_r30"], out["lag0_action_group_r30"], 5
    )
    out["last3_action_group_transition_r30"] = np.where(
        out["lag2_action_group_r30"].ge(0),
        out["lag2_action_group_r30"].astype(int) * 25
        + out["lag1_action_group_r30"].astype(int) * 5
        + out["lag0_action_group_r30"].astype(int),
        -1,
    ).astype(np.int16)
    out["action_group_attack_to_control_r30"] = (
        out["lag1_action_group_r30"].eq(1) & out["lag0_action_group_r30"].eq(2)
    ).astype(np.int8)
    out["action_group_control_to_attack_r30"] = (
        out["lag1_action_group_r30"].eq(2) & out["lag0_action_group_r30"].eq(1)
    ).astype(np.int8)

    run = np.ones(len(out), dtype=np.int8)
    lag0_group = out["lag0_action_group_r30"].to_numpy(dtype=int)
    for lag in range(1, 6):
        exists = out[f"lag{lag}_exists"].to_numpy(dtype=bool)
        same = out[f"lag{lag}_action_group_r30"].to_numpy(dtype=int) == lag0_group
        still_same = exists & same & (run == lag)
        run = np.where(still_same, run + 1, run)
    out["last_action_group_run_len_cap6_r30"] = run.astype(np.int8)
    return out


def add_incoming_signature_features(out: pd.DataFrame) -> pd.DataFrame:
    out["incoming_depth_r30"] = point_depth(out["lag0_pointId"])
    out["incoming_side_r30"] = point_side(out["lag0_pointId"])
    out["incoming_depth_spin_pair_r30"] = encode_pair(out["incoming_depth_r30"], out["lag0_spinId"], 6)
    out["incoming_side_action_group_pair_r30"] = encode_pair(
        out["incoming_side_r30"], out["lag0_action_group_r30"], 5
    )
    out["incoming_depth_strength_pair_r30"] = encode_pair(out["incoming_depth_r30"], out["lag0_strengthId"], 4)
    out["incoming_zone_response_signature_r30"] = np.where(
        out["incoming_depth_r30"].ge(0) & out["incoming_side_r30"].ge(0),
        out["incoming_depth_r30"].astype(int) * 2_000
        + out["incoming_side_r30"].astype(int) * 500
        + out["lag0_spinId"].astype(int) * 80
        + out["lag0_action_group_r30"].astype(int) * 10
        + out["lag0_strengthId"].astype(int),
        -1,
    ).astype(np.int32)
    return out


def add_score_structure_features(out: pd.DataFrame) -> pd.DataFrame:
    score_total = out["scoreTotal"].astype(int)
    server_score = out["serverScore"].astype(int)
    receiver_score = out["receiverScore"].astype(int)
    is_deuce = server_score.ge(10) & receiver_score.ge(10)

    out["serve_rotation_is_deuce_r30"] = is_deuce.astype(np.int8)
    out["serve_pair_idx_pre_deuce_r30"] = np.where(is_deuce, -1, (score_total // 2) % 2).astype(np.int8)
    out["serve_point_in_pair_pre_deuce_r30"] = np.where(is_deuce, -1, score_total % 2).astype(np.int8)
    out["serve_deuce_parity_r30"] = np.where(is_deuce, score_total % 2, -1).astype(np.int8)
    out["server_points_to_11_r30"] = np.maximum(0, 11 - server_score).astype(np.int8)
    out["receiver_points_to_11_r30"] = np.maximum(0, 11 - receiver_score).astype(np.int8)
    out["score_lead_abs_r30"] = np.abs(server_score - receiver_score).astype(np.int8)
    out["deuce_margin_r30"] = np.where(is_deuce, np.maximum(server_score, receiver_score) - 10, -1).astype(np.int8)
    out["can_server_close_next_r30"] = (server_score.ge(10) & (server_score - receiver_score).ge(1)).astype(np.int8)
    out["can_receiver_close_next_r30"] = (
        receiver_score.ge(10) & (receiver_score - server_score).ge(1)
    ).astype(np.int8)
    return out


def add_zero_burden_features(out: pd.DataFrame) -> pd.DataFrame:
    for lag in range(2):
        burden = np.zeros(len(out), dtype=np.int8)
        for field in ZERO_BURDEN_FIELDS:
            burden += (out[f"lag{lag}_{field}"].to_numpy(dtype=int) == 0).astype(np.int8)
        burden = np.where(out[f"lag{lag}_exists"].to_numpy(dtype=bool), burden, 0)
        out[f"lag{lag}_zero_burden_r30"] = burden.astype(np.int8)
    out["last_zero_burden_r30"] = out["lag0_zero_burden_r30"].astype(np.int8)
    out["zero_burden_last2_delta_r30"] = (
        out["lag0_zero_burden_r30"].astype(int) - out["lag1_zero_burden_r30"].astype(int)
    ).astype(np.int8)
    return out


def add_r30_features(prefix: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = prefix.copy()
    aggregates = build_r30_prefix_aggregates(raw)
    out = out.merge(aggregates, on=["rally_uid", "prefix_len"], how="left")

    out = add_zero_burden_features(out)
    out = add_run_length_features(out)
    out = add_time_since_features(out)
    out = add_strength_switch_features(out)
    out = add_action_group_features(out)
    out = add_incoming_signature_features(out)
    out = add_score_structure_features(out)

    if out.isna().any().any():
        bad_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"R30 feature table contains NaN in {bad_cols[:20]}")
    return out


def make_recommendation(
    metrics: dict[str, float],
    r7_metrics: dict[str, float] | None,
    cv_path: str,
    prefix_path: str,
    submission_path: str,
) -> str:
    lines = [
        "# R30 Recommendation",
        "",
        "R30 adds only low-risk structural prefix features on top of the V3/R7 protocol.",
        "It does not use old-test labels, future scoreboard information, target encoding, raw player IDs, or decoder tuning.",
        "",
        "## Selected OOF Metrics",
        "",
        f"- overall: {metrics['overall']:.6f}",
        f"- action: {metrics['action_macro_f1']:.6f}",
        f"- point: {metrics['point_macro_f1']:.6f}",
        f"- server: {metrics['server_auc']:.6f}",
    ]
    if r7_metrics:
        lines.extend(
            [
                "",
                "## Delta vs R7 Feature Report",
                "",
                f"- overall: {metrics['overall'] - r7_metrics['overall']:+.6f}",
                f"- action: {metrics['action_macro_f1'] - r7_metrics['action_macro_f1']:+.6f}",
                f"- point: {metrics['point_macro_f1'] - r7_metrics['point_macro_f1']:+.6f}",
                f"- server: {metrics['server_auc'] - r7_metrics['server_auc']:+.6f}",
            ]
        )
    verdict = "not_submit"
    if metrics["overall"] >= 0.316 and metrics["point_macro_f1"] >= 0.205:
        verdict = "candidate"
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- status: `{verdict}`",
            f"- cv report: `{cv_path}`",
            f"- prefix report: `{prefix_path}`",
            f"- submission candidate: `{submission_path}`",
            "",
            "Submit only if this branch beats the current safe reference under the same local comparison.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_metrics(path: str) -> dict[str, float] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = data.get("selected", {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {key: float(value) for key, value in metrics.items()}


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
    r7_feature_count = len([c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"])

    print("adding R30 low-risk structural features...")
    prefix_df = add_r30_features(prefix_df, train)
    test_prefix = add_r30_features(test_prefix, test)

    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    r30_features = [c for c in features if c.endswith("_r30")]

    print(f"train prefix rows: {len(prefix_df):,}")
    print(f"test prediction rows: {len(test_prefix):,}")
    print(f"feature count: {len(features)} ({len(r30_features)} new R30 features)")
    print(f"seeds: {args.seeds}")

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
        pickle.dump({**oof, "tuning": tuning}, f)

    print("selected tuning:")
    print(
        json.dumps(
            {
                "action_ngram_weight": tuning.action_ngram_weight,
                "point_ngram_weight": tuning.point_ngram_weight,
                "server_weights": tuning.server_weights,
                **tuning.metrics,
            },
            indent=2,
        )
    )

    print("training full-data models...")
    full_pred = full_predict(prefix_df, test_prefix, features, args)
    submission = write_submission(test_prefix, full_pred, tuning, Path(args.submission))

    metadata = {
        "experiment": "R30 low-risk structural features",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_df)),
        "test_prediction_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "r7_feature_count": int(r7_feature_count),
        "r30_feature_count": int(len(r30_features)),
        "r30_features": r30_features,
        "features": features,
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "ngram_key_levels": NGRAM_KEY_LEVELS,
        "ngram_alpha": args.ngram_alpha,
        "remaining_classes": REMAINING_CLASSES,
        "selected": {
            "action_ngram_weight": tuning.action_ngram_weight,
            "point_ngram_weight": tuning.point_ngram_weight,
            "server_weights": tuning.server_weights,
            "action_multipliers": tuning.action_multipliers,
            "point_multipliers": tuning.point_multipliers,
            "bins_mode": tuning.bins_mode,
            "metrics": tuning.metrics,
        },
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    r7_metrics = load_metrics("feature_report_r7.json")
    Path(args.recommendation).write_text(
        make_recommendation(tuning.metrics, r7_metrics, args.cv_report, args.prefix_len_report, args.submission),
        encoding="utf-8",
    )

    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")
    print(f"wrote {args.recommendation}")


if __name__ == "__main__":
    main()
