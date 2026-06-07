"""Generate full-test candidates for the R33 safe OOF ensemble.

Produces:
- submission_r33_oof_selected.csv: follows R33 OOF selected action/point/server.
- submission_r33_safe_point.csv: same action/server, but point fixed to V3.
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
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, full_predict
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
    parser = argparse.ArgumentParser(description="Generate R33 full-test candidates.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--selected", default="r33_safe_oof_ensemble/r33_selected.json")
    parser.add_argument("--r7-full-proba", default="r7_full_lgbm_proba.pkl")
    parser.add_argument("--submission-selected", default="submission_r33_oof_selected.csv")
    parser.add_argument("--submission-safe-point", default="submission_r33_safe_point.csv")
    parser.add_argument("--feature-report", default="feature_report_r33.json")
    parser.add_argument("--reuse-r7-proba", action="store_true")
    return parser.parse_args()


def compose_r7_full(train: pd.DataFrame, test: pd.DataFrame, r7_tuning: V3Tuning, out_path: Path) -> dict[str, object]:
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    pred = full_predict(prefix_df, test_prefix, features, SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0))
    action = blend_probs(pred["lgbm_action"], pred["ngram_action"], r7_tuning.action_ngram_weight)
    point = blend_probs(pred["lgbm_point"], pred["ngram_point"], r7_tuning.point_ngram_weight)
    sw = r7_tuning.server_weights
    server = (
        sw["direct"] * pred["lgbm_server"]
        + sw["ngram"] * pred["ngram_server"]
        + sw["parity"] * pred["parity_server"]
        + sw["remaining"] * pred["remaining_server"]
    )
    out = {
        "test_prefix": test_prefix[["rally_uid", "prefix_len"]].copy(),
        "r7_action": action,
        "r7_point": point,
        "r7_server": np.clip(server, 1e-6, 1.0 - 1e-6),
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    return out


def write_submission(
    test_meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    action_mult: dict,
    point_mult: dict,
    point_mode: str,
    path: Path,
) -> pd.DataFrame:
    action_pred = apply_segmented_multipliers(test_meta, action_prob, action_mult, ACTION_CLASSES, "two")
    point_pred = apply_segmented_multipliers(test_meta, point_prob, point_mult, POINT_CLASSES, point_mode)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if sub.isna().any().any():
        raise ValueError(f"{path} contains NaN.")
    sub.to_csv(path, index=False, float_format="%.8f")
    return sub


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))

    with open(args.v3_oof, "rb") as f:
        v3_oof = pickle.load(f)
    with open(args.r7_oof, "rb") as f:
        r7_oof = pickle.load(f)
    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10_full = pickle.load(f)

    test_prefix, _, v3_point, v3_server = compose_v3_full(train, test, v3_oof["tuning"])
    if args.reuse_r7_proba and Path(args.r7_full_proba).exists():
        with open(args.r7_full_proba, "rb") as f:
            r7_full = pickle.load(f)
    else:
        r7_full = compose_r7_full(train, test, r7_oof["tuning"], Path(args.r7_full_proba))

    test_meta = r1_seq["test_meta"].reset_index(drop=True)
    for name, ids in [
        ("V3", test_prefix["rally_uid"].reset_index(drop=True)),
        ("R7", r7_full["test_prefix"]["rally_uid"].reset_index(drop=True)),
        ("V10B", v10_full["test_meta"]["rally_uid"].reset_index(drop=True)),
    ]:
        if not test_meta["rally_uid"].reset_index(drop=True).equals(ids):
            raise ValueError(f"{name} full-test rows are not aligned.")

    r1_action = 0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"]
    r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]

    # R33 selected: action base+0.05*r7+0.1*v5, server base+0.15*v10b+0.15*r7.
    action_prob = 0.85 * r1_action + 0.05 * r7_full["r7_action"] + 0.10 * r1_seq["gru_action"]
    action_prob = action_prob / action_prob.sum(axis=1, keepdims=True)
    server_prob = 0.70 * r1_server + 0.15 * v10_full["v10_server"] + 0.15 * r7_full["r7_server"]
    server_prob = np.clip(server_prob, 1e-6, 1.0 - 1e-6)

    point_selected = blend_probs(v3_point, v10_full["v10_point"], 0.10)
    selected_sub = write_submission(
        test_meta,
        action_prob,
        point_selected,
        server_prob,
        selected["action_multipliers"],
        selected["point_multipliers"],
        "two",
        Path(args.submission_selected),
    )
    safe_sub = write_submission(
        test_meta,
        action_prob,
        v3_point,
        server_prob,
        selected["action_multipliers"],
        v3_oof["tuning"].point_multipliers,
        v3_oof["tuning"].bins_mode,
        Path(args.submission_safe_point),
    )

    metadata = {
        "source": "R33 safe OOF ensemble",
        "oof_selected": {k: v for k, v in selected.items() if not k.endswith("multipliers")},
        "full_action_weights": {"r1": 0.85, "r7": 0.05, "v5_gru": 0.10},
        "full_server_weights": {"r1": 0.70, "v10b": 0.15, "r7": 0.15},
        "selected_point_policy": "v3 + 0.10*v10b with R33 selected point multipliers",
        "safe_point_policy": "v3 point probabilities and v3 point multipliers",
        "rows_selected": int(len(selected_sub)),
        "rows_safe": int(len(safe_sub)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.submission_selected} ({len(selected_sub):,} rows)")
    print(f"wrote {args.submission_safe_point} ({len(safe_sub):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
