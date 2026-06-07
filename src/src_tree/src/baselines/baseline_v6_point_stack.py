"""V6 point stacking baseline.

This script targets the current bottleneck: pointId Macro-F1. It uses OOF
action probabilities from V3 tabular and V5 GRU models as leakage-safe stacking
features for a LightGBM point model. The final point probabilities are blended
with the V3 hierarchical point baseline and tuned with segmented multipliers.
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
from baseline_v3 import (
    add_remaining_bucket,
    apply_segmented_multipliers,
    full_predict as v3_full_predict,
    tune_segmented_multipliers,
)


# Compatibility for unpickling OOF files that were written when scripts ran as
# __main__.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V6 point stacking model.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--base-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--gru-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--submission", default="submission_v6.csv")
    parser.add_argument("--cv-report", default="cv_report_v6.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v6.csv")
    parser.add_argument("--class-report-point", default="class_report_v6_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v6.json")
    parser.add_argument("--oof-proba", default="oof_proba_v6.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "three", "five"], default="three")
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


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def add_prob_features(df: pd.DataFrame, prefix: str, prob: np.ndarray, classes: list[int]) -> pd.DataFrame:
    out = df.copy()
    for idx, cls in enumerate(classes):
        out[f"{prefix}_prob_{cls}"] = prob[:, idx]
    order = np.argsort(-prob, axis=1)
    out[f"{prefix}_top1"] = np.asarray(classes)[order[:, 0]]
    out[f"{prefix}_top2"] = np.asarray(classes)[order[:, 1]]
    out[f"{prefix}_entropy"] = entropy(prob)
    out[f"{prefix}_max"] = prob.max(axis=1)
    return out


def point_stack_features(
    base_features: pd.DataFrame,
    base_action: np.ndarray,
    base_point: np.ndarray,
    gru_action: np.ndarray | None,
    action_gru_weight: float,
    use_gru: bool,
) -> pd.DataFrame:
    x = base_features.reset_index(drop=True).copy()
    x = add_prob_features(x, "base_action", base_action, ACTION_CLASSES)
    x = add_prob_features(x, "base_point", base_point, POINT_CLASSES)
    x["terminal_prob"] = base_point[:, 0]
    if use_gru and gru_action is not None:
        blend_action = blend_probs(base_action, gru_action, action_gru_weight)
        x = add_prob_features(x, "gru_action", gru_action, ACTION_CLASSES)
        x = add_prob_features(x, "blend_action", blend_action, ACTION_CLASSES)
    return x


def stack_feature_columns(df: pd.DataFrame) -> list[str]:
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


def load_oofs(base_path: str, gru_path: str) -> dict[str, object]:
    with open(base_path, "rb") as f:
        base = pickle.load(f)
    with open(gru_path, "rb") as f:
        gru = pickle.load(f)
    base_action, base_point, base_server = compose_v3_predictions(base, base["tuning"])
    meta = base["valid_meta"].reset_index(drop=True)
    gru_meta = gru["valid_meta"].reset_index(drop=True)
    if not meta[["rally_uid", "prefix_len"]].equals(gru_meta[["rally_uid", "prefix_len"]]):
        raise ValueError("V3 and V5 OOF rows are not aligned.")
    return {
        "meta": meta,
        "base_action": base_action,
        "base_point": base_point,
        "base_server": base_server,
        "gru_action": gru["gru_action"],
        "gru_point": gru["gru_point"],
        "gru_server": gru["gru_server"],
        "gru_tuning": gru["tuning"],
        "base_tuning": base["tuning"],
    }


def merge_prefix_features(prefix_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keys = ["rally_uid", "prefix_len"]
    merged = meta[keys].merge(prefix_df, on=keys, how="left", validate="one_to_one")
    if merged.isna().any().any():
        missing = int(merged.isna().any(axis=1).sum())
        raise ValueError(f"Missing prefix features for {missing} OOF rows.")
    return merged


def make_point_model(n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    model = make_lgbm("multiclass", n_estimators, seed, num_class=len(POINT_CLASSES))
    model.set_params(
        learning_rate=0.035,
        num_leaves=47,
        min_child_samples=30,
        reg_alpha=0.08,
        reg_lambda=1.2,
    )
    return model


def run_stack_oof(
    stack_df: pd.DataFrame,
    features: list[str],
    folds: int,
    n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    out = np.zeros((len(stack_df), len(POINT_CLASSES)), dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(stack_df, groups=stack_df["match"]), start=1):
        train_part = stack_df.iloc[tr_idx]
        valid_part = stack_df.iloc[va_idx]
        if set(train_part["match"]) & set(valid_part["match"]):
            raise RuntimeError("Stacking fold match leakage.")
        model = make_point_model(n_estimators, seed + fold * 100)
        model.fit(
            train_part[features],
            train_part["next_pointId"],
            sample_weight=class_weight_sample(train_part["next_pointId"]),
        )
        prob = aligned_proba(model, valid_part[features], POINT_CLASSES)
        out[va_idx] = prob
        pred = np.asarray(POINT_CLASSES)[np.argmax(prob, axis=1)]
        point_f1 = f1_score(valid_part["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
        fold_rows.append({"fold": fold, "stack_point_macro_f1": float(point_f1), "valid_rows": int(len(valid_part))})
        print(f"stack fold {fold}: point_f1={point_f1:.6f}")
    return out, pd.DataFrame(fold_rows)


def bin_masks(meta: pd.DataFrame, mode: str) -> list[tuple[str, np.ndarray]]:
    prefix = meta["prefix_len"]
    if mode == "global":
        return [("global", np.ones(len(meta), dtype=bool))]
    if mode == "two":
        return [("le2", prefix.le(2).to_numpy()), ("ge3", prefix.ge(3).to_numpy())]
    if mode == "three":
        return [
            ("1", prefix.eq(1).to_numpy()),
            ("2", prefix.eq(2).to_numpy()),
            ("ge3", prefix.ge(3).to_numpy()),
        ]
    return [
        ("1", prefix.eq(1).to_numpy()),
        ("2", prefix.eq(2).to_numpy()),
        ("3", prefix.eq(3).to_numpy()),
        ("4-6", prefix.between(4, 6).to_numpy()),
        ("7+", prefix.ge(7).to_numpy()),
    ]


def blend_by_bins(meta: pd.DataFrame, base: np.ndarray, stack: np.ndarray, weights: dict[str, float], mode: str) -> np.ndarray:
    out = np.zeros_like(base)
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        w = weights[label]
        out[idx] = blend_probs(base[idx], stack[idx], w)
    return out


def tune_stack_weights(meta: pd.DataFrame, base: np.ndarray, stack: np.ndarray, mode: str) -> dict[str, float]:
    grid = [round(x, 1) for x in np.arange(0.0, 0.8, 0.1)]
    weights: dict[str, float] = {}
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) < 200:
            weights[label] = 0.0
            continue
        y = meta.iloc[idx]["next_pointId"].to_numpy()
        best = max(
            grid,
            key=lambda w: f1_score(
                y,
                np.asarray(POINT_CLASSES)[np.argmax(blend_probs(base[idx], stack[idx], w), axis=1)],
                average="macro",
                labels=POINT_CLASSES,
                zero_division=0,
            ),
        )
        weights[label] = float(best)
    return weights


def tune_point_multipliers(meta: pd.DataFrame, point_prob: np.ndarray, mode: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    # Include a finer terminal grid by letting the greedy search consider 0.4/0.6.
    values = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    global_mult = tune_segmented_multipliers(meta, point_prob, POINT_CLASSES, "point", "global")["global"]
    if mode == "global":
        # Re-run with the terminal-focused grid for all classes.
        result["global"] = _greedy_point(meta["next_pointId"].to_numpy(), point_prob, values)
        return result
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) < 250:
            result[label] = global_mult
        else:
            result[label] = _greedy_point(meta.iloc[idx]["next_pointId"].to_numpy(), point_prob[idx], values)
    return result


def _greedy_point(y: np.ndarray, prob: np.ndarray, values: list[float], passes: int = 2) -> list[float]:
    mult = np.ones(len(POINT_CLASSES), dtype=float)

    def metric(m: np.ndarray) -> float:
        pred = np.asarray(POINT_CLASSES)[np.argmax(prob * m[None, :], axis=1)]
        return float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0))

    best = metric(mult)
    for _ in range(passes):
        improved = False
        for idx in range(len(POINT_CLASSES)):
            old = mult[idx]
            local_best = best
            local_value = old
            for value in values:
                mult[idx] = value
                score = metric(mult)
                if score > local_best + 1e-12:
                    local_best = score
                    local_value = value
            mult[idx] = local_value
            if local_best > best + 1e-12:
                best = local_best
                improved = True
            else:
                mult[idx] = old
        if not improved:
            break
    return mult.tolist()


def apply_point_multipliers(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict[str, list[float]], mode: str) -> np.ndarray:
    pred = np.zeros(len(meta), dtype=int)
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        mult = np.asarray(multipliers[label], dtype=float)
        pred[idx] = np.asarray(POINT_CLASSES)[np.argmax(prob[idx] * mult[None, :], axis=1)]
    return pred


def point_metrics(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict[str, list[float]], mode: str) -> tuple[float, np.ndarray]:
    pred = apply_point_multipliers(meta, prob, multipliers, mode)
    f1 = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    return float(f1), pred


def overall_metrics(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    point_pred: np.ndarray,
    action_multipliers: dict[str, list[float]] | None = None,
    action_bins_mode: str = "global",
) -> dict[str, float]:
    if action_multipliers is None:
        action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    else:
        action_pred = apply_segmented_multipliers(
            meta, action_prob, action_multipliers, ACTION_CLASSES, action_bins_mode
        )
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def fit_full_stack(stack_df: pd.DataFrame, features: list[str], n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    model = make_point_model(n_estimators, seed)
    model.fit(stack_df[features], stack_df["next_pointId"], sample_weight=class_weight_sample(stack_df["next_pointId"]))
    return model


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("loading OOF probabilities...")
    oofs = load_oofs(args.base_oof, args.gru_oof)
    meta = oofs["meta"].copy()

    print("building prefix features...")
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    base_feature_df = merge_prefix_features(prefix_df, meta)
    base_cols = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]

    variants = {}
    base_point_pred = apply_segmented_multipliers(
        meta,
        oofs["base_point"],
        oofs["base_tuning"].point_multipliers,
        POINT_CLASSES,
        oofs["base_tuning"].bins_mode,
    )
    base_point_f1 = f1_score(
        meta["next_pointId"], base_point_pred, average="macro", labels=POINT_CLASSES, zero_division=0
    )
    variants["base_v3_point"] = {
        "features": [],
        "stack_prob": oofs["base_point"],
        "stack_fold": pd.DataFrame(),
        "stack_weights": {"global": 0.0},
        "multipliers": oofs["base_tuning"].point_multipliers,
        "point_f1": float(base_point_f1),
        "metrics": overall_metrics(
            meta,
            oofs["base_action"],
            oofs["base_point"],
            oofs["base_server"],
            base_point_pred,
            oofs["base_tuning"].action_multipliers,
            oofs["base_tuning"].bins_mode,
        ),
        "use_gru": False,
        "stack_df": pd.DataFrame(),
        "is_base": True,
    }
    for use_gru in [False, True]:
        name = "with_gru_action" if use_gru else "base_action_only"
        x = point_stack_features(
            base_feature_df[["rally_uid", "match", "next_pointId"] + base_cols],
            oofs["base_action"],
            oofs["base_point"],
            oofs["gru_action"],
            float(oofs["gru_tuning"].action_gru_weight),
            use_gru,
        )
        x["next_pointId"] = meta["next_pointId"].to_numpy()
        x["match"] = meta["match"].to_numpy()
        x["rally_uid"] = meta["rally_uid"].to_numpy()
        features = stack_feature_columns(x)
        print(f"running stack OOF variant={name}, features={len(features)}")
        stack_prob, stack_fold = run_stack_oof(x, features, args.folds, args.n_estimators, args.seed)
        stack_weights = tune_stack_weights(meta, oofs["base_point"], stack_prob, args.multiplier_bins)
        blended_point = blend_by_bins(meta, oofs["base_point"], stack_prob, stack_weights, args.multiplier_bins)
        multipliers = tune_point_multipliers(meta, blended_point, args.multiplier_bins)
        point_f1, point_pred = point_metrics(meta, blended_point, multipliers, args.multiplier_bins)
        metrics = overall_metrics(
            meta,
            oofs["base_action"],
            blended_point,
            oofs["base_server"],
            point_pred,
            oofs["base_tuning"].action_multipliers,
            oofs["base_tuning"].bins_mode,
        )
        variants[name] = {
            "features": features,
            "stack_prob": stack_prob,
            "stack_fold": stack_fold,
            "stack_weights": stack_weights,
            "multipliers": multipliers,
            "point_f1": point_f1,
            "metrics": metrics,
            "use_gru": use_gru,
            "stack_df": x,
            "is_base": False,
        }
        print(f"{name}: point={point_f1:.6f} overall={metrics['overall']:.6f} weights={stack_weights}")

    best_name = max(variants, key=lambda k: variants[k]["metrics"]["overall"])
    best = variants[best_name]
    print("selected variant:")
    print(json.dumps({"name": best_name, "metrics": best["metrics"], "weights": best["stack_weights"]}, indent=2))

    if best.get("is_base"):
        point_pred = apply_segmented_multipliers(
            meta,
            oofs["base_point"],
            best["multipliers"],
            POINT_CLASSES,
            oofs["base_tuning"].bins_mode,
        )
    else:
        point_pred = apply_point_multipliers(
            meta,
            blend_by_bins(meta, oofs["base_point"], best["stack_prob"], best["stack_weights"], args.multiplier_bins),
            best["multipliers"],
            args.multiplier_bins,
        )
    class_report = pd.DataFrame(
        classification_report(meta["next_pointId"], point_pred, labels=POINT_CLASSES, zero_division=0, output_dict=True)
    ).T
    class_report.to_csv(args.class_report_point)
    rows = []
    if best.get("is_base"):
        blended_best = oofs["base_point"]
    else:
        blended_best = blend_by_bins(meta, oofs["base_point"], best["stack_prob"], best["stack_weights"], args.multiplier_bins)
    report_mode = oofs["base_tuning"].bins_mode if best.get("is_base") else args.multiplier_bins
    for label, mask in bin_masks(meta, report_mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        mult = np.asarray(best["multipliers"][label], dtype=float)
        pred = np.asarray(POINT_CLASSES)[np.argmax(blended_best[idx] * mult[None, :], axis=1)]
        pf1 = f1_score(
            meta.iloc[idx]["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0
        )
        rows.append({"prefix_len_bin": label, "count": int(len(idx)), "point_macro_f1": pf1})
    pd.DataFrame(rows).to_csv(args.prefix_len_report, index=False)

    # CV report
    report_rows = []
    for name, data in variants.items():
        row = {"variant": name, **data["metrics"]}
        row.update({f"stack_weight_{k}": v for k, v in data["stack_weights"].items()})
        report_rows.append(row)
    pd.DataFrame(report_rows).to_csv(args.cv_report, index=False)

    # Build test stack features and submission for the selected variant.
    print("training full stack model and writing submission...")
    test_prefix = build_test_prefix_table(test, args.max_lag)
    test_features_base = test_prefix[["rally_uid", "match"] + base_cols].copy()
    v3_args = SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0)
    full_features_for_v3 = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    full_test_prefix = test_prefix[["rally_uid", "match"] + full_features_for_v3]
    tab_pred = v3_full_predict(prefix_df, full_test_prefix, full_features_for_v3, v3_args)
    tab_action_test, tab_point_test, tab_server_test = compose_v3_predictions(tab_pred, oofs["base_tuning"])
    # V6 does not train a full GRU here. If GRU-action variant wins, use the
    # V5-selected blend weight but fall back to tabular action for test-time GRU
    # placeholders; this keeps submission generation deterministic and avoids a
    # hidden second sequence training run. CV is the source of truth for whether
    # this variant should be submitted.
    test_x = point_stack_features(
        test_features_base,
        tab_action_test,
        tab_point_test,
        None,
        float(oofs["gru_tuning"].action_gru_weight),
        False,
    )
    if best.get("is_base"):
        point_pred_test = apply_segmented_multipliers(
            test_prefix,
            tab_point_test,
            oofs["base_tuning"].point_multipliers,
            POINT_CLASSES,
            oofs["base_tuning"].bins_mode,
        )
    elif best["use_gru"]:
        print("selected CV variant used GRU action; submission falls back to base-action stack features.")
        # Refit compatible no-GRU stack for a conservative submission.
        submission_variant = variants["base_action_only"]
        submission_features = submission_variant["features"]
        model = fit_full_stack(submission_variant["stack_df"], submission_features, args.n_estimators, args.seed)
        stack_prob_test = aligned_proba(model, test_x[submission_features], POINT_CLASSES)
        stack_weights = submission_variant["stack_weights"]
        multipliers = submission_variant["multipliers"]
        point_test = blend_by_bins(test_prefix, tab_point_test, stack_prob_test, stack_weights, args.multiplier_bins)
        point_pred_test = apply_point_multipliers(test_prefix, point_test, multipliers, args.multiplier_bins)
    else:
        model = fit_full_stack(best["stack_df"], best["features"], args.n_estimators, args.seed)
        stack_prob_test = aligned_proba(model, test_x[best["features"]], POINT_CLASSES)
        stack_weights = best["stack_weights"]
        multipliers = best["multipliers"]
        point_test = blend_by_bins(test_prefix, tab_point_test, stack_prob_test, stack_weights, args.multiplier_bins)
        point_pred_test = apply_point_multipliers(test_prefix, point_test, multipliers, args.multiplier_bins)

    # Preserve V3 action/server predictions for submission.
    action_pred_test = apply_segmented_multipliers(
        test_prefix, tab_action_test, oofs["base_tuning"].action_multipliers, ACTION_CLASSES, oofs["base_tuning"].bins_mode
    )
    submission = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred_test.astype(int),
            "pointId": point_pred_test.astype(int),
            "serverGetPoint": np.round(np.clip(tab_server_test, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    submission.to_csv(args.submission, index=False, float_format="%.8f")

    with open(args.oof_proba, "wb") as f:
        pickle.dump(
            {
                "meta": meta,
                "variants": {k: {kk: vv for kk, vv in v.items() if kk != "stack_df"} for k, v in variants.items()},
                "selected": best_name,
            },
            f,
        )
    metadata = {
        "selected": best_name,
        "metrics": best["metrics"],
        "stack_weights": best["stack_weights"],
        "multipliers": best["multipliers"],
        "base_cols": base_cols,
        "args": vars(args),
        "note": "Submission uses V3 action/server; point uses selected stack unless GRU-action variant requires unavailable full GRU features.",
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
