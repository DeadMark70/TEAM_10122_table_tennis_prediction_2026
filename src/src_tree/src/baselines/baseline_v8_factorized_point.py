"""V8 receiver-relative factorized point model.

pointId is not an absolute left/right court coordinate. It is defined from the
receiving player's forehand/backhand perspective. This script models pointId as:
- terminal vs non-terminal
- direct non-terminal point 1..9
- receiver-relative depth: short / half-long / long
- receiver-relative side: forehand / middle / backhand

It only changes point probabilities; action/server are preserved from V3.
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
    POINT_NONTERMINAL_CLASSES,
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


DEPTH_MAP = {
    1: 0, 2: 0, 3: 0,  # short
    4: 1, 5: 1, 6: 1,  # half-long
    7: 2, 8: 2, 9: 2,  # long
}
SIDE_MAP = {
    1: 0, 4: 0, 7: 0,  # receiver-relative forehand
    2: 1, 5: 1, 8: 1,  # middle
    3: 2, 6: 2, 9: 2,  # receiver-relative backhand
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V8 factorized receiver-relative point model.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--base-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--submission", default="submission_v8.csv")
    parser.add_argument("--cv-report", default="cv_report_v8.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_v8.csv")
    parser.add_argument("--class-report-point", default="class_report_v8_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v8.json")
    parser.add_argument("--oof-proba", default="oof_proba_v8.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--alpha-bins", choices=["global", "two", "three", "five"], default="three")
    parser.add_argument("--multiplier-bins", choices=["global", "two", "three", "five"], default="two")
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


def load_base(path: str) -> dict[str, object]:
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
    for col in ["match", "next_pointId"]:
        if f"{col}_x" in merged.columns:
            merged[col] = merged[f"{col}_x"]
            merged = merged.drop(columns=[c for c in [f"{col}_x", f"{col}_y"] if c in merged.columns])
    if merged.isna().any().any():
        raise ValueError("Missing merged prefix features.")
    return merged


def add_receiver_relative_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["point_terminal"] = out["next_pointId"].eq(0).astype(int)
    out["point_nonterminal"] = out["next_pointId"].clip(lower=1)
    out["point_depth"] = out["next_pointId"].map(DEPTH_MAP).fillna(-1).astype(int)
    out["point_side"] = out["next_pointId"].map(SIDE_MAP).fillna(-1).astype(int)
    return out


def model_features(df: pd.DataFrame) -> list[str]:
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
        "point_terminal",
        "point_nonterminal",
        "point_depth",
        "point_side",
    }
    return [c for c in df.columns if c not in forbidden]


def point_model(objective: str, n_estimators: int, seed: int, num_class: int | None = None) -> lgb.LGBMClassifier:
    model = make_lgbm(objective, n_estimators, seed, num_class=num_class)
    model.set_params(
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=25,
        reg_alpha=0.06,
        reg_lambda=1.1,
    )
    return model


def fit_factorized_oof(df: pd.DataFrame, features: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(df)
    direct = np.zeros((n, 9), dtype=float)
    depth = np.zeros((n, 3), dtype=float)
    side = np.zeros((n, 3), dtype=float)
    terminal = np.zeros(n, dtype=float)
    rows = []
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, groups=df["match"]), start=1):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]
        term = point_model("binary", args.n_estimators, args.seed + fold)
        term.fit(tr[features], tr["point_terminal"])
        terminal[va_idx] = term.predict_proba(va[features])[:, 1]

        tr_nt = tr[tr["next_pointId"].gt(0)].copy()
        direct_model = point_model("multiclass", args.n_estimators, args.seed + 100 + fold, num_class=9)
        direct_model.fit(
            tr_nt[features],
            tr_nt["next_pointId"],
            sample_weight=class_weight_sample(tr_nt["next_pointId"]),
        )
        direct[va_idx] = aligned_proba(direct_model, va[features], POINT_NONTERMINAL_CLASSES)

        depth_model = point_model("multiclass", args.n_estimators, args.seed + 200 + fold, num_class=3)
        depth_model.fit(
            tr_nt[features],
            tr_nt["point_depth"],
            sample_weight=class_weight_sample(tr_nt["point_depth"]),
        )
        depth[va_idx] = aligned_proba(depth_model, va[features], [0, 1, 2])

        side_model = point_model("multiclass", args.n_estimators, args.seed + 300 + fold, num_class=3)
        side_model.fit(
            tr_nt[features],
            tr_nt["point_side"],
            sample_weight=class_weight_sample(tr_nt["point_side"]),
        )
        side[va_idx] = aligned_proba(side_model, va[features], [0, 1, 2])

        rows.append({"fold": fold, "valid_rows": int(len(va))})
        print(f"fold {fold} factorized models done")

    factor = np.zeros((n, 9), dtype=float)
    for point_id in range(1, 10):
        factor[:, point_id - 1] = depth[:, DEPTH_MAP[point_id]] * side[:, SIDE_MAP[point_id]]
    factor = factor / factor.sum(axis=1, keepdims=True)
    return direct, compose_full_point(terminal, factor), pd.DataFrame(rows)


def compose_full_point(terminal: np.ndarray, nonterminal9: np.ndarray) -> np.ndarray:
    terminal = np.clip(terminal.astype(float), 1e-6, 1.0 - 1e-6)
    p = np.zeros((len(terminal), 10), dtype=float)
    p[:, 0] = terminal
    p[:, 1:] = (1.0 - terminal[:, None]) * nonterminal9
    return p / p.sum(axis=1, keepdims=True)


def bin_masks(meta: pd.DataFrame, mode: str) -> list[tuple[str, np.ndarray]]:
    prefix = meta["prefix_len"]
    if mode == "global":
        return [("global", np.ones(len(meta), dtype=bool))]
    if mode == "two":
        return [("le2", prefix.le(2).to_numpy()), ("ge3", prefix.ge(3).to_numpy())]
    if mode == "three":
        return [("1", prefix.eq(1).to_numpy()), ("2", prefix.eq(2).to_numpy()), ("ge3", prefix.ge(3).to_numpy())]
    return [
        ("1", prefix.eq(1).to_numpy()),
        ("2", prefix.eq(2).to_numpy()),
        ("3", prefix.eq(3).to_numpy()),
        ("4-6", prefix.between(4, 6).to_numpy()),
        ("7+", prefix.ge(7).to_numpy()),
    ]


def blend_factorized(meta: pd.DataFrame, direct_full: np.ndarray, factor_full: np.ndarray, alphas: dict[str, float], mode: str) -> np.ndarray:
    out = np.zeros_like(direct_full)
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        alpha = alphas[label]
        out[idx] = alpha * direct_full[idx] + (1.0 - alpha) * factor_full[idx]
    return out / out.sum(axis=1, keepdims=True)


def tune_alpha(meta: pd.DataFrame, direct_full: np.ndarray, factor_full: np.ndarray, mode: str) -> dict[str, float]:
    grid = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
    result = {}
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) < 200:
            result[label] = 1.0
            continue
        y = meta.iloc[idx]["next_pointId"]
        best = max(
            grid,
            key=lambda a: f1_score(
                y,
                np.asarray(POINT_CLASSES)[np.argmax(a * direct_full[idx] + (1.0 - a) * factor_full[idx], axis=1)],
                average="macro",
                labels=POINT_CLASSES,
                zero_division=0,
            ),
        )
        result[label] = float(best)
    return result


def greedy_point(y: np.ndarray, prob: np.ndarray, values: list[float] | None = None) -> list[float]:
    if values is None:
        values = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    mult = np.ones(10, dtype=float)

    def metric(m):
        pred = np.asarray(POINT_CLASSES)[np.argmax(prob * m[None, :], axis=1)]
        return float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0))

    best = metric(mult)
    for _ in range(2):
        improved = False
        for i in range(10):
            old = mult[i]
            local_best = best
            local_value = old
            for value in values:
                mult[i] = value
                score = metric(mult)
                if score > local_best + 1e-12:
                    local_best = score
                    local_value = value
            mult[i] = local_value
            if local_best > best + 1e-12:
                best = local_best
                improved = True
            else:
                mult[i] = old
        if not improved:
            break
    return mult.tolist()


def tune_multipliers(meta: pd.DataFrame, point: np.ndarray, mode: str) -> dict[str, list[float]]:
    global_m = greedy_point(meta["next_pointId"].to_numpy(), point)
    if mode == "global":
        return {"global": global_m}
    result = {}
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) < 250:
            result[label] = global_m
        else:
            result[label] = greedy_point(meta.iloc[idx]["next_pointId"].to_numpy(), point[idx])
    return result


def apply_mult(meta: pd.DataFrame, point: np.ndarray, mults: dict[str, list[float]], mode: str) -> np.ndarray:
    pred = np.zeros(len(meta), dtype=int)
    for label, mask in bin_masks(meta, mode):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        m = np.asarray(mults[label], dtype=float)
        pred[idx] = np.asarray(POINT_CLASSES)[np.argmax(point[idx] * m[None, :], axis=1)]
    return pred


def score(meta, action, point, server, tuning, point_mults, point_mode):
    action_pred = apply_segmented_multipliers(meta, action, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_mult(meta, point, point_mults, point_mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }, point_pred


def fit_full_models(df: pd.DataFrame, features: list[str], args: argparse.Namespace):
    term = point_model("binary", args.n_estimators, args.seed)
    term.fit(df[features], df["point_terminal"])
    nt = df[df["next_pointId"].gt(0)].copy()
    direct = point_model("multiclass", args.n_estimators, args.seed + 1000, num_class=9)
    direct.fit(nt[features], nt["next_pointId"], sample_weight=class_weight_sample(nt["next_pointId"]))
    depth = point_model("multiclass", args.n_estimators, args.seed + 2000, num_class=3)
    depth.fit(nt[features], nt["point_depth"], sample_weight=class_weight_sample(nt["point_depth"]))
    side = point_model("multiclass", args.n_estimators, args.seed + 3000, num_class=3)
    side.fit(nt[features], nt["point_side"], sample_weight=class_weight_sample(nt["point_side"]))
    return term, direct, depth, side


def predict_full_models(models, x: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    term, direct_model, depth_model, side_model = models
    terminal = term.predict_proba(x[features])[:, 1]
    direct9 = aligned_proba(direct_model, x[features], POINT_NONTERMINAL_CLASSES)
    depth = aligned_proba(depth_model, x[features], [0, 1, 2])
    side = aligned_proba(side_model, x[features], [0, 1, 2])
    factor9 = np.zeros_like(direct9)
    for point_id in range(1, 10):
        factor9[:, point_id - 1] = depth[:, DEPTH_MAP[point_id]] * side[:, SIDE_MAP[point_id]]
    factor9 = factor9 / factor9.sum(axis=1, keepdims=True)
    return compose_full_point(terminal, direct9), compose_full_point(terminal, factor9)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    base = load_base(args.base_oof)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    merged = add_receiver_relative_labels(merge_prefix(prefix_df, base["meta"]))
    features = model_features(merged)
    print(f"V8 features: {len(features)}")
    direct9, factor_full, fold_report = fit_factorized_oof(merged, features, args)
    # Direct full point uses the same terminal model as factorized, but direct 1..9.
    terminal_from_factor = factor_full[:, 0]
    direct_full = compose_full_point(terminal_from_factor, direct9)
    alphas = tune_alpha(base["meta"], direct_full, factor_full, args.alpha_bins)
    point = blend_factorized(base["meta"], direct_full, factor_full, alphas, args.alpha_bins)
    mults = tune_multipliers(base["meta"], point, args.multiplier_bins)
    v8_metrics, point_pred = score(base["meta"], base["action"], point, base["server"], base["tuning"], mults, args.multiplier_bins)
    base_metrics, base_pred = score(base["meta"], base["action"], base["point"], base["server"], base["tuning"], base["tuning"].point_multipliers, base["tuning"].bins_mode)
    print("base:", json.dumps(base_metrics, indent=2))
    print("v8:", json.dumps({**v8_metrics, "alphas": alphas}, indent=2))

    rows = [
        {"variant": "base_v3", **base_metrics},
        {"variant": "v8_factorized", **v8_metrics},
    ]
    pd.DataFrame(rows).to_csv(args.cv_report, index=False)
    pd.DataFrame(
        classification_report(base["meta"]["next_pointId"], point_pred, labels=POINT_CLASSES, zero_division=0, output_dict=True)
    ).T.to_csv(args.class_report_point)
    pr_rows = []
    for label, mask in bin_masks(base["meta"], args.multiplier_bins):
        idx = np.where(mask)[0]
        if len(idx):
            pf1 = f1_score(base["meta"].iloc[idx]["next_pointId"], point_pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0)
            pr_rows.append({"prefix_len_bin": label, "count": int(len(idx)), "point_macro_f1": float(pf1)})
    pd.DataFrame(pr_rows).to_csv(args.prefix_len_report, index=False)

    # Submission: only apply V8 if it improves OOF point; otherwise fallback to V3.
    test_prefix = build_test_prefix_table(test, args.max_lag)
    v3_args = SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0)
    full_features_for_v3 = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    full_test_prefix = test_prefix[["rally_uid", "match"] + full_features_for_v3]
    tab_pred = v3_full_predict(prefix_df, full_test_prefix, full_features_for_v3, v3_args)
    action_test, point_test, server_test = compose_v3_predictions(tab_pred, base["tuning"])
    selected = "base_v3"
    if v8_metrics["point_macro_f1"] > base_metrics["point_macro_f1"]:
        test_merged = add_receiver_relative_labels(test_prefix.copy())
        models = fit_full_models(merged, features, args)
        direct_test, factor_test = predict_full_models(models, test_merged, features)
        point_test = blend_factorized(test_prefix, direct_test, factor_test, alphas, args.alpha_bins)
        selected = "v8_factorized"
        point_pred_test = apply_mult(test_prefix, point_test, mults, args.multiplier_bins)
    else:
        point_pred_test = apply_segmented_multipliers(test_prefix, point_test, base["tuning"].point_multipliers, POINT_CLASSES, base["tuning"].bins_mode)
    action_pred_test = apply_segmented_multipliers(test_prefix, action_test, base["tuning"].action_multipliers, ACTION_CLASSES, base["tuning"].bins_mode)
    submission = pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred_test.astype(int),
            "pointId": point_pred_test.astype(int),
            "serverGetPoint": np.round(np.clip(server_test, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    submission.to_csv(args.submission, index=False, float_format="%.8f")
    with open(args.oof_proba, "wb") as f:
        pickle.dump({"direct": direct_full, "factor": factor_full, "point": point, "alphas": alphas, "mults": mults, "metrics": v8_metrics, "selected": selected}, f)
    metadata = {
        "selected": selected,
        "base_metrics": base_metrics,
        "v8_metrics": v8_metrics,
        "alphas": alphas,
        "multipliers": mults,
        "features": features,
        "args": vars(args),
        "note": "Receiver-relative side means forehand/middle/backhand, not absolute left/middle/right.",
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"selected: {selected}")
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.class_report_point}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
