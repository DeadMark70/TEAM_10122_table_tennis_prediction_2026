"""R17 public/internal transition supervised adaptation.

Uses only observed prefix-internal transitions:

  prefix 1      -> observed stroke 2
  prefix 1..k   -> observed stroke k+1
  ...
  prefix 1..L-1 -> observed stroke L

For CV, fold-valid rallies are treated like public test prefixes: once a
test-like prefix length L is sampled, rows with prefix_len < L are allowed as
transductive/internal adaptation rows. The hidden validation target row
prefix_len == L is never added to training.

This script evaluates action/point adaptation only. Server is kept fixed to
the existing V10B-safe branch because public/internal rows do not have legal
server labels.
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    POINT_NONTERMINAL_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    make_lgbm,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, tune_segmented_multipliers


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
    parser = argparse.ArgumentParser(description="Run R17 internal transition adaptation.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--cv-report", default="cv_report_r17.csv")
    parser.add_argument("--adapt-search", default="r17_adapt_search.csv")
    parser.add_argument("--selected", default="r17_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r17.json")
    parser.add_argument("--oof-proba", default="oof_proba_r17.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--internal-weights", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5])
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=float)
    for src_idx, cls in enumerate([int(c) for c in model.classes_]):
        if cls in classes:
            out[:, classes.index(cls)] = proba[:, src_idx]
    zero = out.sum(axis=1) <= 0
    if zero.any():
        out[zero] = 1.0 / len(classes)
    return normalize_rows(out)


def fit_action_point_models(
    train_df: pd.DataFrame,
    features: list[str],
    seed: int,
    n_estimators: int,
    internal_weight: float,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMClassifier, lgb.LGBMClassifier]:
    x = train_df[features]
    domain_weight = np.where(train_df["is_internal_adapt"].to_numpy(dtype=int) == 1, internal_weight, 1.0)

    action_model = make_lgbm("multiclass", n_estimators, seed, num_class=len(ACTION_CLASSES))
    action_weight = class_weight_sample(train_df["next_actionId"]) * domain_weight
    action_model.fit(x, train_df["next_actionId"], sample_weight=action_weight)

    terminal_model = make_lgbm("binary", n_estimators, seed + 1)
    terminal_model.fit(x, train_df["next_is_terminal"], sample_weight=domain_weight)

    point_train = train_df[train_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
    point_domain_weight = np.where(point_train["is_internal_adapt"].to_numpy(dtype=int) == 1, internal_weight, 1.0)
    point_weight = class_weight_sample(point_train["next_pointId"]) * point_domain_weight
    point_model = make_lgbm("multiclass", n_estimators, seed + 2, num_class=len(POINT_NONTERMINAL_CLASSES))
    point_model.fit(point_train[features], point_train["next_pointId"], sample_weight=point_weight)
    return action_model, terminal_model, point_model


def predict_action_point(
    models: tuple[lgb.LGBMClassifier, lgb.LGBMClassifier, lgb.LGBMClassifier],
    df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    action_model, terminal_model, point_model = models
    x = df[features]
    action_prob = aligned_proba(action_model, x, ACTION_CLASSES)
    term_raw = terminal_model.predict_proba(x)
    terminal = term_raw[:, 1] if term_raw.ndim == 2 else term_raw
    terminal = np.clip(terminal.astype(float), 1e-6, 1.0 - 1e-6)
    point_nonterm = aligned_proba(point_model, x, POINT_NONTERMINAL_CLASSES)
    point = np.zeros((len(df), len(POINT_CLASSES)), dtype=float)
    point[:, 0] = terminal
    point[:, 1:] = (1.0 - terminal[:, None]) * point_nonterm
    return action_prob, normalize_rows(point)


def internal_rows_from_valid(valid_pool: pd.DataFrame, fold_valid: pd.DataFrame) -> pd.DataFrame:
    limits = fold_valid[["rally_uid", "prefix_len"]].rename(columns={"prefix_len": "observed_prefix_len"})
    merged = valid_pool.merge(limits, on="rally_uid", how="inner")
    internal = merged[merged["prefix_len"].lt(merged["observed_prefix_len"])].copy()
    internal = internal.drop(columns=["observed_prefix_len"])
    return internal


def internal_rows_from_public_test(test_prefix_like: pd.DataFrame) -> pd.DataFrame:
    # build_train_prefix_table on public test-like data already creates rows
    # prefix 1 -> stroke2 ... prefix L-1 -> stroke L. These labels are observed.
    out = test_prefix_like.copy()
    out["serverGetPoint"] = 0
    out["num_prefixes_in_rally"] = 1
    out["final_parity_even"] = 0
    out["remaining_len"] = 1
    out["next_is_terminal"] = 0
    return out


def cv_oof(
    prefix_df: pd.DataFrame,
    test_prefix_lengths: np.ndarray,
    features: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    parts: dict[str, list] = {"valid_meta": []}
    for weight in args.internal_weights:
        parts[f"action_w{weight}"] = []
        parts[f"point_w{weight}"] = []
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[train_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"])
        if set(rally_meta.iloc[train_idx]["match"]) & set(rally_meta.iloc[valid_idx]["match"]):
            raise RuntimeError("GroupKFold leakage.")
        fold_train_base = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
        valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled = sample_validation_prefixes(valid_pool, test_prefix_lengths, args.seed + fold)
        fold_valid = valid_pool.loc[sampled].copy()
        internal = internal_rows_from_valid(valid_pool, fold_valid)
        fold_train_base["is_internal_adapt"] = 0
        internal["is_internal_adapt"] = 1
        fold_train = pd.concat([fold_train_base, internal], ignore_index=True)

        row = {
            "fold": fold,
            "train_rows": len(fold_train_base),
            "internal_rows": len(internal),
            "valid_rows": len(fold_valid),
            "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
        }
        for weight in args.internal_weights:
            models = fit_action_point_models(
                fold_train, features, args.seed + fold * 19 + int(weight * 1000), args.n_estimators, weight
            )
            action, point = predict_action_point(models, fold_valid, features)
            parts[f"action_w{weight}"].append(action)
            parts[f"point_w{weight}"].append(point)
            row[f"action_argmax_w{weight}"] = float(
                f1_score(
                    fold_valid["next_actionId"],
                    np.asarray(ACTION_CLASSES)[np.argmax(action, axis=1)],
                    average="macro",
                    labels=ACTION_CLASSES,
                    zero_division=0,
                )
            )
            row[f"point_argmax_w{weight}"] = float(
                f1_score(
                    fold_valid["next_pointId"],
                    np.asarray(POINT_CLASSES)[np.argmax(point, axis=1)],
                    average="macro",
                    labels=POINT_CLASSES,
                    zero_division=0,
                )
            )
        print(
            f"fold {fold}: internal={len(internal):,} "
            + " ".join([f"w{w}=A{row[f'action_argmax_w{w}']:.4f}/P{row[f'point_argmax_w{w}']:.4f}" for w in args.internal_weights])
        )
        keep = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
        parts["valid_meta"].append(fold_valid[keep].reset_index(drop=True))
        fold_rows.append(row)
    out: dict[str, object] = {
        "valid_meta": pd.concat(parts["valid_meta"], ignore_index=True),
        "fold_report": pd.DataFrame(fold_rows),
    }
    for weight in args.internal_weights:
        out[f"action_w{weight}"] = np.vstack(parts[f"action_w{weight}"])
        out[f"point_w{weight}"] = np.vstack(parts[f"point_w{weight}"])
    return out


def build_safe_server(v3, v5, v7, v10, selected_v10) -> np.ndarray:
    _, _, v3_server = compose_v3(v3)
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    return (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]


def search_adaptation(oof: dict, v3: dict, v5: dict, v7: dict, v10: dict, selected_v10: dict, weights: list[float]):
    meta = normalize_meta(oof["valid_meta"])
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    v3_action, v3_point, _ = compose_v3(v3)
    safe_server = build_safe_server(v3, v5, v7, v10, selected_v10)
    server_auc = roc_auc_score(meta["serverGetPoint"], safe_server)
    rows = []
    best = None
    for w in weights:
        adapt_action = oof[f"action_w{w}"]
        adapt_point = oof[f"point_w{w}"]
        for action_blend in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]:
            action_prob = blend_probs(r1_action, adapt_action, action_blend)
            action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", "two")
            action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, "two")
            action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
            action_churn = float(
                (
                    action_pred
                    != apply_segmented_multipliers(meta, r1_action, tune_segmented_multipliers(meta, r1_action, ACTION_CLASSES, "action", "two"), ACTION_CLASSES, "two")
                ).mean()
            )
            for point_blend in [0.0, 0.02, 0.05, 0.1, 0.2]:
                point_prob = blend_probs(v3_point, adapt_point, point_blend)
                point_pred = apply_segmented_multipliers(
                    meta, point_prob, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode
                )
                point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
                overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
                row = {
                    "internal_weight": w,
                    "action_blend": action_blend,
                    "point_blend": point_blend,
                    "action_macro_f1": float(action_f1),
                    "point_macro_f1": float(point_f1),
                    "server_auc": float(server_auc),
                    "overall": float(overall),
                    "action_churn_vs_r1": action_churn,
                }
                rows.append(row)
                eligible = overall >= 0.316 and action_f1 >= 0.272 and point_f1 >= 0.203
                objective = overall - 0.01 * action_churn
                if eligible and (best is None or objective > best["objective"]):
                    best = {"objective": objective, "row": row, "action_mult": action_mult}
    report = pd.DataFrame(rows).sort_values(["overall", "action_macro_f1"], ascending=False)
    if best is None:
        selected = report.iloc[0].to_dict()
        selected["submit_recommendation"] = False
        selected["selected_policy"] = "diagnostic_only"
        return report, selected
    best["row"]["submit_recommendation"] = True
    best["row"]["selected_policy"] = "internal_transition_adaptation"
    best["row"]["action_multipliers"] = best["action_mult"]
    return report, best["row"]


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_prefix = build_test_prefix_table(test, args.max_lag)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    print(f"train prefixes={len(prefix_df):,} test_internal_rows={sum(test.groupby('rally_uid').size()-1):,}")

    oof = cv_oof(prefix_df, test_prefix["prefix_len"].to_numpy(dtype=int), features, args)
    with open(args.oof_proba, "wb") as f:
        pickle.dump(oof, f)
    oof["fold_report"].to_csv(args.cv_report, index=False)

    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    assert_aligned(normalize_meta(v3["valid_meta"]), oof["valid_meta"], "R17")
    report, selected = search_adaptation(oof, v3, v5, v7, v10, selected_v10, args.internal_weights)
    report.to_csv(args.adapt_search, index=False)
    out = {
        "selected": selected,
        "server_policy": "fixed_v10b_safe_server",
        "point_policy": "v3_point_with_optional_small_blend",
        "protocol": "fold-valid internal transitions only, hidden sampled target excluded",
    }
    Path(args.selected).write_text(json.dumps(out, indent=2), encoding="utf-8")
    metadata = {
        "cv_report": args.cv_report,
        "adapt_search": args.adapt_search,
        "selected": out,
        "features": features,
        "internal_weights": args.internal_weights,
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("R17 selected:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
