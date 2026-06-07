"""R32 ablation for R31 empirical prior families.

R31 showed that using all empirical prior features together hurts action and
point.  This script reruns the same fold-safe protocol with narrow prior
families so useful pieces can be kept without carrying the full feature pack.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
from analysis_r31_foldsafe_empirical_priors import (
    CategoricalPrior,
    add_prior_features_from_source,
    fit_prior_table,
    load_metrics,
    prepare_prior_frame,
    prior_specs,
    transform_prior_table,
    write_recommendation,
)
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import NGRAM_KEY_LEVELS, fit_ngram_bundle, predict_ngram_bundle
from baseline_v3 import (
    REMAINING_CLASSES,
    add_remaining_bucket,
    average_lgbm_predictions,
    blend_probs,
    full_predict,
    prefix_len_report,
    tune_v3,
    write_submission,
)


VARIANTS: dict[str, list[str]] = {
    "terminal_server": ["terminal", "remaining3", "final_parity"],
    "rare89": ["rare89", "action8_9"],
    "next_action": ["next_action"],
    "point_nonterminal": ["point_nonterminal"],
    "rare_next_action": ["rare89", "action8_9", "next_action"],
    "terminal_point": ["terminal", "remaining3", "point_nonterminal"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R32 narrow prior-family ablations.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--output-dir", default="r32_prior_ablation")
    parser.add_argument("--variants", nargs="+", default=["all"], choices=["all", *VARIANTS.keys()])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def select_specs(names: list[str]) -> list[CategoricalPrior]:
    wanted = set(names)
    specs = [spec for spec in prior_specs() if spec.name in wanted]
    found = {spec.name for spec in specs}
    missing = wanted - found
    if missing:
        raise ValueError(f"Unknown R31 prior spec names: {sorted(missing)}")
    return specs


def add_prior_features_selected(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    specs: list[CategoricalPrior],
) -> pd.DataFrame:
    source_prepared = prepare_prior_frame(source_df)
    target_prepared = prepare_prior_frame(target_df)
    parts = [target_df.reset_index(drop=True)]
    for spec in specs:
        bundle = fit_prior_table(source_prepared, spec)
        parts.append(transform_prior_table(target_prepared, bundle))
    out = pd.concat(parts, axis=1)
    if out.isna().any().any():
        bad_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"R32 prior features contain NaN in {bad_cols[:20]}")
    return out


def add_inner_oof_prior_features_selected(
    df: pd.DataFrame,
    specs: list[CategoricalPrior],
    inner_folds: int,
) -> pd.DataFrame:
    meta = df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    n_splits = min(inner_folds, int(meta["match"].nunique()))
    if n_splits < 2:
        return add_prior_features_selected(df, df, specs)

    splitter = GroupKFold(n_splits=n_splits)
    parts: list[pd.DataFrame] = []
    for inner_fold, (tr_idx, va_idx) in enumerate(splitter.split(meta, groups=meta["match"]), start=1):
        train_rallies = set(meta.iloc[tr_idx]["rally_uid"])
        valid_rallies = set(meta.iloc[va_idx]["rally_uid"])
        inner_train = df[df["rally_uid"].isin(train_rallies)].copy()
        inner_valid = df[df["rally_uid"].isin(valid_rallies)].copy()
        transformed = add_prior_features_selected(inner_train, inner_valid, specs)
        transformed["_r32_original_index"] = inner_valid.index.to_numpy()
        transformed["_r32_inner_fold"] = inner_fold
        parts.append(transformed)
    out = pd.concat(parts, ignore_index=True).sort_values("_r32_original_index")
    out = out.drop(columns=["_r32_original_index", "_r32_inner_fold"]).reset_index(drop=True)
    return out


def run_cv_with_selected_priors(
    prefix_df: pd.DataFrame,
    test_prefix_lengths: np.ndarray,
    features: list[str],
    args: argparse.Namespace,
    specs: list[CategoricalPrior],
) -> dict[str, object]:
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    parts: dict[str, list] = {
        "valid_meta": [],
        "lgbm_action": [],
        "lgbm_point": [],
        "lgbm_server": [],
        "ngram_action": [],
        "ngram_point": [],
        "ngram_server": [],
        "parity_server": [],
        "remaining_server": [],
    }
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_rally_idx, valid_rally_idx) in enumerate(
        splitter.split(rally_meta, groups=rally_meta["match"]), start=1
    ):
        train_rallies = set(rally_meta.iloc[train_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_rally_idx]["rally_uid"])
        if set(rally_meta.iloc[train_rally_idx]["match"]) & set(rally_meta.iloc[valid_rally_idx]["match"]):
            raise RuntimeError("GroupKFold leakage: train/valid match overlap.")

        fold_train_base = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
        valid_pool_base = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool_base, test_prefix_lengths, args.seeds[0] + fold)
        fold_valid_base = valid_pool_base.loc[sampled_idx].copy()

        fold_train = add_inner_oof_prior_features_selected(fold_train_base, specs, args.inner_folds)
        fold_valid = add_prior_features_selected(fold_train_base, fold_valid_base, specs)
        for col in features:
            if col not in fold_train.columns:
                fold_train[col] = 0
            if col not in fold_valid.columns:
                fold_valid[col] = 0

        lgbm_action, lgbm_point, lgbm_server, parity_server, remaining_server = average_lgbm_predictions(
            fold_train, fold_valid, features, args.n_estimators, args.seeds, fold
        )
        ngram_bundle = fit_ngram_bundle(fold_train, args.ngram_alpha)
        ngram_action, ngram_point, ngram_server = predict_ngram_bundle(ngram_bundle, fold_valid)

        v2_like = 0.4 * f1_score(
            fold_valid["next_actionId"],
            np.asarray(ACTION_CLASSES)[np.argmax(lgbm_action, axis=1)],
            average="macro",
            labels=ACTION_CLASSES,
            zero_division=0,
        ) + 0.4 * f1_score(
            fold_valid["next_pointId"],
            np.asarray(POINT_CLASSES)[np.argmax(lgbm_point, axis=1)],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        ) + 0.2 * roc_auc_score(fold_valid["serverGetPoint"], lgbm_server)
        parity_auc = roc_auc_score(fold_valid["serverGetPoint"], parity_server)
        remaining_auc = roc_auc_score(fold_valid["serverGetPoint"], remaining_server)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "valid_rows": len(fold_valid),
                "valid_prefix_len_mean": float(fold_valid["prefix_len"].mean()),
                "base_lgbm_overall": float(v2_like),
                "direct_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], lgbm_server)),
                "ngram_server_auc": float(roc_auc_score(fold_valid["serverGetPoint"], ngram_server)),
                "parity_server_auc": float(parity_auc),
                "remaining_server_auc": float(remaining_auc),
            }
        )
        print(
            f"  fold {fold}: base_lgbm={v2_like:.6f} "
            f"server direct={fold_rows[-1]['direct_server_auc']:.6f} "
            f"parity={parity_auc:.6f} remaining={remaining_auc:.6f}"
        )

        keep_cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
        parts["valid_meta"].append(fold_valid[keep_cols].reset_index(drop=True))
        parts["lgbm_action"].append(lgbm_action)
        parts["lgbm_point"].append(lgbm_point)
        parts["lgbm_server"].append(lgbm_server)
        parts["ngram_action"].append(ngram_action)
        parts["ngram_point"].append(ngram_point)
        parts["ngram_server"].append(ngram_server)
        parts["parity_server"].append(parity_server)
        parts["remaining_server"].append(remaining_server)

    oof: dict[str, object] = {
        "valid_meta": pd.concat(parts["valid_meta"], ignore_index=True),
        "fold_report": pd.DataFrame(fold_rows),
    }
    for key in ["lgbm_action", "lgbm_point", "ngram_action", "ngram_point"]:
        oof[key] = np.vstack(parts[key])
    for key in ["lgbm_server", "ngram_server", "parity_server", "remaining_server"]:
        oof[key] = np.concatenate(parts[key])

    fold_report = oof["fold_report"]
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0}
    for col in fold_report.columns:
        if col not in mean_row and col != "fold":
            mean_row[col] = float(fold_report[col].mean())
    oof["fold_report"] = pd.concat([fold_report, pd.DataFrame([mean_row])], ignore_index=True)
    return oof


def evaluate_variant(
    variant: str,
    spec_names: list[str],
    prefix_base: pd.DataFrame,
    test_base: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | str | int]:
    specs = select_specs(spec_names)
    prefix_full = add_prior_features_selected(prefix_base, prefix_base, specs)
    test_full = add_prior_features_selected(prefix_base, test_base, specs)
    features = [c for c in feature_columns(prefix_full) if c != "remaining_len_bucket"]
    test_full = test_full[["rally_uid", "match"] + features]
    r32_features = [c for c in features if c.startswith("r31_")]

    print(f"R32 {variant}: {len(r32_features)} prior features from {spec_names}")
    oof = run_cv_with_selected_priors(prefix_base, test_full["prefix_len"].to_numpy(dtype=int), features, args, specs)
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

    cv_path = output_dir / f"cv_report_r32_{variant}.csv"
    prefix_path = output_dir / f"prefix_len_report_r32_{variant}.csv"
    oof_path = output_dir / f"oof_proba_r32_{variant}.pkl"
    feature_path = output_dir / f"feature_report_r32_{variant}.json"
    submission_path = output_dir / f"submission_r32_{variant}.csv"
    recommendation_path = output_dir / f"r32_{variant}_recommendation.md"

    fold_report = oof["fold_report"].copy()
    fold_report["selected_action_ngram_weight"] = tuning.action_ngram_weight
    fold_report["selected_point_ngram_weight"] = tuning.point_ngram_weight
    for name, value in tuning.server_weights.items():
        fold_report[f"selected_server_weight_{name}"] = value
    for name, value in tuning.metrics.items():
        fold_report[f"selected_{name}"] = value
    fold_report.to_csv(cv_path, index=False)
    prefix_report.to_csv(prefix_path, index=False)
    with open(oof_path, "wb") as f:
        pickle.dump({**oof, "tuning": tuning}, f)

    full_pred = full_predict(prefix_full, test_full, features, args)
    submission = write_submission(test_full, full_pred, tuning, submission_path)
    metadata = {
        "experiment": f"R32 {variant}",
        "variant": variant,
        "spec_names": spec_names,
        "train_prefix_rows": int(len(prefix_full)),
        "test_prediction_rows": int(len(test_full)),
        "feature_count": int(len(features)),
        "r32_feature_count": int(len(r32_features)),
        "r32_features": r32_features,
        "features": features,
        "seeds": args.seeds,
        "inner_folds": args.inner_folds,
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
    feature_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_recommendation(
        recommendation_path,
        tuning.metrics,
        {
            "V3": load_metrics("feature_report_v3.json"),
            "R7": load_metrics("feature_report_r7.json"),
            "R30": load_metrics("feature_report_r30.json"),
        },
        str(cv_path),
        str(prefix_path),
        str(submission_path),
    )

    row = {
        "variant": variant,
        "spec_names": ",".join(spec_names),
        "feature_count": int(len(features)),
        "r32_feature_count": int(len(r32_features)),
        "submission": str(submission_path),
        "cv_report": str(cv_path),
        "feature_report": str(feature_path),
        "rows": int(len(submission)),
        **{k: float(v) for k, v in tuning.metrics.items()},
    }
    print(
        f"R32 {variant} selected: overall={row['overall']:.6f} "
        f"action={row['action_macro_f1']:.6f} point={row['point_macro_f1']:.6f} "
        f"server={row['server_auc']:.6f}"
    )
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building V3/R7 base prefix tables...")
    prefix_base = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_base = build_test_prefix_table(test, args.max_lag)
    prefix_base = add_phase_features(prefix_base, train)
    test_base = add_phase_features(test_base, test)

    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    summary_rows: list[dict[str, float | str | int]] = []
    for variant in variants:
        summary_rows.append(evaluate_variant(variant, VARIANTS[variant], prefix_base, test_base, args, output_dir))

    summary = pd.DataFrame(summary_rows).sort_values("overall", ascending=False)
    summary_path = output_dir / "r32_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary[["variant", "overall", "action_macro_f1", "point_macro_f1", "server_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
