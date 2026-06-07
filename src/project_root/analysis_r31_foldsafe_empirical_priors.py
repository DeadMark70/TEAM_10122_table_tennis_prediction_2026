"""R31 fold-safe empirical prior features.

This is the second-batch feature audit after R30:
- terminal / remaining-length / final-parity context priors
- non-terminal pointId priors
- rare actionId 8/9 log-odds priors
- next-action empirical prior vector

All validation features are fit only from the corresponding fold-train rows.
Training features inside each outer fold are generated with an inner
match-group OOF pass, so target-prior features are not row-wise self labels.

This script intentionally does not use old-test labels, future scoreboard
features, raw player IDs, or decoder changes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
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
    run_cv,
    tune_v3,
    write_submission,
)


@dataclass(frozen=True)
class CategoricalPrior:
    name: str
    key_cols: list[str]
    target_col: str
    classes: list[int]
    alpha: float
    filter_col: str | None = None
    filter_value: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R31 fold-safe empirical prior feature audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--submission", default="submission_r31.csv")
    parser.add_argument("--cv-report", default="cv_report_r31.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_r31.csv")
    parser.add_argument("--feature-report", default="feature_report_r31.json")
    parser.add_argument("--oof-proba", default="oof_proba_r31.pkl")
    parser.add_argument("--recommendation", default="r31_recommendation.md")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--ngram-alpha", type=float, default=20.0)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--smoke-baseline", action="store_true", help="Also run plain R7 protocol for same script sanity.")
    return parser.parse_args()


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def top_gap(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return probs[:, 0]
    part = np.partition(probs, -2, axis=1)
    return part[:, -1] - part[:, -2]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def prefix_bin(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    return np.select(
        [arr <= 1, arr == 2, arr == 3, (arr >= 4) & (arr <= 6), arr >= 7],
        [1, 2, 3, 4, 5],
        default=0,
    ).astype(np.int8)


def prepare_prior_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r31_prefix_bin"] = prefix_bin(out["prefix_len"])
    out["r31_score_total_bin"] = np.minimum(out["scoreTotal"].astype(int), 20).astype(np.int8)
    out["r31_server_lead_bin"] = np.clip(out["serverScoreDiff"].astype(int), -5, 5).astype(np.int8)
    out["r31_lag0_point_depth_side"] = (out["lag0_point_depth"].astype(int) * 4 + out["lag0_point_side"].astype(int)).astype(
        np.int16
    )
    out["r31_lag0_action_spin"] = (out["lag0_actionId"].astype(int) * 6 + out["lag0_spinId"].astype(int)).astype(
        np.int16
    )
    out["r31_lag0_action_point"] = (out["lag0_actionId"].astype(int) * 10 + out["lag0_pointId"].astype(int)).astype(
        np.int16
    )
    out["r31_remaining_bucket3"] = np.minimum(out["remaining_len"].astype(int), 3).astype(np.int8) if "remaining_len" in out else -1
    out["r31_point_nonterminal"] = out["next_pointId"].ne(0).astype(np.int8) if "next_pointId" in out else -1
    out["r31_rare89"] = (
        out["next_actionId"].isin([8, 9]).astype(np.int8) if "next_actionId" in out else -1
    )
    return out


def prior_specs() -> list[CategoricalPrior]:
    return [
        CategoricalPrior(
            name="terminal",
            key_cols=[
                "r31_prefix_bin",
                "next_hitter_is_server",
                "lag0_actionId",
                "lag0_pointId",
                "lag0_spinId",
                "r31_score_total_bin",
                "r31_server_lead_bin",
            ],
            target_col="next_is_terminal",
            classes=[0, 1],
            alpha=60.0,
        ),
        CategoricalPrior(
            name="remaining3",
            key_cols=[
                "r31_prefix_bin",
                "next_hitter_is_server",
                "lag0_actionId",
                "lag0_pointId",
                "lag0_spinId",
                "r31_score_total_bin",
                "r31_server_lead_bin",
            ],
            target_col="r31_remaining_bucket3",
            classes=[1, 2, 3],
            alpha=80.0,
        ),
        CategoricalPrior(
            name="final_parity",
            key_cols=[
                "r31_prefix_bin",
                "next_hitter_is_server",
                "lag0_actionId",
                "lag0_pointId",
                "lag0_spinId",
                "r31_score_total_bin",
                "r31_server_lead_bin",
            ],
            target_col="final_parity_even",
            classes=[0, 1],
            alpha=80.0,
        ),
        CategoricalPrior(
            name="point_nonterminal",
            key_cols=[
                "phase_id",
                "r31_prefix_bin",
                "lag0_actionId",
                "lag0_pointId",
                "lag0_spinId",
                "lag0_point_depth",
                "lag0_point_side",
                "sex",
            ],
            target_col="next_pointId",
            classes=list(range(1, 10)),
            alpha=60.0,
            filter_col="r31_point_nonterminal",
            filter_value=1,
        ),
        CategoricalPrior(
            name="rare89",
            key_cols=[
                "phase_id",
                "next_hitter_is_server",
                "lag0_actionId",
                "lag0_spinId",
                "lag0_pointId",
                "lag0_positionId",
            ],
            target_col="r31_rare89",
            classes=[0, 1],
            alpha=80.0,
        ),
        CategoricalPrior(
            name="action8_9",
            key_cols=[
                "phase_id",
                "next_hitter_is_server",
                "lag0_actionId",
                "lag0_spinId",
                "lag0_pointId",
                "lag0_positionId",
            ],
            target_col="next_actionId",
            classes=[8, 9],
            alpha=120.0,
            filter_col="r31_rare89",
            filter_value=1,
        ),
        CategoricalPrior(
            name="next_action",
            key_cols=[
                "phase_id",
                "r31_prefix_bin",
                "lag0_actionId",
                "lag0_pointId",
                "lag0_spinId",
                "r31_lag0_action_point",
            ],
            target_col="next_actionId",
            classes=list(range(19)),
            alpha=80.0,
        ),
    ]


def fit_prior_table(df: pd.DataFrame, spec: CategoricalPrior) -> dict[str, object]:
    source = df
    if spec.filter_col is not None:
        source = source[source[spec.filter_col].eq(spec.filter_value)]
    if len(source) == 0:
        global_probs = np.full(len(spec.classes), 1.0 / len(spec.classes), dtype=float)
        table = pd.DataFrame(columns=spec.key_cols + ["r31_support_" + spec.name])
        return {"spec": spec, "global": global_probs, "table": table}

    counts = source[spec.target_col].value_counts().reindex(spec.classes, fill_value=0).astype(float)
    global_probs = (counts.to_numpy(dtype=float) + spec.alpha / len(spec.classes)) / (
        float(counts.sum()) + spec.alpha
    )

    grouped = (
        source.groupby(spec.key_cols + [spec.target_col], dropna=False)
        .size()
        .unstack(spec.target_col, fill_value=0)
        .reindex(columns=spec.classes, fill_value=0)
        .reset_index()
    )
    class_counts = grouped[spec.classes].to_numpy(dtype=float)
    support = class_counts.sum(axis=1)
    probs = (class_counts + spec.alpha * global_probs[None, :]) / (support[:, None] + spec.alpha)

    table = grouped[spec.key_cols].copy()
    for idx, cls in enumerate(spec.classes):
        table[f"r31_{spec.name}_prior_{cls}"] = probs[:, idx]
    table[f"r31_{spec.name}_support"] = support.astype(np.float32)
    table[f"r31_{spec.name}_entropy"] = entropy(probs)
    table[f"r31_{spec.name}_top_gap"] = top_gap(probs)
    if len(spec.classes) == 2:
        table[f"r31_{spec.name}_logodds_1"] = logit(probs[:, 1])
    return {"spec": spec, "global": global_probs, "table": table}


def transform_prior_table(df: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    spec = bundle["spec"]
    assert isinstance(spec, CategoricalPrior)
    table = bundle["table"]
    assert isinstance(table, pd.DataFrame)
    global_probs = np.asarray(bundle["global"], dtype=float)

    out = df[spec.key_cols].merge(table, on=spec.key_cols, how="left")
    feature_cols = [c for c in out.columns if c.startswith(f"r31_{spec.name}_")]
    prob_cols = [f"r31_{spec.name}_prior_{cls}" for cls in spec.classes]
    missing = out[prob_cols[0]].isna().to_numpy()
    for idx, cls in enumerate(spec.classes):
        col = f"r31_{spec.name}_prior_{cls}"
        out[col] = out[col].fillna(float(global_probs[idx]))
    support_col = f"r31_{spec.name}_support"
    out[support_col] = out[support_col].fillna(0.0)
    probs = out[prob_cols].to_numpy(dtype=float)
    out[f"r31_{spec.name}_entropy"] = out[f"r31_{spec.name}_entropy"].fillna(pd.Series(entropy(probs)))
    out[f"r31_{spec.name}_top_gap"] = out[f"r31_{spec.name}_top_gap"].fillna(pd.Series(top_gap(probs)))
    if len(spec.classes) == 2:
        out[f"r31_{spec.name}_logodds_1"] = out[f"r31_{spec.name}_logodds_1"].fillna(
            float(logit(np.array([global_probs[1]]))[0])
        )
    out[f"r31_{spec.name}_is_backoff"] = missing.astype(np.int8)
    feature_cols = [c for c in out.columns if c.startswith(f"r31_{spec.name}_")]
    return out[feature_cols].reset_index(drop=True)


def add_prior_features_from_source(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    source_prepared = prepare_prior_frame(source_df)
    target_prepared = prepare_prior_frame(target_df)
    parts = [target_df.reset_index(drop=True)]
    for spec in prior_specs():
        bundle = fit_prior_table(source_prepared, spec)
        parts.append(transform_prior_table(target_prepared, bundle))
    out = pd.concat(parts, axis=1)
    if out.isna().any().any():
        bad_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"R31 prior features contain NaN in {bad_cols[:20]}")
    return out


def add_inner_oof_prior_features(df: pd.DataFrame, inner_folds: int, seed: int) -> pd.DataFrame:
    meta = df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    n_splits = min(inner_folds, int(meta["match"].nunique()))
    if n_splits < 2:
        return add_prior_features_from_source(df, df)

    splitter = GroupKFold(n_splits=n_splits)
    parts: list[pd.DataFrame] = []
    for inner_fold, (tr_idx, va_idx) in enumerate(splitter.split(meta, groups=meta["match"]), start=1):
        train_rallies = set(meta.iloc[tr_idx]["rally_uid"])
        valid_rallies = set(meta.iloc[va_idx]["rally_uid"])
        inner_train = df[df["rally_uid"].isin(train_rallies)].copy()
        inner_valid = df[df["rally_uid"].isin(valid_rallies)].copy()
        transformed = add_prior_features_from_source(inner_train, inner_valid)
        transformed["_r31_original_index"] = inner_valid.index.to_numpy()
        transformed["_r31_inner_fold"] = inner_fold
        parts.append(transformed)
    out = pd.concat(parts, ignore_index=True).sort_values("_r31_original_index")
    out = out.drop(columns=["_r31_original_index", "_r31_inner_fold"]).reset_index(drop=True)
    return out


def run_cv_with_priors(
    prefix_df: pd.DataFrame,
    test_prefix_lengths: np.ndarray,
    features: list[str],
    args: argparse.Namespace,
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

        fold_train = add_inner_oof_prior_features(fold_train_base, args.inner_folds, args.seeds[0] + fold)
        fold_valid = add_prior_features_from_source(fold_train_base, fold_valid_base)
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
            f"fold {fold}: base_lgbm={v2_like:.6f} "
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


def load_metrics(path: str) -> dict[str, float] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = data.get("selected", {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {key: float(value) for key, value in metrics.items()}


def write_recommendation(
    path: Path,
    metrics: dict[str, float],
    refs: dict[str, dict[str, float] | None],
    cv_path: str,
    prefix_path: str,
    submission_path: str,
) -> None:
    lines = [
        "# R31 Recommendation",
        "",
        "R31 adds fold-safe empirical prior features on top of the V3/R7 protocol.",
        "It does not use old-test labels, future scoreboard information, raw player IDs, or decoder tuning.",
        "",
        "## Selected OOF Metrics",
        "",
        f"- overall: {metrics['overall']:.6f}",
        f"- action: {metrics['action_macro_f1']:.6f}",
        f"- point: {metrics['point_macro_f1']:.6f}",
        f"- server: {metrics['server_auc']:.6f}",
    ]
    for name, ref in refs.items():
        if ref:
            lines.extend(
                [
                    "",
                    f"## Delta vs {name}",
                    "",
                    f"- overall: {metrics['overall'] - ref['overall']:+.6f}",
                    f"- action: {metrics['action_macro_f1'] - ref['action_macro_f1']:+.6f}",
                    f"- point: {metrics['point_macro_f1'] - ref['point_macro_f1']:+.6f}",
                    f"- server: {metrics['server_auc'] - ref['server_auc']:+.6f}",
                ]
            )
    verdict = "not_submit"
    if metrics["overall"] >= 0.3145 and metrics["point_macro_f1"] >= 0.205:
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
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building V3 prefix tables...")
    prefix_base = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_base = build_test_prefix_table(test, args.max_lag)

    print("adding R7 phase-aware features...")
    prefix_base = add_phase_features(prefix_base, train)
    test_base = add_phase_features(test_base, test)
    base_feature_count = len([c for c in feature_columns(prefix_base) if c != "remaining_len_bucket"])

    print("adding full-train R31 priors for final training/test feature schema...")
    prefix_full = add_prior_features_from_source(prefix_base, prefix_base)
    test_full = add_prior_features_from_source(prefix_base, test_base)
    features = [c for c in feature_columns(prefix_full) if c != "remaining_len_bucket"]
    test_full = test_full[["rally_uid", "match"] + features]
    r31_features = [c for c in features if c.startswith("r31_")]

    print(f"train prefix rows: {len(prefix_full):,}")
    print(f"test prediction rows: {len(test_full):,}")
    print(f"feature count: {len(features)} ({len(r31_features)} R31 prior/key features)")
    print(f"seeds: {args.seeds}")

    if args.smoke_baseline:
        print("running plain R7 protocol sanity baseline...")
        _ = run_cv(prefix_base, test_full["prefix_len"].to_numpy(dtype=int), [c for c in features if c in prefix_base], args)

    print("running R31 fold-safe CV...")
    oof = run_cv_with_priors(prefix_base, test_full["prefix_len"].to_numpy(dtype=int), features, args)
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
    full_pred = full_predict(prefix_full, test_full, features, args)
    submission = write_submission(test_full, full_pred, tuning, Path(args.submission))

    metadata = {
        "experiment": "R31 fold-safe empirical priors",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prefix_rows": int(len(prefix_full)),
        "test_prediction_rows": int(len(test_full)),
        "feature_count": int(len(features)),
        "base_feature_count": int(base_feature_count),
        "r31_feature_count": int(len(r31_features)),
        "r31_features": r31_features,
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
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_recommendation(
        Path(args.recommendation),
        tuning.metrics,
        {
            "V3": load_metrics("feature_report_v3.json"),
            "R7": load_metrics("feature_report_r7.json"),
            "R30": load_metrics("feature_report_r30.json"),
        },
        args.cv_report,
        args.prefix_len_report,
        args.submission,
    )

    print(f"wrote {args.cv_report}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.submission} ({len(submission):,} rows)")
    print(f"wrote {args.feature_report}")
    print(f"wrote {args.recommendation}")


if __name__ == "__main__":
    main()
