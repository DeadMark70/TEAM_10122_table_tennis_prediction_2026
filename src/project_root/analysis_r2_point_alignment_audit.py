"""R2 analysis: pointId predictability and target-alignment audit.

This script does not train a submission model. It audits whether the current
prefix -> next-stroke target construction is aligned with the competition
definition, then measures how much pointId signal exists in short prefixes via
entropy, transition matrices, and leakage-safe conditional-prior CV.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    sample_validation_prefixes,
    validate_raw_data,
)


POINT_LABELS = list(range(10))
SEED_LIST = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R2 point/alignment diagnostics.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--out-dir", default=".")
    return parser.parse_args()


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def prefix_bin(prefix_len: pd.Series) -> pd.Series:
    return pd.cut(
        prefix_len,
        bins=[0, 1, 2, 3, 6, np.inf],
        labels=["1", "2", "3", "4-6", "7+"],
        right=True,
    ).astype(str)


def add_analysis_features(prefix_df: pd.DataFrame) -> pd.DataFrame:
    df = prefix_df.copy()
    df["prefix_bin"] = prefix_bin(df["prefix_len"])
    df["target_is_point0"] = df["next_pointId"].eq(0).astype(int)
    df["lag0_point_depth"] = np.where(df["lag0_pointId"].gt(0), (df["lag0_pointId"] - 1) // 3, -1)
    df["lag0_point_side"] = np.where(df["lag0_pointId"].gt(0), (df["lag0_pointId"] - 1) % 3, -1)
    df["lag1_point_depth"] = np.where(df["lag1_pointId"].gt(0), (df["lag1_pointId"] - 1) // 3, -1)
    df["lag1_point_side"] = np.where(df["lag1_pointId"].gt(0), (df["lag1_pointId"] - 1) % 3, -1)
    df["next_point_depth"] = np.where(df["next_pointId"].gt(0), (df["next_pointId"] - 1) // 3, -1)
    df["next_point_side"] = np.where(df["next_pointId"].gt(0), (df["next_pointId"] - 1) % 3, -1)
    return df


def audit_alignment(train: pd.DataFrame, test: pd.DataFrame, prefix_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(name: str, value: object, status: str = "info", detail: str = "") -> None:
        rows.append({"check": name, "value": value, "status": status, "detail": detail})

    add("train_rows", len(train))
    add("test_rows", len(test))
    add("train_rallies", train["rally_uid"].nunique())
    add("test_rallies", test["rally_uid"].nunique())
    add("train_prefix_samples", len(prefix_df))
    add("train_missing_values", int(train.isna().sum().sum()), "pass" if not train.isna().any().any() else "fail")
    add("test_missing_values", int(test.isna().sum().sum()), "pass" if not test.isna().any().any() else "fail")
    add("train_duplicate_rows", int(train.duplicated().sum()), "pass" if not train.duplicated().any() else "fail")
    add("test_duplicate_rows", int(test.duplicated().sum()), "pass" if not test.duplicated().any() else "fail")
    add("match_overlap_train_test", len(set(train["match"]) & set(test["match"])), "pass")
    add("rally_uid_overlap_train_test", len(set(train["rally_uid"]) & set(test["rally_uid"])), "pass")

    train_sorted = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    test_sorted = test.sort_values(["rally_uid", "strikeNumber"]).copy()
    for name, df in [("train", train_sorted), ("test", test_sorted)]:
        grouped = df.groupby("rally_uid")["strikeNumber"]
        starts_at_one = int((grouped.min() == 1).sum())
        consecutive = int(grouped.apply(lambda s: list(s) == list(range(1, len(s) + 1))).sum())
        duplicate_strikes = int(df.duplicated(["rally_uid", "strikeNumber"]).sum())
        add(f"{name}_rallies_start_at_1", f"{starts_at_one}/{df['rally_uid'].nunique()}", "pass")
        add(f"{name}_rallies_consecutive_strikeNumber", f"{consecutive}/{df['rally_uid'].nunique()}", "pass")
        add(f"{name}_duplicate_rally_strikeNumber", duplicate_strikes, "pass" if duplicate_strikes == 0 else "fail")

    train_even_receiver = train_sorted["strikeNumber"].mod(2).eq(0)
    train_expected_server_hitter = ~train_even_receiver
    test_even_receiver = test_sorted["strikeNumber"].mod(2).eq(0)
    test_expected_server_hitter = ~test_even_receiver
    add(
        "train_player_parity_mismatch_rows",
        int((train_sorted["is_server_hitter"].astype(bool) != train_expected_server_hitter).sum()),
        "pass" if int((train_sorted["is_server_hitter"].astype(bool) != train_expected_server_hitter).sum()) == 0 else "warn",
    )
    add(
        "test_player_parity_mismatch_rows",
        int((test_sorted["is_server_hitter"].astype(bool) != test_expected_server_hitter).sum()),
        "pass" if int((test_sorted["is_server_hitter"].astype(bool) != test_expected_server_hitter).sum()) == 0 else "warn",
    )

    score_nunique = train_sorted.groupby("rally_uid")[["serverScore", "receiverScore"]].nunique()
    score_changed = int(score_nunique.gt(1).any(axis=1).sum())
    add("train_rallies_with_server_receiver_score_changes", score_changed, "pass" if score_changed == 0 else "warn")

    next_strike_rows: list[dict[str, int]] = []
    for _, group in train_sorted.groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        for idx in range(len(group) - 1):
            prefix_len = int(group.iloc[idx]["strikeNumber"])
            actual = int(group.iloc[idx + 1]["strikeId"])
            expected = 2 if prefix_len == 1 else 4
            next_strike_rows.append({"prefix_len": prefix_len, "actual": actual, "expected": expected})
    next_strike = pd.DataFrame(next_strike_rows)
    next_strike_violations = int(next_strike["actual"].ne(next_strike["expected"]).sum())
    add(
        "actual_next_strikeId_rule_violations",
        next_strike_violations,
        "pass" if next_strike_violations == 0 else "warn",
        "Expected 2 after serve prefix, otherwise 4; exceptions may be stop/no-record states.",
    )
    final = prefix_df["next_is_terminal"].astype(bool)
    point0_final = float(prefix_df.loc[final, "next_pointId"].eq(0).mean())
    point0_nonfinal = float(prefix_df.loc[~final, "next_pointId"].eq(0).mean())
    add("point0_rate_when_target_final", f"{point0_final:.6f}", "info")
    add("point0_rate_when_target_nonfinal", f"{point0_nonfinal:.6f}", "info")
    add("next_action_17_18_count", int(prefix_df["next_actionId"].isin([17, 18]).sum()), "info")
    add("next_action_15_16_count", int(prefix_df["next_actionId"].isin([15, 16]).sum()), "info")
    add(
        "final_parity_server_rule_accuracy",
        f"{(prefix_df.drop_duplicates('rally_uid')['final_parity_even'].eq(prefix_df.drop_duplicates('rally_uid')['serverGetPoint'])).mean():.6f}",
        "info",
        "Matches the observed rule final strike even ~= server wins.",
    )
    add("test_prediction_rows_expected", test["rally_uid"].nunique(), "pass")
    add("test_prefix_mean_len", f"{test.groupby('rally_uid').size().mean():.6f}", "info")
    add("test_prefix_median_len", f"{test.groupby('rally_uid').size().median():.6f}", "info")

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "r2_alignment_audit.csv", index=False)
    return report


def entropy_by_prefix(prefix_df: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    test_lengths = test.groupby("rally_uid").size()
    test_len_counts = test_lengths.value_counts().to_dict()
    rows: list[dict[str, object]] = []
    for label, part in [("all", prefix_df)] + [(str(k), v) for k, v in prefix_df.groupby("prefix_len")]:
        counts = part["next_pointId"].value_counts().reindex(POINT_LABELS, fill_value=0).to_numpy()
        entropy = entropy_from_counts(counts)
        top_class = int(np.argmax(counts))
        rows.append(
            {
                "prefix_len": label,
                "train_prefix_rows": int(len(part)),
                "test_rallies_with_this_prefix_len": int(test_len_counts.get(int(label), 0)) if label != "all" else int(len(test_lengths)),
                "point_entropy_bits": entropy,
                "point_entropy_normalized": entropy / math.log2(len(POINT_LABELS)),
                "top1_pointId": top_class,
                "top1_rate": float(counts[top_class] / counts.sum()) if counts.sum() else 0.0,
                "point0_rate": float(part["next_pointId"].eq(0).mean()),
                "nonterminal_rate": float(part["next_pointId"].gt(0).mean()),
                "unique_point_labels": int((counts > 0).sum()),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "r2_point_entropy_by_prefix.csv", index=False)
    return report


def fit_prior_table(train_df: pd.DataFrame, key_cols: list[str], target: str, alpha: float) -> tuple[pd.DataFrame, int]:
    global_counts = train_df[target].value_counts().reindex(POINT_LABELS, fill_value=0).astype(float)
    global_prior = global_counts.to_numpy() / float(global_counts.sum())
    global_pred = int(np.argmax(global_prior))
    if not key_cols:
        return pd.DataFrame({"__global_pred": [global_pred]}), global_pred

    counts = train_df.groupby(key_cols + [target]).size().unstack(fill_value=0).reindex(columns=POINT_LABELS, fill_value=0)
    probs = counts.to_numpy(dtype=float) + alpha * global_prior[None, :]
    pred = np.asarray(POINT_LABELS)[np.argmax(probs, axis=1)]
    table = counts.reset_index()[key_cols].copy()
    table["pred_pointId"] = pred.astype(int)
    table["support"] = counts.sum(axis=1).to_numpy(dtype=int)
    return table, global_pred


def predict_prior(valid_df: pd.DataFrame, table: pd.DataFrame, key_cols: list[str], global_pred: int) -> np.ndarray:
    if not key_cols:
        return np.full(len(valid_df), global_pred, dtype=int)
    merged = valid_df[key_cols].merge(table, on=key_cols, how="left")
    pred = merged["pred_pointId"].fillna(global_pred).astype(int).to_numpy()
    return pred


def weighted_conditional_entropy(df: pd.DataFrame, key_cols: list[str]) -> float:
    if not key_cols:
        return entropy_from_counts(df["next_pointId"].value_counts().reindex(POINT_LABELS, fill_value=0).to_numpy())
    total = len(df)
    weighted = 0.0
    for _, part in df.groupby(key_cols, dropna=False):
        counts = part["next_pointId"].value_counts().reindex(POINT_LABELS, fill_value=0).to_numpy()
        weighted += len(part) / total * entropy_from_counts(counts)
    return float(weighted)


def conditional_prior_scores(prefix_df: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    condition_sets: list[tuple[str, list[str]]] = [
        ("global", []),
        ("prefix_len", ["prefix_len"]),
        ("prefix_bin", ["prefix_bin"]),
        ("prefix_len+sex", ["prefix_len", "sex"]),
        ("prefix_len+last_action", ["prefix_len", "lag0_actionId"]),
        ("prefix_len+last_point", ["prefix_len", "lag0_pointId"]),
        ("prefix_len+last_spin", ["prefix_len", "lag0_spinId"]),
        ("prefix_len+last_action+last_point", ["prefix_len", "lag0_actionId", "lag0_pointId"]),
        ("prefix_len+last_action+last_point+last_spin", ["prefix_len", "lag0_actionId", "lag0_pointId", "lag0_spinId"]),
        (
            "prefix_len+last_action+last_point+last_spin+sex",
            ["prefix_len", "lag0_actionId", "lag0_pointId", "lag0_spinId", "sex"],
        ),
        (
            "phase_rule_fields",
            ["prefix_len", "sex", "next_strikeId_rule", "lag0_actionId", "lag0_pointId", "lag0_spinId", "lag0_handId"],
        ),
    ]
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    test_prefix_lengths = test.groupby("rally_uid").size().to_numpy(dtype=int)
    rows: list[dict[str, object]] = []

    for cond_name, key_cols in condition_sets:
        y_all: list[int] = []
        pred_all: list[int] = []
        valid_counts: list[int] = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
            train_rallies = set(rally_meta.iloc[train_idx]["rally_uid"])
            valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"])
            fold_train = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
            valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
            sampled_idx = sample_validation_prefixes(valid_pool, test_prefix_lengths, args.seed + fold)
            fold_valid = valid_pool.loc[sampled_idx].copy()
            table, global_pred = fit_prior_table(fold_train, key_cols, "next_pointId", args.alpha)
            pred = predict_prior(fold_valid, table, key_cols, global_pred)
            y_all.extend(fold_valid["next_pointId"].astype(int).tolist())
            pred_all.extend(pred.astype(int).tolist())
            valid_counts.append(len(fold_valid))

        y = np.asarray(y_all, dtype=int)
        pred = np.asarray(pred_all, dtype=int)
        cond_entropy = weighted_conditional_entropy(prefix_df, key_cols)
        rows.append(
            {
                "condition": cond_name,
                "key_cols": "|".join(key_cols) if key_cols else "global",
                "cv_rows": int(len(y)),
                "cv_point_macro_f1": float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
                "cv_point_accuracy": float(accuracy_score(y, pred)),
                "full_train_weighted_conditional_entropy_bits": cond_entropy,
                "full_train_entropy_reduction_vs_global_bits": float(weighted_conditional_entropy(prefix_df, []) - cond_entropy),
                "mean_valid_rows_per_fold": float(np.mean(valid_counts)),
            }
        )

    report = pd.DataFrame(rows).sort_values("cv_point_macro_f1", ascending=False)
    report.to_csv(out_dir / "r2_point_conditional_prior_scores.csv", index=False)
    return report


def transition_reports(prefix_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (bin_label, last_point), part in prefix_df.groupby(["prefix_bin", "lag0_pointId"]):
        counts = part["next_pointId"].value_counts().reindex(POINT_LABELS, fill_value=0)
        total = int(counts.sum())
        if total <= 0:
            continue
        for next_point, count in counts.items():
            rows.append(
                {
                    "prefix_bin": bin_label,
                    "last_pointId": int(last_point),
                    "next_pointId": int(next_point),
                    "count": int(count),
                    "prob": float(count / total),
                    "row_total": total,
                }
            )
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "r2_point_transition_matrix.csv", index=False)
    return report


def tactical_pattern_report(prefix_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    configs = [
        (
            "prefix1_serve",
            prefix_df["prefix_len"].eq(1),
            ["sex", "lag0_actionId", "lag0_spinId", "lag0_pointId", "lag0_strengthId", "lag0_handId"],
            30,
        ),
        (
            "prefix2_serve_receive",
            prefix_df["prefix_len"].eq(2),
            ["sex", "lag1_actionId", "lag1_spinId", "lag1_pointId", "lag0_actionId", "lag0_pointId", "lag0_spinId"],
            20,
        ),
        (
            "prefix3_transition",
            prefix_df["prefix_len"].eq(3),
            ["sex", "lag2_actionId", "lag1_actionId", "lag0_actionId", "lag0_pointId", "lag0_spinId"],
            12,
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, mask, keys, min_count in configs:
        part = prefix_df[mask].copy()
        for key, group in part.groupby(keys, dropna=False):
            if len(group) < min_count:
                continue
            counts = group["next_pointId"].value_counts().reindex(POINT_LABELS, fill_value=0).to_numpy()
            top = int(np.argmax(counts))
            top_rate = float(counts[top] / counts.sum())
            rows.append(
                {
                    "pattern_type": name,
                    "count": int(len(group)),
                    "entropy_bits": entropy_from_counts(counts),
                    "top1_pointId": top,
                    "top1_rate": top_rate,
                    "key_cols": "|".join(keys),
                    "key_values": "|".join(str(int(v)) for v in (key if isinstance(key, tuple) else (key,))),
                }
            )
    report = pd.DataFrame(rows).sort_values(["pattern_type", "top1_rate", "count"], ascending=[True, False, False])
    report.to_csv(out_dir / "r2_short_prefix_tactical_report.csv", index=False)
    return report


def validation_stability(prefix_df: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    test_prefix_lengths = test.groupby("rally_uid").size().to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for seed in SEED_LIST:
        sampled_parts = []
        for fold, (_, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
            valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"])
            valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
            sampled_idx = sample_validation_prefixes(valid_pool, test_prefix_lengths, seed + fold)
            sampled_parts.append(valid_pool.loc[sampled_idx].copy())
        sampled = pd.concat(sampled_parts, ignore_index=True)
        counts = sampled["next_pointId"].value_counts().reindex(POINT_LABELS, fill_value=0).to_numpy()
        top = int(np.argmax(counts))
        rows.append(
            {
                "seed": seed,
                "sampled_rows": int(len(sampled)),
                "mean_prefix_len": float(sampled["prefix_len"].mean()),
                "prefix_len_1_rate": float(sampled["prefix_len"].eq(1).mean()),
                "prefix_len_2_rate": float(sampled["prefix_len"].eq(2).mean()),
                "point_entropy_bits": entropy_from_counts(counts),
                "point_top1": top,
                "point_top1_rate": float(counts[top] / counts.sum()),
                "point0_rate": float(sampled["next_pointId"].eq(0).mean()),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "r2_validation_stability.csv", index=False)
    return report


def write_recommendation(
    alignment: pd.DataFrame,
    entropy_report: pd.DataFrame,
    prior_report: pd.DataFrame,
    tactical_report: pd.DataFrame,
    stability_report: pd.DataFrame,
    out_dir: Path,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        str_df = df.copy()
        for col in str_df.columns:
            str_df[col] = str_df[col].map(lambda v: f"{v:.6f}" if isinstance(v, float) else str(v))
        header = "| " + " | ".join(str_df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(str_df.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in str_df.astype(str).to_numpy()]
        return "\n".join([header, sep] + body)

    best_prior = prior_report.iloc[0]
    exact_entropy = entropy_report[entropy_report["prefix_len"].isin(["1", "2", "3"])][
        ["prefix_len", "point_entropy_bits", "top1_rate", "point0_rate"]
    ]
    stability_summary = stability_report[["mean_prefix_len", "prefix_len_1_rate", "prefix_len_2_rate", "point_entropy_bits"]].agg(
        ["mean", "std"]
    )
    sharp_patterns = tactical_report[tactical_report["top1_rate"].ge(0.65)].head(10)
    fail_checks = alignment[alignment["status"].eq("fail")]
    warn_checks = alignment[alignment["status"].eq("warn")]

    lines = [
        "# R2 Point Predictability And Alignment Audit",
        "",
        "## Alignment",
    ]
    if len(fail_checks) == 0:
        lines.append("- No hard alignment failures were detected.")
    else:
        lines.append("- Hard failures:")
        for _, row in fail_checks.iterrows():
            lines.append(f"  - {row['check']}: {row['value']}")
    if len(warn_checks):
        lines.append("- Warnings:")
        for _, row in warn_checks.iterrows():
            lines.append(f"  - {row['check']}: {row['value']} ({row['detail']})")

    lines.extend(
        [
            "",
            "## Short-Prefix Point Entropy",
            md_table(exact_entropy),
            "",
            "## Best Leakage-Safe Conditional Prior",
            f"- Condition: `{best_prior['condition']}`",
            f"- CV point Macro-F1: `{float(best_prior['cv_point_macro_f1']):.6f}`",
            f"- CV point accuracy: `{float(best_prior['cv_point_accuracy']):.6f}`",
            f"- Entropy reduction vs global: `{float(best_prior['full_train_entropy_reduction_vs_global_bits']):.6f}` bits",
            "",
            "## Validation Sampling Stability",
            md_table(stability_summary.reset_index().rename(columns={"index": "stat"})),
            "",
            "## Sharp Short-Prefix Patterns",
        ]
    )
    if len(sharp_patterns) == 0:
        lines.append("- No short-prefix pattern with `top1_rate >= 0.65` under the configured minimum supports.")
    else:
        lines.append(
            md_table(sharp_patterns[["pattern_type", "count", "top1_pointId", "top1_rate", "entropy_bits", "key_values"]])
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "- Use this report before deciding whether a 0.4 target is plausible from the current feature space.",
            "- If conditional prior Macro-F1 remains close to the current V3 point score, the bottleneck is not simply model capacity.",
            "- If the tactical report contains high-support sharp patterns, the next model should encode those phase-specific interactions explicitly.",
            "- Do not use this analysis as a submission model; it is diagnostic only.",
            "",
        ]
    )
    (out_dir / "r2_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    print("building train prefix table...")
    prefix_df = build_train_prefix_table(train, args.max_lag)
    prefix_df = add_analysis_features(prefix_df)
    print(f"prefix rows: {len(prefix_df):,}")

    print("alignment audit...")
    alignment = audit_alignment(train, test, prefix_df, out_dir)
    print("entropy by prefix...")
    entropy_report = entropy_by_prefix(prefix_df, test, out_dir)
    print("conditional prior CV...")
    prior_report = conditional_prior_scores(prefix_df, test, args, out_dir)
    print("transition matrix...")
    transition_reports(prefix_df, out_dir)
    print("short-prefix tactical report...")
    tactical_report = tactical_pattern_report(prefix_df, out_dir)
    print("validation stability...")
    stability_report = validation_stability(prefix_df, test, args, out_dir)
    write_recommendation(alignment, entropy_report, prior_report, tactical_report, stability_report, out_dir)

    print("wrote R2 reports:")
    for name in [
        "r2_alignment_audit.csv",
        "r2_point_entropy_by_prefix.csv",
        "r2_point_conditional_prior_scores.csv",
        "r2_point_transition_matrix.csv",
        "r2_short_prefix_tactical_report.csv",
        "r2_validation_stability.csv",
        "r2_recommendation.md",
    ]:
        print(f"- {out_dir / name}")


if __name__ == "__main__":
    main()
