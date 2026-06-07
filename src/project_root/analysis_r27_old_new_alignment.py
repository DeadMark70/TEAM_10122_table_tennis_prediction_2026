"""R27 old-test / new-test alignment diagnostic.

This script does not generate a submission.  It quantifies which information
from the reference-only old test file can be treated as observed training data,
and which information would be direct target leakage for the current test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = ["rally_uid", "strikeNumber"]
RALLY_META_COLS = [
    "rally_uid",
    "sex",
    "match",
    "numberGame",
    "rally_id",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
]
LABEL_COLS = ["actionId", "pointId", "strikeId", "handId", "strengthId", "spinId", "positionId"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R27 old/new test alignment audit.")
    parser.add_argument("--test-old", default="test_old.csv")
    parser.add_argument("--test-new", default="test_new.csv")
    parser.add_argument("--summary", default="r27_old_new_alignment_summary.csv")
    parser.add_argument("--relation-report", default="r27_old_new_overlap_by_relation.csv")
    parser.add_argument("--visible-target-report", default="r27_old_target_visible_report.csv")
    parser.add_argument("--direct-leak-report", default="r27_new_target_direct_leak_report.csv")
    parser.add_argument("--server-report", default="r27_old_server_alignment_report.csv")
    parser.add_argument("--internal-report", default="r27_new_internal_transition_report.csv")
    parser.add_argument("--feature-report", default="feature_report_r27.json")
    parser.add_argument("--recommendation", default="r27_recommendation.md")
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(KEY_COLS).reset_index(drop=True)


def rally_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    agg = {
        "strikeNumber": ["size", "max"],
        "sex": "first",
        "match": "first",
        "numberGame": "first",
        "rally_id": "first",
        "scoreSelf": "first",
        "scoreOther": "first",
        "gamePlayerId": "first",
        "gamePlayerOtherId": "first",
    }
    if "serverGetPoint" in df.columns:
        agg["serverGetPoint"] = "first"
    out = df.groupby("rally_uid", sort=False).agg(agg)
    out.columns = [
        f"{name}_rows",
        f"{name}_prefix_len",
        f"{name}_sex",
        f"{name}_match",
        f"{name}_numberGame",
        f"{name}_rally_id",
        f"{name}_scoreSelf0",
        f"{name}_scoreOther0",
        f"{name}_server_id0",
        f"{name}_receiver_id0",
    ] + ([f"{name}_serverGetPoint"] if "serverGetPoint" in df.columns else [])
    return out.reset_index()


def compare_observed_rows(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    common_cols = [c for c in old_df.columns if c in new_df.columns and c not in KEY_COLS]
    merged = old_df[KEY_COLS + common_cols].merge(
        new_df[KEY_COLS + common_cols],
        on=KEY_COLS,
        how="left",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    found = merged["_merge"].eq("both")
    exact = found.copy()
    mismatch_counts = {}
    for col in common_cols:
        same = merged[f"{col}_old"].eq(merged[f"{col}_new"])
        exact &= same
        mismatch_counts[col] = int((found & ~same).sum())
    return {
        "old_observed_rows": int(len(old_df)),
        "old_observed_rows_found_in_new": int(found.sum()),
        "old_observed_rows_exact_in_new": int(exact.sum()),
        "old_observed_row_found_ratio": float(found.mean()) if len(old_df) else np.nan,
        "old_observed_row_exact_ratio": float(exact.mean()) if len(old_df) else np.nan,
        "row_mismatch_counts": mismatch_counts,
    }


def build_relation(old_sum: pd.DataFrame, new_sum: pd.DataFrame) -> pd.DataFrame:
    rel = old_sum.merge(new_sum, on="rally_uid", how="outer", indicator=True)
    rel["old_in_new"] = rel["_merge"].isin(["both"])
    rel["new_in_old"] = rel["_merge"].isin(["both"])
    rel["prefix_relation"] = np.select(
        [
            rel["_merge"].eq("left_only"),
            rel["_merge"].eq("right_only"),
            rel["old_prefix_len"].lt(rel["new_prefix_len"]),
            rel["old_prefix_len"].eq(rel["new_prefix_len"]),
            rel["old_prefix_len"].gt(rel["new_prefix_len"]),
        ],
        ["old_only", "new_only", "old_shorter", "same_len", "old_longer"],
        default="unknown",
    )
    rel["old_target_strike"] = rel["old_prefix_len"] + 1
    rel["new_target_strike"] = rel["new_prefix_len"] + 1
    rel["old_hidden_target_visible_in_new_prefix"] = (
        rel["_merge"].eq("both") & rel["new_prefix_len"].ge(rel["old_target_strike"])
    )
    rel["new_hidden_target_visible_in_old_prefix"] = (
        rel["_merge"].eq("both") & rel["old_prefix_len"].ge(rel["new_target_strike"])
    )
    return rel


def extract_rows_at_strike(df: pd.DataFrame, strikes: pd.DataFrame, strike_col: str) -> pd.DataFrame:
    keys = strikes[["rally_uid", strike_col]].rename(columns={strike_col: "strikeNumber"})
    return keys.merge(df, on=["rally_uid", "strikeNumber"], how="left")


def label_distribution(rows: pd.DataFrame, prefix: str) -> dict:
    out = {}
    for col in ["actionId", "pointId"]:
        if col not in rows.columns:
            continue
        counts = rows[col].dropna().astype(int).value_counts().sort_index()
        out[f"{prefix}_{col}_counts"] = {str(int(k)): int(v) for k, v in counts.items()}
    return out


def internal_transition_report(new_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, group in new_df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")
        max_strike = int(group["strikeNumber"].max())
        for prefix_len in range(1, max_strike):
            target = group[group["strikeNumber"].eq(prefix_len + 1)]
            if target.empty:
                continue
            target = target.iloc[0]
            rows.append(
                {
                    "rally_uid": int(uid),
                    "prefix_len": int(prefix_len),
                    "target_strike": int(prefix_len + 1),
                    "target_actionId": int(target["actionId"]),
                    "target_pointId": int(target["pointId"]),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["prefix_len", "rows", "unique_rallies"])
    return (
        detail.groupby("prefix_len")
        .agg(rows=("rally_uid", "size"), unique_rallies=("rally_uid", "nunique"))
        .reset_index()
    )


def write_recommendation(path: Path, stats: dict) -> None:
    lines = [
        "# R27 old/new alignment recommendation",
        "",
        "This is a diagnostic report only; no submission was generated.",
        "",
        "## Key findings",
        "",
        f"- Old rallies: {stats['old_rallies']}",
        f"- New rallies: {stats['new_rallies']}",
        f"- Matched rally_uids: {stats['matched_rallies']} ({stats['matched_old_ratio']:.2%} of old, {stats['matched_new_ratio']:.2%} of new)",
        f"- Old observed rows exactly present in new: {stats['old_observed_rows_exact_in_new']} / {stats['old_observed_rows']} ({stats['old_observed_row_exact_ratio']:.2%})",
        f"- Old hidden target visible inside new prefix: {stats['old_hidden_target_visible_count']} rallies",
        f"- New hidden target visible inside old prefix: {stats['new_hidden_target_visible_count']} rallies",
        f"- Old serverGetPoint directly covers new rally_uid: {stats['old_server_covers_new_count']} new rallies",
        f"- New observed internal supervised transitions: {stats['new_internal_transition_rows']} rows",
        "",
        "## Interpretation",
        "",
    ]
    if stats["new_hidden_target_visible_count"] > 0:
        lines.append(
            "- Some current hidden targets are directly visible in old test prefixes. Treat this as direct leakage and do not use it for final predictions without explicit organizer approval."
        )
    else:
        lines.append(
            "- No current hidden next-stroke targets were found inside old test observed prefixes. Old test does not directly reveal action/point labels for current test rows under this key check."
        )
    if stats["old_hidden_target_visible_count"] > 0:
        lines.append(
            "- Many old-test hidden targets are now visible inside test_new observed prefixes. These are usable as observed transition/domain-adaptation rows, not as current hidden labels."
        )
    lines.extend(
        [
            "- Old serverGetPoint labels align by rally_uid for matched rallies. Directly copying them into the current submission is high-sensitivity; safer uses are teacher/distillation/diagnostic training or organizer-confirmed usage.",
            "- The clean low-risk path remains: train on observed internal transitions and keep old-server usage separate from action/point lookup logic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    old_df = load_csv(args.test_old)
    new_df = load_csv(args.test_new)

    old_sum = rally_summary(old_df, "old")
    new_sum = rally_summary(new_df, "new")
    rel = build_relation(old_sum, new_sum)

    row_cmp = compare_observed_rows(old_df, new_df)

    relation_report = (
        rel.groupby("prefix_relation")
        .agg(
            rallies=("rally_uid", "size"),
            old_hidden_target_visible_in_new_prefix=("old_hidden_target_visible_in_new_prefix", "sum"),
            new_hidden_target_visible_in_old_prefix=("new_hidden_target_visible_in_old_prefix", "sum"),
        )
        .reset_index()
    )
    relation_report.to_csv(args.relation_report, index=False)

    visible_old = rel[rel["old_hidden_target_visible_in_new_prefix"]].copy()
    visible_old_rows = extract_rows_at_strike(new_df, visible_old, "old_target_strike")
    visible_old_rows.to_csv(args.visible_target_report, index=False)

    direct_new = rel[rel["new_hidden_target_visible_in_old_prefix"]].copy()
    direct_new_rows = extract_rows_at_strike(old_df.drop(columns=["serverGetPoint"], errors="ignore"), direct_new, "new_target_strike")
    direct_new_rows.to_csv(args.direct_leak_report, index=False)

    matched = rel[rel["_merge"].eq("both")].copy()
    old_server_covers_new = matched["old_serverGetPoint"].notna()
    server_report = matched[
        [
            "rally_uid",
            "old_prefix_len",
            "new_prefix_len",
            "old_serverGetPoint",
            "prefix_relation",
            "old_hidden_target_visible_in_new_prefix",
            "new_hidden_target_visible_in_old_prefix",
        ]
    ].copy()
    server_report.to_csv(args.server_report, index=False)

    internal_report = internal_transition_report(new_df)
    internal_report.to_csv(args.internal_report, index=False)

    summary_rows = [
        {"metric": "old_rows", "value": int(len(old_df))},
        {"metric": "new_rows", "value": int(len(new_df))},
        {"metric": "old_rallies", "value": int(old_sum["rally_uid"].nunique())},
        {"metric": "new_rallies", "value": int(new_sum["rally_uid"].nunique())},
        {"metric": "matched_rallies", "value": int(matched.shape[0])},
        {"metric": "matched_old_ratio", "value": float(matched.shape[0] / old_sum.shape[0])},
        {"metric": "matched_new_ratio", "value": float(matched.shape[0] / new_sum.shape[0])},
        {"metric": "old_observed_rows_found_in_new", "value": row_cmp["old_observed_rows_found_in_new"]},
        {"metric": "old_observed_rows_exact_in_new", "value": row_cmp["old_observed_rows_exact_in_new"]},
        {"metric": "old_observed_row_found_ratio", "value": row_cmp["old_observed_row_found_ratio"]},
        {"metric": "old_observed_row_exact_ratio", "value": row_cmp["old_observed_row_exact_ratio"]},
        {"metric": "old_hidden_target_visible_in_new_prefix", "value": int(rel["old_hidden_target_visible_in_new_prefix"].sum())},
        {"metric": "new_hidden_target_visible_in_old_prefix", "value": int(rel["new_hidden_target_visible_in_old_prefix"].sum())},
        {"metric": "old_server_covers_new_count", "value": int(old_server_covers_new.sum())},
        {"metric": "new_internal_transition_rows", "value": int(internal_report["rows"].sum()) if not internal_report.empty else 0},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary, index=False)

    stats = {row["metric"]: row["value"] for row in summary_rows}
    stats.update(
        {
            "old_observed_rows": row_cmp["old_observed_rows"],
            "old_observed_rows_exact_in_new": row_cmp["old_observed_rows_exact_in_new"],
            "old_observed_row_exact_ratio": row_cmp["old_observed_row_exact_ratio"],
            "old_hidden_target_visible_count": int(rel["old_hidden_target_visible_in_new_prefix"].sum()),
            "new_hidden_target_visible_count": int(rel["new_hidden_target_visible_in_old_prefix"].sum()),
            "old_server_covers_new_count": int(old_server_covers_new.sum()),
            "matched_old_ratio": float(matched.shape[0] / old_sum.shape[0]),
            "matched_new_ratio": float(matched.shape[0] / new_sum.shape[0]),
            "visible_old_target_distribution": label_distribution(visible_old_rows, "visible_old_target"),
            "direct_new_target_distribution": label_distribution(direct_new_rows, "direct_new_target"),
            "row_mismatch_counts": row_cmp["row_mismatch_counts"],
        }
    )
    Path(args.feature_report).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_recommendation(Path(args.recommendation), stats)

    print(summary_df.to_string(index=False))
    print(f"Wrote {args.summary}, {args.relation_report}, {args.feature_report}, {args.recommendation}")


if __name__ == "__main__":
    main()
