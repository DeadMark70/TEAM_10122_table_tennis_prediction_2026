"""Export a human-readable inventory of currently used feature families.

The output is intended for external review/feature ideation.  It lists exact
base feature names where available, pattern-expanded style/profile features,
and meta features used by later reranker/server experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add(rows: list[dict], pipeline: str, category: str, feature: str, status: str, notes: str = "") -> None:
    rows.append(
        {
            "pipeline": pipeline,
            "category": category,
            "feature": feature,
            "status": status,
            "notes": notes,
        }
    )


def categorize_base(feature: str) -> str:
    if feature.startswith("lag"):
        return "lag/history"
    if feature.startswith("count_") or feature.startswith("nunique_"):
        return "in-prefix counts"
    if "Score" in feature or feature in {"numberGame", "rally_id", "scoreTotal"}:
        return "score/game state"
    if feature.startswith("prefix") or feature.startswith("next_") or feature == "is_server_hitter":
        return "phase/role"
    return "base raw/context"


def style_features() -> list[str]:
    out = ["style_hitter_seen_count", "style_hitter_log_seen_count"]
    for field, vals in {
        "actionId": range(19),
        "pointId": range(10),
        "spinId": range(6),
        "handId": range(3),
        "strengthId": range(4),
    }.items():
        out.extend([f"style_hitter_{field}_{v}_rate" for v in vals])
    out.extend(["style_receiver_seen_count", "style_receiver_log_seen_count"])
    for field, vals in {"pointId": range(10), "spinId": range(6)}.items():
        out.extend([f"style_receiver_{field}_{v}_rate" for v in vals])
    return out


def main() -> None:
    rows: list[dict] = []

    v3 = read_json("feature_report_v3.json")
    for feat in v3["features"]:
        add(rows, "V3/R1 LightGBM base", categorize_base(feat), feat, "active", "Used by V3 point/server and many later tabular branches.")

    r7 = read_json("feature_report_r7.json")
    for feat in r7["r7_added_features"]:
        add(rows, "R7 phase-aware features", "phase/tactical derived", feat, "tested; action-helpful, point-risky", "R7 full submission hurt PL; action-only branches may reuse action probabilities.")

    for feat in style_features():
        add(rows, "R19 observed-style profile", "test-time/profile rates", feat, "tested; PL negative", "Fold-safe observed prefix style profile; full R19 submission underperformed public LB.")

    r26 = read_json("feature_report_r26.json")
    for feat in r26["candidate_columns"]:
        add(rows, "R20/R26 action reranker", "candidate/meta feature", feat, "candidate branch", "Used for action top-k candidate reranking, not direct base model features.")

    semantic_features = [
        "AICUP_action_family",
        "AICUP_action_technique",
        "action_similarity_same_technique",
        "action_similarity_same_family",
        "semantic_smoothed_action_prob",
        "semantic_smoothed_action_rank",
    ]
    for feat in semantic_features:
        add(rows, "R25 semantic smoothing", "action semantic prior", feat, "candidate branch", "Coarse action family/technique smoothing.")

    external_prior_features = [
        "canonical_family",
        "canonical_technique",
        "OpenTT_transition_prior_by_phase",
        "OpenTT_transition_prior_by_last_family",
        "OpenTT_transition_prior_by_last_technique",
        "r22_prior_prob",
        "r22_prior_rank",
        "r22_blend_prob",
        "r22_blend_rank",
    ]
    for feat in external_prior_features:
        add(rows, "R22 OpenTTGames prior", "external/canonical prior", feat, "tested; weak", "External prior did not improve enough as standalone; reused as reranker feature in R26.")

    sequence_fields = [
        "stroke_sequence[strikeId]",
        "stroke_sequence[handId]",
        "stroke_sequence[strengthId]",
        "stroke_sequence[spinId]",
        "stroke_sequence[pointId]",
        "stroke_sequence[actionId]",
        "stroke_sequence[positionId]",
        "stroke_sequence[is_server_hitter]",
        "stroke_sequence[sex]",
        "stroke_sequence[serverScore]",
        "stroke_sequence[receiverScore]",
        "stroke_sequence[serverScoreDiff]",
        "stroke_sequence[strikeNumber/prefix_position]",
        "h_last",
        "h_pool",
        "concat(h_last,h_pool)",
        "masked_field_pretrain_targets",
        "causal_next_stroke_targets",
    ]
    for feat in sequence_fields:
        add(rows, "V5/V7/V10/V12/V24 sequence models", "sequence/embedding", feat, "tested; action-helpful", "Sequence branches mostly improve action, not point.")

    server_features = [
        "old_serverGetPoint_label_by_rally_uid",
        "old_server_teacher_probability",
        "old_server_soft_pseudo_probability",
        "old_server_direct_diagnostic_probability",
        "scoreboard_future_gap",
        "scoreboard_future_server_score_count",
        "scoreboard_future_receiver_score_count",
        "scoreboard_future_server_score_rate",
        "scoreboard_score_valid_flag",
        "old_covered_flag",
        "new_only_flag",
        "next_strike_final_parity_compatibility",
        "point0_terminal_gate_adjustment_beta",
    ]
    for feat in server_features:
        add(rows, "R28/R29 server-scoreboard", "server/terminal structure", feat, "sensitive candidate", "Rule-sensitive; keep separated until organizer reply.")

    label_targets = [
        "next_actionId",
        "next_is_terminal",
        "next_pointId_nonterminal",
        "serverGetPoint",
        "remaining_len_bucket",
        "final_parity_even",
        "point_depth_aux",
        "point_receiver_relative_side_aux",
    ]
    for feat in label_targets:
        add(rows, "Training targets / auxiliary heads", "target/loss", feat, "varies by experiment", "Used as supervised or auxiliary target, not an inference feature.")

    df = pd.DataFrame(rows)
    df.to_csv("current_feature_inventory.csv", index=False)

    summary = df.groupby(["pipeline", "status"]).size().reset_index(name="count")
    summary_lines = ["| pipeline | status | count |", "| --- | --- | ---: |"]
    for r in summary.itertuples(index=False):
        summary_lines.append(f"| {r.pipeline} | {r.status} | {int(r.count)} |")

    lines = [
        "# Current Feature Inventory",
        "",
        "This inventory lists features and feature families used in the current experiments.  It separates active production features from branches that were tested but risky/weak.",
        "",
        "## Summary by Pipeline",
        "",
        "\n".join(summary_lines),
        "",
    ]
    for pipeline, part in df.groupby("pipeline", sort=False):
        lines.extend([f"## {pipeline}", ""])
        for category, cat_part in part.groupby("category", sort=False):
            lines.append(f"### {category}")
            for r in cat_part.itertuples(index=False):
                note = f" - {r.notes}" if r.notes else ""
                lines.append(f"- `{r.feature}` ({r.status}){note}")
            lines.append("")
    Path("current_feature_inventory.md").write_text("\n".join(lines), encoding="utf-8")

    compact = {
        "total_rows": int(len(df)),
        "pipelines": df.groupby("pipeline").size().to_dict(),
        "outputs": ["current_feature_inventory.csv", "current_feature_inventory.md"],
    }
    Path("current_feature_inventory_summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(df.groupby(["pipeline", "status"]).size().reset_index(name="count").to_string(index=False))
    print("wrote current_feature_inventory.csv, current_feature_inventory.md")


if __name__ == "__main__":
    main()
