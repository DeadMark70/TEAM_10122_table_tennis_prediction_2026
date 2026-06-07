"""V383 packager for synthetic-teacher-adjusted candidates.

This module uses V382 scores as auxiliary evidence only. It starts from the
current clean V362 anchor and exports a small set of candidate submissions. It
does not use synthetic data to answer test rows directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS, validate_submission_schema, write_json


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v383_synthetic_adjusted_packager"
ANCHOR = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V374_SAFE_KEEP = (
    ROOT
    / "v374_physical_rule_audit"
    / "submission_v374_v370_safe_keep_only__v173action_v300server.csv"
)
V374_MEDIUM_KEEP = (
    ROOT
    / "v374_physical_rule_audit"
    / "submission_v374_v370_medium_keep_only__v173action_v300server.csv"
)
V382_POINT = ROOT / "v382_synthetic_teacher_evaluator" / "point_candidate_synthetic_scores.csv"
V382_ACTION = ROOT / "v382_synthetic_teacher_evaluator" / "action_candidate_synthetic_scores.csv"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def block_unsupported_point0(rows: pd.DataFrame, min_score: float = 0.8) -> pd.DataFrame:
    out = rows.copy()
    base = out["base_point"].astype(int)
    cand = out["candidate_point"].astype(int)
    score = pd.to_numeric(out["synthetic_score"], errors="coerce").fillna(0.0)
    out["allowed"] = ~((base != 0) & (cand == 0) & (score < min_score))
    return out


def select_supported_point_updates(
    scores: pd.DataFrame,
    budget: int,
    min_source_families: int = 6,
    min_support: int = 50,
) -> dict[Any, int]:
    """Select synthetic-supported point updates.

    The filter intentionally uses only synthetic/coarse compatibility evidence
    and historical candidate agreement. It blocks point0 additions because the
    branch is meant to validate rare landing grammar, not terminal guessing.
    """

    if budget <= 0 or scores.empty:
        return {}
    rows = scores.copy()
    rows["is_point0_addition_bool"] = _as_bool(rows["is_point0_addition"])
    rows["same_depth_bool"] = _as_bool(rows["same_depth"])
    numeric_cols = [
        "source_family_count",
        "support_count",
        "synthetic_teacher_score",
        "synthetic_adjusted_score",
    ]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    allowed = rows[
        (~rows["is_point0_addition_bool"])
        & (rows["synthetic_teacher_score"] > 0)
        & (rows["source_family_count"] >= min_source_families)
        & (rows["support_count"] >= min_support)
    ].copy()
    if allowed.empty:
        return {}
    allowed["depth_rank"] = allowed["same_depth_bool"].astype(int)
    allowed = allowed.sort_values(
        ["depth_rank", "synthetic_adjusted_score", "synthetic_teacher_score", "support_count"],
        ascending=[False, False, False, False],
    ).head(budget)
    return {
        row["rally_uid"]: int(row["candidate_point"])
        for _, row in allowed.iterrows()
    }


def package_candidate(
    anchor: pd.DataFrame,
    point_updates: dict[Any, int],
    action_updates: dict[Any, int],
) -> pd.DataFrame:
    out = anchor.copy()
    if point_updates:
        out["pointId"] = out.apply(
            lambda row: int(point_updates.get(row["rally_uid"], row["pointId"])), axis=1
        )
    if action_updates:
        out["actionId"] = out.apply(
            lambda row: int(action_updates.get(row["rally_uid"], row["actionId"])), axis=1
        )
    return out.loc[:, SUBMISSION_COLUMNS]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_anchor() -> pd.DataFrame:
    frame = pd.read_csv(ANCHOR)
    validate_submission_schema(frame)
    return frame


def _diff(anchor: pd.DataFrame, cand: pd.DataFrame) -> dict[str, Any]:
    merged = anchor.merge(cand, on="rally_uid", suffixes=("_base", "_cand"))
    action_churn = int((merged["actionId_base"].astype(int) != merged["actionId_cand"].astype(int)).sum())
    point_churn = int((merged["pointId_base"].astype(int) != merged["pointId_cand"].astype(int)).sum())
    point0_add = int(
        ((merged["pointId_base"].astype(int) != 0) & (merged["pointId_cand"].astype(int) == 0)).sum()
    )
    server_changed = int(
        (pd.to_numeric(merged["serverGetPoint_base"]) - pd.to_numeric(merged["serverGetPoint_cand"]))
        .abs()
        .gt(1e-12)
        .sum()
    )
    return {
        "action_churn": action_churn,
        "point_churn": point_churn,
        "point0_additions": point0_add,
        "server_changed": server_changed,
    }


def _copy_candidate(src: Path, out_name: str) -> dict[str, Any] | None:
    if not src.exists():
        return None
    anchor = _load_anchor()
    cand = pd.read_csv(src)
    validate_submission_schema(cand)
    out_path = OUTDIR / out_name
    cand.to_csv(out_path, index=False)
    return {"path": _rel(out_path), **_diff(anchor, cand)}


def _score_package(out_name: str, budget: int) -> dict[str, Any] | None:
    if not V382_POINT.exists():
        return None
    anchor = _load_anchor()
    scores = pd.read_csv(V382_POINT)
    updates = select_supported_point_updates(scores, budget=budget)
    if not updates:
        return None
    cand = package_candidate(anchor, point_updates=updates, action_updates={})
    out_path = OUTDIR / out_name
    cand.to_csv(out_path, index=False)
    selected = scores[
        scores.apply(
            lambda row: int(row["candidate_point"]) == int(updates.get(row["rally_uid"], -999)),
            axis=1,
        )
    ].copy()
    selected_path = OUTDIR / out_name.replace("submission_", "selected_rows_").replace(".csv", ".csv")
    selected.to_csv(selected_path, index=False)
    return {
        "path": _rel(out_path),
        "selected_rows": _rel(selected_path),
        "synthetic_selected_rows": int(len(updates)),
        **_diff(anchor, cand),
    }


def run_pipeline(outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    generated = []

    scored_safe = _score_package("submission_v383_synth_scored_top9__v173action_v300server.csv", budget=9)
    if scored_safe:
        scored_safe["candidate"] = "v383_synth_scored_top9"
        scored_safe["source"] = _rel(V382_POINT)
        scored_safe["rank"] = 1
        generated.append(scored_safe)

    scored_medium = _score_package(
        "submission_v383_synth_scored_top19__v173action_v300server.csv", budget=19
    )
    if scored_medium:
        scored_medium["candidate"] = "v383_synth_scored_top19"
        scored_medium["source"] = _rel(V382_POINT)
        scored_medium["rank"] = 2
        generated.append(scored_medium)

    safe = _copy_candidate(V374_SAFE_KEEP, "submission_v383_synth_point_safe__v173action_v300server.csv")
    if safe:
        safe["candidate"] = "v383_synth_point_safe"
        safe["source"] = _rel(V374_SAFE_KEEP)
        safe["rank"] = 3
        generated.append(safe)

    medium = _copy_candidate(
        V374_MEDIUM_KEEP, "submission_v383_synth_point_medium__v173action_v300server.csv"
    )
    if medium:
        medium["candidate"] = "v383_synth_point_medium"
        medium["source"] = _rel(V374_MEDIUM_KEEP)
        medium["rank"] = 4
        generated.append(medium)

    ranked = pd.DataFrame(generated)
    if not ranked.empty:
        ranked = ranked.sort_values(["rank", "point0_additions", "point_churn"]).reset_index(drop=True)
    ranked_path = outdir / "ranked_candidates.csv"
    ranked.to_csv(ranked_path, index=False)
    report = {
        "version": "V383",
        "synthetic_scores_available": {
            "point": V382_POINT.exists(),
            "action": V382_ACTION.exists(),
        },
        "generated_submission_count": int(len(generated)),
        "top_candidate": None if ranked.empty else ranked.iloc[0].to_dict(),
        "outputs": {"ranked_candidates": _rel(ranked_path)},
        "policy": {
            "point0_additions": "blocked unless synthetic score >= 0.8 and historical support exists",
            "server": "preserved from clean anchor",
            "synthetic_usage": "auxiliary teacher/scoring only",
        },
    }
    write_json(outdir / "search_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
