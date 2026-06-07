"""V387 packager for V386 expanded synthetic contrastive scores.

The synthetic grammar is used only as auxiliary ranking evidence. Packaging
starts from the clean V362 anchor and preserves untouched targets unless a
selected V386 candidate explicitly updates that target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v387_expanded_synthetic_packager"
ANCHOR = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V386_POINT = ROOT / "v386_synthetic_contrastive_scorer" / "point_candidate_contrastive_scores.csv"
V386_ACTION = ROOT / "v386_synthetic_contrastive_scorer" / "action_candidate_contrastive_scores.csv"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _score_column(frame: pd.DataFrame) -> str:
    for column in ("contrastive_score", "synthetic_compatibility_score", "synthetic_adjusted_score"):
        if column in frame.columns:
            return column
    raise ValueError("score frame must include contrastive_score or synthetic_compatibility_score")


def _allowed_mask(frame: pd.DataFrame) -> pd.Series:
    if "synthetic_allowed" not in frame.columns:
        return pd.Series([True] * len(frame), index=frame.index)
    return _as_bool(frame["synthetic_allowed"])


def _point0_mask(frame: pd.DataFrame) -> pd.Series:
    if "is_point0_addition" in frame.columns:
        return _as_bool(frame["is_point0_addition"])
    if {"base_point", "candidate_point"}.issubset(frame.columns):
        base = pd.to_numeric(frame["base_point"], errors="coerce")
        cand = pd.to_numeric(frame["candidate_point"], errors="coerce")
        return (base != 0) & (cand == 0)
    return pd.Series([False] * len(frame), index=frame.index)


def select_contrastive_point_updates(scores: pd.DataFrame, budget: int) -> dict[Any, int]:
    """Select highest-ranked V386 point updates without point0 additions."""

    if budget <= 0 or scores.empty:
        return {}
    required = {"rally_uid", "candidate_point"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"point scores missing required columns: {sorted(missing)}")

    score_col = _score_column(scores)
    rows = scores.copy()
    rows[score_col] = pd.to_numeric(rows[score_col], errors="coerce").fillna(float("-inf"))
    rows["candidate_point"] = pd.to_numeric(rows["candidate_point"], errors="coerce")
    rows = rows[_allowed_mask(rows) & (~_point0_mask(rows)) & rows["candidate_point"].between(0, 9)]
    if rows.empty:
        return {}

    rows = rows.sort_values([score_col, "rally_uid"], ascending=[False, True])
    rows = rows.drop_duplicates(subset=["rally_uid"], keep="first").head(budget)
    return {row["rally_uid"]: int(row["candidate_point"]) for _, row in rows.iterrows()}


def select_contrastive_action_updates(scores: pd.DataFrame, budget: int) -> dict[Any, int]:
    """Select highest-ranked V386 action updates without introducing serves 15-18."""

    if budget <= 0 or scores.empty:
        return {}
    required = {"rally_uid", "candidate_action"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"action scores missing required columns: {sorted(missing)}")

    score_col = _score_column(scores)
    rows = scores.copy()
    rows[score_col] = pd.to_numeric(rows[score_col], errors="coerce").fillna(float("-inf"))
    rows["candidate_action"] = pd.to_numeric(rows["candidate_action"], errors="coerce")
    rows = rows[
        _allowed_mask(rows)
        & rows["candidate_action"].between(0, 18)
        & (~rows["candidate_action"].isin(SERVE_ACTION_CLASSES))
    ]
    if rows.empty:
        return {}

    rows = rows.sort_values([score_col, "rally_uid"], ascending=[False, True])
    rows = rows.drop_duplicates(subset=["rally_uid"], keep="first").head(budget)
    return {row["rally_uid"]: int(row["candidate_action"]) for _, row in rows.iterrows()}


def _lookup_update(updates: dict[Any, int], key: Any, fallback: Any) -> Any:
    if key in updates:
        return updates[key]
    text_key = str(key)
    if text_key in updates:
        return updates[text_key]
    return fallback


def package_candidate(
    anchor: pd.DataFrame,
    point_updates: dict[Any, int],
    action_updates: dict[Any, int],
) -> pd.DataFrame:
    """Apply selected point/action updates while preserving server values."""

    out = anchor.copy()
    if point_updates:
        out["pointId"] = out.apply(
            lambda row: int(_lookup_update(point_updates, row["rally_uid"], row["pointId"])), axis=1
        )
    if action_updates:
        out["actionId"] = out.apply(
            lambda row: int(_lookup_update(action_updates, row["rally_uid"], row["actionId"])), axis=1
        )
    return out.loc[:, SUBMISSION_COLUMNS]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_anchor(anchor_path: Path = ANCHOR) -> pd.DataFrame:
    frame = pd.read_csv(anchor_path)
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
    serve_delta = int(
        cand["actionId"].astype(int).isin(SERVE_ACTION_CLASSES).sum()
        - anchor["actionId"].astype(int).isin(SERVE_ACTION_CLASSES).sum()
    )
    return {
        "action_churn": action_churn,
        "point_churn": point_churn,
        "point0_additions": point0_add,
        "server_changed": server_changed,
        "serve_15_18_delta": serve_delta,
    }


def _selected_rows(scores: pd.DataFrame, updates: dict[Any, int], candidate_col: str) -> pd.DataFrame:
    if not updates:
        return scores.head(0).copy()
    update_by_key = {str(key): int(value) for key, value in updates.items()}
    rows = scores.copy()
    candidate = pd.to_numeric(rows[candidate_col], errors="coerce")
    selected = rows.apply(
        lambda row: str(row["rally_uid"]) in update_by_key
        and int(candidate.loc[row.name]) == update_by_key[str(row["rally_uid"])],
        axis=1,
    )
    return rows[selected].copy()


def _write_candidate(
    *,
    outdir: Path,
    anchor: pd.DataFrame,
    scores: pd.DataFrame,
    updates: dict[Any, int],
    candidate_name: str,
    output_name: str,
    selected_prefix: str,
    point_updates: bool,
    rank: int,
    source: Path,
) -> dict[str, Any] | None:
    if not updates:
        return None

    if point_updates:
        cand = package_candidate(anchor, point_updates=updates, action_updates={})
        selected = _selected_rows(scores, updates, "candidate_point")
    else:
        cand = package_candidate(anchor, point_updates={}, action_updates=updates)
        selected = _selected_rows(scores, updates, "candidate_action")

    validate_submission_schema(cand)
    out_path = outdir / output_name
    cand.to_csv(out_path, index=False)
    selected_path = outdir / f"{selected_prefix}_{candidate_name}.csv"
    selected.to_csv(selected_path, index=False)
    return {
        "rank": rank,
        "candidate": candidate_name,
        "path": _rel(out_path),
        "source": _rel(source),
        "selected_rows": _rel(selected_path),
        "selected_row_count": int(len(selected)),
        **_diff(anchor, cand),
    }


def run_pipeline(
    outdir: Path = OUTDIR,
    point_scores_path: Path = V386_POINT,
    action_scores_path: Path = V386_ACTION,
    anchor_path: Path = ANCHOR,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    point_scores_path = Path(point_scores_path)
    action_scores_path = Path(action_scores_path)

    point_available = point_scores_path.exists()
    action_available = action_scores_path.exists()
    generated: list[dict[str, Any]] = []

    anchor = _load_anchor(anchor_path)
    if point_available:
        point_scores = pd.read_csv(point_scores_path)
        for rank, budget in enumerate((9, 19, 36), start=1):
            updates = select_contrastive_point_updates(point_scores, budget=budget)
            item = _write_candidate(
                outdir=outdir,
                anchor=anchor,
                scores=point_scores,
                updates=updates,
                candidate_name=f"v387_contrastive_point_top{budget}",
                output_name=(
                    f"submission_v387_contrastive_point_top{budget}"
                    "__v173action_v300server.csv"
                ),
                selected_prefix="selected_rows",
                point_updates=True,
                rank=rank,
                source=point_scores_path,
            )
            if item:
                generated.append(item)

    if action_available:
        action_scores = pd.read_csv(action_scores_path)
        updates = select_contrastive_action_updates(action_scores, budget=5)
        item = _write_candidate(
            outdir=outdir,
            anchor=anchor,
            scores=action_scores,
            updates=updates,
            candidate_name="v387_contrastive_action_top5",
            output_name="submission_v387_contrastive_action_top5__v362point_v300server.csv",
            selected_prefix="selected_rows",
            point_updates=False,
            rank=4,
            source=action_scores_path,
        )
        if item:
            generated.append(item)

    ranked = pd.DataFrame(generated)
    if not ranked.empty:
        ranked = ranked.sort_values(["rank", "point0_additions", "point_churn", "action_churn"])
    ranked_path = outdir / "ranked_candidates.csv"
    ranked.to_csv(ranked_path, index=False)

    report = {
        "version": "V387",
        "anchor": _rel(anchor_path),
        "v386_scores_available": {"point": point_available, "action": action_available},
        "generated_submission_count": int(len(generated)),
        "generated_candidates": generated,
        "top_candidate": None if ranked.empty else ranked.iloc[0].to_dict(),
        "outputs": {"ranked_candidates": _rel(ranked_path)},
        "policy": {
            "point_candidates": "server/action preserved, point0 additions blocked, synthetic_allowed required",
            "action_candidates": "point/server preserved, serve 15-18 candidates blocked",
            "synthetic_usage": "V386 ranking evidence only; no direct synthetic answer generation",
        },
    }
    write_json(outdir / "search_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
