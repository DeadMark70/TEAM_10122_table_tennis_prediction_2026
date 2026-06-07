"""V391 OOF-gated submission packager.

Packages larger V362-anchor candidates only when both V389 OOF/proxy evidence
and V390 augmented scorer evidence pass hard safety gates.
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
OUTDIR = ROOT / "v391_oof_gated_submission_packager"
ANCHOR = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V389_POINT = ROOT / "v389_synthetic_oof_proxy_lab" / "ranked_point_pool.csv"
V389_ACTION = ROOT / "v389_synthetic_oof_proxy_lab" / "ranked_action_pool.csv"
V390_POINT = ROOT / "v390_synthetic_augmented_scorer" / "point_augmented_scores.csv"
V390_ACTION = ROOT / "v390_synthetic_augmented_scorer" / "action_augmented_scores.csv"

POINT_BUDGETS = (36, 72, 120, 180)
POINT_SCORE_THRESHOLD = 0.50
ACTION_STRONG_THRESHOLD = 0.75


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _score_column(frame: pd.DataFrame, *, preferred: tuple[str, ...]) -> str:
    for column in preferred:
        if column in frame.columns:
            return column
    raise ValueError(f"score frame missing one of: {list(preferred)}")


def _point0_mask(frame: pd.DataFrame) -> pd.Series:
    if "is_point0_addition" in frame.columns:
        return _as_bool(frame["is_point0_addition"])
    if {"base_point", "candidate_point"}.issubset(frame.columns):
        base = pd.to_numeric(frame["base_point"], errors="coerce")
        cand = pd.to_numeric(frame["candidate_point"], errors="coerce")
        return (base != 0) & (cand == 0)
    return pd.Series([False] * len(frame), index=frame.index)


def _serve_addition_mask(frame: pd.DataFrame) -> pd.Series:
    if "is_serve_15_18_addition" in frame.columns:
        explicit = _as_bool(frame["is_serve_15_18_addition"])
    else:
        explicit = pd.Series([False] * len(frame), index=frame.index)
    cand = pd.to_numeric(frame.get("candidate_action", pd.Series(index=frame.index)), errors="coerce")
    return explicit | cand.isin(SERVE_ACTION_CLASSES)


def _merge_evidence(
    ranked: pd.DataFrame,
    augmented: pd.DataFrame,
    *,
    candidate_col: str,
    ranked_score_names: tuple[str, ...],
    augmented_score_names: tuple[str, ...],
) -> pd.DataFrame:
    required_ranked = {"rally_uid", candidate_col, "pass_gate"}
    required_augmented = {"rally_uid", candidate_col}
    missing_ranked = required_ranked - set(ranked.columns)
    missing_augmented = required_augmented - set(augmented.columns)
    if missing_ranked:
        raise ValueError(f"ranked evidence missing required columns: {sorted(missing_ranked)}")
    if missing_augmented:
        raise ValueError(f"augmented evidence missing required columns: {sorted(missing_augmented)}")

    ranked_score = _score_column(ranked, preferred=ranked_score_names)
    augmented_score = _score_column(augmented, preferred=augmented_score_names)

    left = ranked.copy()
    right = augmented.copy()
    left[candidate_col] = pd.to_numeric(left[candidate_col], errors="coerce")
    right[candidate_col] = pd.to_numeric(right[candidate_col], errors="coerce")
    left["v389_score"] = pd.to_numeric(left[ranked_score], errors="coerce").fillna(float("-inf"))
    right["v390_score"] = pd.to_numeric(right[augmented_score], errors="coerce").fillna(float("-inf"))

    merged = left.merge(
        right.loc[:, ["rally_uid", candidate_col, "v390_score"]],
        on=["rally_uid", candidate_col],
        how="inner",
    )
    merged["pass_gate"] = _as_bool(merged["pass_gate"])
    return merged


def select_point_rows(ranked: pd.DataFrame, augmented: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Select point rows with passing V389/V390 evidence and no point0 additions."""

    if budget <= 0 or ranked.empty or augmented.empty:
        return ranked.head(0).copy()

    rows = _merge_evidence(
        ranked,
        augmented,
        candidate_col="candidate_point",
        ranked_score_names=("oof_proxy_score", "proxy_score", "rank_score", "combined_score", "score"),
        augmented_score_names=(
            "risk_adjusted_score",
            "augmented_model_score",
            "augmented_score",
            "synthetic_augmented_score",
            "model_score",
            "combined_score",
            "score",
        ),
    )
    rows = rows[
        rows["pass_gate"]
        & rows["candidate_point"].between(0, 9)
        & (~_point0_mask(rows))
        & rows["v390_score"].ge(POINT_SCORE_THRESHOLD)
    ].copy()
    if rows.empty:
        return rows

    rows = rows.sort_values(["v389_score", "v390_score", "rally_uid"], ascending=[False, False, True])
    return rows.drop_duplicates(subset=["rally_uid"], keep="first").head(budget).reset_index(drop=True)


def select_action_rows(ranked: pd.DataFrame, augmented: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Select very-strong action rows without introducing serve classes 15-18."""

    if budget <= 0 or ranked.empty or augmented.empty:
        return ranked.head(0).copy()

    rows = _merge_evidence(
        ranked,
        augmented,
        candidate_col="candidate_action",
        ranked_score_names=("oof_proxy_score", "proxy_score", "rank_score", "combined_score", "score"),
        augmented_score_names=(
            "risk_adjusted_score",
            "augmented_model_score",
            "augmented_score",
            "synthetic_augmented_score",
            "model_score",
            "combined_score",
            "score",
        ),
    )
    rows = rows[
        rows["pass_gate"]
        & rows["candidate_action"].between(0, 18)
        & (~_serve_addition_mask(rows))
        & rows["v390_score"].ge(ACTION_STRONG_THRESHOLD)
    ].copy()
    if rows.empty:
        return rows

    rows = rows.sort_values(["v389_score", "v390_score", "rally_uid"], ascending=[False, False, True])
    return rows.drop_duplicates(subset=["rally_uid"], keep="first").head(budget).reset_index(drop=True)


def _updates(rows: pd.DataFrame, candidate_col: str) -> dict[str, int]:
    if rows.empty:
        return {}
    return {
        str(row["rally_uid"]): int(row[candidate_col])
        for _, row in rows.loc[:, ["rally_uid", candidate_col]].iterrows()
    }


def _lookup(updates: dict[str, int], key: Any, fallback: Any) -> Any:
    return updates.get(str(key), fallback)


def package_candidate(
    anchor: pd.DataFrame,
    point_rows: pd.DataFrame,
    action_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Apply selected rows to a submission while preserving unselected targets."""

    out = anchor.copy()
    point_updates = _updates(point_rows, "candidate_point") if not point_rows.empty else {}
    action_updates = _updates(action_rows, "candidate_action") if not action_rows.empty else {}
    if point_updates:
        out["pointId"] = out.apply(lambda row: int(_lookup(point_updates, row["rally_uid"], row["pointId"])), axis=1)
    if action_updates:
        out["actionId"] = out.apply(
            lambda row: int(_lookup(action_updates, row["rally_uid"], row["actionId"])), axis=1
        )
    return out.loc[:, SUBMISSION_COLUMNS]


def _diff(anchor: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    merged = anchor.merge(candidate, on="rally_uid", suffixes=("_base", "_cand"))
    action_churn = int((merged["actionId_base"].astype(int) != merged["actionId_cand"].astype(int)).sum())
    point_churn = int((merged["pointId_base"].astype(int) != merged["pointId_cand"].astype(int)).sum())
    point0_additions = int(
        ((merged["pointId_base"].astype(int) != 0) & (merged["pointId_cand"].astype(int) == 0)).sum()
    )
    server_changed = int(
        (pd.to_numeric(merged["serverGetPoint_base"]) - pd.to_numeric(merged["serverGetPoint_cand"]))
        .abs()
        .gt(1e-12)
        .sum()
    )
    serve_delta = int(
        candidate["actionId"].astype(int).isin(SERVE_ACTION_CLASSES).sum()
        - anchor["actionId"].astype(int).isin(SERVE_ACTION_CLASSES).sum()
    )
    return {
        "action_churn": action_churn,
        "point_churn": point_churn,
        "point0_additions": point0_additions,
        "server_changed": server_changed,
        "serve_15_18_delta": serve_delta,
    }


def _write_submission(
    *,
    outdir: Path,
    anchor: pd.DataFrame,
    point_rows: pd.DataFrame,
    action_rows: pd.DataFrame,
    candidate_name: str,
    output_name: str,
    rank: int,
) -> dict[str, Any] | None:
    if point_rows.empty and action_rows.empty:
        return None

    candidate = package_candidate(anchor, point_rows, action_rows)
    validate_submission_schema(candidate, expected_rows=None)
    stats = _diff(anchor, candidate)
    if stats["point0_additions"] != 0 or stats["server_changed"] != 0:
        return None
    if not action_rows.empty and stats["serve_15_18_delta"] > 0:
        return None

    out_path = outdir / output_name
    selected_path = outdir / f"selected_rows_{candidate_name}.csv"
    candidate.to_csv(out_path, index=False)
    selected = pd.concat(
        [
            point_rows.assign(selected_target="point") if not point_rows.empty else point_rows,
            action_rows.assign(selected_target="action") if not action_rows.empty else action_rows,
        ],
        ignore_index=True,
        sort=False,
    )
    selected.to_csv(selected_path, index=False)
    return {
        "rank": rank,
        "candidate": candidate_name,
        "path": _rel(out_path),
        "selected_rows": _rel(selected_path),
        "selected_row_count": int(len(selected)),
        **stats,
    }


def _empty_ranked_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank",
            "candidate",
            "path",
            "selected_rows",
            "selected_row_count",
            "action_churn",
            "point_churn",
            "point0_additions",
            "server_changed",
            "serve_15_18_delta",
        ]
    )


def run_pipeline(
    outdir: Path = OUTDIR,
    anchor_path: Path = ANCHOR,
    ranked_point_path: Path = V389_POINT,
    ranked_action_path: Path = V389_ACTION,
    augmented_point_path: Path = V390_POINT,
    augmented_action_path: Path = V390_ACTION,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "ranked_point_pool": Path(ranked_point_path),
        "ranked_action_pool": Path(ranked_action_path),
        "point_augmented_scores": Path(augmented_point_path),
        "action_augmented_scores": Path(augmented_action_path),
    }
    missing_inputs = {name: _rel(path) for name, path in paths.items() if not path.exists()}

    anchor = pd.read_csv(anchor_path)
    validate_submission_schema(anchor, expected_rows=None)

    generated: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}

    if not missing_inputs:
        ranked_point = pd.read_csv(paths["ranked_point_pool"])
        ranked_action = pd.read_csv(paths["ranked_action_pool"])
        augmented_point = pd.read_csv(paths["point_augmented_scores"])
        augmented_action = pd.read_csv(paths["action_augmented_scores"])

        selected_point72 = pd.DataFrame()
        for rank, budget in enumerate(POINT_BUDGETS, start=1):
            selected = select_point_rows(ranked_point, augmented_point, budget=budget)
            selected_counts[f"point_top{budget}"] = int(len(selected))
            item = _write_submission(
                outdir=outdir,
                anchor=anchor,
                point_rows=selected,
                action_rows=pd.DataFrame(),
                candidate_name=f"v391_oof_point_top{budget}",
                output_name=f"submission_v391_oof_point_top{budget}__v173action_v300server.csv",
                rank=rank,
            )
            if item:
                generated.append(item)
            if budget == 72:
                selected_point72 = selected

        selected_action5 = select_action_rows(ranked_action, augmented_action, budget=5)
        selected_counts["action_top5"] = int(len(selected_action5))
        item = _write_submission(
            outdir=outdir,
            anchor=anchor,
            point_rows=selected_point72,
            action_rows=selected_action5,
            candidate_name="v391_oof_point72_action5",
            output_name="submission_v391_oof_point72_action5__v300server.csv",
            rank=len(POINT_BUDGETS) + 1,
        )
        if item:
            generated.append(item)

    ranked = pd.DataFrame(generated) if generated else _empty_ranked_candidates()
    if not ranked.empty:
        ranked = ranked.sort_values(["rank", "point0_additions", "server_changed"]).reset_index(drop=True)
    ranked_path = outdir / "ranked_candidates.csv"
    ranked.to_csv(ranked_path, index=False)

    report = {
        "version": "V391",
        "anchor": _rel(Path(anchor_path)),
        "missing_inputs": missing_inputs,
        "generated_submission_count": int(len(generated)),
        "generated_candidates": generated,
        "selected_counts": selected_counts,
        "outputs": {"ranked_candidates": _rel(ranked_path)},
        "policy": {
            "point": "requires V389 pass_gate, V390 score threshold, zero point0 additions, unchanged action/server",
            "action": "requires very strong V390 score and blocks serve 15-18 additions",
            "synthetic_usage": "V389/V390 evidence only; no direct synthetic answer generation",
        },
    }
    write_json(outdir / "search_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
