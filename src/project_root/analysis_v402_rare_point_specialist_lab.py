"""V402 rare point specialist lab.

This script builds point-only specialist submissions against the public-proven
V362 anchor. It uses deterministic row evidence gates, train-derived backoff
support, and optional V401 compatibility rows when available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v402_rare_point_specialist_lab"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
V362_SCORED = ROOT / "v362_point_hierarchical_specialists" / "scored_candidates.csv"
V362_SOURCE_VOTES = ROOT / "v362_point_hierarchical_specialists" / "source_votes.csv"
V401_DIR = ROOT / "v401_action_point_compatibility"

SPECIALIST_GROUPS = {
    "short_control": {1, 2, 3},
    "half_long_boundary": {4, 5, 6},
    "long_side": {7, 8, 9},
}
SUBMISSION_NAMES = {
    "long_side": "submission_v402_longside_top9__v173action_v300server.csv",
    "short_control": "submission_v402_short_control_top9__v173action_v300server.csv",
    "half_long_boundary": "submission_v402_half_boundary_top9__v173action_v300server.csv",
    "mixed_specialists": "submission_v402_mixed_specialists_top15__v173action_v300server.csv",
}
SELECTED_NAMES = {
    "long_side": "selected_rows_longside_top9.csv",
    "short_control": "selected_rows_short_control_top9.csv",
    "half_long_boundary": "selected_rows_half_boundary_top9.csv",
    "mixed_specialists": "selected_rows_mixed_specialists_top15.csv",
}
SERVE_ACTIONS = {15, 16, 17, 18}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def point_depth(point_id: int) -> str:
    point = int(point_id)
    if point == 0:
        return "terminal"
    if 1 <= point <= 3:
        return "short"
    if 4 <= point <= 6:
        return "half"
    if 7 <= point <= 9:
        return "long"
    raise ValueError(f"pointId outside 0..9: {point_id}")


def point_side(point_id: int) -> str:
    point = int(point_id)
    if point == 0:
        return "terminal"
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return ("left", "middle", "right")[(point - 1) % 3]


def point_group(point_id: int) -> str:
    point = int(point_id)
    if point in SPECIALIST_GROUPS["short_control"]:
        return "short_control"
    if point in SPECIALIST_GROUPS["half_long_boundary"]:
        return "half_long_boundary"
    if point in SPECIALIST_GROUPS["long_side"]:
        return "long_side"
    if point == 0:
        return "terminal"
    raise ValueError(f"pointId outside 0..9: {point_id}")


def infer_phase(strike_number: Any) -> str:
    try:
        strike = int(strike_number)
    except (TypeError, ValueError):
        return "rally"
    if strike <= 1:
        return "receive"
    if strike == 2:
        return "third_ball"
    if strike == 3:
        return "fourth_ball"
    return "rally"


def prefix_len_bin(strike_number: Any) -> str:
    try:
        strike = int(strike_number)
    except (TypeError, ValueError):
        return "unknown"
    if strike <= 1:
        return "1"
    if strike == 2:
        return "2"
    if strike == 3:
        return "3"
    if strike <= 6:
        return "4_6"
    return "7p"


def action_family(action_id: Any) -> str:
    try:
        action = int(action_id)
    except (TypeError, ValueError):
        return "unknown"
    if action in SERVE_ACTIONS:
        return "serve"
    if action in {0, 1, 2, 3, 4}:
        return "control"
    if action in {5, 6, 7, 8, 9}:
        return "attack"
    if action in {10, 11, 12, 13, 14}:
        return "defense"
    return "other"


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_id",
            "rally_uid",
            "base_point",
            "candidate_point",
            "transition",
            "agreement_count",
            "source_diversity_count",
            "source_dirs",
            "sources",
            "source_risks",
        ]
    )


def load_row_candidate_pool(anchor: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize existing row-level point evidence."""
    if V362_SCORED.exists():
        scored = pd.read_csv(V362_SCORED)
        required = {"row_id", "rally_uid", "base_point", "candidate_point", "agreement_count"}
        if required.issubset(scored.columns):
            out = scored.copy()
            if "source_diversity_count" not in out.columns:
                out["source_diversity_count"] = out.get("source_dir_count", 1)
            for col in ["source_dirs", "sources", "source_risks"]:
                if col not in out.columns:
                    out[col] = ""
            return out, {"source": relative_path(V362_SCORED), "available": True, "rows": int(len(out))}

    if not V362_SOURCE_VOTES.exists():
        return _empty_candidates(), {"source": None, "available": False, "rows": 0}

    votes = pd.read_csv(V362_SOURCE_VOTES)
    required = {"row_id", "rally_uid", "base_point", "candidate_point", "source_dir", "source"}
    if not required.issubset(votes.columns):
        return _empty_candidates(), {"source": relative_path(V362_SOURCE_VOTES), "available": False, "rows": 0}

    work = votes.copy()
    work["row_id"] = pd.to_numeric(work["row_id"], errors="coerce")
    work["base_point"] = pd.to_numeric(work["base_point"], errors="coerce")
    work["candidate_point"] = pd.to_numeric(work["candidate_point"], errors="coerce")
    work = work.dropna(subset=["row_id", "base_point", "candidate_point"])
    work["row_id"] = work["row_id"].astype(int)
    work["base_point"] = work["base_point"].astype(int)
    work["candidate_point"] = work["candidate_point"].astype(int)
    work = work[work["row_id"].between(0, len(anchor) - 1)]
    grouped = (
        work.groupby(["row_id", "rally_uid", "base_point", "candidate_point"], as_index=False)
        .agg(
            agreement_count=("source", "nunique"),
            source_diversity_count=("source_dir", "nunique"),
            source_dirs=("source_dir", lambda s: "|".join(sorted(set(map(str, s))))),
            sources=("source", lambda s: "|".join(sorted(set(map(str, s))))),
            source_risks=("source_risk", lambda s: "|".join(sorted(set(map(str, s)))) if "source_risk" in work.columns else ""),
        )
        .reset_index(drop=True)
    )
    grouped["transition"] = grouped["base_point"].astype(str) + "->" + grouped["candidate_point"].astype(str)
    return grouped, {"source": relative_path(V362_SOURCE_VOTES), "available": True, "rows": int(len(grouped))}


def load_test_context(anchor: pd.DataFrame) -> pd.DataFrame:
    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH)
    else:
        test = pd.DataFrame()
    if len(test) != len(anchor) or "rally_uid" not in test.columns:
        test = pd.DataFrame(
            {
                "rally_uid": anchor["rally_uid"].to_numpy(),
                "strikeNumber": np.ones(len(anchor), dtype=int),
                "actionId": anchor["actionId"].to_numpy(),
                "pointId": anchor["pointId"].to_numpy(),
                "spinId": np.zeros(len(anchor), dtype=int),
                "strengthId": np.zeros(len(anchor), dtype=int),
            }
        )
    out = test.reset_index(drop=True).copy()
    out["row_id"] = np.arange(len(out))
    if "actionId" not in out.columns:
        out["actionId"] = anchor["actionId"].to_numpy()
    if "pointId" not in out.columns:
        out["pointId"] = anchor["pointId"].to_numpy()
    for col in ["strikeNumber", "spinId", "strengthId"]:
        if col not in out.columns:
            out[col] = 0
    out["phase"] = out["strikeNumber"].map(infer_phase)
    out["prefix_len_bin"] = out["strikeNumber"].map(prefix_len_bin)
    out["lag0_action_family"] = out["actionId"].map(action_family)
    out["action_family"] = out["lag0_action_family"]
    out["lag0_point_depth"] = out["pointId"].astype(int).map(point_depth)
    out["lag0_point_side"] = out["pointId"].astype(int).map(point_side)
    out["lag0_spin"] = pd.to_numeric(out["spinId"], errors="coerce").fillna(0).astype(int)
    out["lag0_strength"] = pd.to_numeric(out["strengthId"], errors="coerce").fillna(0).astype(int)
    return out


def build_train_backoff_support() -> tuple[dict[tuple[str, str, int], int], dict[int, float], dict[str, Any]]:
    if not TRAIN_PATH.exists():
        return {}, {}, {"available": False, "reason": "missing_train"}
    train = pd.read_csv(TRAIN_PATH)
    required = {"actionId", "pointId", "strikeNumber"}
    if not required.issubset(train.columns):
        return {}, {}, {"available": False, "reason": "missing_columns"}
    work = train.loc[:, ["actionId", "pointId", "strikeNumber"]].copy()
    work["pointId"] = pd.to_numeric(work["pointId"], errors="coerce")
    work = work.dropna(subset=["pointId"])
    work["pointId"] = work["pointId"].astype(int)
    work["action_family"] = work["actionId"].map(action_family)
    work["phase"] = work["strikeNumber"].map(infer_phase)
    counts = work.groupby(["action_family", "phase", "pointId"]).size().to_dict()
    point_counts = work["pointId"].value_counts().to_dict()
    total = float(sum(point_counts.values())) or 1.0
    priors = {int(k): float(v / total) for k, v in point_counts.items()}
    return (
        {(str(a), str(p), int(point)): int(count) for (a, p, point), count in counts.items()},
        priors,
        {"available": True, "rows": int(len(work)), "support_keys": int(len(counts))},
    )


def load_v401_compatibility(v401_dir: Path = V401_DIR) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Return optional compatibility scores keyed by (row_id, candidate_point)."""
    v401_dir = Path(v401_dir)
    if not v401_dir.exists():
        return {}, {"available": False, "fallback": "zero_compatibility", "compatibility_rows": 0}
    frames: list[pd.DataFrame] = []
    for path in sorted(v401_dir.glob("*.csv")):
        if path.name == "ranked_candidates.csv" or path.name.startswith("selected_rows"):
            try:
                frame = pd.read_csv(path)
            except (OSError, pd.errors.ParserError):
                continue
            if {"row_id", "candidate_point"}.issubset(frame.columns):
                frames.append(frame)
    if not frames:
        return {}, {"available": False, "fallback": "zero_compatibility", "compatibility_rows": 0}
    work = pd.concat(frames, ignore_index=True)
    score_col = None
    for col in ["compat_delta", "action_point_compat_score", "compatibility_score", "score"]:
        if col in work.columns:
            score_col = col
            break
    if score_col is None:
        return {}, {"available": False, "fallback": "zero_compatibility", "compatibility_rows": int(len(work))}
    work["row_id"] = pd.to_numeric(work["row_id"], errors="coerce")
    work["candidate_point"] = pd.to_numeric(work["candidate_point"], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["row_id", "candidate_point"])
    compat = {
        (int(row.row_id), int(row.candidate_point)): float(row.score)
        for row in work.rename(columns={score_col: "score"}).itertuples(index=False)
    }
    return compat, {"available": True, "fallback": None, "compatibility_rows": int(len(compat))}


def enrich_candidates(
    candidates: pd.DataFrame,
    context: pd.DataFrame,
    train_support: dict[tuple[str, str, int], int],
    group_priors: dict[int, float],
    compat: dict[tuple[int, int], float],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    work = candidates.copy()
    for col in ["row_id", "base_point", "candidate_point", "agreement_count", "source_diversity_count"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["row_id", "base_point", "candidate_point"])
    work["row_id"] = work["row_id"].astype(int)
    work["base_point"] = work["base_point"].astype(int)
    work["candidate_point"] = work["candidate_point"].astype(int)
    work["agreement_count"] = work["agreement_count"].fillna(1).astype(float)
    work["source_diversity_count"] = work["source_diversity_count"].fillna(1).astype(float)
    work = work[work["base_point"].ne(work["candidate_point"])]
    work = work[work["candidate_point"].between(1, 9)]
    work = work.merge(
        context[
            [
                "row_id",
                "phase",
                "prefix_len_bin",
                "actionId",
                "action_family",
                "lag0_action_family",
                "lag0_point_depth",
                "lag0_point_side",
                "lag0_spin",
                "lag0_strength",
            ]
        ],
        on="row_id",
        how="left",
    )
    work["candidate_group"] = work["candidate_point"].map(point_group)
    work["candidate_depth"] = work["candidate_point"].map(point_depth)
    work["candidate_side"] = work["candidate_point"].map(point_side)
    max_support = max(train_support.values()) if train_support else 1
    support_values = []
    prior_values = []
    compat_values = []
    safety_values = []
    for row in work.itertuples(index=False):
        key = (str(row.action_family), str(row.phase), int(row.candidate_point))
        support = int(train_support.get(key, 0))
        support_values.append(math.log1p(support) / math.log1p(max_support) if max_support > 0 else 0.0)
        prior_values.append(float(group_priors.get(int(row.candidate_point), 0.0)) * 10.0)
        compat_values.append(float(compat.get((int(row.row_id), int(row.candidate_point)), 0.0)))
        action = int(getattr(row, "actionId", -1)) if not pd.isna(getattr(row, "actionId", np.nan)) else -1
        safety_values.append(1.0 if int(row.candidate_point) != 0 and action not in {0} else 0.0)
    work["source_agreement_score"] = (0.7 * np.minimum(work["agreement_count"] / 5.0, 1.0)) + (
        0.3 * np.minimum(work["source_diversity_count"] / 3.0, 1.0)
    )
    work["train_backoff_support_score"] = np.clip(support_values, 0.0, 1.0)
    work["action_point_compat_score"] = np.clip(compat_values, -1.0, 1.0)
    work["group_specific_prior_score"] = np.clip(prior_values, 0.0, 1.0)
    work["nonterminal_safety_score"] = np.asarray(safety_values, dtype=float)
    work["specialist_score"] = (
        0.35 * work["source_agreement_score"]
        + 0.25 * work["train_backoff_support_score"]
        + 0.20 * work["action_point_compat_score"].clip(lower=0.0)
        + 0.10 * work["group_specific_prior_score"]
        + 0.10 * work["nonterminal_safety_score"]
    )
    work["point0_addition"] = (work["base_point"] != 0) & (work["candidate_point"] == 0)
    work["evidence"] = (
        "source_agreement="
        + work["agreement_count"].astype(int).astype(str)
        + ";train_backoff="
        + work["train_backoff_support_score"].round(4).astype(str)
        + ";v401_compat="
        + work["action_point_compat_score"].round(4).astype(str)
    )
    return work.sort_values(
        [
            "specialist_score",
            "agreement_count",
            "source_diversity_count",
            "train_backoff_support_score",
            "row_id",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def apply_specialist_gate(candidates: pd.DataFrame, specialist: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    work = candidates.copy()
    for col in ["base_point", "candidate_point", "agreement_count", "source_diversity_count", "nonterminal_safety_score"]:
        if col not in work.columns:
            if col in {"agreement_count", "source_diversity_count", "nonterminal_safety_score"}:
                work[col] = 1
            else:
                raise ValueError(f"missing required column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["base_point", "candidate_point"])
    work["base_point"] = work["base_point"].astype(int)
    work["candidate_point"] = work["candidate_point"].astype(int)
    work = work[~((work["base_point"] != 0) & (work["candidate_point"] == 0))]

    if specialist == "long_side":
        mask = work["candidate_point"].isin(SPECIALIST_GROUPS["long_side"]) & work["base_point"].ne(0)
    elif specialist == "short_control":
        phase = work.get("phase", pd.Series([""] * len(work), index=work.index)).astype(str)
        incoming_depth = work.get("lag0_point_depth", pd.Series([""] * len(work), index=work.index)).astype(str)
        mask = (
            work["candidate_point"].isin(SPECIALIST_GROUPS["short_control"])
            & work["base_point"].ne(0)
            & (phase.isin({"receive", "third_ball"}) | incoming_depth.eq("short"))
        )
    elif specialist == "half_long_boundary":
        incoming_depth = work.get("lag0_point_depth", pd.Series([""] * len(work), index=work.index)).astype(str)
        candidate_depth = work["candidate_point"].map(point_depth)
        mask = (
            work["candidate_point"].isin(SPECIALIST_GROUPS["half_long_boundary"])
            & work["base_point"].ne(0)
            & (incoming_depth.isin({"half", "long"}) | candidate_depth.eq("half"))
        )
    elif specialist == "terminal_removal":
        mask = (
            work["base_point"].eq(0)
            & work["candidate_point"].between(1, 9)
            & work["agreement_count"].ge(2)
            & work["source_diversity_count"].ge(2)
            & work["nonterminal_safety_score"].gt(0)
        )
    else:
        raise ValueError(f"unknown specialist: {specialist}")

    out = work.loc[mask].copy()
    out["specialist"] = specialist
    return out


def dedupe_mixed_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    work = rows.copy()
    if "specialist_score" not in work.columns:
        work["specialist_score"] = 0.0
    if "agreement_count" not in work.columns:
        work["agreement_count"] = 0
    if "source_diversity_count" not in work.columns:
        work["source_diversity_count"] = 0
    work = work.sort_values(
        ["row_id", "specialist_score", "agreement_count"],
        ascending=[True, False, False],
    )
    return work.drop_duplicates("row_id", keep="first").sort_values(
        ["specialist_score", "agreement_count", "source_diversity_count", "row_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def package_submission(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    if not selected.empty:
        for row in selected.itertuples(index=False):
            row_id = int(row.row_id)
            point = int(row.candidate_point)
            if point == 0 and int(out.at[row_id, "pointId"]) != 0:
                continue
            out.at[row_id, "pointId"] = point
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("actionId changed")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("serverGetPoint changed")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def write_candidate_outputs(
    outdir: Path,
    anchor: pd.DataFrame,
    candidate_name: str,
    selected: pd.DataFrame,
    submission_filename: str,
    selected_filename: str,
) -> dict[str, Any]:
    selected_path = safe_output_path(outdir, selected_filename)
    submission_path = safe_output_path(outdir, submission_filename)
    selected.to_csv(selected_path, index=False)
    submission = package_submission(anchor, selected)
    submission.to_csv(submission_path, index=False)
    point_report = point_distribution_report(anchor["pointId"], submission["pointId"])
    return {
        "candidate": candidate_name,
        "path": submission_filename,
        "selected_rows": selected_filename,
        "selected_row_count": int(len(selected)),
        "action_churn": 0,
        "point_churn": int(point_report["changed_rows"]),
        "point0_additions": int(point_report["point0_additions"]),
        "server_changed": 0,
        "risk": "safe" if int(point_report["point0_additions"]) == 0 else "blocked",
        "evidence": "deterministic_source_agreement_train_backoff_v401_optional",
    }


def run_pipeline(*, outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    anchor = load_submission(ANCHOR_PATH)
    context = load_test_context(anchor)
    raw_candidates, candidate_source_report = load_row_candidate_pool(anchor)
    train_support, group_priors, train_report = build_train_backoff_support()
    compat, v401_report = load_v401_compatibility(V401_DIR)
    candidates = enrich_candidates(raw_candidates, context, train_support, group_priors, compat)

    selected_by_name: dict[str, pd.DataFrame] = {}
    for specialist in ["long_side", "short_control", "half_long_boundary", "terminal_removal"]:
        selected_by_name[specialist] = apply_specialist_gate(candidates, specialist)

    output_rows: dict[str, pd.DataFrame] = {
        "long_side": selected_by_name["long_side"].head(9).copy(),
        "short_control": selected_by_name["short_control"].head(9).copy(),
        "half_long_boundary": selected_by_name["half_long_boundary"].head(9).copy(),
    }
    mixed_source = pd.concat(
        [
            selected_by_name["long_side"],
            selected_by_name["short_control"],
            selected_by_name["half_long_boundary"],
            selected_by_name["terminal_removal"],
        ],
        ignore_index=True,
    )
    output_rows["mixed_specialists"] = dedupe_mixed_candidates(mixed_source).head(15).copy()

    ranked_rows: list[dict[str, Any]] = []
    submission_reports: dict[str, Any] = {}
    for name, selected in output_rows.items():
        report = write_candidate_outputs(
            outdir,
            anchor,
            name,
            selected,
            SUBMISSION_NAMES[name],
            SELECTED_NAMES[name],
        )
        ranked_rows.append(report)
        submission_reports[name] = report

    ranked = pd.DataFrame(ranked_rows)
    ranked.to_csv(safe_output_path(outdir, "ranked_candidates.csv"), index=False)
    candidates.to_csv(safe_output_path(outdir, "scored_specialist_candidates.csv"), index=False)

    report = {
        "script": "analysis_v402_rare_point_specialist_lab.py",
        "anchor": relative_path(ANCHOR_PATH),
        "anchor_rows": int(len(anchor)),
        "candidate_source": candidate_source_report,
        "train_backoff": train_report,
        "v401": v401_report,
        "raw_candidate_rows": int(len(raw_candidates)),
        "scored_candidate_rows": int(len(candidates)),
        "specialist_candidate_rows": {name: int(len(frame)) for name, frame in selected_by_name.items()},
        "submissions": submission_reports,
        "hard_blocks": {
            "point0_additions": "blocked",
            "action_changes": "not_generated",
            "server_changes": "not_generated",
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), json_safe(report))
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run_pipeline()
