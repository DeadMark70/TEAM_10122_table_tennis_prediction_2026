"""V354 independent row evidence generator.

This script builds row-level evidence reports from local candidate sources only.
It never writes submissions, upload candidates, or public-result labels.
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
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v354_independent_row_evidence"
TEST_FEATURES = ROOT / "test_new.csv"
TRAIN_FEATURES = ROOT / "train.csv"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_CANDIDATE = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V343_BANK = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
V347_TRANSITION_SUMMARY = ROOT / "v347_v338_v341_diff_audit" / "transition_summary.csv"

RISK_TAG_PARTS = ("risk", "v341", "v307", "saturated", "negative", "ttmatch", "old_server")

ROW_COLUMNS = [
    "row_id",
    "rally_uid",
    "old_point",
    "new_point",
    "transition",
    "phase",
    "lag0_action",
    "lag0_point",
    "old_depth",
    "new_depth",
    "same_depth",
    "depth_delta",
    "old_side",
    "new_side",
    "side_delta",
    "terminal_transition",
    "support_count_phase_action_old_new",
    "support_count_lag0_point_old_new",
    "transition_prior_new_given_phase_action_lag0",
    "source_dir_count",
    "source_count",
    "changed_in_v338",
    "source_public_tag_count",
    "source_public_tag_safe_count",
    "source_public_tag_risk_count",
    "source_dirs",
    "source_public_tags",
    "point0_addition",
    "no_p0_swap",
    "v341_extra_like_transition",
    "independent_evidence_score",
]


def relative_path(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return (point - 1) // 3


def point_side(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return (point - 1) % 3


def infer_phase(value: Any) -> str:
    try:
        strike = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if strike <= 1:
        return "serve"
    if strike <= 3:
        return "early"
    if strike <= 5:
        return "mid"
    return "late"


def add_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "phase" not in out.columns:
        if "strikeNumber" in out.columns:
            out["phase"] = out["strikeNumber"].map(infer_phase)
        else:
            out["phase"] = "unknown"
    if "lag0_action" not in out.columns:
        out["lag0_action"] = pd.to_numeric(out.get("actionId"), errors="coerce").fillna(-1).astype(int)
    if "lag0_point" not in out.columns:
        out["lag0_point"] = pd.to_numeric(out.get("pointId"), errors="coerce").fillna(-1).astype(int)
    return out


def _is_risk_tag(tag: Any) -> bool:
    text = str(tag).lower()
    return any(part in text for part in RISK_TAG_PARTS)


def _safe_join(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna().astype(str) if str(value)}))


def aggregate_candidate_bank(bank: pd.DataFrame) -> pd.DataFrame:
    if bank.empty:
        return pd.DataFrame(columns=["row_id", "rally_uid", "old_point", "new_point"])
    required = {"row_id", "rally_uid", "anchor_value", "candidate_value", "source", "source_dir"}
    missing = required - set(bank.columns)
    if missing:
        raise ValueError(f"candidate bank missing columns: {sorted(missing)}")

    work = bank.copy()
    work["row_id"] = pd.to_numeric(work["row_id"], errors="coerce")
    work["old_point"] = pd.to_numeric(work["anchor_value"], errors="coerce")
    work["new_point"] = pd.to_numeric(work["candidate_value"], errors="coerce")
    work = work.dropna(subset=["row_id", "old_point", "new_point"])
    work["row_id"] = work["row_id"].astype(int)
    work["old_point"] = work["old_point"].astype(int)
    work["new_point"] = work["new_point"].astype(int)
    if "source_public_tag" not in work.columns:
        work["source_public_tag"] = "unknown"
    if "changed_in_v338" not in work.columns:
        work["changed_in_v338"] = False
    work["is_risk_tag"] = work["source_public_tag"].map(_is_risk_tag)

    grouped = (
        work.groupby(["row_id", "rally_uid", "old_point", "new_point"], as_index=False)
        .agg(
            source_dir_count=("source_dir", "nunique"),
            source_count=("source", "nunique"),
            changed_in_v338=("changed_in_v338", "max"),
            source_public_tag_count=("source_public_tag", "nunique"),
            source_public_tag_safe_count=("is_risk_tag", lambda values: int((~values.astype(bool)).sum())),
            source_public_tag_risk_count=("is_risk_tag", lambda values: int(values.astype(bool).sum())),
            source_dirs=("source_dir", _safe_join),
            source_public_tags=("source_public_tag", _safe_join),
        )
        .sort_values(["row_id", "new_point"])
        .reset_index(drop=True)
    )
    return grouped


def build_train_support(train: pd.DataFrame | None) -> dict[str, Any]:
    if train is None or train.empty or "pointId" not in train.columns:
        return {"phase_action_old_new": {}, "lag0_old_new": {}, "prior": {}}
    work = add_context_columns(train)
    work["pointId"] = pd.to_numeric(work["pointId"], errors="coerce")
    work = work.dropna(subset=["pointId"])
    work["pointId"] = work["pointId"].astype(int)

    phase_action_old_new: dict[tuple[str, int, int, int], int] = {}
    lag0_old_new: dict[tuple[int, int, int], int] = {}
    prior: dict[tuple[str, int, int], dict[str, float]] = {}

    for row in work.itertuples(index=False):
        phase = str(getattr(row, "phase"))
        action = int(getattr(row, "lag0_action"))
        lag0_point = int(getattr(row, "lag0_point"))
        observed_point = int(getattr(row, "pointId"))
        phase_action_old_new[(phase, action, lag0_point, observed_point)] = (
            phase_action_old_new.get((phase, action, lag0_point, observed_point), 0) + 1
        )
        lag0_old_new[(lag0_point, lag0_point, observed_point)] = lag0_old_new.get(
            (lag0_point, lag0_point, observed_point), 0
        ) + 1

    for key_values, group in work.groupby(["phase", "lag0_action", "lag0_point"], dropna=False):
        counts = group["pointId"].value_counts()
        total = int(counts.sum())
        prior[(str(key_values[0]), int(key_values[1]), int(key_values[2]))] = {
            str(int(point)): float(count / total) for point, count in counts.items()
        }
    return {"phase_action_old_new": phase_action_old_new, "lag0_old_new": lag0_old_new, "prior": prior}


def _load_v341_extra_transitions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    summary = pd.read_csv(path)
    required = {"transition", "in_v341", "in_v338"}
    if not required.issubset(summary.columns):
        return set()
    in_v341 = pd.to_numeric(summary["in_v341"], errors="coerce").fillna(0)
    in_v338 = pd.to_numeric(summary["in_v338"], errors="coerce").fillna(0)
    return set(summary.loc[(in_v341 > 0) & (in_v338 <= 0), "transition"].astype(str))


def build_row_evidence(
    test_features: pd.DataFrame,
    candidate_bank: pd.DataFrame,
    *,
    train: pd.DataFrame | None,
    v341_extra_transitions: set[str],
) -> pd.DataFrame:
    features = add_context_columns(test_features)
    features = features.copy()
    features.insert(0, "row_id", np.arange(len(features), dtype=int))
    context_cols = ["row_id", "phase", "lag0_action", "lag0_point"]

    grouped = aggregate_candidate_bank(candidate_bank)
    if grouped.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    evidence = grouped.merge(features.loc[:, context_cols], on="row_id", how="left")

    support = build_train_support(train)
    records: list[dict[str, Any]] = []
    for row in evidence.itertuples(index=False):
        old_point = int(row.old_point)
        new_point = int(row.new_point)
        phase = str(row.phase) if pd.notna(row.phase) else "unknown"
        lag0_action = int(row.lag0_action) if pd.notna(row.lag0_action) else -1
        lag0_point = int(row.lag0_point) if pd.notna(row.lag0_point) else -1
        transition = f"{old_point}->{new_point}"
        old_depth = point_depth(old_point)
        new_depth = point_depth(new_point)
        old_side = point_side(old_point)
        new_side = point_side(new_point)
        prior_map = support["prior"].get((phase, lag0_action, lag0_point), {})

        source_support = min(4.0, float(row.source_dir_count)) + min(2.0, float(row.source_count) / 8.0)
        support_score = (
            min(2.0, float(support["phase_action_old_new"].get((phase, lag0_action, old_point, new_point), 0)) / 5.0)
            + min(1.0, float(support["lag0_old_new"].get((lag0_point, old_point, new_point), 0)) / 5.0)
            + float(prior_map.get(str(new_point), 0.0))
        )
        shape_score = (0.5 if old_depth == new_depth else 0.0) + (0.3 if old_point != 0 and new_point != 0 else 0.0)
        risk_penalty = (
            (1.2 if old_point != 0 and new_point == 0 else 0.0)
            + (0.8 if transition in v341_extra_transitions else 0.0)
            + min(1.0, float(row.source_public_tag_risk_count) / 6.0)
        )
        independent_evidence_score = (
            source_support
            + support_score
            + shape_score
            + (1.5 if bool(row.changed_in_v338) else 0.0)
            - risk_penalty
        )

        records.append(
            {
                "row_id": int(row.row_id),
                "rally_uid": row.rally_uid,
                "old_point": old_point,
                "new_point": new_point,
                "transition": transition,
                "phase": phase,
                "lag0_action": lag0_action,
                "lag0_point": lag0_point,
                "old_depth": old_depth,
                "new_depth": new_depth,
                "same_depth": bool(old_depth == new_depth),
                "depth_delta": int(new_depth - old_depth),
                "old_side": old_side,
                "new_side": new_side,
                "side_delta": int(new_side - old_side),
                "terminal_transition": bool(new_depth == 2),
                "support_count_phase_action_old_new": int(
                    support["phase_action_old_new"].get((phase, lag0_action, old_point, new_point), 0)
                ),
                "support_count_lag0_point_old_new": int(
                    support["lag0_old_new"].get((lag0_point, old_point, new_point), 0)
                ),
                "transition_prior_new_given_phase_action_lag0": float(prior_map.get(str(new_point), 0.0)),
                "source_dir_count": int(row.source_dir_count),
                "source_count": int(row.source_count),
                "changed_in_v338": bool(row.changed_in_v338),
                "source_public_tag_count": int(row.source_public_tag_count),
                "source_public_tag_safe_count": int(row.source_public_tag_safe_count),
                "source_public_tag_risk_count": int(row.source_public_tag_risk_count),
                "source_dirs": row.source_dirs,
                "source_public_tags": row.source_public_tags,
                "point0_addition": bool(old_point != 0 and new_point == 0),
                "no_p0_swap": bool(old_point != 0 and new_point != 0),
                "v341_extra_like_transition": bool(transition in v341_extra_transitions),
                "independent_evidence_score": float(independent_evidence_score),
            }
        )

    out = pd.DataFrame(records, columns=ROW_COLUMNS)
    for col in ["same_depth", "terminal_transition", "changed_in_v338", "point0_addition", "no_p0_swap", "v341_extra_like_transition"]:
        out[col] = out[col].astype(object)
    return out.sort_values(["row_id", "new_point"]).reset_index(drop=True)


def build_evidence_summary(row_evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transition",
        "rows",
        "unique_rows",
        "changed_in_v338_rows",
        "point0_additions",
        "no_p0_swap_rows",
        "v341_extra_like_rows",
        "mean_source_dir_count",
        "mean_support_phase_action_old_new",
        "mean_transition_prior",
    ]
    if row_evidence.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        row_evidence.groupby("transition", as_index=False)
        .agg(
            rows=("row_id", "count"),
            unique_rows=("row_id", "nunique"),
            changed_in_v338_rows=("changed_in_v338", "sum"),
            point0_additions=("point0_addition", "sum"),
            no_p0_swap_rows=("no_p0_swap", "sum"),
            v341_extra_like_rows=("v341_extra_like_transition", "sum"),
            mean_source_dir_count=("source_dir_count", "mean"),
            mean_support_phase_action_old_new=("support_count_phase_action_old_new", "mean"),
            mean_transition_prior=("transition_prior_new_given_phase_action_lag0", "mean"),
        )
        .sort_values(["rows", "transition"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary.loc[:, columns]


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "test_new": root / "test_new.csv",
        "v306_anchor": root / "v306_point0_addition_probe" / V306_ANCHOR.name,
        "v338_candidate": root / "v338_joint_moe_pack" / V338_CANDIDATE.name,
        "v343_candidate_bank": root / "v343_row_candidate_bank" / "candidate_bank.csv",
        "train": root / "train.csv",
        "v347_transition_summary": root / "v347_v338_v341_diff_audit" / "transition_summary.csv",
    }
    required = ["test_new", "v306_anchor", "v338_candidate", "v343_candidate_bank"]
    missing = [relative_path(paths[name], root) for name in required if not paths[name].exists()]
    if missing:
        report = {"version": "V354", "decision": "BLOCKED_MISSING_INPUT", "missing": missing}
        write_json(safe_output_path(outdir, "search_report.json"), report)
        return report

    test_features = pd.read_csv(paths["test_new"])
    v306 = load_submission(paths["v306_anchor"], expected_rows=expected_rows)
    v338 = load_submission(paths["v338_candidate"], expected_rows=expected_rows)
    if not v306["rally_uid"].equals(v338["rally_uid"]):
        raise ValueError("V306 and V338 rally_uid order differs")

    candidate_bank = pd.read_csv(paths["v343_candidate_bank"])
    if not candidate_bank.empty:
        old = v306["pointId"].astype(int).to_numpy()
        new = v338["pointId"].astype(int).to_numpy()

        def _actual_v338_candidate(row: pd.Series) -> bool:
            try:
                idx = int(row["row_id"])
                candidate_value = int(row["candidate_value"])
            except (KeyError, TypeError, ValueError):
                return False
            if idx < 0 or idx >= len(old):
                return False
            return bool(old[idx] != new[idx] and candidate_value == int(new[idx]))

        candidate_bank["changed_in_v338"] = candidate_bank.apply(_actual_v338_candidate, axis=1)

    train = pd.read_csv(paths["train"]) if paths["train"].exists() else None
    v341_extra = _load_v341_extra_transitions(paths["v347_transition_summary"])
    row_evidence = build_row_evidence(test_features, candidate_bank, train=train, v341_extra_transitions=v341_extra)
    evidence_summary = build_evidence_summary(row_evidence)

    row_path = safe_output_path(outdir, "row_evidence.csv")
    summary_path = safe_output_path(outdir, "evidence_summary.csv")
    row_evidence.to_csv(row_path, index=False)
    evidence_summary.to_csv(summary_path, index=False)

    report = {
        "version": "V354",
        "decision": "REPORTS_EXPORTED",
        "outputs": {
            "row_evidence": relative_path(row_path, root),
            "evidence_summary": relative_path(summary_path, root),
            "search_report": relative_path(outdir / "search_report.json", root),
        },
        "inputs": {
            name: relative_path(path, root) if path.exists() else None for name, path in paths.items()
        },
        "train_available": train is not None,
        "v347_transition_summary_available": paths["v347_transition_summary"].exists(),
        "candidate_bank_rows": int(len(candidate_bank)),
        "row_evidence_rows": int(len(row_evidence)),
        "unique_rows": int(row_evidence["row_id"].nunique()) if not row_evidence.empty else 0,
        "changed_in_v338_rows": int(row_evidence["changed_in_v338"].sum()) if not row_evidence.empty else 0,
        "point0_additions": int(row_evidence["point0_addition"].sum()) if not row_evidence.empty else 0,
        "no_p0_swap_rows": int(row_evidence["no_p0_swap"].sum()) if not row_evidence.empty else 0,
        "v341_extra_like_rows": int(row_evidence["v341_extra_like_transition"].sum()) if not row_evidence.empty else 0,
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "manual_row_edits": False,
            "upload_candidates_writes": False,
            "submission_exports": False,
            "reports_only": True,
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), _json_safe(report))
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
