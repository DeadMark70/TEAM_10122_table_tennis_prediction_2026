"""V353 V338 row causal audit.

This report groups rows changed by the public-positive V338 point candidate and
exports local-only leave-group probes that revert selected V338 rows to V306.
It never adds rows beyond V338 and never changes action/server columns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v353_v338_row_causal_audit"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_PUBLIC_POSITIVE = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V347_ROW_DIFF = ROOT / "v347_v338_v341_diff_audit" / "row_diff.csv"
V347_TRANSITION_SUMMARY = ROOT / "v347_v338_v341_diff_audit" / "transition_summary.csv"
V348_ROW_GATE_SCORES = ROOT / "v348_public_risk_row_gate" / "row_gate_scores.csv"
V351_ROW_TRUST_SCORES = ROOT / "v351_v338_pruning_trust_model" / "v338_row_trust_scores.csv"


GROUP_COLUMNS = [
    "group_key",
    "group_type",
    "group_value",
    "rows_to_revert",
    "row_ids",
    "remaining_v338_rows",
    "mean_final_trust_score",
    "min_final_trust_score",
    "mean_gate_trust_score",
    "mean_gate_risk_score",
    "point0_rows",
    "v341_overlap_rows",
    "v341_extra_transition_rows",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False).astype(bool)


def _score_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "row_id" not in frame.columns:
        return pd.DataFrame(columns=["row_id"])
    out = frame.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce")
    out = out.dropna(subset=["row_id"])
    out["row_id"] = out["row_id"].astype(int)
    return out


def _trust_quantiles(scores: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    labels = ["q1_low", "q2_midlow", "q3_midhigh", "q4_high"]
    if numeric.notna().sum() == 0:
        return pd.Series(["q_unknown"] * len(scores), index=scores.index)
    ranked = numeric.fillna(numeric.min() - 1.0).rank(method="first")
    buckets = pd.qcut(ranked, q=min(4, len(ranked)), labels=labels[: min(4, len(ranked))])
    out = buckets.astype(str)
    if len(labels[: min(4, len(ranked))]) < 4:
        out = out.replace({"q1_low": "q1_low", "q2_midlow": "q2_midlow", "q3_midhigh": "q3_midhigh"})
    return out


def _merge_optional_evidence(
    changed: pd.DataFrame,
    trust_scores: pd.DataFrame | None,
    gate_scores: pd.DataFrame | None,
    row_diff: pd.DataFrame | None,
    transition_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    out = changed.copy()
    trust = _score_lookup(trust_scores if trust_scores is not None else pd.DataFrame())
    trust_keep = [
        col
        for col in [
            "row_id",
            "final_trust_score",
            "trust_score",
            "risk_score",
            "agreement_count",
            "gate_decision",
            "subset_support",
            "v341_transition_extra_count",
        ]
        if col in trust.columns
    ]
    if trust_keep:
        trust = trust.loc[:, trust_keep].rename(
            columns={
                "trust_score": "v351_gate_trust_score",
                "risk_score": "v351_gate_risk_score",
                "gate_decision": "v351_gate_decision",
            }
        )
        out = out.merge(trust, on="row_id", how="left")

    gate = _score_lookup(gate_scores if gate_scores is not None else pd.DataFrame())
    gate_keep = [
        col
        for col in [
            "row_id",
            "anchor_value",
            "candidate_value",
            "trust_score",
            "risk_score",
            "gate_decision",
            "agreement_count",
            "source_count",
        ]
        if col in gate.columns
    ]
    if gate_keep:
        gate = gate.loc[:, gate_keep].rename(
            columns={
                "anchor_value": "old_point",
                "candidate_value": "new_point",
                "trust_score": "v348_trust_score",
                "risk_score": "v348_risk_score",
                "gate_decision": "v348_gate_decision",
                "agreement_count": "v348_agreement_count",
            }
        )
        merge_keys = ["row_id"]
        if {"old_point", "new_point"}.issubset(gate.columns):
            gate = gate.drop_duplicates(subset=["row_id", "old_point", "new_point"], keep="first")
            merge_keys = ["row_id", "old_point", "new_point"]
        else:
            gate = gate.drop_duplicates(subset=["row_id"], keep="first")
        out = out.merge(gate, on=merge_keys, how="left")

    diff = _score_lookup(row_diff if row_diff is not None else pd.DataFrame())
    if not diff.empty:
        keep = [col for col in ["row_id", "in_v341", "source_count_from_candidate_bank", "v341_candidate_count"] if col in diff.columns]
        diff = diff.loc[:, keep].drop_duplicates(subset=["row_id"], keep="first")
        if "in_v341" in diff.columns:
            diff["in_v341"] = _coerce_bool(diff["in_v341"])
        out = out.merge(diff, on="row_id", how="left")

    transition_extra: dict[str, int] = {}
    if transition_summary is not None and not transition_summary.empty and "transition" in transition_summary.columns:
        summary = transition_summary.copy()
        if {"rows", "in_v338"}.issubset(summary.columns):
            transition_extra = {
                str(row["transition"]): max(0, int(row["rows"]) - int(row["in_v338"]))
                for _, row in summary.iterrows()
                if pd.notna(row.get("rows")) and pd.notna(row.get("in_v338"))
            }
        elif "in_v341" in summary.columns:
            transition_extra = {
                str(row["transition"]): int(row["in_v341"])
                for _, row in summary.iterrows()
                if pd.notna(row.get("in_v341"))
            }
    out["v341_extra_transition_rows"] = out["transition"].map(lambda key: int(transition_extra.get(str(key), 0)))

    for col in ("final_trust_score", "v351_gate_trust_score", "v351_gate_risk_score", "v348_trust_score", "v348_risk_score"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "in_v341" not in out.columns:
        out["in_v341"] = False
    out["in_v341"] = _coerce_bool(out["in_v341"])
    out["trust_quantile"] = _trust_quantiles(out["final_trust_score"].fillna(out["v348_trust_score"]))
    return out


def identify_v338_changed_rows(
    v306: pd.DataFrame,
    v338: pd.DataFrame,
    *,
    trust_scores: pd.DataFrame | None = None,
    gate_scores: pd.DataFrame | None = None,
    row_diff: pd.DataFrame | None = None,
    transition_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    validate_submission_schema(v306, expected_rows=len(v306))
    validate_submission_schema(v338, expected_rows=len(v306))
    if not v306["rally_uid"].equals(v338["rally_uid"]):
        raise ValueError("submission row order differs")
    old = v306["pointId"].to_numpy(dtype=int)
    new = v338["pointId"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for row_id in np.where(old != new)[0]:
        old_point = int(old[row_id])
        new_point = int(new[row_id])
        old_depth = point_depth(old_point)
        new_depth = point_depth(new_point)
        rows.append(
            {
                "row_id": int(row_id),
                "rally_uid": v306.at[int(row_id), "rally_uid"],
                "old_point": old_point,
                "new_point": new_point,
                "transition": f"{old_point}->{new_point}",
                "old_depth": old_depth,
                "new_depth": new_depth,
                "depth_change": int(new_depth - old_depth),
            }
        )
    changed = pd.DataFrame(rows)
    if changed.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "rally_uid",
                "old_point",
                "new_point",
                "transition",
                "old_depth",
                "new_depth",
                "depth_change",
                "trust_quantile",
            ]
        )
    return _merge_optional_evidence(changed, trust_scores, gate_scores, row_diff, transition_summary).sort_values("row_id").reset_index(drop=True)


def _row_id_list(series: pd.Series) -> list[int]:
    return sorted(pd.to_numeric(series, errors="coerce").dropna().astype(int).unique().tolist())


def _summarize_group(changed: pd.DataFrame, group_type: str, group_value: Any, group: pd.DataFrame) -> dict[str, Any]:
    row_ids = _row_id_list(group["row_id"])
    total_changed = int(len(changed))
    point0_rows = int(((group["old_point"].astype(int) != 0) & (group["new_point"].astype(int) == 0)).sum())
    return {
        "group_key": f"{group_type}:{group_value}",
        "group_type": group_type,
        "group_value": str(group_value),
        "rows_to_revert": int(len(row_ids)),
        "row_ids": " ".join(str(row) for row in row_ids),
        "remaining_v338_rows": int(total_changed - len(row_ids)),
        "mean_final_trust_score": float(pd.to_numeric(group.get("final_trust_score", np.nan), errors="coerce").mean()),
        "min_final_trust_score": float(pd.to_numeric(group.get("final_trust_score", np.nan), errors="coerce").min()),
        "mean_gate_trust_score": float(pd.to_numeric(group.get("v348_trust_score", np.nan), errors="coerce").mean()),
        "mean_gate_risk_score": float(pd.to_numeric(group.get("v348_risk_score", np.nan), errors="coerce").mean()),
        "point0_rows": point0_rows,
        "v341_overlap_rows": int(_coerce_bool(group.get("in_v341", pd.Series(False, index=group.index))).sum()),
        "v341_extra_transition_rows": int(pd.to_numeric(group.get("v341_extra_transition_rows", 0), errors="coerce").fillna(0).sum()),
    }


def build_group_audit(changed_rows: pd.DataFrame) -> pd.DataFrame:
    if changed_rows.empty:
        return pd.DataFrame(columns=GROUP_COLUMNS)
    specs = [
        ("transition", "transition"),
        ("old_point", "old_point"),
        ("new_point", "new_point"),
        ("depth_change", "depth_change"),
        ("trust_quantile", "trust_quantile"),
    ]
    records: list[dict[str, Any]] = []
    for group_type, column in specs:
        if column not in changed_rows.columns:
            continue
        for value, group in changed_rows.groupby(column, dropna=False, sort=True):
            records.append(_summarize_group(changed_rows, group_type, value, group))
    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(columns=GROUP_COLUMNS)
    return out.loc[:, GROUP_COLUMNS].sort_values(
        ["remaining_v338_rows", "mean_final_trust_score", "group_key"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_leave_group_submission(v306: pd.DataFrame, v338: pd.DataFrame, rows_to_revert: Iterable[int]) -> pd.DataFrame:
    validate_submission_schema(v306, expected_rows=len(v306))
    validate_submission_schema(v338, expected_rows=len(v306))
    if not v306["rally_uid"].equals(v338["rally_uid"]):
        raise ValueError("submission row order differs")
    changed = set(np.where(v306["pointId"].to_numpy(dtype=int) != v338["pointId"].to_numpy(dtype=int))[0].astype(int).tolist())
    output = v338.copy()
    for row_id in sorted(set(int(row) for row in rows_to_revert)):
        if row_id not in changed:
            raise ValueError(f"row_id is not a V338 changed row: {row_id}")
        output.loc[row_id, "pointId"] = int(v306.loc[row_id, "pointId"])
    output["actionId"] = v338["actionId"]
    output["serverGetPoint"] = v338["serverGetPoint"]
    return output.loc[:, SUBMISSION_COLUMNS].copy()


def _changed_row_ids(base: pd.DataFrame, candidate: pd.DataFrame) -> set[int]:
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs")
    return set(np.where(base["pointId"].to_numpy(dtype=int) != candidate["pointId"].to_numpy(dtype=int))[0].astype(int).tolist())


def candidate_metrics(v306: pd.DataFrame, v338: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    validate_submission_schema(candidate, expected_rows=len(v306))
    if not v306["rally_uid"].equals(candidate["rally_uid"]) or not v338["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs")
    v338_rows = _changed_row_ids(v306, v338)
    candidate_rows = _changed_row_ids(v306, candidate)
    changed_vs_v338 = _changed_row_ids(v338, candidate)
    new_rows = sorted(candidate_rows - v338_rows)
    base_point = v306["pointId"].to_numpy(dtype=int)
    cand_point = candidate["pointId"].to_numpy(dtype=int)
    point0_additions = [int(row) for row in candidate_rows if base_point[row] != 0 and cand_point[row] == 0]
    return {
        "remaining_v338_rows": int(len(candidate_rows & v338_rows)),
        "rows_reverted_from_v338": int(len(changed_vs_v338)),
        "point_churn_vs_v306": int(len(candidate_rows)),
        "point_churn_vs_v338": int(len(changed_vs_v338)),
        "new_rows_beyond_v338": int(len(new_rows)),
        "new_rows_beyond_v338_list": new_rows,
        "point0_additions_vs_v306": int(len(point0_additions)),
        "action_preserved": bool(candidate["actionId"].equals(v338["actionId"])),
        "server_preserved": bool(candidate["serverGetPoint"].equals(v338["serverGetPoint"])),
    }


def _parse_row_ids(raw: Any) -> list[int]:
    if pd.isna(raw):
        return []
    return [int(part) for part in str(raw).replace(",", " ").split() if part.strip()]


def _safe_candidates(groups: pd.DataFrame) -> pd.DataFrame:
    if groups.empty:
        return groups.copy()
    out = groups.copy()
    return out[out["remaining_v338_rows"].astype(int) >= 2].copy()


def build_candidate_summary(
    v306: pd.DataFrame,
    v338: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    top_n: int | None = None,
    outdir: Path | None = None,
    write_submissions: bool = False,
) -> pd.DataFrame:
    eligible = _safe_candidates(groups)
    if eligible.empty:
        return pd.DataFrame()
    eligible = eligible.sort_values(
        ["mean_final_trust_score", "v341_extra_transition_rows", "rows_to_revert", "group_key"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    if top_n is not None:
        eligible = eligible.head(int(top_n))

    records: list[dict[str, Any]] = []
    for _, group in eligible.iterrows():
        row_ids = _parse_row_ids(group["row_ids"])
        candidate = build_leave_group_submission(v306, v338, row_ids)
        metrics = candidate_metrics(v306, v338, candidate)
        name = f"leave_{str(group['group_key']).replace(':', '_').replace('->', '_to_').replace('-', 'neg')}"
        record = {
            "name": name,
            "group_key": group["group_key"],
            "group_type": group["group_type"],
            "group_value": group["group_value"],
            "path": "",
            "reverted_row_ids": " ".join(str(row) for row in row_ids),
            **metrics,
        }
        record.update(
            {
                "mean_final_trust_score": group.get("mean_final_trust_score"),
                "mean_gate_risk_score": group.get("mean_gate_risk_score"),
                "v341_extra_transition_rows": group.get("v341_extra_transition_rows"),
            }
        )
        if write_submissions:
            if outdir is None:
                raise ValueError("outdir is required when write_submissions=True")
            path = safe_output_path(Path(outdir), f"submission_v353_{name}__v173action_v300server.csv")
            validate_submission_schema(candidate, expected_rows=len(v306))
            candidate.to_csv(path, index=False)
            record["path"] = relative_path(path)
        records.append(record)
    return pd.DataFrame(records)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_pipeline(*, outdir: Path = OUTDIR, expected_rows: int | None = 1845) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = safe_output_path(outdir, "search_report.json")

    missing = [relative_path(path) for path in [V306_ANCHOR, V338_PUBLIC_POSITIVE] if not path.exists()]
    if missing:
        report = {"version": "V353", "decision": "BLOCKED_MISSING_INPUT", "missing": missing}
        write_json(report_path, report)
        return report

    v306 = load_submission(V306_ANCHOR, expected_rows=expected_rows)
    v338 = load_submission(V338_PUBLIC_POSITIVE, expected_rows=len(v306))
    changed = identify_v338_changed_rows(
        v306,
        v338,
        trust_scores=_read_optional_csv(V351_ROW_TRUST_SCORES),
        gate_scores=_read_optional_csv(V348_ROW_GATE_SCORES),
        row_diff=_read_optional_csv(V347_ROW_DIFF),
        transition_summary=_read_optional_csv(V347_TRANSITION_SUMMARY),
    )
    groups = build_group_audit(changed)
    all_candidates = build_candidate_summary(v306, v338, groups, top_n=None, outdir=outdir, write_submissions=False)
    top_candidates = build_candidate_summary(v306, v338, groups, top_n=3, outdir=outdir, write_submissions=True)
    if not all_candidates.empty and not top_candidates.empty:
        path_by_name = dict(zip(top_candidates["name"].astype(str), top_candidates["path"].astype(str)))
        all_candidates["path"] = all_candidates["name"].astype(str).map(path_by_name).fillna("")

    group_path = safe_output_path(outdir, "group_audit.csv")
    summary_path = safe_output_path(outdir, "candidate_summary.csv")
    groups.to_csv(group_path, index=False)
    all_candidates.to_csv(summary_path, index=False)

    report = {
        "version": "V353",
        "decision": "HAS_SAFE_LEAVE_GROUP_SUBMISSIONS" if not top_candidates.empty else "NO_SAFE_LEAVE_GROUP_SUBMISSIONS",
        "inputs": {
            "v306_anchor": relative_path(V306_ANCHOR),
            "v338_public_positive": relative_path(V338_PUBLIC_POSITIVE),
            "v347_row_diff": relative_path(V347_ROW_DIFF) if V347_ROW_DIFF.exists() else None,
            "v347_transition_summary": relative_path(V347_TRANSITION_SUMMARY) if V347_TRANSITION_SUMMARY.exists() else None,
            "v348_row_gate_scores": relative_path(V348_ROW_GATE_SCORES) if V348_ROW_GATE_SCORES.exists() else None,
            "v351_row_trust_scores": relative_path(V351_ROW_TRUST_SCORES) if V351_ROW_TRUST_SCORES.exists() else None,
        },
        "outputs": {
            "group_audit": relative_path(group_path),
            "candidate_summary": relative_path(summary_path),
            "search_report": relative_path(report_path),
            "top_submissions": top_candidates["path"].dropna().astype(str).tolist() if not top_candidates.empty else [],
        },
        "v338_changed_rows": int(len(changed)),
        "groups": int(len(groups)),
        "candidate_count": int(len(all_candidates)),
        "exported_submission_count": int(len(top_candidates)),
        "top_candidates": top_candidates.to_dict("records") if not top_candidates.empty else [],
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "upload_candidates_writes": False,
            "only_reverts_v338_rows": True,
            "no_new_rows_beyond_v338": bool(all_candidates["new_rows_beyond_v338"].max() == 0) if not all_candidates.empty else True,
            "no_point0_additions": bool(all_candidates["point0_additions_vs_v306"].max() == 0) if not all_candidates.empty else True,
            "action_server_preserved": bool(all_candidates["action_preserved"].all() and all_candidates["server_preserved"].all())
            if not all_candidates.empty
            else True,
        },
    }
    write_json(report_path, _json_safe(report))
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
