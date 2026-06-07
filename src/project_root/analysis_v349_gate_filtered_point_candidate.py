"""V349 gate-filtered point candidate builder.

Builds conservative local point-only candidates from the V343 row bank. The
preferred input is V348 row gate scores; when those are absent, this module
uses an interpretable fallback gate from V343 support plus V338/V341 evidence.
Outputs stay under v349_gate_filtered_point_candidate and never write upload
candidate directories.
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
OUTDIR = ROOT / "v349_gate_filtered_point_candidate"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V343_BANK = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
V348_SCORES = ROOT / "v348_public_risk_row_gate" / "row_gate_scores.csv"


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


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return (point - 1) // 3


def same_depth(old: int, new: int) -> bool:
    return point_depth(int(old)) == point_depth(int(new))


def transition_counts(base_point: pd.Series, cand_point: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for old, new in zip(base_point.astype(int), cand_point.astype(int)):
        if old == new:
            continue
        key = f"{old}->{new}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def load_bank() -> pd.DataFrame:
    if not V343_BANK.exists():
        raise FileNotFoundError(f"missing V343 candidate bank: {relative_path(V343_BANK)}")
    return pd.read_csv(V343_BANK)


def _source_text(group: pd.DataFrame) -> str:
    pieces: list[str] = []
    for column in ("source_dir", "source", "source_public_tag"):
        if column in group.columns:
            pieces.extend(str(value).lower() for value in group[column].dropna().tolist())
    return " ".join(pieces)


def _family_count(group: pd.DataFrame) -> int:
    if "source_dir" not in group.columns:
        return 0
    return int(group["source_dir"].astype(str).nunique())


def build_fallback_gate_scores(bank: pd.DataFrame) -> pd.DataFrame:
    """Conservative interpretable row gate when V348 scores are unavailable."""
    work = bank.copy()
    if "task" in work.columns:
        work = work[work["task"].astype(str).str.lower().eq("point")].copy()
    for column in ("row_id", "anchor_value", "candidate_value"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["row_id", "anchor_value", "candidate_value"]).copy()
    work["row_id"] = work["row_id"].astype(int)
    work["anchor_value"] = work["anchor_value"].astype(int)
    work["candidate_value"] = work["candidate_value"].astype(int)
    work = work[work["anchor_value"].ne(work["candidate_value"])].copy()

    rows: list[dict[str, Any]] = []
    keys = ["row_id", "rally_uid", "anchor_value", "candidate_value"]
    present_keys = [key for key in keys if key in work.columns]
    for key_values, group in work.groupby(present_keys, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        record = dict(zip(present_keys, key_values))
        old = int(record["anchor_value"])
        new = int(record["candidate_value"])
        text = _source_text(group)
        changed_in_v338 = bool(_bool_series(group, "changed_in_v338").any())
        point0_addition = bool(old != 0 and new == 0)
        nonterminal_swap = bool(old != 0 and new != 0)
        same_depth_swap = bool(nonterminal_swap and same_depth(old, new))
        source_count = int(group["source"].astype(str).nunique()) if "source" in group.columns else int(len(group))
        family_count = _family_count(group)
        local_delta = (
            pd.to_numeric(group.get("source_local_delta_if_known", pd.Series(0.0, index=group.index)), errors="coerce")
            .fillna(0.0)
            .max()
        )
        v338_family_support = any(token in text for token in ("v333", "v334", "v337", "v338_public_positive"))
        no_p0_support = any(token in text for token in ("v339", "v340"))
        v341_extra_risk = ("v341" in text or "no_p0_expansion" in text) and not changed_in_v338
        v307_extra_p0_risk = ("v307" in text) and point0_addition and not changed_in_v338

        support_score = (
            (3.0 if changed_in_v338 else 0.0)
            + (0.35 * max(0, source_count - 1))
            + (0.45 * max(0, family_count - 1))
            + (0.55 if same_depth_swap else 0.0)
            + (0.8 if v338_family_support else 0.0)
            + (0.35 if no_p0_support else 0.0)
            + min(1.0, max(0.0, float(local_delta)) * 150.0)
        )
        risk_score = (
            (1.8 if v341_extra_risk else 0.0)
            + (1.6 if v307_extra_p0_risk else 0.0)
            + (0.85 if point0_addition and not changed_in_v338 else 0.0)
            + (0.45 if nonterminal_swap and not same_depth_swap and not changed_in_v338 else 0.0)
            + (0.35 if source_count <= 1 and not changed_in_v338 else 0.0)
        )
        trust_score = support_score - risk_score
        record.update(
            {
                "transition": f"{old}->{new}",
                "source_count": source_count,
                "agreement_count": source_count,
                "source_family_count": family_count,
                "changed_in_v338": changed_in_v338,
                "point0_addition": point0_addition,
                "nonterminal_swap": nonterminal_swap,
                "same_depth": same_depth_swap,
                "v338_family_support": bool(v338_family_support),
                "no_p0_support": bool(no_p0_support),
                "v341_extra_risk": bool(v341_extra_risk),
                "v307_extra_p0_risk": bool(v307_extra_p0_risk),
                "source_local_delta_if_known": float(local_delta),
                "support_score": float(support_score),
                "risk_score": float(risk_score),
                "trust_score": float(trust_score),
                "gate_source": "fallback_v343_v338_v341",
            }
        )
        rows.append(record)
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored
    return scored.sort_values(
        ["trust_score", "risk_score", "agreement_count", "row_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def load_gate_scores(bank: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if V348_SCORES.exists():
        scores = pd.read_csv(V348_SCORES)
        source = {"gate_source": "v348", "path": relative_path(V348_SCORES)}
    else:
        scores = build_fallback_gate_scores(bank)
        source = {
            "gate_source": "fallback_v343_v338_v341",
            "reason": "v348_public_risk_row_gate/row_gate_scores.csv not present",
        }
    for column in ("row_id", "anchor_value", "candidate_value"):
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce").astype(int)
    if "trust_score" not in scores.columns:
        scores["trust_score"] = 0.0
    if "risk_score" not in scores.columns:
        scores["risk_score"] = 0.0
    if "agreement_count" not in scores.columns:
        scores["agreement_count"] = scores.get("source_count", 1)
    if "point0_addition" not in scores.columns:
        scores["point0_addition"] = (scores["anchor_value"] != 0) & (scores["candidate_value"] == 0)
    if "nonterminal_swap" not in scores.columns:
        scores["nonterminal_swap"] = (scores["anchor_value"] != 0) & (scores["candidate_value"] != 0)
    if "changed_in_v338" not in scores.columns:
        scores["changed_in_v338"] = False
    return scores, source


def extra_nonp0_pool(scores: pd.DataFrame) -> pd.DataFrame:
    pool = scores.copy()
    pool["changed_in_v338"] = _bool_series(pool, "changed_in_v338")
    pool["nonterminal_swap"] = _bool_series(pool, "nonterminal_swap")
    pool["point0_addition"] = _bool_series(pool, "point0_addition")
    pool = pool[
        (~pool["changed_in_v338"])
        & pool["nonterminal_swap"]
        & (~pool["point0_addition"])
        & (pd.to_numeric(pool["trust_score"], errors="coerce").fillna(-999.0) >= 0.75)
        & (pd.to_numeric(pool["risk_score"], errors="coerce").fillna(999.0) <= 1.25)
    ].copy()
    return pool.sort_values(
        ["trust_score", "risk_score", "agreement_count", "row_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).drop_duplicates("row_id", keep="first")


def point0_pool(scores: pd.DataFrame) -> pd.DataFrame:
    pool = scores.copy()
    pool["changed_in_v338"] = _bool_series(pool, "changed_in_v338")
    pool["point0_addition"] = _bool_series(pool, "point0_addition")
    pool = pool[
        (~pool["changed_in_v338"])
        & pool["point0_addition"]
        & (pd.to_numeric(pool["trust_score"], errors="coerce").fillna(-999.0) >= -0.25)
        & (pd.to_numeric(pool["risk_score"], errors="coerce").fillna(999.0) <= 1.75)
    ].copy()
    return pool.sort_values(
        ["trust_score", "risk_score", "agreement_count", "row_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).drop_duplicates("row_id", keep="first")


def risky_v338_rows(scores: pd.DataFrame, max_rows: int = 6) -> pd.DataFrame:
    pool = scores.copy()
    pool["changed_in_v338"] = _bool_series(pool, "changed_in_v338")
    pool = pool[pool["changed_in_v338"]].copy()
    if pool.empty:
        return pool
    pool = pool.sort_values(
        ["risk_score", "trust_score", "agreement_count", "row_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    risky = pool[(pool["risk_score"] > 0.8) | (pool["trust_score"] < 2.5)].copy()
    return risky.drop_duplicates("row_id", keep="first").head(max_rows)


def build_submission_from_rows(
    start: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    target_column: str = "candidate_value",
) -> pd.DataFrame:
    out = start.copy()
    for _, row in selected.iterrows():
        row_id = int(row["row_id"])
        target = int(row[target_column])
        if row_id < 0 or row_id >= len(out):
            raise IndexError(f"row_id out of bounds: {row_id}")
        out.at[row_id, "pointId"] = target
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["actionId"].equals(start["actionId"]):
        raise AssertionError("action changed")
    if not out["serverGetPoint"].equals(start["serverGetPoint"]):
        raise AssertionError("server changed")
    validate_submission_schema(out, expected_rows=len(start))
    return out


def audit_candidate(
    base: pd.DataFrame,
    public: pd.DataFrame,
    candidate: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    expected_point0_additions: bool,
) -> dict[str, Any]:
    base_point = base["pointId"].to_numpy(dtype=int)
    public_point = public["pointId"].to_numpy(dtype=int)
    cand_point = candidate["pointId"].to_numpy(dtype=int)
    changed_vs_base = base_point != cand_point
    changed_v338 = base_point != public_point
    dist_v306 = point_distribution_report(base_point, cand_point)
    dist_v338 = point_distribution_report(public_point, cand_point)
    duplicate = bool(np.array_equal(public_point, cand_point))
    unexpected_point0 = bool(dist_v338["point0_additions"] > 0 and not expected_point0_additions)
    schema_ok = bool(
        candidate["actionId"].equals(public["actionId"]) and candidate["serverGetPoint"].equals(public["serverGetPoint"])
    )
    export_allowed = bool(schema_ok and not duplicate and not unexpected_point0)
    return {
        "selected_rows": int(len(selected)),
        "point_churn_vs_v306": dist_v306["changed_rows"],
        "point_churn_vs_v338": dist_v338["changed_rows"],
        "point0_additions_vs_v306": dist_v306["point0_additions"],
        "point0_additions_vs_v338": dist_v338["point0_additions"],
        "point0_removals_vs_v338": dist_v338["point0_removals"],
        "transition_counts_vs_v306": transition_counts(base["pointId"], candidate["pointId"]),
        "overlap_with_v338_changed_rows": int(np.sum(changed_vs_base & changed_v338)),
        "new_rows_beyond_v338": int(np.sum(changed_vs_base & ~changed_v338)),
        "duplicate_of_v338": duplicate,
        "unexpected_point0_additions": unexpected_point0,
        "action_preserved": bool(candidate["actionId"].equals(public["actionId"])),
        "server_preserved": bool(candidate["serverGetPoint"].equals(public["serverGetPoint"])),
        "export_allowed": export_allowed,
        "trust_score_min": float(selected["trust_score"].min()) if "trust_score" in selected and not selected.empty else 0.0,
        "trust_score_sum": float(selected["trust_score"].sum()) if "trust_score" in selected and not selected.empty else 0.0,
        "risk_score_max": float(selected["risk_score"].max()) if "risk_score" in selected and not selected.empty else 0.0,
    }


def _write_candidate(
    candidate_name: str,
    submission: pd.DataFrame,
    selected: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    selected_path = safe_output_path(OUTDIR, f"selected_{candidate_name.lower()}.csv")
    selected.to_csv(selected_path, index=False)
    row = {
        "candidate": candidate_name,
        "selected_path": relative_path(selected_path),
        **audit,
    }
    if audit["export_allowed"]:
        filename = f"submission_{candidate_name.lower()}__v173action_v300server.csv"
        out_path = safe_output_path(OUTDIR, filename)
        submission.to_csv(out_path, index=False)
        row["path"] = relative_path(out_path)
    else:
        row["path"] = ""
    return row


def choose_recommended(summary: pd.DataFrame) -> dict[str, Any] | None:
    if summary.empty:
        return None
    eligible = summary[summary["export_allowed"].astype(bool)].copy()
    if eligible.empty:
        return None
    eligible["point0_penalty"] = pd.to_numeric(eligible["point0_additions_vs_v338"], errors="coerce").fillna(0)
    eligible = eligible.sort_values(
        ["point0_penalty", "trust_score_sum", "new_rows_beyond_v338", "risk_score_max"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    return eligible.iloc[0].to_dict()


def run_pipeline(expected_rows: int | None = 1845) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = load_submission(V306_ANCHOR, expected_rows)
    public = load_submission(V338_ANCHOR, expected_rows)
    if not base["rally_uid"].equals(public["rally_uid"]):
        raise ValueError("V306 and V338 anchors have different row order")

    bank = load_bank()
    scores, gate_report = load_gate_scores(bank)
    scores_path = safe_output_path(OUTDIR, "row_gate_scores_used.csv")
    scores.to_csv(scores_path, index=False)

    nonp0 = extra_nonp0_pool(scores)
    p0 = point0_pool(scores)
    risky = risky_v338_rows(scores)

    summary_rows: list[dict[str, Any]] = []

    for name, limit in (("v349a_gate_nonp0_plus06", 6), ("v349b_gate_nonp0_plus12", 12)):
        selected = nonp0.head(limit).copy()
        submission = build_submission_from_rows(public, selected)
        audit = audit_candidate(base, public, submission, selected, expected_point0_additions=False)
        summary_rows.append(_write_candidate(name, submission, selected, audit))

    swap_out = risky.head(6).copy()
    swap_in = nonp0[~nonp0["row_id"].isin(set(swap_out["row_id"].tolist()))].head(len(swap_out)).copy()
    c_start = public.copy()
    if not swap_out.empty:
        removals = swap_out.copy()
        removals["target_value"] = removals["anchor_value"].astype(int)
        c_start = build_submission_from_rows(c_start, removals, target_column="target_value")
    c_selected = pd.concat([swap_out.assign(v349_role="swap_out"), swap_in.assign(v349_role="swap_in")], ignore_index=True)
    c_submission = build_submission_from_rows(c_start, swap_in) if not swap_in.empty else c_start
    c_audit = audit_candidate(base, public, c_submission, c_selected, expected_point0_additions=False)
    summary_rows.append(_write_candidate("v349c_gate_swap_risky", c_submission, c_selected, c_audit))

    d_selected = p0.head(12).copy()
    d_submission = build_submission_from_rows(public, d_selected)
    d_audit = audit_candidate(base, public, d_submission, d_selected, expected_point0_additions=True)
    summary_rows.append(_write_candidate("v349d_gate_point0_k12", d_submission, d_selected, d_audit))

    summary = pd.DataFrame(summary_rows)
    summary_path = safe_output_path(OUTDIR, "candidate_summary.csv")
    summary.to_csv(summary_path, index=False)
    recommended = choose_recommended(summary)
    generated = [
        {"candidate": row["candidate"], "path": row["path"], "selected_rows": int(row["selected_rows"])}
        for row in summary_rows
        if row.get("path")
    ]
    report = {
        "version": "V349",
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "recommended_candidate": recommended,
        "candidate_summary": relative_path(summary_path),
        "row_gate_scores_used": relative_path(scores_path),
        "gate_report": gate_report,
        "pool_sizes": {
            "extra_nonp0": int(len(nonp0)),
            "point0": int(len(p0)),
            "risky_v338": int(len(risky)),
        },
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "manual_row_edits": False,
            "no_upload_candidates_writes": True,
            "action_preserved_required": True,
            "server_preserved_required": True,
            "duplicate_v338_exports_blocked": True,
            "unexpected_point0_additions_blocked": True,
        },
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            _json_safe(
                {
                    "outdir": relative_path(OUTDIR),
                    "decision": report["decision"],
                    "generated_submission_count": report["generated_submission_count"],
                    "generated_submissions": report["generated_submissions"],
                    "recommended_candidate": report["recommended_candidate"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
