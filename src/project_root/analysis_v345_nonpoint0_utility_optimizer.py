"""V345 non-point0 row-level point utility optimizer.

Exports point-only nonzero-to-nonzero swaps from a row candidate bank. If the
V343 bank is not available yet, this script builds a minimal compatible pool
from known local source submissions.
"""

from __future__ import annotations

import json
import math
import re
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
OUTDIR = ROOT / "v345_nonpoint0_utility_optimizer"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V343_BANK = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
BUDGETS = (12, 18, 24, 36)
SOURCE_DIRS = (
    "v306_point0_addition_probe",
    "v307_point0_dose_extension",
    "v311_point0_robust_terminal",
    "v333_hierarchical_point_model",
    "v334_joint_hierarchical_action_point",
    "v337_point_moe",
    "v338_joint_moe_pack",
    "v339_no_p0_point_moe_expand",
    "v340_no_p0_point_agreement_ensemble",
    "v341_no_p0_point_pack",
    "v322_nonterminal_point_modelbank",
    "v329_point_distributional_selector",
    "v272_action_conditioned_point_residual",
)


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


def source_public_tag(source_dir: str, source: str) -> str:
    text = f"{source_dir}/{source}".lower()
    if "v338" in text:
        return "v338_public_positive"
    if "v339" in text or "v340" in text:
        return "v339_v340_support"
    if "v341" in text:
        return "v341_expansion_risk"
    if "v333" in text or "v334" in text or "v337" in text:
        return "v338_family_support"
    return "unknown"


def local_delta_from_name(source_dir: str, source: str) -> float:
    text = f"{source_dir}/{source}".lower()
    if "v338" in text:
        return 0.00320717625047956
    if "v339" in text:
        return 0.004812391705785318
    if "v340" in text:
        return 0.00320717625047956
    if "v322" in text:
        return 0.0002517238455426174
    return 0.0


def filter_nonpoint0_swaps(bank: pd.DataFrame) -> pd.DataFrame:
    out = bank.copy()
    if "task" in out.columns:
        out = out[out["task"].astype(str).eq("point")]
    anchor = pd.to_numeric(out["anchor_value"], errors="coerce")
    candidate = pd.to_numeric(out["candidate_value"], errors="coerce")
    mask = anchor.between(1, 9) & candidate.between(1, 9) & anchor.ne(candidate)
    return out.loc[mask].copy()


def select_budget(bank: pd.DataFrame, budget: int) -> pd.DataFrame:
    if budget <= 0 or bank.empty:
        return bank.iloc[0:0].copy()
    sort_cols = ["utility"]
    ascending = [False]
    if "changed_in_v338" in bank.columns:
        sort_cols.append("changed_in_v338")
        ascending.append(False)
    if "agreement_count" in bank.columns:
        sort_cols.append("agreement_count")
        ascending.append(False)
    if "row_id" in bank.columns:
        sort_cols.append("row_id")
        ascending.append(True)
    ordered = bank.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return ordered.head(int(budget)).copy()


def build_minimal_candidate_bank(expected_rows: int | None = 1845) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = load_submission(V306_ANCHOR, expected_rows)
    public = load_submission(V338_ANCHOR, expected_rows)
    if not base["rally_uid"].equals(public["rally_uid"]):
        raise ValueError("V306 and V338 anchors have different row order")

    base_point = base["pointId"].to_numpy(dtype=int)
    public_point = public["pointId"].to_numpy(dtype=int)
    changed_v338 = base_point != public_point
    rows: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source_dir_name in SOURCE_DIRS:
        source_dir = ROOT / source_dir_name
        if not source_dir.exists():
            skipped.append({"source_dir": source_dir_name, "reason": "missing_dir"})
            continue
        for path in sorted(source_dir.glob("submission*.csv")):
            try:
                frame = load_submission(path, expected_rows)
            except Exception as exc:
                skipped.append({"path": relative_path(path), "reason": f"bad_schema:{exc}"})
                continue
            if not frame["rally_uid"].equals(base["rally_uid"]):
                skipped.append({"path": relative_path(path), "reason": "row_order_mismatch"})
                continue
            if not frame["actionId"].equals(base["actionId"]):
                skipped.append({"path": relative_path(path), "reason": "action_changed"})
                continue
            if not frame["serverGetPoint"].equals(base["serverGetPoint"]):
                skipped.append({"path": relative_path(path), "reason": "server_changed"})
                continue
            cand_point = frame["pointId"].to_numpy(dtype=int)
            changed = np.flatnonzero(cand_point != base_point)
            scanned.append({"path": relative_path(path), "changed_rows": int(len(changed))})
            for row_id in changed:
                old = int(base_point[row_id])
                new = int(cand_point[row_id])
                rows.append(
                    {
                        "row_id": int(row_id),
                        "rally_uid": base.at[int(row_id), "rally_uid"],
                        "task": "point",
                        "anchor_value": old,
                        "candidate_value": new,
                        "source": path.stem,
                        "source_dir": source_dir.name,
                        "transition": f"{old}->{new}",
                        "is_point0_addition": bool(old != 0 and new == 0),
                        "is_point0_removal": bool(old == 0 and new != 0),
                        "is_nonterminal_point_swap": bool(old != 0 and new != 0 and old != new),
                        "is_same_depth_swap": bool(old != 0 and new != 0 and same_depth(old, new)),
                        "changed_in_v338": bool(changed_v338[row_id]),
                        "v338_candidate_value": int(public_point[row_id]),
                        "source_public_tag": source_public_tag(source_dir.name, path.stem),
                        "source_local_delta_if_known": local_delta_from_name(source_dir.name, path.stem),
                    }
                )
    report = {
        "bank_source": "minimal_fallback",
        "candidate_bank_present": False,
        "scanned_submissions": scanned,
        "skipped": skipped,
        "raw_edit_rows": len(rows),
    }
    return pd.DataFrame(rows), report


def load_or_build_bank(expected_rows: int | None = 1845) -> tuple[pd.DataFrame, dict[str, Any]]:
    if V343_BANK.exists():
        bank = pd.read_csv(V343_BANK)
        return bank, {"bank_source": "v343", "candidate_bank": relative_path(V343_BANK), "candidate_bank_present": True}
    return build_minimal_candidate_bank(expected_rows)


def score_candidates(bank: pd.DataFrame) -> pd.DataFrame:
    filtered = filter_nonpoint0_swaps(bank)
    if filtered.empty:
        return filtered.assign(utility=pd.Series(dtype=float), agreement_count=pd.Series(dtype=int))

    for column, default in {
        "source_dir": "",
        "source": "",
        "transition": "",
        "is_same_depth_swap": False,
        "changed_in_v338": False,
        "source_public_tag": "unknown",
        "source_local_delta_if_known": 0.0,
    }.items():
        if column not in filtered.columns:
            filtered[column] = default

    filtered["row_id"] = pd.to_numeric(filtered["row_id"], errors="coerce").astype(int)
    filtered["anchor_value"] = pd.to_numeric(filtered["anchor_value"], errors="coerce").astype(int)
    filtered["candidate_value"] = pd.to_numeric(filtered["candidate_value"], errors="coerce").astype(int)
    filtered["is_same_depth_swap"] = filtered["is_same_depth_swap"].astype(bool)
    filtered["changed_in_v338"] = filtered["changed_in_v338"].astype(bool)
    filtered["source_local_delta_if_known"] = pd.to_numeric(
        filtered["source_local_delta_if_known"], errors="coerce"
    ).fillna(0.0)

    grouped_rows: list[dict[str, Any]] = []
    keys = ["row_id", "rally_uid", "anchor_value", "candidate_value"]
    present_keys = [key for key in keys if key in filtered.columns]
    for key_values, group in filtered.groupby(present_keys, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        record = dict(zip(present_keys, key_values))
        source_dirs = sorted({str(v) for v in group["source_dir"].dropna()})
        sources = sorted({str(v) for v in group["source"].dropna()})
        tags = sorted({str(v) for v in group["source_public_tag"].dropna()})
        agreement = int(group["source"].astype(str).nunique())
        changed_in_v338 = bool(group["changed_in_v338"].any())
        same_depth_flag = bool(group["is_same_depth_swap"].any()) or same_depth(
            int(record["anchor_value"]), int(record["candidate_value"])
        )
        v339_v340_support = any(("v339" in value or "v340" in value) for value in source_dirs + sources + tags)
        v341_expansion_risk = any("v341" in value for value in source_dirs + sources + tags) and not changed_in_v338
        local_delta = float(group["source_local_delta_if_known"].max())
        utility = 0.0
        utility += local_delta * 100.0
        utility += 2.0 if changed_in_v338 else 0.0
        utility += 1.2 if v339_v340_support else 0.0
        utility += 0.35 if same_depth_flag else 0.0
        utility += 0.35 * max(0, agreement - 1)
        utility -= 0.35 if not changed_in_v338 else 0.0
        utility -= 0.8 if v341_expansion_risk else 0.0
        record.update(
            {
                "transition": f"{int(record['anchor_value'])}->{int(record['candidate_value'])}",
                "source_dirs": "|".join(source_dirs),
                "sources": "|".join(sources),
                "source_public_tags": "|".join(tags),
                "agreement_count": agreement,
                "changed_in_v338": changed_in_v338,
                "v339_v340_support": bool(v339_v340_support),
                "is_same_depth_swap": bool(same_depth_flag),
                "v341_expansion_risk": bool(v341_expansion_risk),
                "source_local_delta_if_known": local_delta,
                "utility": float(utility),
            }
        )
        grouped_rows.append(record)
    scored = pd.DataFrame(grouped_rows)
    return scored.sort_values(["utility", "agreement_count", "row_id"], ascending=[False, False, True]).reset_index(drop=True)


def select_for_budget(scored: pd.DataFrame, budget: int) -> pd.DataFrame:
    pool = scored.copy()
    if budget > 24 and not pool.empty:
        pool = pool[(pool["changed_in_v338"].astype(bool)) | (pool["agreement_count"].astype(int) >= 2)].copy()
    return select_budget(pool, budget)


def build_submission(base: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for _, row in selected.iterrows():
        row_id = int(row["row_id"])
        new_value = int(row["candidate_value"])
        old_value = int(out.at[row_id, "pointId"])
        if old_value == 0 or new_value == 0 or old_value == new_value:
            raise ValueError(f"invalid non-point0 swap at row {row_id}: {old_value}->{new_value}")
        out.at[row_id, "pointId"] = new_value
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["actionId"].equals(base["actionId"]):
        raise AssertionError("action changed")
    if not out["serverGetPoint"].equals(base["serverGetPoint"]):
        raise AssertionError("server changed")
    validate_submission_schema(out, expected_rows=len(base))
    return out


def audit_submission(base: pd.DataFrame, public: pd.DataFrame, cand: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    base_point = base["pointId"].to_numpy(dtype=int)
    public_point = public["pointId"].to_numpy(dtype=int)
    cand_point = cand["pointId"].to_numpy(dtype=int)
    changed_vs_base = base_point != cand_point
    changed_v338 = base_point != public_point
    dist_v306 = point_distribution_report(base_point, cand_point)
    dist_v338 = point_distribution_report(public_point, cand_point)
    return {
        "point_churn_vs_v306": dist_v306["changed_rows"],
        "point_churn_vs_v338": dist_v338["changed_rows"],
        "point0_additions": dist_v306["point0_additions"],
        "point0_removals": dist_v306["point0_removals"],
        "transition_counts": transition_counts(base["pointId"], cand["pointId"]),
        "overlap_with_v338_changed_rows": int(np.sum(changed_vs_base & changed_v338)),
        "new_rows_beyond_v338": int(np.sum(changed_vs_base & ~changed_v338)),
        "agreement_count_min": int(selected["agreement_count"].min()) if not selected.empty else 0,
        "agreement_count_mean": float(selected["agreement_count"].mean()) if not selected.empty else 0.0,
        "utility_sum": float(selected["utility"].sum()) if not selected.empty else 0.0,
    }


def run_pipeline(expected_rows: int | None = 1845) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = load_submission(V306_ANCHOR, expected_rows)
    public = load_submission(V338_ANCHOR, expected_rows)
    if not base["rally_uid"].equals(public["rally_uid"]):
        raise ValueError("V306 and V338 anchors have different row order")

    bank, bank_report = load_or_build_bank(expected_rows)
    scored = score_candidates(bank)
    scored_path = safe_output_path(OUTDIR, "scored_candidates.csv")
    scored.to_csv(scored_path, index=False)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        selected = select_for_budget(scored, budget)
        filename = f"submission_v345_nonp0_util_b{budget:02d}__v173action_v300server.csv"
        out_path = safe_output_path(OUTDIR, filename)
        submission = build_submission(base, selected)
        submission.to_csv(out_path, index=False)
        selected_path = safe_output_path(OUTDIR, f"selected_b{budget:02d}.csv")
        selected.to_csv(selected_path, index=False)
        audit = audit_submission(base, public, submission, selected)
        row = {
            "candidate": f"v345_nonp0_util_b{budget:02d}",
            "budget": budget,
            "selected_rows": int(len(selected)),
            "path": relative_path(out_path),
            "selected_path": relative_path(selected_path),
            **audit,
        }
        summary_rows.append(row)
        generated.append({"candidate": row["candidate"], "path": row["path"], "selected_rows": row["selected_rows"]})

    summary_path = safe_output_path(OUTDIR, "candidate_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    recommended = None
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        eligible = summary[(summary["budget"] <= 24) & (summary["point0_additions"] == 0)].copy()
        if eligible.empty:
            eligible = summary.copy()
        eligible["overlap_rate"] = eligible["overlap_with_v338_changed_rows"] / eligible["selected_rows"].clip(lower=1)
        eligible = eligible.sort_values(["overlap_rate", "utility_sum", "budget"], ascending=[False, False, True])
        recommended = eligible.iloc[0].to_dict()

    report = {
        "version": "V345",
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "recommended_candidate": recommended,
        "candidate_summary": relative_path(summary_path),
        "scored_candidates": relative_path(scored_path),
        "bank_report": bank_report,
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "manual_row_edits": False,
            "no_upload_candidates_writes": True,
            "only_nonpoint0_point_swaps": True,
            "action_preserved": True,
            "server_preserved": True,
            "budget_gt_24_new_rows_require_multisource": True,
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
                    "recommended_candidate": report["recommended_candidate"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
