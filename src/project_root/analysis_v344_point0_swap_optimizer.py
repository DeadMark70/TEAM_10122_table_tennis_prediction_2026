"""V344 fixed-budget point0 swap-set optimizer.

This script exports local-only point0 swap submissions. It prefers the V343
row candidate bank when available, and otherwise builds the minimal point0 pool
from known fixed-anchor source submissions.
"""

from __future__ import annotations

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
OUTDIR = ROOT / "v344_point0_swap_optimizer"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V343_BANK = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
BUDGETS = (8, 12, 18)
SOURCE_DIRS = [
    ROOT / "v306_point0_addition_probe",
    ROOT / "v307_point0_dose_extension",
    ROOT / "v311_point0_robust_terminal",
    ROOT / "v333_hierarchical_point_model",
    ROOT / "v334_joint_hierarchical_action_point",
    ROOT / "v337_point_moe",
    ROOT / "v338_joint_moe_pack",
    ROOT / "v339_no_p0_point_moe_expand",
    ROOT / "v340_no_p0_point_agreement_ensemble",
    ROOT / "v341_no_p0_point_pack",
    ROOT / "v322_nonterminal_point_modelbank",
    ROOT / "v329_point_distributional_selector",
    ROOT / "v272_action_conditioned_point_residual",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def transition_counts(base_point: pd.Series, cand_point: pd.Series) -> dict[str, int]:
    base = base_point.to_numpy(dtype=int)
    cand = cand_point.to_numpy(dtype=int)
    rows: dict[str, int] = {}
    for old, new in zip(base, cand):
        if old == new:
            continue
        key = f"{old}->{new}"
        rows[key] = rows.get(key, 0) + 1
    return dict(sorted(rows.items(), key=lambda item: (-item[1], item[0])))


def source_family(value: Any) -> str:
    text = str(value or "").lower()
    match = re.search(r"v\d+", text)
    return match.group(0).upper() if match else text[:32]


def source_bonus(row: pd.Series) -> float:
    text = " ".join(
        str(row.get(col, ""))
        for col in ("source_public_tag", "source", "source_dir")
    ).lower()
    if "v307" in text:
        return -0.2
    if "v306" in text or "v338" in text or "public_positive" in text or "positive" in text:
        return 1.0
    return 0.0


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def compute_utility(bank: pd.DataFrame) -> pd.DataFrame:
    out = bank.copy()
    out["source_family"] = out.apply(
        lambda row: source_family(row.get("source_dir") or row.get("source")), axis=1
    )
    base = _numeric_column(out, "source_local_delta_if_known", 0.0)
    out["source_bonus"] = out.apply(source_bonus, axis=1).astype(float)
    out["same_anchor_count"] = out.groupby("anchor_value")["row_id"].transform("count").astype(float)
    out["same_row_source_count"] = out.groupby(["row_id", "source_family"])["row_id"].transform("count").astype(float)
    out["utility"] = (
        base
        + out["source_bonus"]
        - (0.01 * (out["same_anchor_count"] - 1.0).clip(lower=0.0))
        - (0.02 * (out["same_row_source_count"] - 1.0).clip(lower=0.0))
    )
    return out


def filter_point0_pool(bank: pd.DataFrame) -> pd.DataFrame:
    out = bank.copy()
    if "task" in out.columns:
        out = out[out["task"].astype(str).str.lower().eq("point")]
    if "is_point0_addition" in out.columns:
        mask = out["is_point0_addition"].astype(str).str.lower().isin({"true", "1", "yes"})
        out = out[mask]
    out["anchor_value"] = pd.to_numeric(out["anchor_value"], errors="coerce")
    out["candidate_value"] = pd.to_numeric(out["candidate_value"], errors="coerce")
    out = out[(out["anchor_value"].between(1, 9)) & (out["candidate_value"] == 0)]
    out = out.dropna(subset=["row_id", "anchor_value", "candidate_value"]).copy()
    out["row_id"] = out["row_id"].astype(int)
    out["anchor_value"] = out["anchor_value"].astype(int)
    out["candidate_value"] = out["candidate_value"].astype(int)
    return out


def select_fixed_budget(bank: pd.DataFrame, budget: int) -> pd.DataFrame:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if bank.empty or budget == 0:
        return bank.head(0).copy()
    work = bank.copy()
    if "utility" not in work.columns:
        work["utility"] = 0.0
    work["utility"] = pd.to_numeric(work["utility"], errors="coerce").fillna(0.0)
    work = work.sort_values(["utility", "row_id"], ascending=[False, True])
    work = work.drop_duplicates("row_id", keep="first")
    return work.head(int(budget)).reset_index(drop=True)


def build_submission(base: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for _, row in selected.iterrows():
        row_id = int(row["row_id"])
        candidate_value = int(row["candidate_value"])
        if row_id < 0 or row_id >= len(out):
            raise IndexError(f"row_id out of bounds: {row_id}")
        old_value = int(out.at[row_id, "pointId"])
        if old_value == 0 or candidate_value != 0:
            raise ValueError(f"V344 only allows nonzero -> 0 changes, got {old_value}->{candidate_value}")
        out.at[row_id, "pointId"] = candidate_value
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["actionId"].equals(base["actionId"]):
        raise AssertionError("action changed")
    if not out["serverGetPoint"].equals(base["serverGetPoint"]):
        raise AssertionError("server changed")
    validate_submission_schema(out, expected_rows=len(base))
    return out


def _read_candidate_summary_delta(source_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("candidate_summary.csv", "joint_summary.csv", "summary.csv"):
        path = source_dir / name
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "candidate" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            delta = 0.0
            for key in ("point_oof_delta_vs_v306", "point_oof_delta", "expected_oof_delta", "source_local_delta_if_known"):
                if key in frame.columns and pd.notna(row.get(key)):
                    delta = float(row[key])
                    break
            out[str(row["candidate"])] = delta
    return out


def build_minimal_bank(
    base: pd.DataFrame,
    *,
    expected_rows: int | None = 1845,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    base_point = base["pointId"].to_numpy(dtype=int)
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            scan_rows.append({"source_dir": source_dir.name, "status": "missing"})
            continue
        deltas = _read_candidate_summary_delta(source_dir)
        for path in sorted(source_dir.glob("submission*.csv")):
            status = "ok"
            added = 0
            try:
                cand = load_submission(path, expected_rows=expected_rows)
                if not cand["rally_uid"].equals(base["rally_uid"]):
                    raise ValueError("row order differs from base")
                if not cand["actionId"].equals(base["actionId"]):
                    raise ValueError("action differs from fixed anchor")
                if not cand["serverGetPoint"].equals(base["serverGetPoint"]):
                    raise ValueError("server differs from fixed anchor")
                cand_point = cand["pointId"].to_numpy(dtype=int)
                changed = (base_point != 0) & (cand_point == 0)
                for row_id in np.flatnonzero(changed):
                    rows.append(
                        {
                            "row_id": int(row_id),
                            "rally_uid": str(base.at[int(row_id), "rally_uid"]),
                            "task": "point",
                            "anchor_value": int(base_point[row_id]),
                            "candidate_value": 0,
                            "source": path.stem,
                            "source_dir": source_dir.name,
                            "transition": f"{int(base_point[row_id])}->0",
                            "is_point0_addition": True,
                            "is_point0_removal": False,
                            "source_public_tag": source_family(source_dir.name),
                            "source_local_delta_if_known": deltas.get(path.stem, 0.0),
                        }
                    )
                added = int(changed.sum())
            except Exception as exc:
                status = f"skip:{type(exc).__name__}:{exc}"
            scan_rows.append(
                {
                    "source_dir": source_dir.name,
                    "path": relative_path(path),
                    "status": status,
                    "point0_candidate_rows": added,
                }
            )
    return pd.DataFrame(rows), scan_rows


def load_or_build_bank(
    base: pd.DataFrame,
    *,
    expected_rows: int | None = 1845,
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    if V343_BANK.exists():
        bank = pd.read_csv(V343_BANK)
        return bank, "v343_row_candidate_bank/candidate_bank.csv", []
    bank, scan_rows = build_minimal_bank(base, expected_rows=expected_rows)
    return bank, "minimal_source_submission_scan", scan_rows


def summarize_candidate(
    v306: pd.DataFrame,
    v338: pd.DataFrame,
    submission: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
    budget: int,
) -> dict[str, Any]:
    dist_v306 = point_distribution_report(v306["pointId"], submission["pointId"])
    dist_v338 = point_distribution_report(v338["pointId"], submission["pointId"])
    selected_rows = selected["row_id"].to_numpy(dtype=int) if not selected.empty else np.array([], dtype=int)
    v306_p0_rows = v306["pointId"].to_numpy(dtype=int) == 0
    v338_changed_rows = v306["pointId"].to_numpy(dtype=int) != v338["pointId"].to_numpy(dtype=int)
    return {
        "candidate": f"k{budget:02d}",
        "budget": int(budget),
        "selected_rows": int(len(selected)),
        "point_churn_vs_v306": int(dist_v306["changed_rows"]),
        "point_churn_vs_v338": int(dist_v338["changed_rows"]),
        "point0_additions_vs_v306": int(dist_v306["point0_additions"]),
        "point0_removals_vs_v306": int(dist_v306["point0_removals"]),
        "transition_counts_vs_v338": transition_counts(v338["pointId"], submission["pointId"]),
        "overlap_with_v306_point0_rows": int(np.isin(selected_rows, np.flatnonzero(v306_p0_rows)).sum()),
        "overlap_with_v338_no_p0_rows": int(np.isin(selected_rows, np.flatnonzero(v338_changed_rows)).sum()),
        "path": relative_path(path),
        "mean_selected_utility": float(selected["utility"].mean()) if "utility" in selected and len(selected) else 0.0,
    }


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not V306_ANCHOR.exists() or not V338_ANCHOR.exists():
        missing = [relative_path(p) for p in (V306_ANCHOR, V338_ANCHOR) if not p.exists()]
        report = {"version": "V344", "decision": "BLOCKED_MISSING_ANCHOR", "missing": missing}
        write_json(safe_output_path(outdir, "search_report.json"), report)
        return report

    v306 = load_submission(V306_ANCHOR, expected_rows=expected_rows)
    v338 = load_submission(V338_ANCHOR, expected_rows=expected_rows)
    if not v306["rally_uid"].equals(v338["rally_uid"]):
        raise ValueError("V306 and V338 anchors have different row order")
    if not v306["actionId"].equals(v338["actionId"]):
        raise ValueError("V338 action differs from fixed V306 action anchor")
    if not v306["serverGetPoint"].equals(v338["serverGetPoint"]):
        raise ValueError("V338 server differs from fixed V300 server anchor")

    raw_bank, bank_source, scan_rows = load_or_build_bank(v306, expected_rows=expected_rows)
    point0_pool = filter_point0_pool(raw_bank)
    point0_pool = compute_utility(point0_pool) if not point0_pool.empty else point0_pool
    if not point0_pool.empty:
        point0_pool = point0_pool.sort_values(["utility", "row_id"], ascending=[False, True])
    point0_pool.to_csv(safe_output_path(outdir, "point0_candidate_pool.csv"), index=False)
    pd.DataFrame(scan_rows).to_csv(safe_output_path(outdir, "source_scan_summary.csv"), index=False)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        selected = select_fixed_budget(point0_pool, budget=budget)
        selected_path = safe_output_path(outdir, f"selected_rows_k{budget:02d}.csv")
        selected.to_csv(selected_path, index=False)
        submission = build_submission(v338, selected)
        changed_vs_v338 = int((submission["pointId"].to_numpy(dtype=int) != v338["pointId"].to_numpy(dtype=int)).sum())
        if changed_vs_v338 > budget:
            raise AssertionError(f"budget exceeded for k{budget:02d}: {changed_vs_v338} > {budget}")
        filename = f"submission_v344_point0_swap_k{budget:02d}__v173action_v300server.csv"
        out_path = safe_output_path(outdir, filename)
        submission.to_csv(out_path, index=False, float_format="%.8f")
        row = summarize_candidate(v306, v338, submission, selected, out_path, budget)
        summary_rows.append(row)
        generated.append({"candidate": row["candidate"], "path": row["path"], "selected_rows": row["selected_rows"]})

    pd.DataFrame(summary_rows).to_csv(safe_output_path(outdir, "candidate_summary.csv"), index=False)
    report = {
        "version": "V344",
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "bank_source": bank_source,
        "candidate_pool_rows": int(len(point0_pool)),
        "unique_candidate_rows": int(point0_pool["row_id"].nunique()) if "row_id" in point0_pool else 0,
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "recommended_candidate": "k12" if len(point0_pool) >= 12 else (generated[-1]["candidate"] if generated else None),
        "summary": relative_path(outdir / "candidate_summary.csv"),
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_manual_row_edits": True,
            "upload_candidates_writes": False,
            "only_nonzero_to_zero_point_changes": True,
            "action_preserved": True,
            "server_preserved": True,
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(report["decision"])
    print(f"generated_submission_count={report.get('generated_submission_count', 0)}")
    print(f"recommended_candidate={report.get('recommended_candidate')}")


if __name__ == "__main__":
    main()
