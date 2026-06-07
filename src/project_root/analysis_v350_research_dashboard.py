"""V350 compact research dashboard for V338 gate follow-up work.

The dashboard is intentionally read-only with respect to candidate sources. It
collects whatever V344-V349 reports are present, ranks available local
candidates, and records missing upstream reports so the same command can be
rerun after other workers finish.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    read_json,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v350_research_dashboard"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
SOURCE_DIRS = {
    "V344": ROOT / "v344_point0_swap_optimizer",
    "V345": ROOT / "v345_nonpoint0_utility_optimizer",
    "V346": ROOT / "v346_row_utility_pack",
    "V347": ROOT / "v347_v338_v341_diff_audit",
    "V348": ROOT / "v348_public_risk_row_gate",
    "V349": ROOT / "v349_gate_filtered_point_candidate",
}


def relative_path(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
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


def resolve_path(root: Path, source_dir: Path, value: Any) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [root / raw, source_dir / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_report(source_dir: Path) -> dict[str, Any] | None:
    path = source_dir / "search_report.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def load_optional_csv(root: Path, source_dir: Path, raw_path: Any) -> pd.DataFrame:
    path = resolve_path(root, source_dir, raw_path)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_present(row: pd.Series, names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def report_summary_path(report: dict[str, Any], version: str) -> Any:
    for key in ("candidate_priority", "candidate_summary", "summary", "joint_summary"):
        if report.get(key):
            return report[key]
    if version == "V347":
        return "v347_v338_v341_diff_audit/row_diff.csv"
    if version == "V348":
        return "v348_public_risk_row_gate/row_gate_scores.csv"
    return None


def collect_report_status(root: Path = ROOT) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    status: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for version, source_dir in SOURCE_DIRS.items():
        source_dir = root / source_dir.name
        report = load_report(source_dir)
        if report is None:
            status[version] = {"status": "missing", "source_dir": source_dir.name}
            diagnostics.append({"version": version, "status": "missing", "source_dir": source_dir.name})
            continue
        if "read_error" in report:
            status[version] = {"status": "unreadable", "source_dir": source_dir.name, "error": report["read_error"]}
            diagnostics.append(status[version] | {"version": version})
            continue
        decision = str(report.get("decision", "UNKNOWN"))
        summary_path = report_summary_path(report, version)
        summary = load_optional_csv(root, source_dir, summary_path)
        status[version] = {
            "status": "present",
            "source_dir": source_dir.name,
            "decision": decision,
            "summary_path": summary_path,
            "summary_rows": int(len(summary)),
        }
        diagnostics.append({"version": version, **status[version]})
    return status, diagnostics


def candidate_rows_from_summary(
    root: Path,
    version: str,
    source_dir: Path,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = load_optional_csv(root, source_dir, report_summary_path(report, version))
    rows: list[dict[str, Any]] = []
    if not summary.empty and "candidate" in summary.columns:
        for _, row in summary.iterrows():
            candidate = str(row["candidate"])
            rows.append(
                {
                    "source_version": version,
                    "source_dir": source_dir.name,
                    "candidate": candidate,
                    "path": _first_present(row, ["path", "submission"], None),
                    "selected_rows": _first_present(row, ["selected_rows", "budget"], np.nan),
                    "expected_utility": _first_present(
                        row,
                        ["expected_utility", "utility_sum", "mean_selected_utility", "point_oof_delta_vs_v306"],
                        np.nan,
                    ),
                    "reported_point_churn_vs_v306": _first_present(row, ["point_churn_vs_v306"], np.nan),
                    "reported_point_churn_vs_v338": _first_present(row, ["point_churn_vs_v338"], np.nan),
                    "reported_point0_additions": _first_present(
                        row, ["point0_additions", "point0_additions_vs_v306"], np.nan
                    ),
                    "reported_new_rows_beyond_v338": _first_present(row, ["new_rows_beyond_v338"], np.nan),
                    "summary_selected_path": _first_present(row, ["selected_path"], None),
                }
            )
    seen = {(row["candidate"], str(row.get("path"))) for row in rows}
    for item in report.get("generated_submissions") or []:
        candidate = str(item.get("candidate") or Path(str(item.get("path", "candidate"))).stem)
        key = (candidate, str(item.get("path")))
        if key in seen:
            continue
        rows.append(
            {
                "source_version": version,
                "source_dir": source_dir.name,
                "candidate": candidate,
                "path": item.get("path") or item.get("submission"),
                "selected_rows": item.get("selected_rows", np.nan),
                "expected_utility": item.get("expected_utility", np.nan),
                "reported_point_churn_vs_v306": np.nan,
                "reported_point_churn_vs_v338": np.nan,
                "reported_point0_additions": np.nan,
                "reported_new_rows_beyond_v338": np.nan,
                "summary_selected_path": None,
            }
        )
    return rows


def collect_candidates(root: Path = ROOT) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for version in ("V344", "V345", "V346", "V349"):
        source_dir = root / SOURCE_DIRS[version].name
        report = load_report(source_dir)
        if not report or "read_error" in report:
            continue
        out.extend(candidate_rows_from_summary(root, version, source_dir, report))
    return out


def changed_rows(base: pd.Series, candidate: pd.Series) -> set[int]:
    base_arr = base.to_numpy(dtype=int)
    cand_arr = candidate.to_numpy(dtype=int)
    return {int(i) for i in np.flatnonzero(base_arr != cand_arr)}


def load_v341_extra_rows(root: Path = ROOT) -> set[int] | None:
    path = root / "v347_v338_v341_diff_audit" / "row_diff.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if not {"row_id", "in_v338", "in_v341"}.issubset(frame.columns):
        return None
    in_v338 = frame["in_v338"].astype(str).str.lower().isin({"true", "1", "yes"})
    in_v341 = frame["in_v341"].astype(str).str.lower().isin({"true", "1", "yes"})
    return set(pd.to_numeric(frame.loc[in_v341 & ~in_v338, "row_id"], errors="coerce").dropna().astype(int))


def score_candidate(
    row: dict[str, Any],
    root: Path,
    v306: pd.DataFrame | None,
    v338: pd.DataFrame | None,
    v341_extra_rows: set[int] | None,
    expected_rows: int | None,
) -> dict[str, Any]:
    out = dict(row)
    source_dir = root / str(row["source_dir"])
    path = resolve_path(root, source_dir, row.get("path"))
    out["resolved_path"] = relative_path(path, root) if path else ""
    out["candidate_available"] = bool(path)
    out["action_preserved"] = True
    out["server_preserved"] = True
    out["point_churn_vs_v306"] = row.get("reported_point_churn_vs_v306", np.nan)
    out["point_churn_vs_v338"] = row.get("reported_point_churn_vs_v338", np.nan)
    out["point0_additions"] = row.get("reported_point0_additions", np.nan)
    out["new_rows_beyond_v338"] = row.get("reported_new_rows_beyond_v338", np.nan)
    out["v338_changed_overlap"] = np.nan
    out["v338_similarity"] = np.nan
    out["v341_extra_overlap"] = np.nan
    out["v341_extra_overlap_rate"] = np.nan

    candidate_changed_vs_v306: set[int] = set()
    if path and v306 is not None:
        try:
            cand = load_submission(path, expected_rows=expected_rows)
            if not cand["rally_uid"].equals(v306["rally_uid"]):
                raise ValueError("row order differs from V306")
            out["action_preserved"] = bool(cand["actionId"].equals(v306["actionId"]))
            out["server_preserved"] = bool(cand["serverGetPoint"].equals(v306["serverGetPoint"]))
            candidate_changed_vs_v306 = changed_rows(v306["pointId"], cand["pointId"])
            out["point_churn_vs_v306"] = len(candidate_changed_vs_v306)
            out["point0_additions"] = int(
                np.sum((v306["pointId"].to_numpy(dtype=int) != 0) & (cand["pointId"].to_numpy(dtype=int) == 0))
            )
            if v338 is not None and cand["rally_uid"].equals(v338["rally_uid"]):
                changed_v338 = changed_rows(v306["pointId"], v338["pointId"])
                changed_vs_v338 = changed_rows(v338["pointId"], cand["pointId"])
                overlap = len(candidate_changed_vs_v306 & changed_v338)
                out["point_churn_vs_v338"] = len(changed_vs_v338)
                out["new_rows_beyond_v338"] = len(candidate_changed_vs_v306 - changed_v338)
                out["v338_changed_overlap"] = overlap
                out["v338_similarity"] = overlap / len(candidate_changed_vs_v306) if candidate_changed_vs_v306 else 0.0
        except Exception as exc:
            out["candidate_available"] = False
            out["read_error"] = f"{type(exc).__name__}: {exc}"

    if v341_extra_rows is not None:
        overlap = len(candidate_changed_vs_v306 & v341_extra_rows)
        out["v341_extra_overlap"] = overlap
        out["v341_extra_overlap_rate"] = overlap / len(candidate_changed_vs_v306) if candidate_changed_vs_v306 else 0.0

    similarity = float(out["v338_similarity"]) if pd.notna(out["v338_similarity"]) else 0.0
    novelty = float(out["point_churn_vs_v338"]) if pd.notna(out["point_churn_vs_v338"]) else 0.0
    new_rows = float(out["new_rows_beyond_v338"]) if pd.notna(out["new_rows_beyond_v338"]) else 0.0
    point0 = float(out["point0_additions"]) if pd.notna(out["point0_additions"]) else 0.0
    v341_overlap = float(out["v341_extra_overlap"]) if pd.notna(out["v341_extra_overlap"]) else 0.0
    policy_penalty = 0.0 if out["action_preserved"] and out["server_preserved"] else 200.0
    point0_penalty = 25.0 if point0 > 0 else 0.0
    v341_penalty = 8.0 * v341_overlap
    novelty_bonus = min(novelty, 12.0) * 2.0
    if novelty == 0:
        novelty_bonus -= 15.0
    if new_rows > 12:
        novelty_bonus -= (new_rows - 12.0) * 2.0
    out["priority_score"] = round((100.0 * similarity) + novelty_bonus - point0_penalty - v341_penalty - policy_penalty, 6)
    out["recommendation_tier"] = recommendation_tier(out)
    return out


def recommendation_tier(row: dict[str, Any]) -> str:
    if not row.get("candidate_available", False):
        return "hold_missing_file"
    if not row.get("action_preserved", False) or not row.get("server_preserved", False):
        return "reject_policy"
    if float(row.get("point0_additions") or 0) > 0:
        return "hold_point0_addition"
    if float(row.get("v341_extra_overlap") or 0) > 0:
        return "hold_v341_overlap"
    novelty = float(row.get("point_churn_vs_v338") or 0)
    if novelty == 0:
        return "baseline_reference"
    if novelty <= 12:
        return "top_next_upload_priority"
    return "review_larger_novelty"


def rank_candidates(root: Path = ROOT, expected_rows: int | None = 1845) -> pd.DataFrame:
    candidates = collect_candidates(root)
    columns = [
        "rank",
        "recommendation_tier",
        "priority_score",
        "source_version",
        "source_dir",
        "candidate",
        "resolved_path",
        "candidate_available",
        "action_preserved",
        "server_preserved",
        "point_churn_vs_v306",
        "point_churn_vs_v338",
        "new_rows_beyond_v338",
        "point0_additions",
        "v338_changed_overlap",
        "v338_similarity",
        "v341_extra_overlap",
        "v341_extra_overlap_rate",
        "expected_utility",
        "selected_rows",
    ]
    if not candidates:
        return pd.DataFrame(columns=columns)
    v306 = load_submission(root / relative_path(V306_ANCHOR), expected_rows) if (root / relative_path(V306_ANCHOR)).exists() else None
    v338 = load_submission(root / relative_path(V338_ANCHOR), expected_rows) if (root / relative_path(V338_ANCHOR)).exists() else None
    v341_extra = load_v341_extra_rows(root)
    rows = [score_candidate(row, root, v306, v338, v341_extra, expected_rows) for row in candidates]
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["priority_score", "point_churn_vs_v338", "source_version", "candidate"],
        ascending=[False, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, columns]


def write_recommendation(path: Path, priority: pd.DataFrame, status_rows: list[dict[str, Any]]) -> None:
    missing = [row["version"] for row in status_rows if row.get("status") != "present"]
    present = [row["version"] for row in status_rows if row.get("status") == "present"]
    lines = [
        "# V350 research dashboard recommendation",
        "",
        "## Status",
        "",
    ]
    if missing:
        lines.append(f"- Partial dashboard: missing or unreadable upstream reports: {', '.join(missing)}.")
        lines.append("- Rerun `python analysis_v350_research_dashboard.py` after those workers finish.")
    else:
        lines.append("- All V344-V349 report directories were readable.")
    if present:
        lines.append(f"- Readable upstream reports: {', '.join(present)}.")
    lines.extend(
        [
            "- Policy: no TTMATCH, no old-server, no upload_candidates writes.",
            "",
            "## Next upload priority",
            "",
        ]
    )
    if priority.empty:
        lines.append("- No candidate reports are currently available.")
    else:
        top = priority[priority["recommendation_tier"].eq("top_next_upload_priority")]
        if top.empty:
            top = priority
        top = top.drop_duplicates("candidate", keep="first").head(3)
        for _, row in top.iterrows():
            lines.append(
                "- "
                f"`{row['candidate']}` from {row['source_version']} "
                f"(tier `{row['recommendation_tier']}`, score {row['priority_score']}, "
                f"novelty vs V338 {row['point_churn_vs_v338']}, "
                f"point0 additions {row['point0_additions']}, "
                f"V341-extra overlap {row['v341_extra_overlap']})."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prefer V338-family non-point0 edits that preserve action/server columns and add limited novelty.",
            "- Treat point0-addition candidates as hold/review even when gate-filtered; the public-positive evidence is still strongest for non-point0 V338-family rows.",
            "- Keep the V345 b24/V338-equivalent row set as a reference, not the next upload, because it has no novelty versus V338.",
        ]
    )
    if "V347" in missing:
        lines.append("- V341-extra overlap is pending because V347 row-diff output is not available yet.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(root: Path = ROOT, outdir: Path = OUTDIR, expected_rows: int | None = 1845) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    status, status_rows = collect_report_status(root)
    priority = rank_candidates(root, expected_rows=expected_rows)
    priority_path = safe_output_path(outdir, "candidate_priority.csv")
    priority.to_csv(priority_path, index=False)
    recommendation_path = safe_output_path(outdir, "recommendation.md")
    write_recommendation(recommendation_path, priority, status_rows)
    missing_versions = [version for version, item in status.items() if item.get("status") != "present"]
    report = {
        "version": "V350",
        "decision": "PARTIAL_RERUN_LATER" if missing_versions else "COMPLETE_DASHBOARD",
        "candidate_count": int(len(priority)),
        "missing_or_unreadable_versions": missing_versions,
        "candidate_priority": relative_path(priority_path, root),
        "recommendation": relative_path(recommendation_path, root),
        "report_status": status,
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_candidates_writes": True,
            "read_only_sources": True,
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), _json_safe(report))
    return report


def main() -> None:
    report = run_pipeline()
    print(
        {
            "outdir": relative_path(OUTDIR),
            "decision": report["decision"],
            "candidate_count": report["candidate_count"],
            "missing": report["missing_or_unreadable_versions"],
        }
    )


if __name__ == "__main__":
    main()
