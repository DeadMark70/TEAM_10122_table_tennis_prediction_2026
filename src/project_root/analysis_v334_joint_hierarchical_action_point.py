"""V334 joint hierarchical action-point packager.

V334 consumes V332/V333 local reports and only packages combinations whose
component evidence already passed.  It never repairs rows manually and never
uses old-server or TTMATCH inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v334_joint_hierarchical_action_point"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V332_REPORT = ROOT / "v332_hierarchical_action_model" / "v332_report.json"
V333_REPORT = ROOT / "v333_hierarchical_point_model" / "v333_report.json"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
BANNED_WRITE_PARTS = {"upload_candidates", "upload_candidates_20260519", "selected", "submissions"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def protected_output_path(outdir: Path, filename: str) -> Path:
    root = Path(outdir)
    path = root / filename
    parts = {part.lower() for part in path.parts}
    if parts & BANNED_WRITE_PARTS:
        raise ValueError(f"V334 refuses upload/selected/submissions path: {path}")
    if path.parent != root:
        raise ValueError(f"V334 outputs must stay directly under {root}: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": relative_path(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def load_anchor() -> pd.DataFrame:
    anchor = pd.read_csv(ANCHOR_SUBMISSION)
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"anchor columns {list(anchor.columns)} != {SUBMISSION_COLUMNS}")
    return anchor


def passing_v332(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("reviewable_candidates", [])
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if row.get("evidence_pass") in (1, True) and float(row.get("action_oof_delta", 0.0)) > 0.0:
            out.append(row)
    return out


def passing_v333(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("verdict") in {"NO_EXPORT", "NO_EXPORT_NO_EVIDENCE", "NO_EXPORT_ANCHOR_FALLBACK"} and "best_candidate" not in report:
        return []
    rows = []
    best = report.get("best_candidate")
    if isinstance(best, dict):
        rows.append(best)
    for item in report.get("generated_submissions", []):
        if isinstance(item, dict) and item.get("candidate") != (best or {}).get("candidate"):
            # Search CSV carries the full metrics; report generated list is a path index.
            pass
    search_path = ROOT / "v333_hierarchical_point_model" / "v333_point_search.csv"
    if search_path.exists():
        rows = pd.read_csv(search_path).to_dict("records")
    return [
        row
        for row in rows
        if bool(row.get("evidence_pass", False))
        and float(row.get("point_oof_delta_vs_v306", 0.0)) > 0.0
        and str(row.get("path", ""))
    ]


def compatibility_allows(action_row: dict[str, Any], point_row: dict[str, Any]) -> bool:
    """Conservative joint compatibility gate."""
    action_changed = int(action_row.get("changed_action_rows", 0))
    point_changed = int(point_row.get("test_changed_rows", 0))
    if action_changed <= 0 or point_changed <= 0:
        return False
    if int(action_row.get("serve_action_rows", 0)) > 0:
        return False
    if float(action_row.get("action_oof_delta", 0.0)) <= 0.0:
        return False
    if float(point_row.get("point_oof_delta_vs_v306", 0.0)) <= 0.0:
        return False
    return True


def copy_candidate(src: Path, dst_name: str) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    frame = pd.read_csv(src)
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{src} columns {list(frame.columns)} != {SUBMISSION_COLUMNS}")
    path = protected_output_path(OUTDIR, dst_name)
    frame.to_csv(path, index=False, float_format="%.8f")
    return relative_path(path)


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor()
    v332 = load_json(V332_REPORT)
    v333 = load_json(V333_REPORT)
    action_rows = passing_v332(v332)
    point_rows = passing_v333(v333)
    generated: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    if action_rows:
        best_action = sorted(action_rows, key=lambda r: float(r.get("action_oof_delta", 0.0)), reverse=True)[0]
        src = ROOT / "v332_hierarchical_action_model" / str(best_action.get("candidate_file", ""))
        if src.exists():
            generated.append(
                {
                    "kind": "action_only",
                    "source_candidate": best_action.get("candidate"),
                    "path": copy_candidate(src, "submission_v334_action_only_from_v332__v306point_v300server.csv"),
                }
            )
        else:
            decisions.append({"kind": "action_only", "decision": "SOURCE_FILE_MISSING", "source": relative_path(src)})
    else:
        decisions.append({"kind": "action_only", "decision": "SKIP_NO_PASSING_V332"})

    if point_rows:
        # Prefer lower point0-add dose for first public probe when available; otherwise best delta.
        sorted_points = sorted(
            point_rows,
            key=lambda r: (
                int(r.get("test_point0_additions", 999)),
                -float(r.get("point_oof_delta_vs_v306", 0.0)),
                int(r.get("test_changed_rows", 999)),
            ),
        )
        for row in sorted_points[:3]:
            src = ROOT / str(row.get("path", ""))
            dst = f"submission_v334_point_only_{row.get('candidate')}__v173action_v300server.csv"
            generated.append(
                {
                    "kind": "point_only",
                    "source_candidate": row.get("candidate"),
                    "point_oof_delta_vs_v306": float(row.get("point_oof_delta_vs_v306", 0.0)),
                    "test_changed_rows": int(row.get("test_changed_rows", 0)),
                    "test_point0_additions": int(row.get("test_point0_additions", 0)),
                    "path": copy_candidate(src, dst),
                }
            )
    else:
        decisions.append({"kind": "point_only", "decision": "SKIP_NO_PASSING_V333"})

    if action_rows and point_rows and compatibility_allows(action_rows[0], point_rows[0]):
        decisions.append({"kind": "joint", "decision": "DEFER_UNTIL_INDIVIDUAL_PUBLIC_PROBES"})
    else:
        decisions.append({"kind": "joint", "decision": "SKIP_NO_SAFE_ACTION_POINT_COMPONENT_PAIR"})

    report = {
        "version": "V334",
        "anchor_submission": relative_path(ANCHOR_SUBMISSION),
        "anchor_rows": int(len(anchor)),
        "v332_decision": v332.get("decision", "missing"),
        "v333_verdict": v333.get("verdict", "missing"),
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "decisions": decisions,
        "recommendation": "REVIEW_POINT_ONLY" if generated else "DO_NOT_UPLOAD",
        "policy": {
            "no_old_server": True,
            "no_ttmatch": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
        },
    }
    (OUTDIR / "v334_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    pd.DataFrame(generated or decisions).to_csv(OUTDIR / "v334_summary.csv", index=False)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": relative_path(OUTDIR),
                "recommendation": report["recommendation"],
                "generated_submission_count": report["generated_submission_count"],
                "generated_submissions": [item["path"] for item in report["generated_submissions"]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
