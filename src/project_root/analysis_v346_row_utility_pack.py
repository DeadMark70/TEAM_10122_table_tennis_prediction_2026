"""V346 packer for row-level utility candidates from V344/V345."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    point_distribution_report,
    read_json,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v346_row_utility_pack"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = ROOT / "v338_joint_moe_pack" / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
SOURCE_DIRS = [ROOT / "v344_point0_swap_optimizer", ROOT / "v345_nonpoint0_utility_optimizer"]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slug(value: Any) -> str:
    text = str(value or "candidate").lower().strip()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "candidate"


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def point0_additions(base: pd.Series, cand: pd.Series) -> int:
    base_arr = base.to_numpy(dtype=int)
    cand_arr = cand.to_numpy(dtype=int)
    return int(np.sum((base_arr != 0) & (cand_arr == 0)))


def new_rows_beyond_v338(v306: pd.Series, v338: pd.Series, cand: pd.Series) -> int:
    base = v306.to_numpy(dtype=int)
    public = v338.to_numpy(dtype=int)
    candidate = cand.to_numpy(dtype=int)
    public_changed = base != public
    candidate_changed = base != candidate
    return int(np.sum(candidate_changed & ~public_changed))


def resolve_path(source_dir: Path, item: dict[str, Any]) -> Path | None:
    raw = item.get("path") or item.get("submission")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        candidates = [ROOT / path, source_dir / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return path.resolve() if path.exists() else None


def load_report(source_dir: Path) -> dict[str, Any] | None:
    path = source_dir / "search_report.json"
    if not path.exists():
        return None
    return read_json(path)


def report_candidates(source_dir: Path) -> list[dict[str, Any]]:
    report = load_report(source_dir)
    if not report:
        return []
    if report.get("decision") not in {"HAS_EXPORT", "POINT_ONLY", "EXPORT_LOCAL", "REVIEW_POINT"}:
        return []
    candidate_rows: dict[str, dict[str, Any]] = {}
    summary_path = report.get("summary") or report.get("candidate_summary")
    if summary_path:
        path = Path(summary_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            summary = pd.read_csv(path)
            if "candidate" in summary.columns:
                candidate_rows = {str(row["candidate"]): row.to_dict() for _, row in summary.iterrows()}
    out = []
    for item in report.get("generated_submissions") or []:
        path = resolve_path(source_dir, item)
        if path is None:
            continue
        candidate = item.get("candidate") or Path(path).stem
        candidate_row = candidate_rows.get(str(candidate), {})
        utility = (
            item.get("expected_utility")
            or item.get("expected_oof_delta")
            or item.get("point_oof_delta_vs_v306")
            or item.get("utility")
            or candidate_row.get("expected_utility")
            or candidate_row.get("utility_sum")
            or candidate_row.get("point_oof_delta_vs_v306")
            or 0.0
        )
        out.append({"source": source_dir.name, "candidate": str(candidate), "path": path, "utility": float(utility)})
    return out


def pack_candidate(src: Path, outdir: Path, name: str, expected_rows: int | None = 1845) -> tuple[Path, pd.DataFrame]:
    frame = load_submission(src, expected_rows)
    out_path = safe_output_path(outdir, f"submission_v346_{slug(name)}__v173action_v300server.csv")
    shutil.copyfile(src, out_path)
    return out_path, frame


def run_pipeline(expected_rows: int | None = 1845) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    v306 = load_submission(V306_ANCHOR, expected_rows)
    v338 = load_submission(V338_ANCHOR, expected_rows)
    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for source_dir in SOURCE_DIRS:
        for item in report_candidates(source_dir):
            path, frame = pack_candidate(item["path"], OUTDIR, f"{item['source']}_{item['candidate']}", expected_rows)
            if not frame["rally_uid"].equals(v306["rally_uid"]):
                raise ValueError(f"{item['path']} row order differs from V306")
            if not frame["actionId"].equals(v306["actionId"]):
                raise ValueError(f"{item['path']} changed action")
            if not frame["serverGetPoint"].equals(v306["serverGetPoint"]):
                raise ValueError(f"{item['path']} changed server")
            dist_v306 = point_distribution_report(v306["pointId"], frame["pointId"])
            dist_v338 = point_distribution_report(v338["pointId"], frame["pointId"])
            row = {
                "candidate": item["candidate"],
                "source": item["source"],
                "expected_utility": item["utility"],
                "point_churn_vs_v306": dist_v306["changed_rows"],
                "point_churn_vs_v338": dist_v338["changed_rows"],
                "point0_additions": point0_additions(v306["pointId"], frame["pointId"]),
                "new_rows_beyond_v338": new_rows_beyond_v338(v306["pointId"], v338["pointId"], frame["pointId"]),
                "path": relative_path(path),
            }
            summary_rows.append(row)
            generated.append({"candidate": item["candidate"], "source": item["source"], "path": relative_path(path)})

    pd.DataFrame(summary_rows).to_csv(safe_output_path(OUTDIR, "joint_summary.csv"), index=False)
    report = {
        "version": "V346",
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "summary": relative_path(OUTDIR / "joint_summary.csv"),
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "action_preserved": True,
            "server_preserved": True,
        },
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print({"outdir": relative_path(OUTDIR), "decision": report["decision"], "generated": report["generated_submission_count"]})


if __name__ == "__main__":
    main()
