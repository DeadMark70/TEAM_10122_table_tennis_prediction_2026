"""V341 packer for no-point0-add point extension candidates."""

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
OUTDIR = ROOT / "v341_no_p0_point_pack"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
SOURCE_DIRS = [
    ROOT / "v339_no_p0_point_moe_expand",
    ROOT / "v340_no_p0_point_agreement_ensemble",
]


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


def pack_point_only(base: pd.DataFrame, point: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["actionId"].equals(base["actionId"]):
        raise AssertionError("action changed")
    if not out["serverGetPoint"].equals(base["serverGetPoint"]):
        raise AssertionError("server changed")
    validate_submission_schema(out, expected_rows=len(base))
    return out


def transition_counts(base_point: pd.Series, cand_point: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for old, new in zip(base_point.astype(int), cand_point.astype(int)):
        if old == new:
            continue
        key = f"{old}->{new}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def no_p0_additions(base_point: pd.Series, cand_point: pd.Series) -> bool:
    base = base_point.to_numpy(dtype=int)
    cand = cand_point.to_numpy(dtype=int)
    return not bool(np.any((base != 0) & (cand == 0)))


def load_report(source_dir: Path) -> dict[str, Any] | None:
    path = source_dir / "search_report.json"
    if not path.exists():
        return None
    return read_json(path)


def resolve_candidate_path(source_dir: Path, item: dict[str, Any]) -> Path | None:
    raw = item.get("path") or item.get("submission")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        candidates = [ROOT / path, source_dir / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return path.resolve() if path.exists() else None


def passing_candidates(source_dir: Path) -> list[dict[str, Any]]:
    report = load_report(source_dir)
    if not report:
        return []
    if report.get("decision") not in {"HAS_EXPORT", "POINT_ONLY", "REVIEW_POINT", "EXPORT_LOCAL"}:
        return []
    candidate_rows: dict[str, dict[str, Any]] = {}
    summary_path = report.get("candidate_summary")
    if summary_path:
        path = Path(summary_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            summary = pd.read_csv(path)
            if "candidate" in summary.columns:
                candidate_rows = {str(row["candidate"]): row.to_dict() for _, row in summary.iterrows()}
    out: list[dict[str, Any]] = []
    for item in report.get("generated_submissions") or []:
        path = resolve_candidate_path(source_dir, item)
        if path is None:
            continue
        candidate = item.get("candidate") or path.stem
        candidate_row = candidate_rows.get(str(candidate), {})
        delta = (
            item.get("point_oof_delta_vs_v306")
            or item.get("point_oof_delta")
            or item.get("expected_oof_delta")
            or candidate_row.get("point_oof_delta_vs_v306")
            or candidate_row.get("evidence_delta_proxy")
            or candidate_row.get("expected_oof_delta")
            or 0.0
        )
        out.append(
            {
                "candidate": str(candidate),
                "source_dir": source_dir.name,
                "path": path,
                "delta": float(delta),
            }
        )
    return out


def run_pipeline(expected_rows: int | None = 1845) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    v306 = load_submission(V306_ANCHOR, expected_rows)
    v338 = load_submission(V338_ANCHOR, expected_rows)
    if not v306["rally_uid"].equals(v338["rally_uid"]):
        raise ValueError("V306 and V338 anchors have different row order")

    generated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for source_dir in SOURCE_DIRS:
        for item in passing_candidates(source_dir):
            frame = load_submission(item["path"], expected_rows)
            if not frame["rally_uid"].equals(v306["rally_uid"]):
                raise ValueError(f"{item['path']} row order differs from V306")
            if not frame["actionId"].equals(v306["actionId"]):
                raise ValueError(f"{item['path']} changed action")
            if not frame["serverGetPoint"].equals(v306["serverGetPoint"]):
                raise ValueError(f"{item['path']} changed server")
            if not no_p0_additions(v306["pointId"], frame["pointId"]):
                rows.append(
                    {
                        "source_dir": item["source_dir"],
                        "candidate": item["candidate"],
                        "decision": "SKIP_POINT0_ADDITION",
                        "path": relative_path(item["path"]),
                    }
                )
                continue

            filename = f"submission_v341_{slug(item['source_dir'])}_{slug(item['candidate'])}__v173action_v300server.csv"
            out_path = safe_output_path(OUTDIR, filename)
            shutil.copyfile(item["path"], out_path)
            dist_v306 = point_distribution_report(v306["pointId"], frame["pointId"])
            dist_v338 = point_distribution_report(v338["pointId"], frame["pointId"])
            changed_v306 = v306["pointId"].to_numpy(dtype=int) != frame["pointId"].to_numpy(dtype=int)
            changed_v338 = v338["pointId"].to_numpy(dtype=int) != frame["pointId"].to_numpy(dtype=int)
            overlap = int(np.sum(changed_v306 & (v306["pointId"].to_numpy(dtype=int) != v338["pointId"].to_numpy(dtype=int))))
            new_beyond_v338 = int(np.sum(changed_v338))
            row = {
                "source_dir": item["source_dir"],
                "candidate": item["candidate"],
                "decision": "EXPORT_LOCAL",
                "expected_oof_delta": item["delta"],
                "point_churn_vs_v306": dist_v306["changed_rows"],
                "point_churn_vs_v338": dist_v338["changed_rows"],
                "overlap_with_v338_changed_rows": overlap,
                "new_rows_beyond_v338": new_beyond_v338,
                "point0_additions_vs_v306": dist_v306["point0_additions"],
                "transition_counts_vs_v306": transition_counts(v306["pointId"], frame["pointId"]),
                "path": relative_path(out_path),
            }
            rows.append(row)
            generated.append({"candidate": item["candidate"], "source_dir": item["source_dir"], "path": relative_path(out_path)})

    summary = pd.DataFrame(rows)
    summary.to_csv(safe_output_path(OUTDIR, "joint_summary.csv"), index=False)
    report = {
        "version": "V341",
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "summary": relative_path(OUTDIR / "joint_summary.csv"),
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_point0_additions": True,
            "action_preserved": True,
            "server_preserved": True,
        },
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        {
            "outdir": relative_path(OUTDIR),
            "decision": report["decision"],
            "generated_submission_count": report["generated_submission_count"],
        }
    )


if __name__ == "__main__":
    main()
