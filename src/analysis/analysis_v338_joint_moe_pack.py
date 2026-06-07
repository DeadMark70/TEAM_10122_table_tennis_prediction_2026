"""V338 joint packer for locally passing lightweight MoE components.

This script only repackages V336/V337 outputs that already passed their own
local gates. It keeps server predictions fixed and writes only inside
v338_joint_moe_pack.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    action_distribution_report,
    point_distribution_report,
    read_json,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v338_joint_moe_pack"
V336_DIR = ROOT / "v336_action_moe"
V337_DIR = ROOT / "v337_point_moe"
BASE_ANCHOR_PATH = (
    ROOT
    / "v306_point0_addition_probe"
    / "submission_v306_p0_cap0p01__v173action_v300server.csv"
)
COMPAT_THRESHOLD = 0.05


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: Any) -> str:
    text = str(value or "candidate").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "candidate"


def _report_path(component_dir: Path) -> Path:
    return Path(component_dir) / "search_report.json"


def _load_report(component_dir: Path) -> tuple[dict[str, Any] | None, str]:
    path = _report_path(component_dir)
    if not path.exists():
        return None, "missing_report"
    try:
        return read_json(path), "ok"
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, "bad_report"


def _resolve_submission_path(component_dir: Path, item: dict[str, Any]) -> Path:
    raw = item.get("path") or item.get("submission")
    if not raw:
        raise ValueError(f"generated submission entry has no path: {item}")
    path = Path(raw)
    if not path.is_absolute():
        candidates = [ROOT / path, Path(component_dir) / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return path.resolve()


def _passing_submissions(
    component_dir: Path,
    report: dict[str, Any] | None,
    *,
    component: str,
) -> list[dict[str, Any]]:
    if not report:
        return []
    generated = report.get("generated_submissions") or []
    if not generated:
        return []
    if component == "action":
        if report.get("decision") not in {"REVIEW_ACTION", "HAS_EXPORT", "EXPORT_LOCAL"}:
            return []
        delta_keys = ("action_oof_delta", "action_oof_delta_vs_v173")
    elif component == "point":
        if report.get("decision") not in {"HAS_EXPORT", "EXPORT_LOCAL", "REVIEW_POINT"}:
            return []
        delta_keys = ("point_oof_delta_vs_v306", "point_oof_delta")
    else:
        raise ValueError(component)

    best = report.get("best_candidate") or {}
    candidate_rows: dict[str, dict[str, Any]] = {}
    summary_path = report.get("candidate_summary")
    if summary_path:
        path = Path(summary_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            try:
                summary = pd.read_csv(path)
                if "candidate" in summary.columns:
                    candidate_rows = {
                        str(row["candidate"]): row.to_dict()
                        for _, row in summary.iterrows()
                    }
            except Exception:
                candidate_rows = {}
    out = []
    for item in generated:
        candidate = item.get("candidate") or best.get("candidate") or Path(str(item.get("path", ""))).stem
        candidate_row = candidate_rows.get(str(candidate), {})
        delta = 0.0
        for key in delta_keys:
            if key in item:
                delta = float(item[key])
                break
            if key in candidate_row:
                delta = float(candidate_row[key])
                break
            if key in best:
                delta = float(best[key])
                break
        path = _resolve_submission_path(component_dir, item)
        if path.exists():
            out.append({"candidate": candidate, "path": path, "delta": delta})
    return out


def _load_submission(path: Path, expected_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def _load_base_anchor(expected_rows: int | None) -> pd.DataFrame | None:
    if not BASE_ANCHOR_PATH.exists():
        return None
    try:
        return _load_submission(BASE_ANCHOR_PATH, expected_rows)
    except ValueError:
        # Unit tests and ad-hoc fixtures can use smaller row counts. In that
        # case churn against the production anchor is not meaningful.
        return None


def apply_compatibility_veto(rows: pd.DataFrame, threshold: float = COMPAT_THRESHOLD) -> pd.Series:
    if "compat_score" not in rows:
        raise ValueError("compat_score column is required")
    score = pd.to_numeric(rows["compat_score"], errors="coerce").fillna(0.0)
    changed = (rows["base_action"] != rows["cand_action"]) | (rows["base_point"] != rows["cand_point"])
    return (~changed) | (score >= float(threshold))


def _compat_score(action: np.ndarray, point: np.ndarray) -> np.ndarray:
    action_arr = np.asarray(action, dtype=int)
    point_arr = np.asarray(point, dtype=int)
    score = np.full(len(action_arr), 0.80, dtype=float)
    terminal_action = np.isin(action_arr, [10, 11, 12, 13, 14])
    score[terminal_action & (point_arr != 0)] = 0.01
    score[(~terminal_action) & (point_arr == 0)] = 0.20
    return score


def pack_joint_submission(base: pd.DataFrame, action: np.ndarray, point: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    out["actionId"] = np.asarray(action, dtype=int)
    out["pointId"] = np.asarray(point, dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["serverGetPoint"].equals(base["serverGetPoint"]):
        raise AssertionError("V338 export changed serverGetPoint")
    validate_submission_schema(out, expected_rows=len(base))
    return out


def _write_submission(outdir: Path, filename: str, frame: pd.DataFrame, expected_rows: int | None) -> Path:
    validate_submission_schema(frame, expected_rows=expected_rows)
    path = safe_output_path(outdir, filename)
    frame.to_csv(path, index=False, float_format="%.8f")
    return path


def _copy_submission(outdir: Path, filename: str, src: Path, expected_rows: int | None) -> tuple[Path, pd.DataFrame]:
    frame = _load_submission(src, expected_rows)
    path = _write_submission(outdir, filename, frame, expected_rows)
    return path, frame


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    v336_dir: Path = V336_DIR,
    v337_dir: Path = V337_DIR,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = safe_output_path(outdir, "search_report.json")

    action_report, action_status = _load_report(Path(v336_dir))
    point_report, point_status = _load_report(Path(v337_dir))
    action_candidates = _passing_submissions(Path(v336_dir), action_report, component="action")
    point_candidates = _passing_submissions(Path(v337_dir), point_report, component="point")
    base_anchor = _load_base_anchor(expected_rows)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    if point_status == "missing_report":
        decision = "WAITING_FOR_V337"
    elif point_report and point_report.get("decision") == "BLOCKED_MISSING_ANCHOR" and not action_candidates:
        decision = "NO_COMPONENTS"
    elif not action_candidates and not point_candidates:
        decision = "NO_COMPONENTS"
    else:
        decision = "NO_COMPONENTS"

        for cand in action_candidates:
            name = _slug(cand["candidate"])
            filename = f"submission_v338_action_only_{name}__v306point_v300server.csv"
            path, frame = _copy_submission(outdir, filename, cand["path"], expected_rows)
            action_churn = (
                action_distribution_report(base_anchor["actionId"], frame["actionId"])["changed_rows"]
                if base_anchor is not None
                else 0
            )
            point_churn = (
                point_distribution_report(base_anchor["pointId"], frame["pointId"])["changed_rows"]
                if base_anchor is not None
                else 0
            )
            generated.append({"kind": "action_only", "candidate": cand["candidate"], "path": relative_path(path)})
            summary_rows.append(
                {
                    "kind": "action_only",
                    "candidate": cand["candidate"],
                    "expected_oof_delta": cand["delta"],
                    "action_churn": action_churn,
                    "point_churn": point_churn,
                    "path": relative_path(path),
                }
            )
            decision = "ACTION_ONLY"

        for cand in point_candidates:
            name = _slug(cand["candidate"])
            filename = f"submission_v338_point_only_{name}__v173action_v300server.csv"
            path, frame = _copy_submission(outdir, filename, cand["path"], expected_rows)
            action_churn = (
                action_distribution_report(base_anchor["actionId"], frame["actionId"])["changed_rows"]
                if base_anchor is not None
                else 0
            )
            point_churn = (
                point_distribution_report(base_anchor["pointId"], frame["pointId"])["changed_rows"]
                if base_anchor is not None
                else 0
            )
            generated.append({"kind": "point_only", "candidate": cand["candidate"], "path": relative_path(path)})
            summary_rows.append(
                {
                    "kind": "point_only",
                    "candidate": cand["candidate"],
                    "expected_oof_delta": cand["delta"],
                    "action_churn": action_churn,
                    "point_churn": point_churn,
                    "path": relative_path(path),
                }
            )
            decision = "POINT_ONLY" if decision in {"NO_COMPONENTS", "ACTION_ONLY"} else decision

        if action_candidates and point_candidates:
            action = action_candidates[0]
            point = point_candidates[0]
            action_frame = _load_submission(action["path"], expected_rows)
            point_frame = _load_submission(point["path"], expected_rows)
            if not action_frame["rally_uid"].equals(point_frame["rally_uid"]):
                raise ValueError("action and point submissions have different rally_uid order")
            if not action_frame["serverGetPoint"].equals(point_frame["serverGetPoint"]):
                raise ValueError("action and point submissions have different serverGetPoint")

            base = point_frame.copy()
            rows = pd.DataFrame(
                {
                    "base_action": point_frame["actionId"].to_numpy(dtype=int),
                    "cand_action": action_frame["actionId"].to_numpy(dtype=int),
                    "base_point": action_frame["pointId"].to_numpy(dtype=int),
                    "cand_point": point_frame["pointId"].to_numpy(dtype=int),
                }
            )
            rows["compat_score"] = _compat_score(rows["cand_action"].to_numpy(), rows["cand_point"].to_numpy())
            keep = apply_compatibility_veto(rows, threshold=COMPAT_THRESHOLD)
            expected_delta = float(action["delta"]) + float(point["delta"])
            component_floor = max(float(action["delta"]), float(point["delta"])) - 0.0002
            joint_allowed = bool(keep.all() and expected_delta >= component_floor)
            if joint_allowed:
                joint = pack_joint_submission(
                    base,
                    action=action_frame["actionId"].to_numpy(dtype=int),
                    point=point_frame["pointId"].to_numpy(dtype=int),
                )
                action_churn = action_distribution_report(base["actionId"], joint["actionId"])
                point_churn = point_distribution_report(base["pointId"], joint["pointId"])
                action_name = _slug(action["candidate"])
                point_name = _slug(point["candidate"])
                filename = f"submission_v338_joint_{action_name}_{point_name}__v300server.csv"
                path = _write_submission(outdir, filename, joint, expected_rows)
                generated.append(
                    {
                        "kind": "joint",
                        "action_candidate": action["candidate"],
                        "point_candidate": point["candidate"],
                        "path": relative_path(path),
                    }
                )
                summary_rows.append(
                    {
                        "kind": "joint",
                        "candidate": f"{action['candidate']} + {point['candidate']}",
                        "expected_oof_delta": expected_delta,
                        "action_churn": action_churn["changed_rows"],
                        "point_churn": point_churn["changed_rows"],
                        "compatibility_veto_pass": True,
                        "path": relative_path(path),
                    }
                )
                decision = "JOINT"
            else:
                summary_rows.append(
                    {
                        "kind": "joint_vetoed",
                        "candidate": f"{action['candidate']} + {point['candidate']}",
                        "expected_oof_delta": expected_delta,
                        "compatibility_veto_pass": bool(keep.all()),
                        "vetoed_rows": int((~keep).sum()),
                        "component_floor": component_floor,
                    }
                )

    pd.DataFrame(summary_rows).to_csv(safe_output_path(outdir, "joint_summary.csv"), index=False)
    report = {
        "version": "V338",
        "decision": decision,
        "component_status": {
            "v336_report": action_status,
            "v336_decision": None if action_report is None else action_report.get("decision"),
            "v336_passing_submissions": len(action_candidates),
            "v337_report": point_status,
            "v337_decision": None if point_report is None else point_report.get("decision"),
            "v337_passing_submissions": len(point_candidates),
        },
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "joint_summary": relative_path(outdir / "joint_summary.csv"),
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
            "server_preserved": True,
        },
    }
    write_json(report_path, report)
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
