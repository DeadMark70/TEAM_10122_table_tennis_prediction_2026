"""V343 row-level point candidate bank from local schema-valid submissions.

This module audits existing local submission CSVs and converts pointId
differences against V306 into a row-level candidate table. It writes reports
only under v343_row_candidate_bank and never exports submission files.
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
OUTDIR = ROOT / "v343_row_candidate_bank"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
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
BANK_COLUMNS = [
    "row_id",
    "rally_uid",
    "task",
    "anchor_value",
    "candidate_value",
    "source",
    "source_dir",
    "transition",
    "is_point0_addition",
    "is_point0_removal",
    "is_nonterminal_point_swap",
    "is_same_depth_swap",
    "changed_in_v338",
    "source_public_tag",
    "source_local_delta_if_known",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def is_same_depth_point_swap(anchor_value: int, candidate_value: int) -> bool:
    old = int(anchor_value)
    new = int(candidate_value)
    return old != new and 1 <= old <= 9 and 1 <= new <= 9 and point_depth(old) == point_depth(new)


def source_public_tag(source_dir: str, source: str) -> str:
    text = f"{source_dir}/{source}".lower()
    if "v338" in text:
        return "v338_public_positive"
    if "v306" in text:
        return "v306_point0_probe"
    if "v307" in text:
        return "v307_saturated_p0"
    if "v339" in text or "v340" in text or "v341" in text:
        return "no_p0_expansion"
    if "v333" in text or "v334" in text or "v337" in text:
        return "v338_family_support"
    if "v322" in text or "v329" in text or "v272" in text:
        return "historical_point_model"
    return "unknown"


def _candidate_keys(path: Path, source_dir: Path) -> set[str]:
    rel = relative_path(path)
    return {
        path.name,
        path.stem,
        rel,
        path.as_posix(),
        str(path),
        f"{source_dir.name}/{path.name}",
        f"{source_dir.name}/{path.stem}",
    }


def _coerce_delta(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _best_delta_from_dict(item: dict[str, Any]) -> float | None:
    preferred = [
        "source_local_delta_if_known",
        "point_oof_delta_vs_v306",
        "point_oof_delta",
        "expected_oof_delta",
        "literal_oof_delta",
        "evidence_delta_proxy",
        "delta",
        "public_delta",
    ]
    for key in preferred:
        if key in item:
            value = _coerce_delta(item.get(key))
            if value is not None:
                return value
    return None


def _walk_report_items(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if any(key in value for key in ("path", "submission", "candidate", "name")):
            found.append(value)
        for child in value.values():
            found.extend(_walk_report_items(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_report_items(child))
    return found


def load_source_delta_map(source_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for report_path in sorted(source_dir.glob("*.json")):
        try:
            report = _json_load(report_path)
        except (OSError, json.JSONDecodeError):
            continue
        for item in _walk_report_items(report):
            delta = _best_delta_from_dict(item)
            if delta is None:
                continue
            keys = [item.get("path"), item.get("submission"), item.get("candidate"), item.get("name")]
            for raw in keys:
                if raw:
                    text = str(raw)
                    out[text] = delta
                    out[Path(text).name] = delta
                    out[Path(text).stem] = delta
    return out


def source_delta_for_path(path: Path, source_dir: Path, delta_map: dict[str, float]) -> float | None:
    for key in _candidate_keys(path, source_dir):
        if key in delta_map:
            return delta_map[key]
    return None


def extract_point_edits(
    base: pd.DataFrame,
    cand: pd.DataFrame,
    source: str,
    *,
    source_dir: str = "",
    changed_in_v338: pd.Series | np.ndarray | None = None,
    source_public_tag_value: str | None = None,
    source_local_delta_if_known: float | None = None,
) -> pd.DataFrame:
    if len(base) != len(cand):
        raise ValueError("base and candidate row counts differ")
    if "rally_uid" in cand and not base["rally_uid"].equals(cand["rally_uid"]):
        raise ValueError("base and candidate rally_uid order differs")

    base_point = base["pointId"].astype(int).to_numpy()
    cand_point = cand["pointId"].astype(int).to_numpy()
    mask = base_point != cand_point
    if changed_in_v338 is None:
        changed_v338 = np.zeros(len(base), dtype=bool)
    else:
        changed_v338 = np.asarray(changed_in_v338, dtype=bool)
        if len(changed_v338) != len(base):
            raise ValueError("changed_in_v338 length differs from base")

    rows: list[dict[str, Any]] = []
    tag = source_public_tag_value if source_public_tag_value is not None else source_public_tag(source_dir, source)
    for row_id in np.where(mask)[0]:
        anchor_value = int(base_point[row_id])
        candidate_value = int(cand_point[row_id])
        rows.append(
            {
                "row_id": int(row_id),
                "rally_uid": base["rally_uid"].iloc[row_id],
                "task": "point",
                "anchor_value": anchor_value,
                "candidate_value": candidate_value,
                "source": source,
                "source_dir": source_dir,
                "transition": f"{anchor_value}->{candidate_value}",
                "is_point0_addition": bool(anchor_value != 0 and candidate_value == 0),
                "is_point0_removal": bool(anchor_value == 0 and candidate_value != 0),
                "is_nonterminal_point_swap": bool(1 <= anchor_value <= 9 and 1 <= candidate_value <= 9),
                "is_same_depth_swap": bool(is_same_depth_point_swap(anchor_value, candidate_value)),
                "changed_in_v338": bool(changed_v338[row_id]),
                "source_public_tag": tag,
                "source_local_delta_if_known": source_local_delta_if_known,
            }
        )
    return pd.DataFrame(rows, columns=BANK_COLUMNS)


def filter_point0_policy(rows: pd.DataFrame, allow_p0_add: bool) -> pd.DataFrame:
    if allow_p0_add:
        return rows.copy()
    if rows.empty:
        return rows.copy()
    return rows.loc[~((rows["anchor_value"].astype(int) != 0) & (rows["candidate_value"].astype(int) == 0))].copy()


def discover_submission_paths(source_dirs: list[Path] | None = None) -> list[Path]:
    dirs = SOURCE_DIRS if source_dirs is None else source_dirs
    paths: list[Path] = []
    for source_dir in dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        for path in sorted(source_dir.glob("submission*.csv")):
            if "ttmatch" in path.name.lower() or "old_server" in path.name.lower():
                continue
            paths.append(path)
    return paths


def build_candidate_bank(
    base: pd.DataFrame,
    public_anchor: pd.DataFrame,
    submission_paths: list[Path],
    expected_rows: int | None = 1845,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if not base["rally_uid"].equals(public_anchor["rally_uid"]):
        raise ValueError("V306 and V338 anchors have different row order")
    changed_in_v338 = base["pointId"].astype(int).to_numpy() != public_anchor["pointId"].astype(int).to_numpy()

    banks: list[pd.DataFrame] = []
    source_stats: dict[str, dict[str, Any]] = {}
    scan_rows: list[dict[str, Any]] = []
    delta_maps: dict[Path, dict[str, float]] = {}

    for path in submission_paths:
        source_dir = path.parent
        source_dir_name = source_dir.name
        stat = source_stats.setdefault(
            source_dir_name,
            {
                "source_dir": source_dir_name,
                "submission_files_seen": 0,
                "schema_valid_files": 0,
                "included_files": 0,
                "point_edit_rows": 0,
                "point0_additions": 0,
                "point0_removals": 0,
                "nonterminal_point_swaps": 0,
                "same_depth_swaps": 0,
                "changed_in_v338_rows": 0,
            },
        )
        stat["submission_files_seen"] += 1
        rel = relative_path(path)
        try:
            cand = load_submission(path, expected_rows=expected_rows)
            stat["schema_valid_files"] += 1
        except Exception as exc:  # noqa: BLE001 - scan report should keep invalid-file reason.
            scan_rows.append({"path": rel, "source_dir": source_dir_name, "status": "SKIP_INVALID_SCHEMA", "reason": str(exc)})
            continue
        if not base["rally_uid"].equals(cand["rally_uid"]):
            scan_rows.append({"path": rel, "source_dir": source_dir_name, "status": "SKIP_ROW_ORDER", "reason": "rally_uid differs from V306"})
            continue
        if source_dir not in delta_maps:
            delta_maps[source_dir] = load_source_delta_map(source_dir)
        delta = source_delta_for_path(path, source_dir, delta_maps[source_dir])
        edits = extract_point_edits(
            base,
            cand,
            source=path.stem,
            source_dir=source_dir_name,
            changed_in_v338=changed_in_v338,
            source_public_tag_value=source_public_tag(source_dir_name, path.stem),
            source_local_delta_if_known=delta,
        )
        stat["included_files"] += 1
        stat["point_edit_rows"] += int(len(edits))
        if not edits.empty:
            stat["point0_additions"] += int(edits["is_point0_addition"].sum())
            stat["point0_removals"] += int(edits["is_point0_removal"].sum())
            stat["nonterminal_point_swaps"] += int(edits["is_nonterminal_point_swap"].sum())
            stat["same_depth_swaps"] += int(edits["is_same_depth_swap"].sum())
            stat["changed_in_v338_rows"] += int(edits["changed_in_v338"].sum())
            banks.append(edits)
        scan_rows.append(
            {
                "path": rel,
                "source_dir": source_dir_name,
                "status": "INCLUDED",
                "reason": "",
                "point_edit_rows": int(len(edits)),
                "source_public_tag": source_public_tag(source_dir_name, path.stem),
                "source_local_delta_if_known": delta,
            }
        )

    if banks:
        bank = pd.concat(banks, ignore_index=True)
    else:
        bank = pd.DataFrame(columns=BANK_COLUMNS)
    bank = bank.loc[:, BANK_COLUMNS]
    source_summary = pd.DataFrame(source_stats.values()).sort_values("source_dir").reset_index(drop=True)
    return bank, source_summary, scan_rows


def run_pipeline(expected_rows: int | None = 1845) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not V306_ANCHOR.exists() or not V338_ANCHOR.exists():
        missing = [relative_path(path) for path in (V306_ANCHOR, V338_ANCHOR) if not path.exists()]
        report = {"version": "V343", "decision": "BLOCKED_MISSING_ANCHOR", "missing": missing}
        write_json(safe_output_path(OUTDIR, "search_report.json"), report)
        return report

    base = load_submission(V306_ANCHOR, expected_rows=expected_rows)
    public_anchor = load_submission(V338_ANCHOR, expected_rows=expected_rows)
    paths = discover_submission_paths()
    bank, source_summary, scan_rows = build_candidate_bank(base, public_anchor, paths, expected_rows=expected_rows)

    bank_path = safe_output_path(OUTDIR, "candidate_bank.csv")
    summary_path = safe_output_path(OUTDIR, "source_summary.csv")
    bank.to_csv(bank_path, index=False)
    source_summary.to_csv(summary_path, index=False)

    by_tag = bank["source_public_tag"].value_counts(dropna=False).sort_index().to_dict() if not bank.empty else {}
    report = {
        "version": "V343",
        "decision": "BANK_BUILT",
        "anchor": relative_path(V306_ANCHOR),
        "public_anchor": relative_path(V338_ANCHOR),
        "candidate_bank": relative_path(bank_path),
        "source_summary": relative_path(summary_path),
        "submission_files_discovered": len(paths),
        "schema_valid_files": int(source_summary["schema_valid_files"].sum()) if not source_summary.empty else 0,
        "included_files": int(source_summary["included_files"].sum()) if not source_summary.empty else 0,
        "candidate_bank_rows": int(len(bank)),
        "unique_candidate_rows": int(bank["row_id"].nunique()) if not bank.empty else 0,
        "point0_additions": int(bank["is_point0_addition"].sum()) if not bank.empty else 0,
        "point0_removals": int(bank["is_point0_removal"].sum()) if not bank.empty else 0,
        "nonterminal_point_swaps": int(bank["is_nonterminal_point_swap"].sum()) if not bank.empty else 0,
        "same_depth_swaps": int(bank["is_same_depth_swap"].sum()) if not bank.empty else 0,
        "changed_in_v338_rows": int(bank["changed_in_v338"].sum()) if not bank.empty else 0,
        "source_public_tag_counts": {str(k): int(v) for k, v in by_tag.items()},
        "scanned_files": _json_safe(scan_rows),
        "policy": {
            "no_ttmatch": True,
            "no_old_server_filename": True,
            "manual_row_edits": False,
            "upload_candidates_writes": False,
            "submission_exports": False,
        },
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
