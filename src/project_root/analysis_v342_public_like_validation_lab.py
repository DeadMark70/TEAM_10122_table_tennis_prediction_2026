"""V342 public-like validation lab for row-level point candidate audits.

This module is report-only: it scans existing local submission CSVs, compares
their point predictions against V306 and V338 anchors, and writes audit reports
under ``v342_public_like_validation_lab``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v342_public_like_validation_lab"
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

KNOWN_PUBLIC_RECORDS = [
    {"version": "V338", "public_delta": 0.0012136},
    {"version": "V341", "public_delta": 0.0003196},
    {"version": "V191", "public_delta": -0.0064370},
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return relative_path(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_submission(path: Path) -> pd.DataFrame:
    """Load and validate a Kaggle submission-shaped CSV."""
    if not path.exists():
        raise FileNotFoundError(f"missing submission: {path}")
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=None)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def transition_counts(base_point, cand_point) -> dict[str, int]:
    base = pd.Series(base_point).astype(int).to_numpy()
    cand = pd.Series(cand_point).astype(int).to_numpy()
    if len(base) != len(cand):
        raise ValueError("base_point and cand_point length mismatch")
    counts: dict[str, int] = {}
    for old, new in zip(base, cand):
        if old == new:
            continue
        key = f"{old}->{new}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def density_ratio_from_counts(keys, train_counts, test_counts, clip=(0.25, 4.0)) -> np.ndarray:
    """Return clipped per-key test/train count ratios."""
    low, high = clip
    weights: list[float] = []
    for key in keys:
        train = float(train_counts.get(key, 0))
        test = float(test_counts.get(key, 0))
        if train <= 0:
            ratio = high if test > 0 else 1.0
        else:
            ratio = test / train
        weights.append(float(np.clip(ratio, low, high)))
    return np.asarray(weights, dtype=float)


def _ensure_aligned(*frames: pd.DataFrame) -> None:
    first = frames[0]["rally_uid"].astype(str).reset_index(drop=True)
    for frame in frames[1:]:
        if not first.equals(frame["rally_uid"].astype(str).reset_index(drop=True)):
            raise ValueError("submission row order differs by rally_uid")


def _family_tag(base_point: np.ndarray, cand_point: np.ndarray) -> str:
    changed = base_point != cand_point
    p0_add = (base_point != 0) & (cand_point == 0)
    non_p0_change = changed & ~p0_add
    if bool(p0_add.any()) and bool(non_p0_change.any()):
        return "mixed"
    if bool(p0_add.any()):
        return "p0_addition"
    if bool(changed.any()):
        return "no_p0_nonterminal"
    return "no_change"


def point_candidate_audit(base: pd.DataFrame, public_anchor: pd.DataFrame, cand: pd.DataFrame) -> dict[str, Any]:
    """Audit a candidate's point column against V306 base and V338 public anchor."""
    _ensure_aligned(base, public_anchor, cand)
    base_point = base["pointId"].astype(int).to_numpy()
    public_point = public_anchor["pointId"].astype(int).to_numpy()
    cand_point = cand["pointId"].astype(int).to_numpy()

    changed_vs_base = base_point != cand_point
    changed_v338_rows = base_point != public_point
    return {
        "point_churn_vs_v306": int(np.sum(changed_vs_base)),
        "point_churn_vs_v338": int(np.sum(public_point != cand_point)),
        "point0_additions": int(np.sum((base_point != 0) & (cand_point == 0))),
        "point0_removals": int(np.sum((base_point == 0) & (cand_point != 0))),
        "transition_counts": transition_counts(base_point, cand_point),
        "overlap_with_v338_changed_rows": int(np.sum(changed_vs_base & changed_v338_rows)),
        "new_rows_beyond_v338": int(np.sum(changed_vs_base & ~changed_v338_rows)),
        "family": _family_tag(base_point, cand_point),
    }


def historical_sanity(records: list[dict]) -> dict[str, Any]:
    by_version = {str(row["version"]).upper(): float(row["public_delta"]) for row in records}
    positives = [delta for version, delta in by_version.items() if version != "V191" and delta > 0]
    checks = {
        "v338_above_v341": by_version.get("V338", float("-inf")) > by_version.get("V341", float("-inf")),
        "positive_above_v191": bool(positives)
        and min(positives) > by_version.get("V191", float("-inf")),
    }
    ranked = sorted(by_version, key=lambda version: by_version[version], reverse=True)
    return {
        **checks,
        "passed": all(checks.values()),
        "ranked_versions": ranked,
        "public_delta_by_version": by_version,
    }


def _candidate_paths(source_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        paths.extend(sorted(source_dir.glob("submission*.csv")))
    return sorted(paths, key=lambda path: relative_path(path))


def _audit_path(path: Path, base: pd.DataFrame, public_anchor: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        cand = load_submission(path)
        if len(cand) != len(base):
            raise ValueError(f"row count {len(cand)} != anchor row count {len(base)}")
        audit = point_candidate_audit(base, public_anchor, cand)
        audit.update(
            {
                "source_dir": path.parent.name,
                "candidate": path.stem,
                "path": relative_path(path),
                "action_churn_vs_v306": int(
                    np.sum(base["actionId"].astype(int).to_numpy() != cand["actionId"].astype(int).to_numpy())
                ),
                "server_churn_vs_v306": int(
                    np.sum(
                        pd.to_numeric(base["serverGetPoint"]).to_numpy(dtype=float)
                        != pd.to_numeric(cand["serverGetPoint"]).to_numpy(dtype=float)
                    )
                ),
            }
        )
        return audit, None
    except Exception as exc:
        return None, {"path": relative_path(path), "source_dir": path.parent.name, "error": str(exc)}


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    missing_anchors = [relative_path(path) for path in (V306_ANCHOR, V338_ANCHOR) if not path.exists()]
    if missing_anchors:
        report = {
            "version": "V342",
            "decision": "BLOCKED_MISSING_ANCHOR",
            "missing_anchors": missing_anchors,
            "generated_submission_count": 0,
            "policy": {"report_only": True, "no_submissions_exported": True},
        }
        write_json(safe_output_path(OUTDIR, "search_report.json"), report)
        write_json(safe_output_path(OUTDIR, "v342_historical_sanity.json"), historical_sanity(KNOWN_PUBLIC_RECORDS))
        pd.DataFrame().to_csv(safe_output_path(OUTDIR, "v342_candidate_audit.csv"), index=False)
        return report

    base = load_submission(V306_ANCHOR)
    public_anchor = load_submission(V338_ANCHOR)
    if len(base) != len(public_anchor):
        raise ValueError("V306 and V338 anchor row counts differ")
    _ensure_aligned(base, public_anchor)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    paths = _candidate_paths(SOURCE_DIRS)
    for path in paths:
        audit, skip = _audit_path(path, base, public_anchor)
        if audit is not None:
            rows.append(audit)
        if skip is not None:
            skipped.append(skip)

    audit_frame = pd.DataFrame(rows)
    if not audit_frame.empty:
        audit_frame = audit_frame.sort_values(
            ["family", "point_churn_vs_v306", "source_dir", "candidate"],
            ascending=[True, False, True, True],
        )
        audit_frame["transition_counts"] = audit_frame["transition_counts"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    audit_path = safe_output_path(OUTDIR, "v342_candidate_audit.csv")
    audit_frame.to_csv(audit_path, index=False)

    sanity = historical_sanity(KNOWN_PUBLIC_RECORDS)
    sanity_path = safe_output_path(OUTDIR, "v342_historical_sanity.json")
    write_json(sanity_path, sanity)

    family_counts = audit_frame["family"].value_counts().sort_index().to_dict() if not audit_frame.empty else {}
    report = {
        "version": "V342",
        "decision": "REPORT_ONLY",
        "generated_submission_count": 0,
        "candidate_csvs_scanned": len(paths),
        "schema_valid_candidates": len(rows),
        "skipped_candidates": skipped,
        "family_counts": family_counts,
        "top_churn_candidates": audit_frame.head(10).to_dict(orient="records") if not audit_frame.empty else [],
        "historical_sanity": sanity,
        "outputs": {
            "candidate_audit": relative_path(audit_path),
            "historical_sanity": relative_path(sanity_path),
            "search_report": relative_path(OUTDIR / "search_report.json"),
        },
        "policy": {
            "report_only": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "no_manual_row_edits": True,
            "no_upload_candidates_writes": True,
            "no_submissions_exported": True,
        },
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": relative_path(OUTDIR),
                "decision": report["decision"],
                "schema_valid_candidates": report.get("schema_valid_candidates", 0),
                "family_counts": report.get("family_counts", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
