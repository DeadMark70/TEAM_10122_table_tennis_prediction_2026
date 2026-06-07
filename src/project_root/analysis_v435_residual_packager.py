"""V435 residual packager for anchor-aware clean candidates.

Packages row-level action/point proposals back onto the current clean public
anchor while enforcing conservative competition-safe guards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V434_DIR = ROOT / "v434_anchor_aware_moe_gate"
OUTDIR = ROOT / "v435_residual_packager"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _uid_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _normalize_candidate_table(candidates: pd.DataFrame, candidate_col: str) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=["rally_uid", candidate_col, "utility"])
    out = candidates.copy()
    if "rally_uid" not in out.columns:
        raise ValueError("candidate table must include rally_uid")
    if candidate_col not in out.columns and "candidate_value" in out.columns:
        out[candidate_col] = out["candidate_value"]
    if candidate_col not in out.columns:
        raise ValueError(f"candidate table must include {candidate_col} or candidate_value")
    if "utility" not in out.columns:
        score_cols = [col for col in ("score", "expected_delta", "margin", "confidence") if col in out.columns]
        out["utility"] = pd.to_numeric(out[score_cols[0]], errors="coerce") if score_cols else 1.0
    out["utility"] = pd.to_numeric(out["utility"], errors="coerce").fillna(-np.inf)
    out[candidate_col] = pd.to_numeric(out[candidate_col], errors="coerce")
    out = out.loc[out[candidate_col].notna()].copy()
    out[candidate_col] = out[candidate_col].astype(int)
    return out.sort_values(["utility", "rally_uid"], ascending=[False, True]).reset_index(drop=True)


def apply_ranked_candidates(
    anchor: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    target_col: str,
    candidate_col: str,
    max_changes: int,
    allow_point0_additions: bool = False,
    allow_serve_additions: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply candidate labels to an anchor with strict point/action guards."""

    if target_col not in {"actionId", "pointId"}:
        raise ValueError(f"unsupported target_col: {target_col}")
    out = anchor.copy()
    table = _normalize_candidate_table(candidates, candidate_col)
    by_uid = {_uid_key(uid): idx for idx, uid in enumerate(out["rally_uid"])}
    applied = 0
    blocked_point0 = 0
    blocked_serve = 0
    skipped_same = 0
    skipped_missing = 0
    seen: set[str] = set()

    for _, row in table.iterrows():
        if applied >= max_changes:
            break
        uid = _uid_key(row["rally_uid"])
        if uid in seen:
            continue
        seen.add(uid)
        if uid not in by_uid:
            skipped_missing += 1
            continue
        idx = by_uid[uid]
        old_value = int(out.at[idx, target_col])
        new_value = int(row[candidate_col])
        if new_value == old_value:
            skipped_same += 1
            continue
        if target_col == "pointId" and new_value == 0 and old_value != 0 and not allow_point0_additions:
            blocked_point0 += 1
            continue
        if (
            target_col == "actionId"
            and new_value in SERVE_ACTION_CLASSES
            and old_value not in SERVE_ACTION_CLASSES
            and not allow_serve_additions
        ):
            blocked_serve += 1
            continue
        out.at[idx, target_col] = new_value
        applied += 1

    report = {
        "target_col": target_col,
        "candidate_rows": int(len(table)),
        "max_changes": int(max_changes),
        "applied_changes": int(applied),
        "blocked_point0_additions": int(blocked_point0),
        "blocked_serve_additions": int(blocked_serve),
        "skipped_same": int(skipped_same),
        "skipped_missing_rally_uid": int(skipped_missing),
    }
    return out, report


def package_residual_submission(
    anchor: pd.DataFrame,
    *,
    action_candidates: pd.DataFrame | None = None,
    point_candidates: pd.DataFrame | None = None,
    action_top: int = 0,
    point_top: int = 0,
    name: str = "candidate",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    submission = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    reports: dict[str, Any] = {"name": name}
    if action_candidates is not None and action_top > 0:
        submission, action_report = apply_ranked_candidates(
            submission,
            action_candidates,
            target_col="actionId",
            candidate_col="candidate_actionId",
            max_changes=action_top,
        )
        reports["action"] = action_report
    if point_candidates is not None and point_top > 0:
        submission, point_report = apply_ranked_candidates(
            submission,
            point_candidates,
            target_col="pointId",
            candidate_col="candidate_pointId",
            max_changes=point_top,
        )
        reports["point"] = point_report

    base = anchor.loc[:, SUBMISSION_COLUMNS]
    changed_mask = (submission["actionId"].astype(int).to_numpy() != base["actionId"].astype(int).to_numpy()) | (
        submission["pointId"].astype(int).to_numpy() != base["pointId"].astype(int).to_numpy()
    )
    reports["total_changed_rows"] = int(changed_mask.sum())
    reports["server_preserved"] = bool(
        np.allclose(
            pd.to_numeric(submission["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(base["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
        )
    )
    validate_submission_schema(submission, expected_rows=None if len(submission) != 1845 else 1845)
    return submission, reports


def _read_optional_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_packager() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_schema(anchor)
    action_candidates = _read_optional_candidates(V434_DIR / "moe_action_candidates.csv")
    point_candidates = _read_optional_candidates(V434_DIR / "moe_point_candidates.csv")

    configs = [
        ("top5", 5, 5),
        ("top10", 10, 10),
        ("top20", 20, 20),
        ("point_top20", 0, 20),
        ("action_top20", 20, 0),
    ]
    reports: list[dict[str, Any]] = []
    exported: list[str] = []
    for name, action_top, point_top in configs:
        submission, report = package_residual_submission(
            anchor,
            action_candidates=action_candidates,
            point_candidates=point_candidates,
            action_top=action_top,
            point_top=point_top,
            name=name,
        )
        filename = f"submission_v435_{name}__v362anchor.csv"
        path = safe_output_path(OUTDIR, filename)
        submission.to_csv(path, index=False)
        report["filename"] = filename
        reports.append(report)
        exported.append(str(path))

    pd.DataFrame(reports).to_csv(OUTDIR / "packaging_report.csv", index=False)
    summary = {
        "anchor": str(ANCHOR_PATH),
        "action_candidate_rows": int(len(action_candidates)),
        "point_candidate_rows": int(len(point_candidates)),
        "exports": exported,
        "reports": reports,
    }
    write_json(OUTDIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_packager()
    print(json.dumps(_json_safe({"outdir": str(OUTDIR), "exports": len(result["exports"])}), indent=2))
