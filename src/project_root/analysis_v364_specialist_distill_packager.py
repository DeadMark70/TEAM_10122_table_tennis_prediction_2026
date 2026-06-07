"""V364 distill packager for hierarchical specialist candidates.

This script does not train models. It scans V361/V362 specialist outputs,
checks them against the current V338 clean anchor, applies the V360-style
policy, deduplicates equivalent prediction files, and writes a ranked upload
recommendation. It intentionally blocks TTMATCH and old-server candidates.
"""

from __future__ import annotations

import hashlib
import json
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
OUTDIR = ROOT / "v364_specialist_distill_packager"
ANCHOR_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
CANDIDATE_DIRS = [
    ROOT / "v362_point_hierarchical_specialists",
    ROOT / "v361_action_hierarchical_specialists",
]
BANNED_NAME_PARTS = ("ttmatch", "oldserver", "old_server", "oldhard", "oldsharpen")


def candidate_name_allowed(name: str) -> bool:
    lower = str(name).lower()
    return not any(part in lower for part in BANNED_NAME_PARTS)


def prediction_signature(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, SUBMISSION_COLUMNS].copy()
    ordered = ordered.sort_values("rally_uid", kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _load_anchor(path: Path = ANCHOR_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing V338 anchor: {path}")
    frame = pd.read_csv(path)
    validate_submission_schema(frame)
    return frame


def _candidate_family(path: Path) -> str:
    lower = path.as_posix().lower()
    if "v362" in lower:
        if "research" in lower:
            return "point_research"
        return "point_only_safe"
    if "v361" in lower:
        if "research" in lower:
            return "action_research"
        return "action_only_safe"
    return "unknown"


def _diff(anchor: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base = anchor.sort_values("rally_uid", kind="mergesort").reset_index(drop=True)
    cand = candidate.sort_values("rally_uid", kind="mergesort").reset_index(drop=True)
    if not base["rally_uid"].equals(cand["rally_uid"]):
        raise ValueError("candidate rally_uid order/set does not match anchor")

    action_changed = base["actionId"].astype(int) != cand["actionId"].astype(int)
    point_changed = base["pointId"].astype(int) != cand["pointId"].astype(int)
    server_changed = ~np.isclose(
        pd.to_numeric(base["serverGetPoint"], errors="coerce").to_numpy(float),
        pd.to_numeric(cand["serverGetPoint"], errors="coerce").to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )
    point0_add = (base["pointId"].astype(int) != 0) & (cand["pointId"].astype(int) == 0)
    serve_like = cand["actionId"].astype(int).isin([15, 16, 17, 18])
    base_serve_like = base["actionId"].astype(int).isin([15, 16, 17, 18])
    return {
        "action_churn_vs_v338": int(action_changed.sum()),
        "point_churn_vs_v338": int(point_changed.sum()),
        "server_changed": int(server_changed.sum()),
        "point0_additions": int(point0_add.sum()),
        "serve_like_delta": int(serve_like.sum() - base_serve_like.sum()),
        "changed_rows": int((action_changed | point_changed | server_changed).sum()),
    }


def _policy_score(row: dict[str, Any]) -> float:
    score = 0.0
    family = row.get("family", "")
    if family == "point_only_safe":
        score += 8.0
    elif family == "action_only_safe":
        score += 4.0
    elif "research" in family:
        score -= 10.0
    score -= float(row.get("point0_additions", 0)) * 4.0
    score -= float(row.get("server_changed", 0)) * 100.0
    score -= max(0.0, float(row.get("serve_like_delta", 0))) * 5.0
    score -= float(row.get("point_churn_vs_v338", 0)) * 0.08
    score -= float(row.get("action_churn_vs_v338", 0)) * 0.04
    score += float(row.get("local_score_hint", 0.0))
    return score


def _risk_tier(row: dict[str, Any]) -> str:
    if row.get("policy_blocked"):
        return "blocked"
    if row.get("server_changed", 0) > 0:
        return "blocked"
    if row.get("point0_additions", 0) > 0:
        return "research"
    if row.get("point_churn_vs_v338", 0) <= 5 and row.get("action_churn_vs_v338", 0) <= 10:
        return "safe"
    if row.get("point_churn_vs_v338", 0) <= 15 and row.get("action_churn_vs_v338", 0) <= 40:
        return "normal"
    return "research"


def _scan_candidate_csvs(candidate_dirs: list[Path] | None = None) -> list[Path]:
    dirs = candidate_dirs or CANDIDATE_DIRS
    out: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        out.extend(sorted(directory.glob("submission_*.csv")))
    return out


def build_candidate_table(anchor: pd.DataFrame, candidate_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidate_paths:
        name = path.name
        allowed = candidate_name_allowed(name)
        try:
            frame = pd.read_csv(path)
            validate_submission_schema(frame)
            sig = prediction_signature(frame)
            duplicate = sig in seen
            if not duplicate:
                seen.add(sig)
            diff = _diff(anchor, frame)
            row = {
                "name": path.stem,
                "path": _rel(path),
                "family": _candidate_family(path),
                "signature": sig,
                "duplicate_prediction": bool(duplicate),
                "policy_blocked": not allowed,
                **diff,
            }
        except Exception as exc:
            row = {
                "name": path.stem,
                "path": _rel(path),
                "family": _candidate_family(path),
                "signature": "",
                "duplicate_prediction": False,
                "policy_blocked": True,
                "error": f"{type(exc).__name__}: {exc}",
                "action_churn_vs_v338": 999999,
                "point_churn_vs_v338": 999999,
                "server_changed": 999999,
                "point0_additions": 999999,
                "serve_like_delta": 999999,
                "changed_rows": 999999,
            }
        row["risk_tier"] = _risk_tier(row)
        row["score"] = _policy_score(row)
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "name",
                "path",
                "family",
                "score",
                "risk_tier",
                "policy_blocked",
                "duplicate_prediction",
            ]
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["policy_blocked", "duplicate_prediction", "score", "changed_rows"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return table


def recommendation_text(ranked: pd.DataFrame) -> str:
    usable = ranked[
        (~ranked.get("policy_blocked", pd.Series(False, index=ranked.index)).astype(bool))
        & (~ranked.get("duplicate_prediction", pd.Series(False, index=ranked.index)).astype(bool))
        & (ranked.get("risk_tier", pd.Series("", index=ranked.index)) != "research")
    ].copy()
    lines = [
        "# V364 Specialist Distill Recommendation",
        "",
        "Current public best remains V338 unless a listed candidate is uploaded and beats it.",
        "",
    ]
    if usable.empty:
        lines.extend(
            [
                "Recommendation: HOLD V338.",
                "",
                "No clean V361/V362 candidate passed the V364 policy gate.",
            ]
        )
    else:
        lines.append("Top local-gate candidates:")
        lines.append("")
        for idx, row in usable.head(5).iterrows():
            lines.append(
                f"{idx + 1}. `{row['path']}` "
                f"(tier={row['risk_tier']}, score={row['score']:.3f}, "
                f"point_churn={int(row.get('point_churn_vs_v338', 0))}, "
                f"action_churn={int(row.get('action_churn_vs_v338', 0))})"
            )
        lines.append("")
        lines.append("Use scarce quota only on the first candidate unless new evidence arrives.")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(
    outdir: Path = OUTDIR,
    anchor_path: Path = ANCHOR_PATH,
    candidate_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    anchor = _load_anchor(anchor_path)
    candidate_paths = _scan_candidate_csvs(candidate_dirs)
    ranked = build_candidate_table(anchor, candidate_paths)

    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    priority_path = safe_output_path(outdir, "next_upload_priority.csv")
    reco_path = safe_output_path(outdir, "recommendation.md")
    report_path = safe_output_path(outdir, "search_report.json")

    ranked.to_csv(ranked_path, index=False)
    usable = ranked[
        (~ranked.get("policy_blocked", pd.Series(False, index=ranked.index)).astype(bool))
        & (~ranked.get("duplicate_prediction", pd.Series(False, index=ranked.index)).astype(bool))
        & (ranked.get("risk_tier", pd.Series("", index=ranked.index)) != "research")
    ].head(5)
    usable.to_csv(priority_path, index=False)
    reco_path.write_text(recommendation_text(ranked), encoding="utf-8")

    report = {
        "version": "V364",
        "anchor": _rel(anchor_path),
        "candidate_count": int(len(ranked)),
        "usable_candidate_count": int(len(usable)),
        "top_candidate": None if usable.empty else usable.iloc[0].to_dict(),
        "outputs": {
            "ranked_candidates": _rel(ranked_path),
            "next_upload_priority": _rel(priority_path),
            "recommendation": _rel(reco_path),
        },
        "v360_report": _read_json(ROOT / "v360_hierarchical_specialist_gate" / "search_report.json"),
        "v363_report": _read_json(ROOT / "v363_clean_representation_features" / "search_report.json"),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
