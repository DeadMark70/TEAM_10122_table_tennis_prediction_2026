"""V373 final packager for the breakthrough ensemble lab.

The packager scans V370/V371/V372 candidate submissions, validates them
against the current clean anchor, blocks policy-violating candidates, dedupes
identical predictions, and writes a ranked recommendation. It does not train
models or edit predictions manually.
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
OUTDIR = ROOT / "v373_breakthrough_packager"
PRIMARY_ANCHOR = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
FALLBACK_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
CANDIDATE_DIRS = [
    ROOT / "v370_point_breakthrough_pool",
    ROOT / "v371_joint_causal_consistency_lab",
    ROOT / "v372_action_weakness_redux",
    ROOT / "v374_physical_rule_audit",
    ROOT / "v362_point_hierarchical_specialists",
]
BANNED_PARTS = ("ttmatch", "oldserver", "old_server", "oldhard", "oldsharpen")


def candidate_allowed(name: str) -> bool:
    lower = str(name).lower()
    return not any(part in lower for part in BANNED_PARTS)


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


def _load_anchor() -> tuple[pd.DataFrame, Path]:
    path = PRIMARY_ANCHOR if PRIMARY_ANCHOR.exists() else FALLBACK_ANCHOR
    if not path.exists():
        raise FileNotFoundError(f"missing anchor: {PRIMARY_ANCHOR} or {FALLBACK_ANCHOR}")
    frame = pd.read_csv(path)
    validate_submission_schema(frame)
    return frame, path


def _scan_candidates() -> list[Path]:
    paths: list[Path] = []
    for directory in CANDIDATE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("submission_*.csv")):
            if path.resolve() in {PRIMARY_ANCHOR.resolve(), FALLBACK_ANCHOR.resolve()}:
                continue
            # Avoid rescoring every V362 research output if not explicitly useful.
            paths.append(path)
    return paths


def _family(path: Path) -> str:
    text = path.as_posix().lower()
    if "v374" in text:
        if "v370" in path.name.lower():
            return "physical_point_audit"
        if "v372" in path.name.lower():
            return "physical_action_audit"
        return "physical_audit"
    if "v370" in text:
        return "point_pool"
    if "v371" in text:
        return "joint_consistency" if "joint" in path.name.lower() else "point_consistency"
    if "v372" in text:
        return "action_weakness"
    if "v362" in text:
        return "v362_reference"
    return "unknown"


def _diff(anchor: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base = anchor.sort_values("rally_uid", kind="mergesort").reset_index(drop=True)
    cand = candidate.sort_values("rally_uid", kind="mergesort").reset_index(drop=True)
    if not base["rally_uid"].equals(cand["rally_uid"]):
        raise ValueError("candidate rally_uid set/order differs from anchor")
    action_changed = base["actionId"].astype(int) != cand["actionId"].astype(int)
    point_changed = base["pointId"].astype(int) != cand["pointId"].astype(int)
    server_changed = ~np.isclose(
        pd.to_numeric(base["serverGetPoint"], errors="coerce").to_numpy(float),
        pd.to_numeric(cand["serverGetPoint"], errors="coerce").to_numpy(float),
        rtol=0,
        atol=1e-12,
    )
    point0_add = (base["pointId"].astype(int) != 0) & (cand["pointId"].astype(int) == 0)
    serve_delta = int(cand["actionId"].astype(int).isin([15, 16, 17, 18]).sum()) - int(
        base["actionId"].astype(int).isin([15, 16, 17, 18]).sum()
    )
    return {
        "action_churn_vs_anchor": int(action_changed.sum()),
        "point_churn_vs_anchor": int(point_changed.sum()),
        "server_changed": int(server_changed.sum()),
        "point0_additions": int(point0_add.sum()),
        "serve_like_delta": serve_delta,
        "changed_rows": int((action_changed | point_changed | server_changed).sum()),
    }


def _score(row: dict[str, Any]) -> float:
    score = 0.0
    family = row.get("family", "")
    if family in {"point_pool", "point_consistency", "v362_reference", "physical_point_audit"}:
        score += 8.0
    elif family == "joint_consistency":
        score += 4.0
    elif family in {"action_weakness", "physical_action_audit"}:
        score += 2.0
    if family.startswith("physical"):
        score += 1.0
    if "safe" in row.get("name", "").lower():
        score += 1.0
    if "medium" in row.get("name", "").lower():
        score -= 0.5
    if "aggressive" in row.get("name", "").lower() or "research" in row.get("name", "").lower():
        score -= 8.0
    score -= float(row.get("point0_additions", 0)) * 5.0
    score -= float(row.get("server_changed", 0)) * 100.0
    score -= max(0, float(row.get("serve_like_delta", 0))) * 5.0
    score -= float(row.get("point_churn_vs_anchor", 0)) * 0.05
    score -= float(row.get("action_churn_vs_anchor", 0)) * 0.08
    return score


def _risk(row: dict[str, Any]) -> str:
    if row.get("policy_blocked") or row.get("server_changed", 0) > 0:
        return "blocked"
    if "research" in row.get("name", "").lower() or "aggressive" in row.get("name", "").lower():
        return "research"
    if row.get("point0_additions", 0) > 0:
        return "research"
    if row.get("point_churn_vs_anchor", 0) <= 20 and row.get("action_churn_vs_anchor", 0) <= 10:
        return "safe"
    if row.get("point_churn_vs_anchor", 0) <= 50 and row.get("action_churn_vs_anchor", 0) <= 40:
        return "normal"
    return "research"


def build_ranked_table(anchor: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path)
            validate_submission_schema(frame)
            sig = prediction_signature(frame)
            row = {
                "name": path.stem,
                "path": _rel(path),
                "family": _family(path),
                "signature": sig,
                "duplicate_prediction": sig in seen,
                "policy_blocked": not candidate_allowed(path.name),
                **_diff(anchor, frame),
            }
            seen.add(sig)
        except Exception as exc:
            row = {
                "name": path.stem,
                "path": _rel(path),
                "family": _family(path),
                "signature": "",
                "duplicate_prediction": False,
                "policy_blocked": True,
                "error": f"{type(exc).__name__}: {exc}",
                "action_churn_vs_anchor": 999999,
                "point_churn_vs_anchor": 999999,
                "server_changed": 999999,
                "point0_additions": 999999,
                "serve_like_delta": 999999,
                "changed_rows": 999999,
            }
        row["risk_tier"] = _risk(row)
        row["score"] = _score(row)
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["name", "path", "family", "risk_tier", "score", "policy_blocked"]
        )
    table = pd.DataFrame(rows)
    table["tier_rank"] = table["risk_tier"].map(
        {"safe": 0, "normal": 1, "research": 2, "blocked": 3}
    ).fillna(3)
    return table.sort_values(
        ["policy_blocked", "duplicate_prediction", "tier_rank", "score", "changed_rows"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def top_usable_candidates(ranked: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()
    mask = (
        ~ranked.get("policy_blocked", pd.Series(False, index=ranked.index)).astype(bool)
        & ~ranked.get("duplicate_prediction", pd.Series(False, index=ranked.index)).astype(bool)
        & (ranked.get("risk_tier", pd.Series("", index=ranked.index)) != "research")
        & (pd.to_numeric(ranked.get("changed_rows", pd.Series(0, index=ranked.index)), errors="coerce").fillna(0) > 0)
    )
    usable = ranked.loc[mask].copy()
    if "tier_rank" not in usable.columns:
        usable["tier_rank"] = usable["risk_tier"].map(
            {"safe": 0, "normal": 1, "research": 2, "blocked": 3}
        ).fillna(3)
    return usable.sort_values(
        ["score", "tier_rank", "changed_rows"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(limit)


def _recommendation(ranked: pd.DataFrame, anchor_path: Path) -> str:
    usable = top_usable_candidates(ranked, limit=5)
    lines = [
        "# V373 Breakthrough Packager Recommendation",
        "",
        f"Anchor: `{_rel(anchor_path)}`",
        "",
    ]
    if usable.empty:
        lines.append("Recommendation: HOLD current anchor. No usable candidate passed policy.")
    else:
        lines.append("Top candidates:")
        lines.append("")
        for pos, (_, row) in enumerate(usable.iterrows(), start=1):
            lines.append(
                f"{pos}. `{row['path']}` "
                f"(family={row['family']}, tier={row['risk_tier']}, "
                f"score={row['score']:.3f}, point_churn={int(row['point_churn_vs_anchor'])}, "
                f"action_churn={int(row['action_churn_vs_anchor'])})"
            )
        lines.append("")
        lines.append("Only upload the first candidate if quota is scarce.")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    anchor, anchor_path = _load_anchor()
    paths = _scan_candidates()
    ranked = build_ranked_table(anchor, paths)

    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    priority_path = safe_output_path(outdir, "next_upload_priority.csv")
    reco_path = safe_output_path(outdir, "recommendation.md")
    report_path = safe_output_path(outdir, "search_report.json")

    ranked.to_csv(ranked_path, index=False)
    usable = top_usable_candidates(ranked, limit=5)
    usable.to_csv(priority_path, index=False)
    reco_path.write_text(_recommendation(ranked, anchor_path), encoding="utf-8")

    report = {
        "version": "V373",
        "anchor": _rel(anchor_path),
        "candidate_count": int(len(ranked)),
        "usable_candidate_count": int(len(usable)),
        "top_candidate": None if usable.empty else usable.iloc[0].to_dict(),
        "source_reports": {
            "v370": _read_json(ROOT / "v370_point_breakthrough_pool" / "search_report.json"),
            "v371": _read_json(ROOT / "v371_joint_causal_consistency_lab" / "search_report.json"),
            "v372": _read_json(ROOT / "v372_action_weakness_redux" / "search_report.json"),
        },
        "outputs": {
            "ranked_candidates": _rel(ranked_path),
            "next_upload_priority": _rel(priority_path),
            "recommendation": _rel(reco_path),
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
