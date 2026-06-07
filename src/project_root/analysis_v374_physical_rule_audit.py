"""V374 physical-rule audit for V370/V371/V372 candidates.

This is a read-only research/audit script. It does not generate upload
submissions. It compares candidate changes against the current clean anchor and
scores whether each row-level change is tactically plausible from action/point
depth/side/context features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v374_physical_rule_audit"
ANCHOR = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
TEST_NEW = ROOT / "test_new.csv"
CANDIDATES = {
    "v370_safe": ROOT / "v370_point_breakthrough_pool" / "submission_v370_point_pool_safe__v173action_v300server.csv",
    "v370_medium": ROOT / "v370_point_breakthrough_pool" / "submission_v370_point_pool_medium__v173action_v300server.csv",
    "v371_joint_low": ROOT / "v371_joint_causal_consistency_lab" / "submission_v371_joint_consistency_low__v300server.csv",
    "v372_action_b05": ROOT / "v372_action_weakness_redux" / "submission_v372_action_weak_safe_b05__v338point_v300server.csv",
}


def point_depth(point: int) -> str:
    if point == 0:
        return "terminal"
    if point in {1, 2, 3}:
        return "short"
    if point in {4, 5, 6}:
        return "half"
    return "long"


def point_side(point: int) -> str:
    if point == 0:
        return "terminal"
    rem = (int(point) - 1) % 3
    return ["left", "middle", "right"][rem]


def action_family(action: int) -> str:
    action = int(action)
    if action == 0:
        return "zero"
    if 1 <= action <= 7:
        return "attack"
    if 8 <= action <= 11:
        return "control"
    if 12 <= action <= 14:
        return "defensive"
    if 15 <= action <= 18:
        return "serve"
    return "unknown"


def point_action_physical_score(action: int, point: int) -> tuple[float, str]:
    family = action_family(action)
    depth = point_depth(point)
    if point == 0:
        if family in {"zero", "attack", "defensive"}:
            return 1.0, "terminal-compatible"
        return -0.5, "terminal-with-control"
    if family == "zero":
        return -2.0, "zero-action-nonterminal"
    if family == "serve":
        return -2.0, "serve-like-hidden-action"
    if family == "attack":
        return (1.0, "attack-long") if depth == "long" else (0.2, f"attack-{depth}")
    if family == "control":
        if action == 10 and depth == "long":
            return 1.0, "long-push-long"
        if action == 11 and depth == "short":
            return 1.0, "drop-short"
        return (0.6, f"control-{depth}") if depth in {"short", "half"} else (0.1, "control-long")
    if family == "defensive":
        return (0.8, f"defensive-{depth}") if depth in {"long", "terminal"} else (0.1, f"defensive-{depth}")
    return 0.0, "unknown"


def transition_physical_score(old_point: int, new_point: int) -> tuple[float, str]:
    old_depth, new_depth = point_depth(old_point), point_depth(new_point)
    old_side, new_side = point_side(old_point), point_side(new_point)
    if old_point == new_point:
        return 0.0, "no-change"
    if old_point == 0 and new_point != 0:
        return -0.8, "point0-removal-risk"
    if old_point != 0 and new_point == 0:
        return -1.0, "point0-addition-risk"
    if old_depth == new_depth and old_side != new_side:
        if new_side == "middle":
            return 0.9, "same-depth-to-middle"
        return 0.7, "same-depth-side-shift"
    if old_side == new_side and old_depth != new_depth:
        if new_depth == "long":
            return 0.5, "same-side-deeper"
        return 0.2, "same-side-shorter"
    if old_depth != new_depth and old_side != new_side:
        return -0.1, "depth-and-side-change"
    return 0.0, "neutral"


def verdict(score: float) -> str:
    if score >= 1.4:
        return "keep"
    if score <= -0.2:
        return "reject"
    return "uncertain"


def latest_context() -> pd.DataFrame:
    test = pd.read_csv(TEST_NEW)
    latest = (
        test.sort_values(["rally_uid", "strikeNumber"], kind="mergesort")
        .groupby("rally_uid", as_index=False)
        .tail(1)
        .copy()
    )
    latest = latest.rename(
        columns={
            "actionId": "last_action",
            "pointId": "last_point",
            "spinId": "last_spin",
            "strengthId": "last_strength",
            "strikeNumber": "prefix_len",
        }
    )
    keep = [
        "rally_uid",
        "match",
        "numberGame",
        "prefix_len",
        "scoreSelf",
        "scoreOther",
        "last_action",
        "last_point",
        "last_spin",
        "last_strength",
        "positionId",
    ]
    return latest.loc[:, [c for c in keep if c in latest.columns]]


def candidate_audit(name: str, path: Path, anchor: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    cand = pd.read_csv(path)
    merged = anchor.merge(cand, on="rally_uid", suffixes=("_base", "_cand"))
    changed = merged[
        (merged["actionId_base"] != merged["actionId_cand"])
        | (merged["pointId_base"] != merged["pointId_cand"])
    ].copy()
    if changed.empty:
        return pd.DataFrame()
    changed = changed.merge(context, on="rally_uid", how="left")
    rows: list[dict[str, Any]] = []
    for _, row in changed.iterrows():
        base_action = int(row["actionId_base"])
        cand_action = int(row["actionId_cand"])
        base_point = int(row["pointId_base"])
        cand_point = int(row["pointId_cand"])
        trans_score, trans_reason = transition_physical_score(base_point, cand_point)
        compat_score, compat_reason = point_action_physical_score(cand_action, cand_point)
        total = trans_score + compat_score
        rows.append(
            {
                "candidate": name,
                "rally_uid": row["rally_uid"],
                "base_action": base_action,
                "cand_action": cand_action,
                "action_transition": f"{base_action}->{cand_action}",
                "base_point": base_point,
                "cand_point": cand_point,
                "point_transition": f"{base_point}->{cand_point}",
                "base_depth": point_depth(base_point),
                "cand_depth": point_depth(cand_point),
                "base_side": point_side(base_point),
                "cand_side": point_side(cand_point),
                "cand_action_family": action_family(cand_action),
                "transition_reason": trans_reason,
                "compat_reason": compat_reason,
                "physical_score": total,
                "verdict": verdict(total),
                "prefix_len": row.get("prefix_len"),
                "last_action": row.get("last_action"),
                "last_point": row.get("last_point"),
                "last_spin": row.get("last_spin"),
                "last_strength": row.get("last_strength"),
                "scoreSelf": row.get("scoreSelf"),
                "scoreOther": row.get("scoreOther"),
            }
        )
    return pd.DataFrame(rows)


def package_from_audit(
    anchor: pd.DataFrame,
    candidate_path: Path,
    audit: pd.DataFrame,
    candidate_name: str,
    allowed_verdicts: set[str],
    output_name: str,
    keep_action: bool = True,
    keep_point: bool = True,
    require_action_change: bool = False,
    require_point_change: bool = False,
) -> dict[str, Any]:
    candidate = pd.read_csv(candidate_path)
    out = anchor.copy()
    allowed = audit[
        (audit["candidate"] == candidate_name) & (audit["verdict"].isin(sorted(allowed_verdicts)))
    ].copy()
    if require_action_change:
        allowed = allowed[allowed["base_action"].astype(int) != allowed["cand_action"].astype(int)]
    if require_point_change:
        allowed = allowed[allowed["base_point"].astype(int) != allowed["cand_point"].astype(int)]
    allowed_uids = set(allowed["rally_uid"].astype(anchor["rally_uid"].dtype, errors="ignore"))
    mask = out["rally_uid"].isin(allowed_uids)
    cand_aligned = out[["rally_uid"]].merge(candidate, on="rally_uid", how="left", suffixes=("", "_cand"))
    if keep_action:
        out.loc[mask, "actionId"] = cand_aligned.loc[mask, "actionId"].astype(int).to_numpy()
    if keep_point:
        out.loc[mask, "pointId"] = cand_aligned.loc[mask, "pointId"].astype(int).to_numpy()
    out_path = OUTDIR / output_name
    out.to_csv(out_path, index=False)
    return {
        "name": output_name,
        "path": str(out_path.relative_to(ROOT)),
        "source_candidate": candidate_name,
        "allowed_verdicts": sorted(allowed_verdicts),
        "selected_rows": int(mask.sum()),
        "action_churn_vs_anchor": int((out["actionId"].astype(int) != anchor["actionId"].astype(int)).sum()),
        "point_churn_vs_anchor": int((out["pointId"].astype(int) != anchor["pointId"].astype(int)).sum()),
        "point0_additions": int(((anchor["pointId"].astype(int) != 0) & (out["pointId"].astype(int) == 0)).sum()),
        "server_changed": int(
            (pd.to_numeric(out["serverGetPoint"]) - pd.to_numeric(anchor["serverGetPoint"])).abs().gt(1e-12).sum()
        ),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR)
    context = latest_context()
    all_rows = []
    for name, path in CANDIDATES.items():
        if path.exists():
            audit = candidate_audit(name, path, anchor, context)
            if not audit.empty:
                all_rows.append(audit)
    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    audit_path = OUTDIR / "physical_rule_audit.csv"
    full.to_csv(audit_path, index=False)
    summary = (
        full.groupby(["candidate", "verdict"], dropna=False)
        .size()
        .reset_index(name="rows")
        if not full.empty
        else pd.DataFrame(columns=["candidate", "verdict", "rows"])
    )
    summary_path = OUTDIR / "physical_rule_summary.csv"
    summary.to_csv(summary_path, index=False)
    generated: list[dict[str, Any]] = []
    if not full.empty:
        generated.append(
            package_from_audit(
                anchor,
                CANDIDATES["v370_safe"],
                full,
                "v370_safe",
                {"keep"},
        "submission_v374_v370_safe_keep_only__v173action_v300server.csv",
        keep_action=False,
        keep_point=True,
        require_point_change=True,
            )
        )
        generated.append(
            package_from_audit(
                anchor,
                CANDIDATES["v370_medium"],
                full,
                "v370_medium",
                {"keep"},
        "submission_v374_v370_medium_keep_only__v173action_v300server.csv",
        keep_action=False,
        keep_point=True,
        require_point_change=True,
            )
        )
        generated.append(
            package_from_audit(
                anchor,
                CANDIDATES["v372_action_b05"],
                full,
                "v372_action_b05",
                {"keep", "uncertain"},
        "submission_v374_v372_action_b05_repack_on_v362point__v300server.csv",
        keep_action=True,
        keep_point=False,
        require_action_change=True,
            )
        )
    report = {
        "version": "V374",
        "anchor": str(ANCHOR.relative_to(ROOT)),
        "audit_rows": int(len(full)),
        "summary": summary.to_dict(orient="records"),
        "outputs": {
            "audit": str(audit_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
        },
        "generated_submissions": generated,
        "note": "Heuristic physical plausibility audit with derived local candidates; public validation still required.",
    }
    (OUTDIR / "search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
