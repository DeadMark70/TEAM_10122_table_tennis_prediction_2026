"""V351 V338 pruning trust model.

This script scores only the rows already changed by the public-positive V338
point candidate, then exports conservative candidates that revert low-trust
rows back to the V306 point anchor. It never adds new point rows and never
changes action/server columns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v351_v338_pruning_trust_model"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_ANCHOR = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V348_SCORES = ROOT / "v348_public_risk_row_gate" / "row_gate_scores.csv"
V347_TRANSITIONS = ROOT / "v347_v338_v341_diff_audit" / "transition_summary.csv"
V345_DIR = ROOT / "v345_nonpoint0_utility_optimizer"


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def changed_row_ids(base: pd.DataFrame, candidate: pd.DataFrame) -> set[int]:
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs")
    return set(np.where(base["pointId"].to_numpy(dtype=int) != candidate["pointId"].to_numpy(dtype=int))[0].astype(int))


def changed_transition_frame(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs")
    rows: list[dict[str, Any]] = []
    base_point = base["pointId"].to_numpy(dtype=int)
    cand_point = candidate["pointId"].to_numpy(dtype=int)
    for row_id in np.where(base_point != cand_point)[0]:
        rows.append(
            {
                "row_id": int(row_id),
                "rally_uid": base["rally_uid"].iloc[int(row_id)],
                "anchor_value": int(base_point[row_id]),
                "candidate_value": int(cand_point[row_id]),
                "transition": f"{int(base_point[row_id])}->{int(cand_point[row_id])}",
            }
        )
    return pd.DataFrame(rows)


def _load_selected_rows(path: Path) -> set[int]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "row_id" not in frame.columns:
        return set()
    return set(pd.to_numeric(frame["row_id"], errors="coerce").dropna().astype(int).tolist())


def load_selected_sets(directory: Path = V345_DIR) -> dict[str, set[int]]:
    return {
        "b12": _load_selected_rows(directory / "selected_b12.csv"),
        "b18": _load_selected_rows(directory / "selected_b18.csv"),
        "b24": _load_selected_rows(directory / "selected_b24.csv"),
        "b36": _load_selected_rows(directory / "selected_b36.csv"),
    }


def load_transition_penalty(path: Path = V347_TRANSITIONS) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "transition" not in frame.columns:
        return {}
    count_column = None
    for candidate in ("v341_extra_count", "extra_v341_count", "v341_extra_rows", "count_v341_extra"):
        if candidate in frame.columns:
            count_column = candidate
            break
    if count_column is None:
        numeric_cols = [col for col in frame.columns if col != "transition" and pd.api.types.is_numeric_dtype(frame[col])]
        count_column = numeric_cols[0] if numeric_cols else None
    if count_column is None:
        return {}
    return {
        str(row["transition"]): int(row[count_column])
        for _, row in frame.iterrows()
        if pd.notna(row.get(count_column)) and int(row[count_column]) > 0
    }


def _gate_lookup(gate_scores: pd.DataFrame) -> pd.DataFrame:
    frame = gate_scores.copy()
    for column in ("row_id", "anchor_value", "candidate_value"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    if "trust_score" not in frame.columns:
        frame["trust_score"] = 0.0
    if "risk_score" not in frame.columns:
        frame["risk_score"] = 0.0
    if "agreement_count" not in frame.columns:
        frame["agreement_count"] = 1
    keep = ["row_id", "anchor_value", "candidate_value", "trust_score", "risk_score", "agreement_count"]
    if "gate_decision" in frame.columns:
        keep.append("gate_decision")
    if "transition" in frame.columns:
        keep.append("transition")
    frame = frame.loc[:, [col for col in keep if col in frame.columns]].dropna(subset=["row_id", "anchor_value", "candidate_value"])
    return frame


def score_v338_rows(
    v306: pd.DataFrame,
    v338: pd.DataFrame,
    gate_scores: pd.DataFrame,
    selected_sets: dict[str, set[int]],
    transition_penalty: dict[str, int],
) -> pd.DataFrame:
    """Score rows already changed by V338, highest first."""

    changed = changed_transition_frame(v306, v338)
    if changed.empty:
        return changed

    gates = _gate_lookup(gate_scores)
    scored = changed.merge(
        gates,
        on=["row_id", "anchor_value", "candidate_value"],
        how="left",
        suffixes=("", "_gate"),
    )
    scored["trust_score"] = pd.to_numeric(scored.get("trust_score", 0.0), errors="coerce").fillna(0.0)
    scored["risk_score"] = pd.to_numeric(scored.get("risk_score", 0.0), errors="coerce").fillna(0.0)
    scored["agreement_count"] = pd.to_numeric(scored.get("agreement_count", 1), errors="coerce").fillna(1).astype(int)

    for name in ("b12", "b18", "b24", "b36"):
        rows = selected_sets.get(name, set())
        scored[f"in_{name}"] = scored["row_id"].astype(int).isin(rows)

    scored["v341_transition_extra_count"] = scored["transition"].map(lambda key: int(transition_penalty.get(str(key), 0)))
    scored["subset_support"] = (
        scored["in_b12"].astype(float) * 1.00
        + scored["in_b18"].astype(float) * 0.65
        + scored["in_b24"].astype(float) * 0.25
        + scored["in_b36"].astype(float) * 0.05
    )
    scored["transition_penalty"] = scored["v341_transition_extra_count"].clip(upper=6).astype(float) * 0.08
    scored["final_trust_score"] = (
        scored["trust_score"].astype(float)
        + scored["subset_support"].astype(float)
        + 0.08 * scored["agreement_count"].astype(float)
        - scored["risk_score"].astype(float)
        - scored["transition_penalty"].astype(float)
    )
    scored["rank_desc"] = scored["final_trust_score"].rank(method="first", ascending=False).astype(int)
    return scored.sort_values(
        ["final_trust_score", "trust_score", "subset_support", "row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_pruned_submission(v306: pd.DataFrame, v338: pd.DataFrame, rows_to_revert: Iterable[int]) -> pd.DataFrame:
    if not v306["rally_uid"].equals(v338["rally_uid"]):
        raise ValueError("submission row order differs")
    output = v338.copy()
    for row_id in sorted(set(int(row) for row in rows_to_revert)):
        if row_id < 0 or row_id >= len(output):
            raise ValueError(f"row_id outside submission: {row_id}")
        output.loc[row_id, "pointId"] = int(v306.loc[row_id, "pointId"])
    return output.loc[:, SUBMISSION_COLUMNS].copy()


def candidate_metrics(v306: pd.DataFrame, v338: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    validate_submission_schema(candidate, expected_rows=len(v306))
    if not v306["rally_uid"].equals(candidate["rally_uid"]) or not v338["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs")
    v306_changed = changed_row_ids(v306, candidate)
    v338_changed = changed_row_ids(v338, candidate)
    v338_rows = changed_row_ids(v306, v338)
    new_rows = sorted(v306_changed - v338_rows)
    base_point = v306["pointId"].to_numpy(dtype=int)
    cand_point = candidate["pointId"].to_numpy(dtype=int)
    point0_additions = [int(row) for row in v306_changed if base_point[row] != 0 and cand_point[row] == 0]
    return {
        "point_churn_vs_v306": int(len(v306_changed)),
        "point_churn_vs_v338": int(len(v338_changed)),
        "new_rows_beyond_v338": int(len(new_rows)),
        "new_rows_beyond_v338_list": new_rows,
        "point0_additions_vs_v306": int(len(point0_additions)),
        "action_preserved": bool(candidate["actionId"].equals(v338["actionId"])),
        "server_preserved": bool(candidate["serverGetPoint"].equals(v338["serverGetPoint"])),
    }


def _write_submission(path: Path, frame: pd.DataFrame) -> str:
    out = safe_output_path(OUTDIR, path.name)
    validate_submission_schema(frame, expected_rows=len(frame))
    frame.to_csv(out, index=False)
    return relative_path(out)


def build_candidates(v306: pd.DataFrame, v338: pd.DataFrame, scored: pd.DataFrame) -> list[dict[str, Any]]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows_low = scored.sort_values(["final_trust_score", "row_id"], ascending=[True, True], kind="mergesort")
    rows_high = scored.sort_values(["final_trust_score", "row_id"], ascending=[False, True], kind="mergesort")
    specs: list[tuple[str, list[int]]] = []
    for k in (2, 4, 6):
        specs.append((f"prune_lowtrust_k{k:02d}", rows_low["row_id"].head(k).astype(int).tolist()))
    for k in (12, 18):
        keep = set(rows_high["row_id"].head(k).astype(int).tolist())
        revert = [int(row) for row in scored["row_id"].astype(int).tolist() if int(row) not in keep]
        specs.append((f"trust_top{k:02d}", revert))

    records: list[dict[str, Any]] = []
    for name, rows_to_revert in specs:
        candidate = build_pruned_submission(v306, v338, rows_to_revert)
        path = OUTDIR / f"submission_v351_{name}__v173action_v300server.csv"
        rel_path = _write_submission(path, candidate)
        metrics = candidate_metrics(v306, v338, candidate)
        records.append(
            {
                "name": name,
                "path": rel_path,
                "rows_reverted_from_v338": len(set(rows_to_revert)),
                "reverted_row_ids": " ".join(str(row) for row in sorted(set(rows_to_revert))),
                **metrics,
            }
        )
    return records


def choose_recommended(candidates: pd.DataFrame) -> str | None:
    if candidates.empty:
        return None
    eligible = candidates[
        candidates["new_rows_beyond_v338"].eq(0)
        & candidates["point0_additions_vs_v306"].eq(0)
        & candidates["action_preserved"]
        & candidates["server_preserved"]
    ].copy()
    if eligible.empty:
        return None
    # Prefer very small pruning first; it tests whether any V338 rows are harmful.
    eligible["priority"] = eligible["rows_reverted_from_v338"].map({2: 0, 4: 1, 6: 2, 12: 3}).fillna(9)
    return str(eligible.sort_values(["priority", "point_churn_vs_v338"], kind="mergesort").iloc[0]["name"])


def main() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    v306 = load_submission(V306_ANCHOR)
    v338 = load_submission(V338_ANCHOR, expected_rows=len(v306))
    gate_scores = pd.read_csv(V348_SCORES) if V348_SCORES.exists() else pd.DataFrame()
    selected_sets = load_selected_sets()
    transition_penalty = load_transition_penalty()
    scored = score_v338_rows(v306, v338, gate_scores, selected_sets, transition_penalty)
    scored_path = OUTDIR / "v338_row_trust_scores.csv"
    scored.to_csv(scored_path, index=False)

    candidates = pd.DataFrame(build_candidates(v306, v338, scored))
    summary_path = OUTDIR / "candidate_summary.csv"
    candidates.to_csv(summary_path, index=False)

    report = {
        "outdir": relative_path(OUTDIR),
        "decision": "HAS_EXPORT" if not candidates.empty else "NO_EXPORT",
        "candidate_count": int(len(candidates)),
        "recommended": choose_recommended(candidates),
        "inputs": {
            "v306_anchor": relative_path(V306_ANCHOR),
            "v338_anchor": relative_path(V338_ANCHOR),
            "v348_scores": relative_path(V348_SCORES) if V348_SCORES.exists() else None,
            "v347_transition_summary": relative_path(V347_TRANSITIONS) if V347_TRANSITIONS.exists() else None,
        },
        "policy": {
            "only_prunes_v338_rows": True,
            "no_new_rows_beyond_v338": bool(candidates["new_rows_beyond_v338"].max() == 0) if not candidates.empty else True,
            "no_point0_additions": bool(candidates["point0_additions_vs_v306"].max() == 0) if not candidates.empty else True,
            "action_server_preserved": bool(candidates["action_preserved"].all() and candidates["server_preserved"].all())
            if not candidates.empty
            else True,
        },
        "files": {
            "row_trust_scores": relative_path(scored_path),
            "candidate_summary": relative_path(summary_path),
        },
    }
    write_json(OUTDIR / "search_report.json", report)
    print(json.dumps(_json_safe({"outdir": OUTDIR, "candidates": len(candidates), "recommended": report["recommended"]}), indent=2))
    return report


if __name__ == "__main__":
    main()
