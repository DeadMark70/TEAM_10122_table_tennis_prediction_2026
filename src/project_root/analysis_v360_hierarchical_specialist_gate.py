"""V360 public-like hierarchical specialist gate.

This module provides shared anchor-contract checks and a conservative policy
score for V361-V364 candidate submissions. It reads the current V338/V173/V300
anchor as immutable input and writes only under ``v360_hierarchical_specialist_gate``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v360_hierarchical_specialist_gate"
ANCHOR_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
SERVE_LIKE_ACTIONS = {15, 16, 17, 18}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return value.as_posix()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def validate_submission_schema(df: pd.DataFrame, expected_rows: int = 1845) -> dict[str, Any]:
    """Validate the required submission contract without raising.

    Returns a compact result dict so workers can include failures in reports
    instead of crashing candidate-ranking jobs.
    """

    errors: list[str] = []
    columns = list(df.columns)
    missing = [col for col in SUBMISSION_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in SUBMISSION_COLUMNS]

    if missing:
        errors.append(f"missing required columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    if not missing and not extra and columns != SUBMISSION_COLUMNS:
        errors.append(f"columns must be exactly {SUBMISSION_COLUMNS}, got {columns}")
    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"row count must be {expected_rows}, got {len(df)}")

    if "rally_uid" in df.columns and df["rally_uid"].duplicated().any():
        errors.append("rally_uid values must be unique")
    if "actionId" in df.columns:
        action = pd.to_numeric(df["actionId"], errors="coerce")
        if action.isna().any() or not action.between(0, 18).all():
            errors.append("actionId must be integer-like values in [0, 18]")
    if "pointId" in df.columns:
        point = pd.to_numeric(df["pointId"], errors="coerce")
        if point.isna().any() or not point.between(0, 9).all():
            errors.append("pointId must be integer-like values in [0, 9]")
    if "serverGetPoint" in df.columns:
        server = pd.to_numeric(df["serverGetPoint"], errors="coerce")
        finite = np.isfinite(server.to_numpy(dtype=float, na_value=np.nan))
        if server.isna().any() or not finite.all():
            errors.append("serverGetPoint must be finite numeric values")
        elif not server.between(0.0, 1.0).all():
            errors.append("serverGetPoint must be in [0, 1]")

    return {
        "ok": not errors,
        "errors": errors,
        "rows": int(len(df)),
        "expected_rows": None if expected_rows is None else int(expected_rows),
        "columns": columns,
    }


def compute_anchor_diff(anchor: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Compare a candidate submission against the immutable anchor by rally_uid."""

    anchor_check = validate_submission_schema(anchor, expected_rows=len(anchor))
    candidate_check = validate_submission_schema(candidate, expected_rows=len(candidate))
    if not anchor_check["ok"]:
        raise ValueError(f"bad anchor schema: {anchor_check['errors']}")
    if not candidate_check["ok"]:
        raise ValueError(f"bad candidate schema: {candidate_check['errors']}")

    merged = anchor.merge(
        candidate,
        on="rally_uid",
        how="outer",
        suffixes=("_anchor", "_candidate"),
        indicator=True,
    )
    common = merged["_merge"] == "both"
    anchor_only = merged["_merge"] == "left_only"
    candidate_only = merged["_merge"] == "right_only"

    common_rows = merged.loc[common].copy()
    action_anchor = pd.to_numeric(common_rows["actionId_anchor"], errors="coerce").astype(int)
    action_candidate = pd.to_numeric(common_rows["actionId_candidate"], errors="coerce").astype(int)
    point_anchor = pd.to_numeric(common_rows["pointId_anchor"], errors="coerce").astype(int)
    point_candidate = pd.to_numeric(common_rows["pointId_candidate"], errors="coerce").astype(int)
    server_anchor = pd.to_numeric(common_rows["serverGetPoint_anchor"], errors="coerce")
    server_candidate = pd.to_numeric(common_rows["serverGetPoint_candidate"], errors="coerce")

    server_changed_mask = ~np.isclose(
        server_anchor.to_numpy(dtype=float),
        server_candidate.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    serve_anchor = action_anchor.isin(SERVE_LIKE_ACTIONS)
    serve_candidate = action_candidate.isin(SERVE_LIKE_ACTIONS)

    return {
        "rows_anchor": int(len(anchor)),
        "rows_candidate": int(len(candidate)),
        "common_rows": int(common.sum()),
        "missing_rows_from_anchor": int(anchor_only.sum()),
        "new_rows_beyond_v338": int(candidate_only.sum()),
        "action_churn": int((action_anchor != action_candidate).sum()),
        "point_churn": int((point_anchor != point_candidate).sum()),
        "server_changed": int(server_changed_mask.sum()),
        "point0_additions": int(((point_anchor != 0) & (point_candidate == 0)).sum()),
        "point0_removals": int(((point_anchor == 0) & (point_candidate != 0)).sum()),
        "serve_like_delta": int(serve_candidate.sum() - serve_anchor.sum()),
        "serve_like_additions": int((~serve_anchor & serve_candidate).sum()),
    }


def score_candidate_policy(row: dict[str, Any]) -> float:
    """Score candidate evidence with conservative churn and safety penalties."""

    ordinary_delta = _as_float(row.get("ordinary_delta"))
    public_like_delta = _as_float(row.get("public_like_delta"))
    weak_class_delta = _as_float(row.get("weak_class_delta"))

    point_churn = _as_float(row.get("point_churn_vs_v338", row.get("point_churn")))
    action_churn = _as_float(row.get("action_churn_vs_v173", row.get("action_churn")))
    point0_additions = _as_float(row.get("point0_additions"))
    new_rows = _as_float(row.get("new_rows_beyond_v338"))
    server_changed = _as_float(row.get("server_changed"))
    class_collapse = bool(row.get("class_collapse", False))
    serve_like_delta = max(
        0.0,
        _as_float(row.get("serve_like_delta", row.get("action_15_18_delta"))),
    )
    serve_like_additions = max(0.0, _as_float(row.get("serve_like_additions")))

    score = (
        (2.0 * ordinary_delta)
        + (5.0 * public_like_delta)
        + (2.0 * weak_class_delta)
    )
    score -= 0.00020 * point_churn
    score -= 0.00008 * action_churn
    score -= 0.01000 * point0_additions
    score -= 0.00200 * new_rows
    score -= 0.05000 * server_changed
    score -= 0.00400 * serve_like_delta
    score -= 0.00600 * serve_like_additions
    if class_collapse:
        score -= 0.10000
    return float(score)


def _risk_level(row: dict[str, Any]) -> str:
    point_churn = _as_float(row.get("point_churn_vs_v338", row.get("point_churn")))
    action_churn = _as_float(row.get("action_churn_vs_v173", row.get("action_churn")))
    point0_additions = _as_float(row.get("point0_additions"))
    new_rows = _as_float(row.get("new_rows_beyond_v338"))
    server_changed = _as_float(row.get("server_changed"))
    class_collapse = bool(row.get("class_collapse", False))

    if (
        point0_additions == 0
        and new_rows == 0
        and server_changed == 0
        and point_churn <= 5
        and action_churn <= 10
        and not class_collapse
    ):
        return "safe"
    if (
        point0_additions == 0
        and new_rows == 0
        and server_changed == 0
        and point_churn <= 15
        and action_churn <= 40
        and not class_collapse
    ):
        return "normal"
    return "research"


def rank_candidates(candidate_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return candidates sorted by V360 policy score descending."""

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidate_rows):
        enriched = dict(row)
        enriched.setdefault("candidate", f"candidate_{idx}")
        enriched["policy_score"] = score_candidate_policy(enriched)
        enriched["risk_level"] = _risk_level(enriched)
        rows.append(enriched)
    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate",
                "policy_score",
                "risk_level",
                "ordinary_delta",
                "public_like_delta",
                "weak_class_delta",
            ]
        )
    ranked = pd.DataFrame(rows)
    return ranked.sort_values(
        ["policy_score", "candidate"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def load_anchor_submission() -> pd.DataFrame:
    """Load the current V338 point/V173 action/V300 server anchor."""

    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"missing anchor submission: {ANCHOR_PATH}")
    return pd.read_csv(ANCHOR_PATH)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor_submission()
    schema = validate_submission_schema(anchor)
    if not schema["ok"]:
        raise ValueError(f"anchor violates V360 submission contract: {schema['errors']}")

    anchor_diff = compute_anchor_diff(anchor, anchor)
    anchor_contract = {
        "version": "V360",
        "anchor_path": ANCHOR_PATH,
        "anchor_components": {
            "action": "V173",
            "point": "V338 point-only MoE no-p0-add budget24",
            "server": "V300",
        },
        "schema": schema,
        "self_diff": anchor_diff,
        "rules": {
            "columns": SUBMISSION_COLUMNS,
            "expected_rows": 1845,
            "no_ttmatch": True,
            "no_old_server": True,
            "preserve_v300_server": True,
        },
    }

    template_rows = [
        {
            "candidate": "anchor_v338_v173_v300",
            "ordinary_delta": 0.0,
            "public_like_delta": 0.0,
            "weak_class_delta": 0.0,
            "point_churn_vs_v338": 0,
            "action_churn_vs_v173": 0,
            "point0_additions": 0,
            "new_rows_beyond_v338": 0,
            "server_changed": 0,
            "serve_like_delta": 0,
            "class_collapse": False,
        }
    ]
    ranked_template = rank_candidates(template_rows)

    _write_json(OUTDIR / "anchor_contract.json", anchor_contract)
    ranked_template.to_csv(OUTDIR / "candidate_policy_template.csv", index=False)
    _write_json(
        OUTDIR / "search_report.json",
        {
            "version": "V360",
            "status": "ok",
            "anchor_path": ANCHOR_PATH,
            "outputs": [
                OUTDIR / "anchor_contract.json",
                OUTDIR / "candidate_policy_template.csv",
                OUTDIR / "search_report.json",
            ],
            "candidate_policy_columns": list(ranked_template.columns),
            "top_template_candidate": ranked_template.iloc[0].to_dict(),
        },
    )


if __name__ == "__main__":
    main()
