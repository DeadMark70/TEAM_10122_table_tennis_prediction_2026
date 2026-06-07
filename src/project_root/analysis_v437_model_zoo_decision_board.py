"""V437 decision board for model-zoo and residual candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v437_model_zoo_decision_board"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)


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


def _bool_value(value: Any, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def candidate_risk_penalty(row: dict[str, Any] | pd.Series) -> float:
    clean = _bool_value(row.get("clean_eligible", False))
    point0 = _float_value(row.get("point0_additions", 0.0))
    serve = _float_value(row.get("serve_additions", 0.0))
    changed = _float_value(row.get("target_changed", 0.0))
    server_churn = _float_value(row.get("server_churn", 0.0))
    penalty = 0.0
    if not clean:
        penalty += 10_000.0
    penalty += point0 * 250.0
    penalty += serve * 500.0
    penalty += max(0.0, changed - 40.0) * 0.5
    penalty += server_churn * 100.0
    return float(penalty)


def rank_upload_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ranked = rows.copy()
    for col in ["local_delta", "public_like_delta", "target_changed", "point0_additions", "serve_additions"]:
        if col not in ranked.columns:
            ranked[col] = 0.0
    if "clean_eligible" not in ranked.columns:
        ranked["clean_eligible"] = True
    if "is_fallback_final" not in ranked.columns:
        ranked["is_fallback_final"] = False
    ranked["risk_penalty"] = [candidate_risk_penalty(row) for _, row in ranked.iterrows()]
    ranked["decision_score"] = (
        ranked["local_delta"].map(_float_value) * 1000.0
        + ranked["public_like_delta"].map(_float_value) * 1200.0
        - ranked["risk_penalty"]
        - ranked["target_changed"].map(_float_value) * 0.03
    )
    ranked["recommendation"] = "hold"
    ranked.loc[ranked["clean_eligible"].map(_bool_value) & (ranked["risk_penalty"] < 1.0), "recommendation"] = "probe"
    ranked.loc[ranked["is_fallback_final"].map(_bool_value), "recommendation"] = "fallback_final"
    return ranked.sort_values(
        ["clean_eligible", "is_fallback_final", "decision_score"],
        ascending=[False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _load_packaging_candidates() -> pd.DataFrame:
    report_path = ROOT / "v435_residual_packager" / "packaging_report.csv"
    if not report_path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(report_path)
    out = pd.DataFrame(
        {
            "candidate": rows["filename"].astype(str).str.replace(".csv", "", regex=False),
            "path": [str(ROOT / "v435_residual_packager" / name) for name in rows["filename"].astype(str)],
            "clean_eligible": True,
            "risk_tier": rows["name"].astype(str),
            "target_changed": pd.to_numeric(rows["total_changed_rows"], errors="coerce").fillna(0).astype(int),
            "action_churn": 0.0,
            "point_churn": 0.0,
            "server_churn": 0.0,
            "point0_additions": 0,
            "serve_additions": 0,
            "local_delta": 0.0,
            "public_like_delta": 0.0,
            "source_families": "V434/V435",
        }
    )
    out["action_churn"] = out["target_changed"] / 1845.0
    out["point_churn"] = out["target_changed"] / 1845.0
    return out


def _load_baseline_candidates() -> pd.DataFrame:
    rows = [
        {
            "candidate": "v362_final_resubmit",
            "path": str(ANCHOR_PATH),
            "clean_eligible": True,
            "risk_tier": "fallback",
            "target_changed": 0,
            "action_churn": 0.0,
            "point_churn": 0.0,
            "server_churn": 0.0,
            "point0_additions": 0,
            "serve_additions": 0,
            "local_delta": 0.0,
            "public_like_delta": 0.0,
            "source_families": "V362 public best",
            "is_fallback_final": True,
        },
        {
            "candidate": "v400_public_agree_top9",
            "path": str(ROOT / "v400_public_component_recombination" / "submission_v400_public_agree_top9__v173action_v300server.csv"),
            "clean_eligible": True,
            "risk_tier": "known_queue",
            "target_changed": 9,
            "action_churn": 0.0,
            "point_churn": 9 / 1845.0,
            "server_churn": 0.0,
            "point0_additions": 0,
            "serve_additions": 0,
            "local_delta": 0.0001,
            "public_like_delta": 0.0,
            "source_families": "historical clean queue",
            "is_fallback_final": False,
        },
        {
            "candidate": "v407_longside_corner",
            "path": str(ROOT / "v407_transition_family_probe_factory" / "submission_v407_longside_corner__v173action_v300server.csv"),
            "clean_eligible": True,
            "risk_tier": "known_queue",
            "target_changed": 0,
            "action_churn": 0.0,
            "point_churn": 0.0,
            "server_churn": 0.0,
            "point0_additions": 0,
            "serve_additions": 0,
            "local_delta": 0.00005,
            "public_like_delta": 0.0,
            "source_families": "historical clean queue",
            "is_fallback_final": False,
        },
    ]
    return pd.DataFrame(rows)


def _load_risk_register() -> dict[str, Any]:
    risk: dict[str, Any] = {}
    for name, path in {
        "v436_ttmatch": ROOT / "v436_ttmatch_quarantined_contrastive" / "quarantine_report.json",
        "v438_sony": ROOT / "v438_sony_nd_audit_only" / "sony_nd_audit_report.json",
    }.items():
        if path.exists():
            try:
                risk[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                risk[name] = {"error": "invalid_json", "path": str(path)}
    return risk


def run_decision_board() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat([_load_packaging_candidates(), _load_baseline_candidates()], ignore_index=True)
    ranked = rank_upload_candidates(candidates)
    ranked.to_csv(OUTDIR / "decision_board.csv", index=False)
    risk = _load_risk_register()
    write_json(OUTDIR / "risk_register.json", risk)

    clean_probe = ranked.loc[(ranked["recommendation"] == "probe") & ranked["path"].map(lambda p: Path(str(p)).exists())].head(5)
    lines = [
        "# V437 Upload Queue",
        "",
        "Clean probe candidates:",
    ]
    if clean_probe.empty:
        lines.append("- No new clean probe candidate with an existing file. Keep V362 final fallback.")
    else:
        for _, row in clean_probe.iterrows():
            lines.append(f"- `{row['path']}` ({row['candidate']}, changed={int(row['target_changed'])})")
    lines.extend(["", "Final fallback:", f"- `{ANCHOR_PATH}`"])
    lines.extend(["", "Never upload:", "- V436 TTMATCH quarantine outputs", "- V438 Sony audit outputs"])
    (OUTDIR / "upload_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "candidate_rows": int(len(ranked)),
        "top_candidate": ranked.iloc[0].to_dict() if not ranked.empty else None,
        "fallback_final": str(ANCHOR_PATH),
        "risk_register_keys": sorted(risk.keys()),
    }
    write_json(OUTDIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_decision_board()
    print(json.dumps(_json_safe({"outdir": str(OUTDIR), **result}), indent=2))
