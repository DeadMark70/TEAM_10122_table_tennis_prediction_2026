"""V446 decision board for the professor-advice training run.

This board is intentionally conservative: new V440-V445 outputs are treated as
exploratory until they get public evidence, while the known clean public-best
V362 file remains the final fallback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v446_professor_run_decision_board"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V445_DIR = ROOT / "v445_full_professor_moe_packager"
V442_DIR = ROOT / "v442_intent_first_sequence_point"


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
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
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


def _int_value(value: Any, default: int = 0) -> int:
    return int(round(_float_value(value, float(default))))


def _public_rank(value: Any) -> int:
    text = str(value or "none").strip().lower()
    if text in {"positive", "public_positive", "best", "known_positive"}:
        return 3
    if text in {"neutral", "same", "tie"}:
        return 2
    if text in {"negative", "public_negative"}:
        return 0
    return 1


def _existing_path_score(path_value: Any) -> int:
    try:
        return int(Path(str(path_value)).exists())
    except OSError:
        return 0


def rank_professor_upload_queue(rows: pd.DataFrame, require_existing_files: bool = False) -> pd.DataFrame:
    """Rank clean exploratory uploads while never promoting risky candidates."""

    if rows.empty:
        return rows.copy()

    ranked = rows.copy()
    defaults: dict[str, Any] = {
        "candidate": "",
        "path": "",
        "clean_eligible": True,
        "public_evidence": "none",
        "changed_rows": 0,
        "risk_penalty": 0.0,
        "point0_additions": 0,
        "serve_additions": 0,
        "server_preserved": True,
        "is_final_fallback": False,
        "local_signal": 0.0,
        "source": "unknown",
    }
    for col, default in defaults.items():
        if col not in ranked.columns:
            ranked[col] = default

    ranked["clean_eligible"] = ranked["clean_eligible"].map(_bool_value)
    ranked["changed_rows"] = ranked["changed_rows"].map(_int_value)
    ranked["point0_additions"] = ranked["point0_additions"].map(_int_value)
    ranked["serve_additions"] = ranked["serve_additions"].map(_int_value)
    ranked["server_preserved"] = ranked["server_preserved"].map(lambda v: _bool_value(v, default=True))
    ranked["is_final_fallback"] = ranked["is_final_fallback"].map(_bool_value)
    ranked["risk_penalty"] = ranked["risk_penalty"].map(_float_value)
    ranked["local_signal"] = ranked["local_signal"].map(_float_value)
    ranked["path_exists"] = ranked["path"].map(_existing_path_score)
    ranked["public_rank"] = ranked["public_evidence"].map(_public_rank)

    safety_penalty = (
        (~ranked["clean_eligible"]).astype(float) * 1_000_000.0
        + ranked["point0_additions"] * 2_000.0
        + ranked["serve_additions"] * 5_000.0
        + (~ranked["server_preserved"]).astype(float) * 20_000.0
        + ranked["risk_penalty"]
        + (
            ranked["path_exists"].eq(0)
            & ~ranked["is_final_fallback"]
            & bool(require_existing_files)
        ).astype(float)
        * 50_000.0
    )
    # Small exploratory probes are preferred before larger private probes.
    ranked["decision_score"] = (
        ranked["clean_eligible"].astype(float) * 100_000.0
        + ranked["public_rank"] * 1_000.0
        + ranked["path_exists"] * 50.0
        + ranked["local_signal"] * 100.0
        - ranked["changed_rows"] * 0.6
        - safety_penalty
    )

    ranked["recommendation"] = "hold"
    ranked.loc[ranked["is_final_fallback"], "recommendation"] = "fallback_final"
    exploratory = (
        ranked["clean_eligible"]
        & ~ranked["is_final_fallback"]
        & (ranked["public_rank"] >= 1)
        & (ranked["point0_additions"] == 0)
        & (ranked["serve_additions"] == 0)
        & ranked["server_preserved"]
    )
    ranked.loc[exploratory, "recommendation"] = "exploratory"
    ranked.loc[ranked["public_rank"].eq(0) & ~ranked["is_final_fallback"], "recommendation"] = "stop_after_negative_public"
    if require_existing_files:
        ranked.loc[ranked["path_exists"].eq(0) & ~ranked["is_final_fallback"], "recommendation"] = "missing_file"
    ranked.loc[~ranked["clean_eligible"], "recommendation"] = "never_upload"

    return ranked.sort_values(
        ["clean_eligible", "path_exists", "recommendation", "public_rank", "decision_score"],
        ascending=[False, False, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _load_v445_candidates() -> pd.DataFrame:
    report_path = V445_DIR / "packaging_report.csv"
    if not report_path.exists():
        return pd.DataFrame()
    report = pd.read_csv(report_path)
    rows: list[dict[str, Any]] = []
    for _, row in report.iterrows():
        name = str(row.get("name", "v445"))
        filename = str(row.get("filename", ""))
        changed = _int_value(row.get("total_changed_rows", 0))
        action_changed = _int_value(row.get("action_changed_rows", 0))
        point_changed = _int_value(row.get("point_changed_rows", 0))
        rows.append(
            {
                "candidate": f"v445_{name}",
                "path": str(V445_DIR / filename),
                "clean_eligible": True,
                "public_evidence": "none",
                "changed_rows": changed,
                "action_changed_rows": action_changed,
                "point_changed_rows": point_changed,
                "point0_additions": _int_value(row.get("blocked_point0_additions", 0)),
                "serve_additions": _int_value(row.get("blocked_serve_additions", 0)),
                "server_preserved": _bool_value(row.get("server_preserved", True), default=True),
                "risk_penalty": 0.0 if changed <= 20 else (changed - 20) * 5.0,
                "local_signal": max(0.0, 25.0 - changed) / 25.0,
                "source": "V445 professor MoE packager",
            }
        )
    return pd.DataFrame(rows)


def _load_v442_candidates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(V442_DIR.glob("submission_v442_*.csv")):
        changed = 5 if "top5" in path.name else 10 if "top10" in path.name else 0
        rows.append(
            {
                "candidate": path.stem,
                "path": str(path),
                "clean_eligible": True,
                "public_evidence": "none",
                "changed_rows": changed,
                "action_changed_rows": 0,
                "point_changed_rows": changed,
                "point0_additions": 0,
                "serve_additions": 0,
                "server_preserved": True,
                "risk_penalty": 0.0,
                "local_signal": 0.1,
                "source": "V442 intent-first point",
            }
        )
    return pd.DataFrame(rows)


def _load_known_queue() -> pd.DataFrame:
    rows = [
        {
            "candidate": "v362_final_fallback",
            "path": str(ANCHOR_PATH),
            "clean_eligible": True,
            "public_evidence": "positive",
            "changed_rows": 0,
            "action_changed_rows": 0,
            "point_changed_rows": 0,
            "point0_additions": 0,
            "serve_additions": 0,
            "server_preserved": True,
            "risk_penalty": 0.0,
            "local_signal": 1.0,
            "source": "known clean public best",
            "is_final_fallback": True,
        },
        {
            "candidate": "v300_prior_public_positive",
            "path": str(ROOT / "v300_best_safe_repack" / "submission_v300_best_safe_repack__v173action_v261point_server.csv"),
            "clean_eligible": True,
            "public_evidence": "positive",
            "changed_rows": 0,
            "action_changed_rows": 0,
            "point_changed_rows": 0,
            "point0_additions": 0,
            "serve_additions": 0,
            "server_preserved": True,
            "risk_penalty": 0.0,
            "local_signal": 0.8,
            "source": "known public positive older anchor",
        },
    ]
    return pd.DataFrame(rows)


def _load_risk_register() -> dict[str, Any]:
    risk: dict[str, Any] = {}
    for name, path in {
        "ttmatch_quarantine": ROOT / "v436_ttmatch_quarantined_contrastive" / "quarantine_report.json",
        "sony_nd_audit": ROOT / "v438_sony_nd_audit_only" / "sony_nd_audit_report.json",
    }.items():
        if path.exists():
            try:
                risk[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                risk[name] = {"error": "invalid_json", "path": str(path)}
    return risk


def collect_professor_candidates() -> pd.DataFrame:
    frames = [_load_v445_candidates(), _load_v442_candidates(), _load_known_queue()]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


def _write_upload_queue(ranked: pd.DataFrame) -> None:
    exploratory = ranked.loc[
        (ranked["recommendation"] == "exploratory")
        & ranked["path_exists"].eq(1)
        & ranked["clean_eligible"].map(_bool_value)
    ].head(5)
    fallback = ranked.loc[ranked["recommendation"].eq("fallback_final")]

    lines = [
        "# V446 Professor Upload Queue",
        "",
        "Exploratory clean candidates:",
    ]
    if exploratory.empty:
        lines.append("- No new clean exploratory candidate with an existing file.")
    else:
        for _, row in exploratory.iterrows():
            lines.append(
                f"- `{row['path']}` ({row['candidate']}, changed={int(row['changed_rows'])}, "
                f"source={row['source']})"
            )

    lines.extend(["", "Final fallback:"])
    if fallback.empty:
        lines.append(f"- `{ANCHOR_PATH}`")
    else:
        for _, row in fallback.head(3).iterrows():
            lines.append(f"- `{row['path']}` ({row['candidate']})")

    lines.extend(
        [
            "",
            "Never upload in clean branch:",
            "- V436 TTMATCH quarantine outputs",
            "- V438 Sony ND audit outputs",
            "- Any V445 candidate after a negative public probe unless a new public-positive signal appears",
        ]
    )
    (OUTDIR / "professor_upload_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_decision_board() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_professor_candidates()
    ranked = rank_professor_upload_queue(candidates, require_existing_files=True)
    ranked.to_csv(OUTDIR / "professor_decision_board.csv", index=False)
    _write_upload_queue(ranked)

    risk_register = _load_risk_register()
    write_json(OUTDIR / "risk_register.json", risk_register)

    exploratory = ranked.loc[ranked["recommendation"].eq("exploratory") & ranked["path_exists"].eq(1)]
    fallback = ranked.loc[ranked["recommendation"].eq("fallback_final")]
    summary = {
        "candidate_rows": int(len(ranked)),
        "exploratory_candidates": int(len(exploratory)),
        "recommended_exploratory": exploratory.iloc[0].to_dict() if not exploratory.empty else None,
        "fallback_final": fallback.iloc[0].to_dict() if not fallback.empty else {"path": str(ANCHOR_PATH)},
        "risk_register_keys": sorted(risk_register.keys()),
        "version": "V446",
    }
    write_json(OUTDIR / "professor_run_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_decision_board()
    print(json.dumps(_json_safe({"outdir": str(OUTDIR), **result}), indent=2))
