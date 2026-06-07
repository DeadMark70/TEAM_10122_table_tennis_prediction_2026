"""V456 decision board for V447-V455 full professor run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v456_professor_full_decision_board"
V455_DIR = ROOT / "v455_professor_full_packager"
V446_BOARD = ROOT / "v446_professor_run_decision_board" / "professor_decision_board.csv"
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


def _path_exists(value: Any) -> int:
    try:
        return int(Path(str(value)).exists())
    except OSError:
        return 0


def _public_rank(value: Any) -> int:
    text = str(value or "none").strip().lower()
    if text in {"positive", "public_positive", "best"}:
        return 3
    if text in {"negative", "public_negative"}:
        return 0
    return 1


def rank_full_professor_queue(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ranked = rows.copy()
    defaults = {
        "candidate": "",
        "path": "",
        "clean_eligible": True,
        "changed_rows": 0,
        "public_evidence": "none",
        "risk_penalty": 0.0,
        "path_exists": None,
        "is_final_fallback": False,
    }
    for col, default in defaults.items():
        if col not in ranked.columns:
            ranked[col] = default
    ranked["clean_eligible"] = ranked["clean_eligible"].map(_bool_value)
    ranked["changed_rows"] = ranked["changed_rows"].map(_int_value)
    ranked["risk_penalty"] = ranked["risk_penalty"].map(_float_value)
    ranked["is_final_fallback"] = ranked["is_final_fallback"].map(_bool_value)
    ranked["path_exists"] = [
        _path_exists(path) if value is None or pd.isna(value) else _int_value(value)
        for path, value in zip(ranked["path"], ranked["path_exists"])
    ]
    ranked["public_rank"] = ranked["public_evidence"].map(_public_rank)
    ranked["recommendation"] = "hold"
    clean_existing = ranked["clean_eligible"] & ranked["path_exists"].eq(1)
    ranked.loc[clean_existing, "recommendation"] = "exploratory"
    ranked.loc[clean_existing & ranked["changed_rows"].gt(25), "recommendation"] = "exploratory_private"
    ranked.loc[ranked["is_final_fallback"], "recommendation"] = "fallback_final"
    ranked.loc[~ranked["clean_eligible"], "recommendation"] = "never_upload"
    ranked.loc[ranked["path_exists"].eq(0) & ~ranked["is_final_fallback"], "recommendation"] = "missing_file"
    ranked["decision_score"] = (
        ranked["clean_eligible"].astype(float) * 100_000
        + ranked["path_exists"].astype(float) * 2_000
        + ranked["public_rank"] * 500
        - ranked["changed_rows"] * 1.5
        - ranked["risk_penalty"]
    )
    return ranked.sort_values(
        ["clean_eligible", "path_exists", "recommendation", "decision_score"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _load_v455_candidates() -> pd.DataFrame:
    report_path = V455_DIR / "packaging_report.csv"
    if not report_path.exists():
        return pd.DataFrame()
    report = pd.read_csv(report_path)
    return pd.DataFrame(
        {
            "candidate": "v455_" + report["name"].astype(str),
            "path": [str(V455_DIR / filename) for filename in report["filename"].astype(str)],
            "clean_eligible": True,
            "changed_rows": pd.to_numeric(report["total_changed_rows"], errors="coerce").fillna(0).astype(int),
            "public_evidence": "none",
            "risk_penalty": np.where(pd.to_numeric(report["total_changed_rows"], errors="coerce").fillna(0) > 25, 50.0, 0.0),
            "source": "V455 full professor packager",
        }
    )


def _load_v446_candidates() -> pd.DataFrame:
    if not V446_BOARD.exists():
        return pd.DataFrame()
    board = pd.read_csv(V446_BOARD)
    keep = board.loc[board["recommendation"].isin(["exploratory", "fallback_final"])].copy()
    keep = keep.loc[~keep["candidate"].astype(str).str.contains("v362_final|fallback", case=False, regex=True)].copy()
    keep = keep.loc[keep["path"].astype(str) != str(ANCHOR_PATH)].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["candidate"] = "prior_" + keep["candidate"].astype(str)
    keep["source"] = "V446 prior professor board"
    return keep[["candidate", "path", "clean_eligible", "changed_rows", "public_evidence", "risk_penalty", "source"]]


def _fallback_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": "v362_final_fallback",
                "path": str(ANCHOR_PATH),
                "clean_eligible": True,
                "changed_rows": 0,
                "public_evidence": "positive",
                "risk_penalty": 0.0,
                "source": "known clean public best",
                "is_final_fallback": True,
            }
        ]
    )


def collect_full_queue() -> pd.DataFrame:
    frames = [_load_v455_candidates(), _load_v446_candidates(), _fallback_row()]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _write_queue(ranked: pd.DataFrame) -> None:
    exploratory = ranked.loc[ranked["recommendation"].eq("exploratory")].head(6)
    private = ranked.loc[ranked["recommendation"].eq("exploratory_private")].head(3)
    fallback = ranked.loc[ranked["recommendation"].eq("fallback_final")].head(1)
    lines = ["# V456 Professor Full Upload Queue", "", "Exploratory clean candidates:"]
    if exploratory.empty:
        lines.append("- No existing clean exploratory candidate.")
    else:
        for _, row in exploratory.iterrows():
            lines.append(f"- `{row['path']}` ({row['candidate']}, changed={int(row['changed_rows'])})")
    lines.extend(["", "Private-risk exploratory candidates:"])
    if private.empty:
        lines.append("- None.")
    else:
        for _, row in private.iterrows():
            lines.append(f"- `{row['path']}` ({row['candidate']}, changed={int(row['changed_rows'])})")
    lines.extend(["", "Final fallback:"])
    for _, row in fallback.iterrows():
        lines.append(f"- `{row['path']}` ({row['candidate']})")
    (OUTDIR / "professor_full_upload_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_decision_board() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    queue = collect_full_queue()
    ranked = rank_full_professor_queue(queue)
    ranked.to_csv(OUTDIR / "professor_full_decision_board.csv", index=False)
    _write_queue(ranked)
    exploratory = ranked.loc[ranked["recommendation"].eq("exploratory")]
    fallback = ranked.loc[ranked["recommendation"].eq("fallback_final")]
    summary = {
        "version": "V456",
        "candidate_rows": int(len(ranked)),
        "recommended_exploratory": exploratory.iloc[0].to_dict() if not exploratory.empty else None,
        "fallback_final": fallback.iloc[0].to_dict() if not fallback.empty else None,
    }
    write_json(OUTDIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_decision_board()
    print(json.dumps(_json_safe({"outdir": str(OUTDIR), **result}), indent=2))
