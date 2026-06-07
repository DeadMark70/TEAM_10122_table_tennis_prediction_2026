"""V392 submission budget recommender.

This module converts V391/V387 ranked candidate metadata into a short upload
queue under the last-submission-counts rule. It deliberately reserves the last
slot for the best public-proven clean resubmission.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v392_submission_budget_recommender"
V387_RANKED = ROOT / "v387_expanded_synthetic_packager" / "ranked_candidates.csv"
V391_RANKED = ROOT / "v391_oof_gated_submission_packager" / "ranked_candidates.csv"
V383_RANKED = ROOT / "v383_synthetic_adjusted_packager" / "ranked_candidates.csv"
EXPERIMENTS_LOG = ROOT / "experiments_log.md"

MAX_EFFECTIVE_SLOTS = 7
PROBE_SLOTS = MAX_EFFECTIVE_SLOTS - 1
BEST_PUBLIC_PROVEN_PATH = (
    "v362_point_hierarchical_specialists/"
    "submission_v362_depth_agree_only__v173action_v300server.csv"
)
BEST_PUBLIC_PROVEN_PL = 0.3590124
QUEUE_COLUMNS = ["slot", "purpose", "candidate_path", "risk", "why", "fallback_if_negative"]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_ranked(path: Path, source_label: str) -> tuple[pd.DataFrame, bool]:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), False
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(), True
    if frame.empty:
        return frame, True
    frame = frame.copy()
    frame["source_label"] = source_label
    frame["source_ranked_path"] = _rel(path)
    return frame, True


def _numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _candidate_path(row: pd.Series) -> str:
    for column in ("path", "candidate_path", "submission_path"):
        value = row.get(column)
        if isinstance(value, str) and value:
            return value
    return ""


def _candidate_signature(path_text: str) -> str:
    """Return a stable signature for duplicate upload files.

    Ranked metadata may contain several candidate names that point to identical
    CSV contents after gates collapse to the same selected rows. Deduping only
    by path would waste upload slots, so prefer file bytes when the submission is
    present locally and fall back to path text for synthetic test fixtures.
    """
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        return f"path:{path_text}"
    import hashlib

    return "md5:" + hashlib.md5(path.read_bytes()).hexdigest()


def _risk(row: pd.Series) -> str:
    action_churn = _numeric(row, "action_churn")
    point_churn = _numeric(row, "point_churn")
    point0_additions = _numeric(row, "point0_additions")
    server_changed = _numeric(row, "server_changed")
    serve_delta = _numeric(row, "serve_15_18_delta")

    if server_changed > 0 or point0_additions > 0 or serve_delta > 0:
        return "blocked"
    if action_churn > 0:
        return "high"
    if point_churn <= 12:
        return "low"
    if point_churn <= 40:
        return "medium"
    return "high"


def _purpose(row: pd.Series) -> str:
    source = str(row.get("source_label", "candidate"))
    risk = _risk(row)
    action_churn = _numeric(row, "action_churn")
    point_churn = _numeric(row, "point_churn")
    if action_churn > 0:
        return f"{source}_action_probe"
    if point_churn > 0:
        return f"{source}_point_probe"
    return f"{source}_candidate_probe"


def _why(row: pd.Series) -> str:
    bits = [
        str(row.get("candidate", "ranked candidate")),
        f"source={row.get('source_label', 'unknown')}",
        f"rank={int(_numeric(row, 'rank', 999))}",
        f"point_churn={int(_numeric(row, 'point_churn'))}",
        f"action_churn={int(_numeric(row, 'action_churn'))}",
    ]
    if row.get("public_evidence"):
        bits.append("public_evidence=true")
    return "; ".join(bits)


def _sort_key(row: pd.Series) -> tuple[int, int, int, int, str]:
    risk_order = {"low": 0, "medium": 1, "high": 2, "blocked": 9}
    source_order = {"v391": 0, "v387": 1, "v383": 2}
    public_priority = 0 if _is_truthy(row.get("public_evidence", False)) else 1
    action_priority = 1 if _numeric(row, "action_churn") > 0 else 0
    rank = int(_numeric(row, "rank", 999))
    path = _candidate_path(row)
    source_priority = source_order.get(str(row.get("source_label", "")), 8)
    return (risk_order.get(_risk(row), 8), public_priority, action_priority, source_priority, rank, path)


def _dedupe_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = [frame for frame in frames if not frame.empty]
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined["candidate_path"] = combined.apply(_candidate_path, axis=1)
    combined = combined[combined["candidate_path"].astype(str).str.len() > 0].copy()
    combined["risk"] = combined.apply(_risk, axis=1)
    combined = combined[combined["risk"] != "blocked"].copy()
    if combined.empty:
        return combined
    combined["_sort_key"] = combined.apply(_sort_key, axis=1)
    combined["candidate_signature"] = combined["candidate_path"].map(_candidate_signature)
    combined = combined.sort_values("_sort_key").drop_duplicates("candidate_signature", keep="first")
    return combined.drop(columns=["_sort_key"])


def _extract_best_public_from_log(path: Path) -> dict[str, Any]:
    result = {
        "path": BEST_PUBLIC_PROVEN_PATH,
        "pl": BEST_PUBLIC_PROVEN_PL,
        "source": "default_v392_policy",
        "experiments_log_available": False,
    }
    path = Path(path)
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="ignore")
    result["experiments_log_available"] = True
    if BEST_PUBLIC_PROVEN_PATH in text and re.search(r"PL\s*=\s*0\.3590124", text):
        result["source"] = _rel(path)
    return result


def _fallback_frames(
    *,
    v387_path: Path,
    v391_path: Path,
    v383_path: Path,
) -> tuple[list[pd.DataFrame], dict[str, bool], list[str]]:
    v391, has_v391 = _read_ranked(v391_path, "v391")
    v387, has_v387 = _read_ranked(v387_path, "v387")
    v383, has_v383 = _read_ranked(v383_path, "v383")
    missing = {
        "missing_v391": (not has_v391) or v391.empty,
        "missing_v387": not has_v387,
        "missing_v383": not has_v383,
    }
    inputs_used: list[str] = []
    frames: list[pd.DataFrame] = []
    if has_v391 and not v391.empty:
        frames.append(v391)
        inputs_used.append(_rel(Path(v391_path)))
    if has_v387 and not v387.empty:
        frames.append(v387)
        inputs_used.append(_rel(Path(v387_path)))
    if has_v383 and not v383.empty:
        frames.append(v383)
        inputs_used.append(_rel(Path(v383_path)))
    return frames, missing, inputs_used


def build_recommended_queue(
    *,
    v387_path: Path = V387_RANKED,
    v391_path: Path = V391_RANKED,
    v383_path: Path = V383_RANKED,
    experiments_log_path: Path = EXPERIMENTS_LOG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames, missing, inputs_used = _fallback_frames(
        v387_path=v387_path,
        v391_path=v391_path,
        v383_path=v383_path,
    )
    candidates = _dedupe_candidates(frames)
    if not candidates.empty:
        candidates["_sort_key"] = candidates.apply(_sort_key, axis=1)
        candidates = candidates.sort_values("_sort_key").drop(columns=["_sort_key"])

    rows: list[dict[str, Any]] = []
    for _, row in candidates.head(PROBE_SLOTS).iterrows():
        rows.append(
            {
                "slot": len(rows) + 1,
                "purpose": _purpose(row),
                "candidate_path": row["candidate_path"],
                "risk": row["risk"],
                "why": _why(row),
                "fallback_if_negative": BEST_PUBLIC_PROVEN_PATH,
            }
        )

    best_public = _extract_best_public_from_log(experiments_log_path)
    rows.append(
        {
            "slot": len(rows) + 1,
            "purpose": "final_resubmit_best_public_proven",
            "candidate_path": best_public["path"],
            "risk": "public_proven",
            "why": f"Reserved final slot; known clean public PL={best_public['pl']}",
            "fallback_if_negative": "last submission counts, so this slot is the fallback",
        }
    )

    queue = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    report = {
        "version": "V392",
        "upload_budget": {
            "today_remaining_after_one_upload": 2,
            "tomorrow": 3,
            "final_day_before_deadline": 2,
            "total_effective": MAX_EFFECTIVE_SLOTS,
            "probe_slots_before_final_resubmit": PROBE_SLOTS,
            "last_submission_counts": True,
        },
        "reserved_final_slot": True,
        "best_public_proven": best_public,
        "missing_inputs": missing,
        "inputs_used": inputs_used,
        "recommended_count": int(len(queue)),
        "policy": {
            "max_recommended_slots": MAX_EFFECTIVE_SLOTS,
            "first_probe_preference": "public-backed or low-risk point-only before action probes",
            "fallback_without_v391": "Use V387, then V383 if available, then final public-proven resubmit",
        },
        "recommended_queue": queue.to_dict(orient="records"),
    }
    return queue, report


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    v387_path: Path = V387_RANKED,
    v391_path: Path = V391_RANKED,
    v383_path: Path = V383_RANKED,
    experiments_log_path: Path = EXPERIMENTS_LOG,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    queue, report = build_recommended_queue(
        v387_path=v387_path,
        v391_path=v391_path,
        v383_path=v383_path,
        experiments_log_path=experiments_log_path,
    )
    queue_path = outdir / "recommended_upload_queue.csv"
    report_path = outdir / "search_report.json"
    queue.to_csv(queue_path, index=False)
    report["outputs"] = {
        "recommended_upload_queue": _rel(queue_path),
        "search_report": _rel(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
