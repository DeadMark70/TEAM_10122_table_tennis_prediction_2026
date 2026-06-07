"""V404 breakthrough decision board.

This script merges V400-V403 ranked candidate metadata into a small upload
queue. It is intentionally conservative because the remaining contest budget is
limited and the last submission counts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v404_breakthrough_decision_board"
BEST_PUBLIC_PROVEN_PATH = (
    "v362_point_hierarchical_specialists/"
    "submission_v362_depth_agree_only__v173action_v300server.csv"
)
BEST_PUBLIC_PROVEN_PL = 0.3590124
RANKED_INPUTS = [
    ("v400", ROOT / "v400_public_component_recombination" / "ranked_candidates.csv"),
    ("v401", ROOT / "v401_action_point_compatibility" / "ranked_candidates.csv"),
    ("v402", ROOT / "v402_rare_point_specialist_lab" / "ranked_candidates.csv"),
    ("v403", ROOT / "v403_neural_posterior_gate" / "ranked_candidates.csv"),
]
QUEUE_COLUMNS = ["slot", "purpose", "candidate_path", "risk", "why", "fallback_if_negative"]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_ranked(path: Path, source: str) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        return pd.DataFrame(), False
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(), True
    if frame.empty:
        return frame, True
    frame = frame.copy()
    frame["source_label"] = source
    frame["source_ranked_path"] = _rel(path)
    return frame, True


def _num(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else float(parsed)


def _candidate_path(row: pd.Series) -> str:
    for name in ("path", "candidate_path", "submission_path"):
        value = row.get(name)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.is_absolute():
                return value
            ranked_path = row.get("source_ranked_path")
            if isinstance(ranked_path, str) and ranked_path:
                parent = Path(ranked_path).parent
                resolved = parent / path
                if (ROOT / resolved).exists():
                    return resolved.as_posix()
            if (ROOT / path).exists():
                return value
            return value
    return ""


def _signature(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        return f"path:{path_text}"
    return "md5:" + hashlib.md5(path.read_bytes()).hexdigest()


def _risk(row: pd.Series) -> str:
    if _num(row, "point0_additions") > 0:
        return "blocked"
    if _num(row, "action_churn") > 0:
        return "blocked"
    if _num(row, "server_changed") > 0:
        return "blocked"
    if _num(row, "point_churn") > 30:
        return "blocked"
    if "v391" in _candidate_path(row).lower():
        return "blocked"
    churn = _num(row, "point_churn")
    if churn <= 12:
        return "low"
    return "medium"


def _score(row: pd.Series) -> tuple[int, float, float, float, float, str]:
    risk_order = {"low": 0, "medium": 1, "blocked": 9}
    source_order = {"v400": 0, "v402": 1, "v401": 2, "v403": 3}
    agreement = _num(row, "source_agreement_count", _num(row, "agreement_count", 0.0))
    public_count = _num(row, "public_positive_component_count", 0.0)
    compat = _num(row, "compatibility_score", _num(row, "compat_delta", 0.0))
    specialist = _num(row, "specialist_score", 0.0)
    rank = _num(row, "rank", 999.0)
    return (
        risk_order.get(_risk(row), 8),
        -public_count,
        -agreement,
        -compat,
        -specialist,
        f"{source_order.get(str(row.get('source_label', '')), 8):02d}-{rank:06.1f}-{_candidate_path(row)}",
    )


def _why(row: pd.Series) -> str:
    return "; ".join(
        [
            str(row.get("candidate", "candidate")),
            f"source={row.get('source_label', 'unknown')}",
            f"rank={int(_num(row, 'rank', 999))}",
            f"point_churn={int(_num(row, 'point_churn'))}",
            f"agreement={_num(row, 'source_agreement_count', _num(row, 'agreement_count', 0.0)):.2f}",
        ]
    )


def build_queue(
    *,
    ranked_inputs: list[tuple[str, Path]] | None = None,
    best_public_path: str = BEST_PUBLIC_PROVEN_PATH,
    max_new_probes: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranked_inputs = ranked_inputs or RANKED_INPUTS
    frames: list[pd.DataFrame] = []
    missing: dict[str, bool] = {}
    for source, path in ranked_inputs:
        frame, present = _read_ranked(path, source)
        missing[f"missing_{source}"] = (not present) or frame.empty
        if present and not frame.empty:
            frames.append(frame)

    if frames:
        candidates = pd.concat(frames, ignore_index=True, sort=False)
        candidates["candidate_path"] = candidates.apply(_candidate_path, axis=1)
        candidates = candidates[candidates["candidate_path"].astype(str).str.len() > 0].copy()
        candidates["risk"] = candidates.apply(_risk, axis=1)
        candidates = candidates[candidates["risk"] != "blocked"].copy()
    else:
        candidates = pd.DataFrame()

    if not candidates.empty:
        candidates["candidate_signature"] = candidates["candidate_path"].map(_signature)
        candidates["_sort_key"] = candidates.apply(_score, axis=1)
        candidates = (
            candidates.sort_values("_sort_key")
            .drop_duplicates("candidate_signature", keep="first")
            .drop(columns=["_sort_key"])
        )

    rows: list[dict[str, Any]] = []
    for _, row in candidates.head(max_new_probes).iterrows():
        rows.append(
            {
                "slot": len(rows) + 1,
                "purpose": f"{row.get('source_label', 'candidate')}_breakthrough_probe",
                "candidate_path": row["candidate_path"],
                "risk": row["risk"],
                "why": _why(row),
                "fallback_if_negative": best_public_path,
            }
        )
    rows.append(
        {
            "slot": len(rows) + 1,
            "purpose": "final_resubmit_best_public_proven",
            "candidate_path": best_public_path,
            "risk": "public_proven",
            "why": f"Reserved final slot; known clean public PL={BEST_PUBLIC_PROVEN_PL}",
            "fallback_if_negative": "last submission counts, so this slot is the fallback",
        }
    )
    queue = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    report = {
        "version": "V404",
        "missing_inputs": missing,
        "candidate_count_after_gates": int(0 if candidates.empty else len(candidates)),
        "max_new_probes": int(max_new_probes),
        "reserved_final_slot": True,
        "best_public_proven": {"path": best_public_path, "pl": BEST_PUBLIC_PROVEN_PL},
        "recommended_queue": queue.to_dict(orient="records"),
    }
    return queue, report


def run_pipeline(*, outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    queue, report = build_queue()
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
