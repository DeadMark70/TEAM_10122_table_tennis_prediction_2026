"""V410 post-V400 decision board.

Merges V400-V408 candidate metadata into a short upload recommendation. The
last slot is always reserved for the current public-proven clean best because
the contest uses the last submitted file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v410_post_v400_decision_board"
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
    ("v405", ROOT / "v405_v362_pruning_lab" / "ranked_candidates.csv"),
    ("v407", ROOT / "v407_transition_family_probe_factory" / "ranked_candidates.csv"),
    ("v408", ROOT / "v408_clean_server_microblend_recheck" / "ranked_candidates.csv"),
]
V406_SCORES = ROOT / "v406_public_response_meta_model" / "candidate_response_scores.csv"
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
                resolved = Path(ranked_path).parent / path
                if (ROOT / resolved).exists():
                    return resolved.as_posix()
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
    source = str(row.get("source_label", ""))
    path = _candidate_path(row).lower()
    if "v391" in path:
        return "blocked"
    if _num(row, "point0_additions") > 0:
        return "blocked"
    if _num(row, "action_churn") > 0:
        return "blocked"
    server_changed = _num(row, "server_changed")
    if server_changed > 0 and source != "v408":
        return "blocked"
    if source == "v408" and _num(row, "server_mad", 99.0) > 0.02:
        return "blocked"
    point_churn = _num(row, "point_churn")
    if source != "v408" and point_churn == 0 and server_changed == 0:
        return "blocked"
    if source != "v408" and 0 < point_churn < 5:
        return "blocked"
    if point_churn > 30:
        return "blocked"
    if source == "v408":
        return "low"
    if point_churn <= 12:
        return "low"
    return "medium"


def _load_v406_scores(path: Path = V406_SCORES) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return {}
    if frame.empty:
        return {}
    scores: dict[str, float] = {}
    for _, row in frame.iterrows():
        candidate_path = row.get("candidate_path", row.get("path", ""))
        if not isinstance(candidate_path, str) or not candidate_path:
            continue
        score = pd.to_numeric(pd.Series([row.get("response_score", row.get("score", 0.0))]), errors="coerce").iloc[0]
        if pd.notna(score):
            scores[candidate_path] = float(score)
    return scores


def _score(row: pd.Series, response_scores: dict[str, float]) -> tuple[int, int, float, float, float, str]:
    path = _candidate_path(row)
    response = response_scores.get(path, response_scores.get(path.replace("\\", "/"), 0.0))
    risk_order = {"low": 0, "medium": 1, "blocked": 9}
    source_order = {"v400": 0, "v405": 1, "v407": 2, "v402": 3, "v408": 4, "v401": 5, "v403": 6}
    return (
        risk_order.get(_risk(row), 8),
        source_order.get(str(row.get("source_label", "")), 8),
        _num(row, "point_churn", 0.0),
        -response,
        _num(row, "server_mad", 0.0),
        path,
    )


def _why(row: pd.Series, response_scores: dict[str, float]) -> str:
    path = _candidate_path(row)
    response = response_scores.get(path, response_scores.get(path.replace("\\", "/"), 0.0))
    return "; ".join(
        [
            str(row.get("candidate", "candidate")),
            f"source={row.get('source_label', 'unknown')}",
            f"point_churn={int(_num(row, 'point_churn'))}",
            f"server_mad={_num(row, 'server_mad', 0.0):.6f}",
            f"response_score={response:.4f}",
        ]
    )


def build_queue(
    *,
    ranked_inputs: list[tuple[str, Path]] | None = None,
    v406_scores_path: Path = V406_SCORES,
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

    response_scores = _load_v406_scores(v406_scores_path)
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
        candidates["_sort_key"] = candidates.apply(lambda row: _score(row, response_scores), axis=1)
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
                "purpose": f"{row.get('source_label', 'candidate')}_post_v400_probe",
                "candidate_path": row["candidate_path"],
                "risk": row["risk"],
                "why": _why(row, response_scores),
                "fallback_if_negative": BEST_PUBLIC_PROVEN_PATH,
            }
        )
    rows.append(
        {
            "slot": len(rows) + 1,
            "purpose": "final_resubmit_best_public_proven",
            "candidate_path": BEST_PUBLIC_PROVEN_PATH,
            "risk": "public_proven",
            "why": f"Reserved final slot; known clean public PL={BEST_PUBLIC_PROVEN_PL}",
            "fallback_if_negative": "last submission counts, so this slot is the fallback",
        }
    )
    queue = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    report = {
        "version": "V410",
        "missing_inputs": missing,
        "v406_scores_available": bool(response_scores),
        "candidate_count_after_gates": int(0 if candidates.empty else len(candidates)),
        "max_new_probes": int(max_new_probes),
        "reserved_final_slot": True,
        "best_public_proven": {"path": BEST_PUBLIC_PROVEN_PATH, "pl": BEST_PUBLIC_PROVEN_PL},
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
