"""V384 report for the synthetic rare grammar branch.

The report is intentionally documentation-only. It summarizes provenance,
synthetic row coverage, candidate packaging, and limits so the branch can be
explained later without implying manual test-row correction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_v335_moe_anchor_contract import read_json, write_json


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v384_synthetic_data_report"
V380_REPORT = ROOT / "v380_synthetic_authenticity_registry" / "search_report.json"
V381_ROWS = ROOT / "v381_rare_synthetic_grammar_generator" / "synthetic_rare_grammar.csv"
V381_REPORT = ROOT / "v381_rare_synthetic_grammar_generator" / "search_report.json"
V382_REPORT = ROOT / "v382_synthetic_teacher_evaluator" / "search_report.json"
V383_RANKED = ROOT / "v383_synthetic_adjusted_packager" / "ranked_candidates.csv"
V383_REPORT = ROOT / "v383_synthetic_adjusted_packager" / "search_report.json"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def collect_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "version": "V384",
        "v380_rules": 0,
        "v381_rows": 0,
        "v382_synthetic_source": None,
        "v382_point_candidates_scored": 0,
        "v382_action_candidates_scored": 0,
        "v383_candidate_count": 0,
        "top_candidate": None,
        "top_point_churn": None,
        "top_point0_additions": None,
        "inputs": {
            "v380_report": _rel(V380_REPORT),
            "v381_rows": _rel(V381_ROWS),
            "v382_report": _rel(V382_REPORT),
            "v383_ranked": _rel(V383_RANKED),
        },
    }
    if V380_REPORT.exists():
        v380 = read_json(V380_REPORT)
        summary["v380_rules"] = int(v380.get("rule_count", v380.get("rules", 0)) or 0)
    if V381_ROWS.exists():
        rows = pd.read_csv(V381_ROWS)
        summary["v381_rows"] = int(len(rows))
        summary["v381_unique_rules"] = int(rows["rule_id"].nunique()) if "rule_id" in rows else 0
        summary["v381_all_synthetic_uid"] = bool(
            rows["rally_uid"].astype(str).str.startswith("synthetic_").all()
        ) if "rally_uid" in rows else False
    if V381_REPORT.exists():
        summary["v381_report"] = read_json(V381_REPORT)
    if V382_REPORT.exists():
        v382 = read_json(V382_REPORT)
        summary["v382_synthetic_source"] = v382.get("synthetic_source")
        summary["v382_point_candidates_scored"] = int(v382.get("point_candidates_scored", 0) or 0)
        summary["v382_action_candidates_scored"] = int(v382.get("action_candidates_scored", 0) or 0)
        summary["v382_missing_v381"] = bool(v382.get("missing_v381", False))
    if V383_RANKED.exists():
        ranked = pd.read_csv(V383_RANKED)
        summary["v383_candidate_count"] = int(len(ranked))
        if not ranked.empty:
            top = ranked.iloc[0].to_dict()
            summary["top_candidate"] = Path(str(top["path"])).name
            summary["top_candidate_path"] = top["path"]
            summary["top_point_churn"] = int(top.get("point_churn", 0))
            summary["top_point0_additions"] = int(top.get("point0_additions", 0))
            summary["top_action_churn"] = int(top.get("action_churn", 0))
            summary["top_server_changed"] = int(top.get("server_changed", 0))
    if V383_REPORT.exists():
        summary["v383_report"] = read_json(V383_REPORT)
    return summary


def build_report_text(summary: dict[str, Any]) -> str:
    top = summary.get("top_candidate") or "none"
    point_churn = summary.get("top_point_churn", "n/a")
    p0_add = summary.get("top_point0_additions", "n/a")
    v381_rows = summary.get("v381_rows", 0)
    return f"""# V384 Synthetic Rare Grammar Data Report

## Purpose

This branch tests self-made synthetic table-tennis grammar as auxiliary
training/teacher evidence for rare action and point cases. It does not use
synthetic data to manually correct test rows.

## Authenticity Controls

- Every generated synthetic rally id must start with `synthetic_`.
- Every synthetic row must include a rule id and provenance.
- No TTMATCH.
- No old-server labels.
- No hidden test labels.
- No external exact mapping to AICUP hidden labels.
- No manual row edits.

## Generated Data

- V381 synthetic grammar rows: `{v381_rows}`.
- Synthetic source: `{summary.get("v382_synthetic_source", "unknown")}`.
- Point candidates scored by V382: `{summary.get("v382_point_candidates_scored", 0)}`.
- Action candidates scored by V382: `{summary.get("v382_action_candidates_scored", 0)}`.

## Candidate Packaging

- V383 generated candidates: `{summary.get("v383_candidate_count", 0)}`.
- Top candidate: `{top}`.
- Top candidate point churn vs V362 anchor: `{point_churn}`.
- Top candidate point0 additions: `{p0_add}`.

## Interpretation

The synthetic data is useful only if it improves candidate ranking while
preserving clean-branch constraints. The current safest use is to score and
filter candidate point updates, not to create new test answers directly.

## Source Notes

The implementation follows standard imbalanced-learning practice: synthetic
minority examples can help rare classes, but they require strict provenance,
temporal/physical compatibility checks, and validation against real held-out
data before being trusted for public upload.
"""


def write_report(summary: dict[str, Any], outdir: Path = OUTDIR) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "synthetic_data_report.md"
    report_path.write_text(build_report_text(summary), encoding="utf-8")
    return report_path


def run_pipeline(outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = collect_summary()
    report_path = write_report(summary, outdir=outdir)
    summary["outputs"] = {"report": _rel(report_path), "summary": _rel(outdir / "search_report.json")}
    write_json(outdir / "search_report.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
