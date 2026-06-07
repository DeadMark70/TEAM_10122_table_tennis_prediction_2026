"""V304 clean candidate decision packager.

Aggregates existing clean-candidate evidence into a decision table for the next
upload choice.  This creates no models and does not copy to upload_candidates.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "v304_clean_decision_packager"
DECISION_TABLE_PATH = OUT_DIR / "v304_decision_table.csv"
REPORT_JSON_PATH = OUT_DIR / "v304_report.json"
REPORT_MD_PATH = OUT_DIR / "v304_report.md"
R200_SUMMARY_PATH = ROOT / "r200_local_validation_dashboard" / "r200_candidate_summary.csv"

PUBLIC_BASELINE_SCORE = 0.3576720
V300_BEST_CANDIDATE = "submission_v300_best_safe_repack__v173action_v261point_server.csv"
V261_ANCHOR_CANDIDATE = "V261 cap1"
V302_PLACEHOLDER = "V302 placeholder (absent)"

PUBLIC_RESULTS = [
    {"candidate": V261_ANCHOR_CANDIDATE, "public_score": PUBLIC_BASELINE_SCORE},
    {"candidate": "V300 best_safe_repack", "public_score": 0.3576975},
    {"candidate": "V272", "public_score": 0.3576159},
    {"candidate": "V277", "public_score": 0.3574825},
    {"candidate": "V291", "public_score": 0.3559391},
]

PUBLIC_SCORE_BY_EXACT_CANDIDATE = {
    V261_ANCHOR_CANDIDATE: PUBLIC_BASELINE_SCORE,
    V300_BEST_CANDIDATE: 0.3576975,
    "V300 best_safe_repack": 0.3576975,
    "V272": 0.3576159,
    "V277": 0.3574825,
    "V291": 0.3559391,
}

SEARCH_TABLE_SPECS = [
    ("V300", Path("v300_clean_server_blend_recycler/v300_server_search.csv")),
    ("V299", Path("v299_point_hybrid_selector/v299_candidate_search.csv")),
    ("V301", Path("v301_action_point_consistency_explorer/v301_pair_search.csv")),
    ("V297", Path("v297_multisource_point_agreement/v297_candidate_search.csv")),
    ("V298", Path("v298_action_point_support_prior/v298_candidate_search.csv")),
    ("V295", Path("v295_true_oof_point_specialists/v295_candidate_search.csv")),
    ("V272", Path("v272_action_conditioned_point_residual/v272_point_search.csv")),
    ("V277", Path("v277_v272b_point_refinement/v277_point_search.csv")),
]

DECISION_COLUMNS = [
    "candidate",
    "task_changed",
    "public_score",
    "public_delta_vs_v261",
    "local_delta",
    "churn",
    "risk",
    "upload_priority",
    "source",
    "decision",
    "path",
    "evidence",
]

POINT_ACTION_SOURCES = {"V299", "V301", "V297", "V298", "V295", "V272", "V277"}


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_float(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row:
            value = _optional_float(row.get(name))
            if value is not None:
                return value
    return None


def _clean_optional_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(object)
            out.loc[pd.isna(out[col]), col] = None
    return out


def public_results_frame() -> pd.DataFrame:
    frame = pd.DataFrame(PUBLIC_RESULTS)
    frame["source"] = "embedded_public"
    frame["public_delta_vs_v261"] = frame["public_score"] - PUBLIC_BASELINE_SCORE
    return frame.loc[:, ["candidate", "source", "public_score", "public_delta_vs_v261"]]


def read_search_tables(root: Path = ROOT) -> tuple[list[tuple[str, Path, pd.DataFrame]], list[dict[str, Any]]]:
    tables: list[tuple[str, Path, pd.DataFrame]] = []
    manifest: list[dict[str, Any]] = []
    for version, rel_path in SEARCH_TABLE_SPECS:
        path = Path(root) / rel_path
        item: dict[str, Any] = {
            "version": version,
            "path": str(rel_path).replace("\\", "/"),
            "status": "missing",
            "rows": 0,
        }
        if path.exists():
            try:
                frame = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - report diagnostics only
                item["status"] = "error"
                item["error"] = str(exc)
            else:
                item["status"] = "loaded"
                item["rows"] = int(len(frame))
                tables.append((version, rel_path, frame))
        manifest.append(item)
    return tables, manifest


def load_r200_summary(path: Path = R200_SUMMARY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _r200_lookup(r200_summary: pd.DataFrame | None) -> dict[str, pd.Series]:
    if r200_summary is None or r200_summary.empty or "candidate" not in r200_summary.columns:
        return {}
    return {
        str(row["candidate"]): row
        for _, row in r200_summary.iterrows()
        if pd.notna(row.get("candidate"))
    }


def _task_changed(source: str, row: pd.Series, r200_row: pd.Series | None = None) -> str:
    action = _first_float(row, ["action_churn", "action_changed_vs_anchor"])
    point = _first_float(row, ["point_churn", "point_changed_vs_anchor"])
    server = _first_float(row, ["server_mad_vs_anchor"])
    if r200_row is not None:
        action = _first_float(r200_row, ["action_churn_vs_anchor"]) if action is None else action
        point = _first_float(r200_row, ["point_churn_vs_anchor"]) if point is None else point
        server = _first_float(r200_row, ["server_mad_vs_anchor"]) if server is None else server

    if source == "V300" or (server is not None and server > 0.0 and not action and not point):
        return "server"
    if (action or 0.0) > 0.0 and (point or 0.0) > 0.0:
        return "action_point"
    if (action or 0.0) > 0.0:
        return "action"
    if source in POINT_ACTION_SOURCES or (point or 0.0) > 0.0:
        return "point"
    return "none"


def _local_delta(source: str, row: pd.Series) -> float | None:
    value = _first_float(
        row,
        [
            "local_delta",
            "proxy_delta_vs_proxy_base",
            "ordinary_delta_vs_base",
            "delta_vs_v261",
            "delta_vs_v294_base",
            "delta_vs_aligned_base",
            "available_source_local_delta",
            "public_like_delta",
        ],
    )
    if value is not None:
        return value
    if source == "V301":
        return _first_float(row, ["local_oof_proxy_delta"])
    return None


def _churn(row: pd.Series, r200_row: pd.Series | None = None) -> float | None:
    values = [
        _first_float(row, ["action_churn", "action_changed_vs_anchor"]),
        _first_float(row, ["point_churn", "point_changed_vs_anchor"]),
        _first_float(row, ["server_mad_vs_anchor"]),
    ]
    if r200_row is not None:
        values.extend(
            [
                _first_float(r200_row, ["action_churn_vs_anchor"]),
                _first_float(r200_row, ["point_churn_vs_anchor"]),
                _first_float(r200_row, ["server_mad_vs_anchor"]),
            ]
        )
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _public_score(candidate: str) -> float | None:
    if candidate in PUBLIC_SCORE_BY_EXACT_CANDIDATE:
        return PUBLIC_SCORE_BY_EXACT_CANDIDATE[candidate]
    return None


def _risk_and_priority(source: str, candidate: str, row: pd.Series) -> tuple[str, int, str]:
    lower = candidate.lower()
    public_score = _public_score(candidate)
    public_delta = None if public_score is None else public_score - PUBLIC_BASELINE_SCORE
    recommendation = str(row.get("recommendation", row.get("upload_recommendation", row.get("verdict", ""))))
    local_delta = _local_delta(source, row)

    if candidate == V300_BEST_CANDIDATE:
        return "CURRENT_CLEAN_BEST", 1, "UPLOAD_NEXT_CLEAN_BEST"
    if candidate == V261_ANCHOR_CANDIDATE:
        return "CLEAN_ANCHOR", 2, "ANCHOR_FALLBACK"
    if candidate == V302_PLACEHOLDER:
        return "UNKNOWN_ABSENT", 3, "WAIT_FOR_ARTIFACT"
    if source == "V300":
        if str(row.get("risk_tier", "")).lower() == "safe":
            return "SERVER_LOW_RISK_REVIEW", 10, "KEEP_AS_SERVER_REVIEW"
        return "SERVER_REVIEW", 20, "REVIEW_ONLY"
    if public_delta is not None and public_delta < 0.0:
        return "REJECT_PUBLIC_WEAK", 90, "DO_NOT_UPLOAD"
    if source in POINT_ACTION_SOURCES:
        return "REJECT_POINT_ACTION_WEAK", 91, "DO_NOT_UPLOAD"
    if any(token in lower for token in ["rare134", "point0", "bank"]):
        return "REJECT_POINT_ACTION_WEAK", 91, "DO_NOT_UPLOAD"
    if local_delta is not None and local_delta < 0.0:
        return "REJECT_LOCAL_NEGATIVE", 92, "DO_NOT_UPLOAD"
    if "do_not_upload" in recommendation.lower():
        return "REJECT_POINT_ACTION_WEAK", 91, "DO_NOT_UPLOAD"
    return "HOLD_UNKNOWN_PUBLIC", 50, "REVIEW_ONLY"


def _evidence_summary(source: str, row: pd.Series, r200_row: pd.Series | None = None) -> str:
    parts = [f"source={source}"]
    for name in [
        "blend_kind",
        "risk_tier",
        "verdict",
        "recommendation",
        "upload_recommendation",
        "decision",
        "proxy_delta_vs_proxy_base",
        "ordinary_delta_vs_base",
        "delta_vs_v294_base",
        "available_source_local_delta",
        "support_delta",
    ]:
        if name in row and pd.notna(row.get(name)):
            parts.append(f"{name}={row.get(name)}")
    if r200_row is not None:
        decision = r200_row.get("decision")
        if pd.notna(decision):
            parts.append(f"r200_decision={decision}")
    return "; ".join(parts)


def _public_decision_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in public_results_frame().to_dict("records"):
        candidate = str(row["candidate"])
        if candidate == "V300 best_safe_repack":
            continue
        series = pd.Series(row)
        risk, priority, decision = _risk_and_priority("embedded_public", candidate, series)
        rows.append(
            {
                "candidate": candidate,
                "task_changed": "none" if candidate == V261_ANCHOR_CANDIDATE else "point",
                "public_score": float(row["public_score"]),
                "public_delta_vs_v261": float(row["public_delta_vs_v261"]),
                "local_delta": None,
                "churn": None,
                "risk": risk,
                "upload_priority": priority,
                "source": "embedded_public",
                "decision": decision,
                "path": None,
                "evidence": "embedded public result",
            }
        )
    return rows


def _v302_placeholder_row(workspace_root: Path) -> dict[str, Any]:
    return {
        "candidate": V302_PLACEHOLDER,
        "task_changed": "unknown",
        "public_score": None,
        "public_delta_vs_v261": None,
        "local_delta": None,
        "churn": None,
        "risk": "UNKNOWN_ABSENT",
        "upload_priority": 3,
        "source": "placeholder",
        "decision": "WAIT_FOR_ARTIFACT",
        "path": None,
        "evidence": "V302 was not part of the requested V304 evidence input set.",
    }


def build_decision_table(
    search_tables: list[tuple[str, Path, pd.DataFrame]] | None = None,
    r200_summary: pd.DataFrame | None = None,
    workspace_root: Path = ROOT,
) -> pd.DataFrame:
    rows = _public_decision_rows()
    r200_by_candidate = _r200_lookup(r200_summary)

    for source, rel_path, table in search_tables or []:
        if table.empty or "candidate" not in table.columns:
            continue
        for _, row in table.iterrows():
            candidate = str(row.get("candidate"))
            r200_row = r200_by_candidate.get(candidate)
            public_score = _public_score(candidate)
            risk, priority, decision = _risk_and_priority(source, candidate, row)
            rows.append(
                {
                    "candidate": candidate,
                    "task_changed": _task_changed(source, row, r200_row),
                    "public_score": public_score,
                    "public_delta_vs_v261": None
                    if public_score is None
                    else public_score - PUBLIC_BASELINE_SCORE,
                    "local_delta": _local_delta(source, row),
                    "churn": _churn(row, r200_row),
                    "risk": risk,
                    "upload_priority": priority,
                    "source": source,
                    "decision": decision,
                    "path": row.get("path") if "path" in row else str(rel_path),
                    "evidence": _evidence_summary(source, row, r200_row),
                }
            )

    rows.append(_v302_placeholder_row(workspace_root))

    table = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    table = _clean_optional_columns(
        table,
        ["public_score", "public_delta_vs_v261", "local_delta", "churn", "path"],
    )
    table = table.sort_values(
        ["upload_priority", "public_delta_vs_v261", "local_delta", "churn", "candidate"],
        ascending=[True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return table.loc[:, DECISION_COLUMNS]


def build_report_payload(
    decision_table: pd.DataFrame,
    manifest: list[dict[str, Any]],
    r200_summary_rows: int,
) -> dict[str, Any]:
    best = decision_table.iloc[0].to_dict() if not decision_table.empty else {}
    return {
        "version": "V304",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Clean candidate decision packager; no new model and no upload copy.",
        "public_baseline_candidate": V261_ANCHOR_CANDIDATE,
        "public_baseline_score": PUBLIC_BASELINE_SCORE,
        "embedded_public_results": PUBLIC_RESULTS,
        "decision_rows": int(len(decision_table)),
        "decision_columns": DECISION_COLUMNS,
        "search_table_manifest": manifest,
        "r200_summary_rows": int(r200_summary_rows),
        "current_best_clean": best.get("candidate"),
        "recommendation": [
            f"Rank {V300_BEST_CANDIDATE} as the current clean best based on the embedded public score 0.3576975.",
            "Keep V261 cap1 as the clean anchor fallback.",
            "Treat V302 as unknown when no artifact/search output is present.",
            "Do not upload point/action weak candidates from V272/V277/V295/V297/V298/V299/V301 without new positive public evidence.",
        ],
        "no_new_model": True,
        "no_upload_candidates_copy": True,
        "outputs": {
            "decision_table": str(DECISION_TABLE_PATH.relative_to(ROOT)),
            "report_json": str(REPORT_JSON_PATH.relative_to(ROOT)),
            "report_md": str(REPORT_MD_PATH.relative_to(ROOT)),
        },
    }


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = [str(col) for col in frame.columns]
    rows = [
        [_escape_markdown_cell(value) for value in row]
        for row in frame.astype(object).where(pd.notna(frame), "").itertuples(index=False, name=None)
    ]
    widths = [
        max([len(columns[idx])] + [len(row[idx]) for row in rows])
        for idx in range(len(columns))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    out = [fmt(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    out.extend(fmt(row) for row in rows)
    return out


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(report: dict[str, Any], decision_table: pd.DataFrame) -> str:
    lines = [
        "# V304 Clean Candidate Decision Packager",
        "",
        "No model training and no upload_candidates copy. This report packages current clean-candidate evidence only.",
        "",
        f"- Decision rows: `{report['decision_rows']}`",
        f"- R200 summary rows: `{report['r200_summary_rows']}`",
        f"- Public baseline: `{report['public_baseline_candidate']}` = `{report['public_baseline_score']:.7f}`",
        f"- Current clean best: `{report['current_best_clean']}`",
        "",
        "## Recommendation",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommendation"])
    lines.extend(["", "## Search Table Manifest", ""])
    lines.extend(
        f"- {item['version']}: {item['status']} rows={item.get('rows', 0)} path={item['path']}"
        for item in report["search_table_manifest"]
    )
    lines.extend(["", "## Top Decision Rows", ""])
    show_cols = [
        "candidate",
        "task_changed",
        "public_score",
        "public_delta_vs_v261",
        "local_delta",
        "churn",
        "risk",
        "upload_priority",
        "source",
    ]
    lines.extend(_markdown_table(decision_table.head(20).loc[:, show_cols]))
    lines.append("")
    return "\n".join(lines)


def run(root: Path = ROOT, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    tables, manifest = read_search_tables(root)
    r200_path = Path(root) / "r200_local_validation_dashboard" / "r200_candidate_summary.csv"
    r200_summary = load_r200_summary(r200_path)
    decision_table = build_decision_table(tables, r200_summary, workspace_root=root)
    report = build_report_payload(decision_table, manifest, len(r200_summary))

    out_dir.mkdir(parents=True, exist_ok=True)
    decision_table.to_csv(out_dir / "v304_decision_table.csv", index=False)
    (out_dir / "v304_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "v304_report.md").write_text(render_markdown_report(report, decision_table), encoding="utf-8")
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "outdir": str(OUT_DIR.relative_to(ROOT)),
                "decision_rows": report["decision_rows"],
                "current_best_clean": report["current_best_clean"],
                "no_upload_candidates_copy": report["no_upload_candidates_copy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
