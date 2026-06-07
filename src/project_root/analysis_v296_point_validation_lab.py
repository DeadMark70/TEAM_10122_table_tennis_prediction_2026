from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v296_point_validation_lab"
PUBLIC_BASELINE_CANDIDATE = "V188 cap2"

PUBLIC_RESULTS = [
    {"candidate": "V188 cap2", "public_score": 0.3573598, "specialist_group": "v188_like"},
    {"candidate": "V188 cap3", "public_score": 0.3573816, "specialist_group": "v188_like"},
    {"candidate": "V188 cap5", "public_score": 0.3573932, "specialist_group": "v188_like"},
    {"candidate": "V261 cap1", "public_score": 0.3576720, "specialist_group": "v261_like"},
    {"candidate": "V272", "public_score": 0.3576159, "specialist_group": "v272_like"},
    {"candidate": "V277", "public_score": 0.3574825, "specialist_group": "v277_like"},
]

SEARCH_TABLE_SPECS = [
    ("V188", Path("v188_point_intent_gru/v188_search.csv")),
    ("V193", Path("v193_v188_calibrated_residual/v193_search.csv")),
    ("V196", Path("v196_point0_calibrated_gru/v196_search.csv")),
    ("V261", Path("v261_action_conditioned_point_residual/v261_point_search.csv")),
    ("V272", Path("v272_action_conditioned_point_residual/v272_point_search.csv")),
    ("V277", Path("v277_v272b_point_refinement/v277_point_search.csv")),
    ("V293", Path("v293_point_weakclass_residual_lab/v293_candidate_search.csv")),
]

PUBLIC_BACKTEST_COLUMNS = [
    "candidate",
    "source",
    "specialist_group",
    "public_score",
    "public_delta_vs_v188_cap2",
    "historical_public_status",
    "local_delta",
    "public_like_delta",
    "point_churn",
    "test_changed_rows",
    "test_point0_rate_delta",
    "long789_mean_delta",
    "rare134_mean_delta",
    "point0_f1_delta",
    "cap",
    "path",
]

RISK_COLUMNS = [
    "candidate",
    "source",
    "specialist_group",
    "risk_label",
    "risk_reasons",
    "public_score",
    "public_delta_vs_v188_cap2",
    "local_delta",
    "public_like_delta",
    "point_churn",
    "test_changed_rows",
    "test_point0_rate_delta",
    "long789_mean_delta",
    "rare134_mean_delta",
    "point0_f1_delta",
]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _status_from_public_delta(public_delta: float | None) -> str:
    if public_delta is None:
        return "UNKNOWN"
    if public_delta > 0.0:
        return "ACCEPTED_VS_V188_CAP2"
    if public_delta < 0.0:
        return "REJECTED_VS_V188_CAP2"
    return "BASELINE"


def infer_specialist_group(candidate: str, fallback: Any = None) -> str:
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    name = str(candidate).lower()
    if "rare134" in name:
        return "rare134"
    if "long789" in name:
        return "long789"
    if "point0" in name or "_p0" in name:
        return "point0"
    if "bank" in name:
        return "bank"
    if "v188" in name:
        return "v188_like"
    if "v261" in name:
        return "v261_like"
    if "v272" in name:
        return "v272_like"
    if "v277" in name:
        return "v277_like"
    return "unknown"


def risk_label(
    *,
    specialist_group: str,
    public_delta: float | None,
    point_churn: float,
    point0_rate_delta: float,
    local_delta: float,
) -> str:
    group = str(specialist_group).lower()
    churn = _finite_float(point_churn)
    p0_delta = _finite_float(point0_rate_delta)
    local = _finite_float(local_delta)

    if "rare134" in group or "bank" in group:
        return "RED"
    if p0_delta > 0.0 or group == "point0":
        return "RED"
    if public_delta is not None and public_delta < 0.0:
        return "RED"
    if local < 0.0:
        return "RED"
    if group == "long789" and churn <= 0.005 and p0_delta <= 0.0:
        return "YELLOW"
    if group in {"v188_like", "v261_like", "v272_like"} and public_delta is not None and public_delta > 0.0:
        return "GREEN"
    if public_delta is not None and public_delta > 0.0 and churn <= 0.005:
        return "GREEN"
    return "YELLOW"


def risk_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    group = str(row.get("specialist_group", "")).lower()
    p0_delta = _finite_float(row.get("test_point0_rate_delta"))
    churn = _finite_float(row.get("point_churn"))
    public_delta = _optional_float(row.get("public_delta_vs_v188_cap2"))
    local_delta = _finite_float(row.get("local_delta"))

    if group in {"rare134", "bank"}:
        reasons.append("rare134/bank expansion has failed historical risk screens")
    if p0_delta > 0.0 or group == "point0":
        reasons.append("point0 expansion is treated as high risk")
    if group == "long789" and churn <= 0.005 and p0_delta <= 0.0:
        reasons.append("long789-only churn <= 0.5% with no point0 increase")
    if public_delta is not None and public_delta > 0.0:
        reasons.append("embedded historical public score improved vs V188 cap2")
    if public_delta is not None and public_delta < 0.0:
        reasons.append("embedded historical public score regressed vs V188 cap2")
    if local_delta < 0.0:
        reasons.append("local/proxy delta is negative")
    if not reasons:
        reasons.append("no direct historical public result; use as diagnostic only")
    return "; ".join(reasons)


def read_search_tables(root: Path = ROOT) -> tuple[list[tuple[str, pd.DataFrame]], list[dict[str, Any]]]:
    tables: list[tuple[str, pd.DataFrame]] = []
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
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                item["status"] = "error"
                item["error"] = str(exc)
            else:
                item["status"] = "loaded"
                item["rows"] = int(len(frame))
                tables.append((str(rel_path).replace("\\", "/"), frame))
        manifest.append(item)
    return tables, manifest


def public_results_frame() -> pd.DataFrame:
    frame = pd.DataFrame(PUBLIC_RESULTS)
    baseline = float(
        frame.loc[frame["candidate"] == PUBLIC_BASELINE_CANDIDATE, "public_score"].iloc[0]
    )
    frame["source"] = "embedded_public"
    frame["public_delta_vs_v188_cap2"] = frame["public_score"] - baseline
    frame["historical_public_status"] = frame["public_delta_vs_v188_cap2"].map(_status_from_public_delta)
    for col in [
        "local_delta",
        "public_like_delta",
        "point_churn",
        "test_changed_rows",
        "test_point0_rate_delta",
        "long789_mean_delta",
        "rare134_mean_delta",
        "point0_f1_delta",
        "cap",
        "path",
    ]:
        frame[col] = None
    return frame.loc[:, PUBLIC_BACKTEST_COLUMNS]


def _row_from_search_table(source: str, row: pd.Series) -> dict[str, Any]:
    candidate = str(row.get("candidate", row.get("name", "unknown_candidate")))
    local_delta = _optional_float(row.get("delta_vs_v261", row.get("local_delta")))
    public_like_delta = _optional_float(row.get("public_like_delta"))
    return {
        "candidate": candidate,
        "source": source,
        "specialist_group": infer_specialist_group(candidate, row.get("specialist_group")),
        "public_score": None,
        "public_delta_vs_v188_cap2": None,
        "historical_public_status": "UNKNOWN",
        "local_delta": local_delta,
        "public_like_delta": public_like_delta,
        "point_churn": _optional_float(row.get("point_churn")),
        "test_changed_rows": _optional_float(row.get("test_changed_rows")),
        "test_point0_rate_delta": _optional_float(row.get("test_point0_rate_delta")),
        "long789_mean_delta": _optional_float(row.get("long789_mean_delta")),
        "rare134_mean_delta": _optional_float(row.get("rare134_mean_delta")),
        "point0_f1_delta": _optional_float(row.get("point0_f1_delta")),
        "cap": _optional_float(row.get("cap")),
        "path": row.get("path"),
    }


def build_public_backtest(search_tables: list[tuple[str, pd.DataFrame]] | None = None) -> pd.DataFrame:
    frames = [public_results_frame()]
    for source, table in search_tables or []:
        if table.empty:
            continue
        rows = [_row_from_search_table(source, row) for _, row in table.iterrows()]
        frames.append(pd.DataFrame(rows, columns=PUBLIC_BACKTEST_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    return out.loc[:, PUBLIC_BACKTEST_COLUMNS]


def build_risk_table(public_backtest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in public_backtest.iterrows():
        public_delta = _optional_float(row.get("public_delta_vs_v188_cap2"))
        label = risk_label(
            specialist_group=str(row.get("specialist_group", "unknown")),
            public_delta=public_delta,
            point_churn=_finite_float(row.get("point_churn")),
            point0_rate_delta=_finite_float(row.get("test_point0_rate_delta")),
            local_delta=_finite_float(row.get("local_delta")),
        )
        item = {col: row.get(col) for col in RISK_COLUMNS if col != "risk_label" and col != "risk_reasons"}
        item["risk_label"] = label
        item["risk_reasons"] = risk_reasons(pd.Series(row))
        rows.append(item)
    risk = pd.DataFrame(rows)
    return risk.loc[:, RISK_COLUMNS]


def key_risk_conclusions(risk: pd.DataFrame) -> list[str]:
    counts = risk["risk_label"].value_counts().to_dict() if not risk.empty else {}
    conclusions = [
        f"Risk label counts: GREEN={int(counts.get('GREEN', 0))}, YELLOW={int(counts.get('YELLOW', 0))}, RED={int(counts.get('RED', 0))}.",
        "V261 remains the strongest embedded public point anchor among the listed historical candidates.",
        "V277 is public-positive vs V188 cap2 but weaker than V261, so it is diagnostic evidence rather than an anchor replacement.",
        "long789-only candidates are YELLOW when churn is <= 0.5% and point0 does not increase.",
        "rare134, bank, and point0-expansion candidates are RED risk until true OOF evidence overturns the historical pattern.",
    ]
    return conclusions


def build_report_payload(
    public_backtest: pd.DataFrame,
    risk: pd.DataFrame,
    search_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "V296",
        "purpose": "Diagnostic public backtest and point-change risk lab; no submissions.",
        "public_baseline_candidate": PUBLIC_BASELINE_CANDIDATE,
        "public_backtest_rows": int(len(public_backtest)),
        "risk_rows": int(len(risk)),
        "public_backtest_columns": PUBLIC_BACKTEST_COLUMNS,
        "risk_columns": RISK_COLUMNS,
        "search_table_manifest": search_manifest,
        "risk_label_counts": {
            str(k): int(v) for k, v in risk["risk_label"].value_counts().sort_index().items()
        },
        "key_risk_conclusions": key_risk_conclusions(risk),
    }


def render_markdown_report(report: dict[str, Any], risk: pd.DataFrame) -> str:
    lines = [
        "# V296 Point Validation Lab",
        "",
        "Diagnostic public backtest and risk lab only. No submissions are generated.",
        "",
        f"- Public backtest rows: {report['public_backtest_rows']}",
        f"- Risk rows: {report['risk_rows']}",
        f"- Public baseline: {report['public_baseline_candidate']}",
        "",
        "## Key Risk Conclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["key_risk_conclusions"])
    lines.extend(["", "## Search Table Manifest", ""])
    lines.extend(
        f"- {item['version']}: {item['status']} rows={item.get('rows', 0)} path={item['path']}"
        for item in report["search_table_manifest"]
    )
    lines.extend(["", "## Risk Label Counts", ""])
    for label, count in report["risk_label_counts"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Highest Signal Rows", ""])
    show = risk.head(12).loc[:, ["candidate", "source", "risk_label", "risk_reasons"]]
    lines.extend(_markdown_table(show))
    lines.append("")
    return "\n".join(lines)


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


def run(root: Path = ROOT, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    tables, manifest = read_search_tables(root)
    public_backtest = build_public_backtest(tables)
    risk = build_risk_table(public_backtest)
    report = build_report_payload(public_backtest, risk, manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    public_backtest.to_csv(out_dir / "v296_public_backtest.csv", index=False)
    risk.to_csv(out_dir / "v296_point_change_risk.csv", index=False)
    (out_dir / "v296_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "v296_report.md").write_text(render_markdown_report(report, risk), encoding="utf-8")
    return report


def main() -> None:
    report = run()
    print(
        "V296 point validation lab wrote "
        f"{report['public_backtest_rows']} public backtest rows to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
