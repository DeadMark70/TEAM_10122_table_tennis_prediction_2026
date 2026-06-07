"""V352 public response lab.

Builds report-only tables from historical public leaderboard results and local
candidate metadata. This script intentionally avoids submission generation,
upload candidate directories, TTMATCH inputs, and old-server branches.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_DIR = Path("v352_public_response_lab")

V338_PL = 0.3590041
V306_PL = 0.3577905

FALLBACK_PUBLIC_RESULTS = [
    {"candidate": "V300", "source": "fallback_known_anchor", "public_pl": 0.3576975},
    {"candidate": "V306", "source": "fallback_known_anchor", "public_pl": 0.3577905},
    {"candidate": "V338", "source": "fallback_known_anchor", "public_pl": 0.3590041},
    {"candidate": "V341", "source": "fallback_known_anchor", "public_pl": 0.3581101},
    {"candidate": "V191", "source": "fallback_known_anchor", "public_pl": 0.3509562},
    {"candidate": "V220", "source": "fallback_known_anchor", "public_pl": 0.3542440},
    {"candidate": "V291", "source": "fallback_known_anchor", "public_pl": 0.3559391},
]

METADATA_INPUTS = [
    Path("v342_public_like_validation_lab/v342_candidate_audit.csv"),
    Path("v350_research_dashboard/candidate_priority.csv"),
    Path("v351_v338_pruning_trust_model/candidate_summary.csv"),
    Path("r200_local_validation_dashboard/r200_candidate_summary.csv"),
]


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _candidate_key(value: object) -> str:
    text = _norm_text(value).lower().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    text = re.sub(r"\.csv$", "", text)
    text = re.sub(r"^submission_", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def classify_family(candidate_name: object) -> str:
    """Map candidate/file names to public-response family labels."""

    key = _candidate_key(candidate_name)
    raw = _norm_text(candidate_name).lower()
    combined = f"{key} {raw}"

    if "ttmatch" in combined or "tt_match" in combined:
        return "ttmatch_blocked"
    if "old_server" in combined or "old-server" in combined:
        return "old_server_blocked"
    if "v300" in key:
        return "anchor_v300"
    if "v306" in key:
        return "anchor_v306"
    if "v338" in key:
        return "v338_positive"
    if "v341" in key:
        return "v341_expansion_negative"
    if "v191" in key or "v166" in key:
        return "v191_v166_action_negative"
    if "v220" in key:
        return "v220_action_repair_negative"
    if "v291" in key or "weakclass" in key:
        return "v291_weakclass_negative"
    if "v272" in key or "v277" in key:
        return "v272_v277_point_micro_negative"
    if "v307" in key:
        return "v307_p0_saturated"
    return "unknown"


def is_clean_recommendation(family: object, candidate_name: object) -> bool:
    text = f"{_norm_text(family)} {_norm_text(candidate_name)}".lower()
    blocked = ("old_server", "old-server", "ttmatch", "tt_match")
    return not any(token in text for token in blocked)


def _parse_table_rows(markdown: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    header: list[str] | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        lowered = [cell.lower() for cell in cells]
        if any("public" in cell and ("pl" in cell or "lb" in cell) for cell in lowered):
            header = lowered
            continue
        if not header or len(cells) != len(header):
            continue
        pl_index = next(
            (
                idx
                for idx, name in enumerate(header)
                if "public" in name and ("pl" in name or "lb" in name)
            ),
            None,
        )
        if pl_index is None:
            continue
        match = re.search(r"\b0\.\d{4,}\b", cells[pl_index])
        if not match:
            continue
        candidate = _candidate_from_table(header, cells)
        if candidate:
            rows.append(
                {
                    "candidate": candidate,
                    "source": "experiments_log_table",
                    "public_pl": float(match.group(0)),
                }
            )
    return rows


def _candidate_from_table(header: list[str], cells: list[str]) -> str | None:
    for wanted in ("id", "version", "name", "candidate"):
        if wanted in header:
            value = cells[header.index(wanted)]
            if value and value.lower() not in {"pending", "not submitted"}:
                return value
    for wanted in ("file", "submission"):
        if wanted in header:
            value = cells[header.index(wanted)]
            if value and value.lower() not in {"pending", "not generated", "not submitted"}:
                return value
    return None


def _parse_inline_rows(markdown: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"(?P<candidate>(?:submission_)?(?:v|V|r|R)\d+[A-Za-z0-9_.\-]*)"
        r"(?:(?!\n).){0,80}?"
        r"(?:Public\s+LB\s*/\s*PL|public\s+PL|PL|score(?:d)?)"
        r"\s*[:=]?\s*"
        r"(?P<pl>0\.\d{4,})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(markdown):
        rows.append(
            {
                "candidate": match.group("candidate").strip("`.,"),
                "source": "experiments_log_inline",
                "public_pl": float(match.group("pl")),
            }
        )
    return rows


def parse_public_results(markdown: str, include_fallback: bool = True) -> list[dict[str, object]]:
    """Extract candidate public PL values from markdown plus known anchors."""

    parsed = _parse_table_rows(markdown) + _parse_inline_rows(markdown)
    if include_fallback:
        parsed.extend(dict(row) for row in FALLBACK_PUBLIC_RESULTS)

    by_key: dict[str, dict[str, object]] = {}
    for row in parsed:
        key = _candidate_key(row["candidate"])
        if not key:
            continue
        current = by_key.get(key)
        if current is None or row["source"] != "fallback_known_anchor":
            by_key[key] = row
    return list(by_key.values())


def _read_optional_metadata(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rel_path in METADATA_INPUTS:
        path = root / rel_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        candidate_col = _find_candidate_column(frame)
        if candidate_col is None:
            continue
        frame = frame.copy()
        frame["candidate_key"] = frame[candidate_col].map(_candidate_key)
        frame["metadata_source"] = str(rel_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["candidate_key"])
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged.groupby("candidate_key", as_index=False).first()


def _find_candidate_column(frame: pd.DataFrame) -> str | None:
    for col in ("candidate", "name", "file", "path", "resolved_path"):
        if col in frame.columns:
            return col
    return None


def _coalesce_columns(frame: pd.DataFrame, target: str, candidates: Iterable[str]) -> None:
    values = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    for col in candidates:
        if col in frame.columns:
            values = values.combine_first(frame[col])
    frame[target] = values


def build_public_response_table(
    root: str | Path = ".",
    log_text: str | None = None,
    include_fallback: bool = True,
) -> pd.DataFrame:
    root = Path(root)
    if log_text is None:
        log_path = root / "experiments_log.md"
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    public_rows = parse_public_results(log_text, include_fallback=include_fallback)
    table = pd.DataFrame(public_rows)
    if table.empty:
        table = pd.DataFrame(columns=["candidate", "source", "public_pl"])
    table["candidate_key"] = table["candidate"].map(_candidate_key)
    table["family"] = table["candidate"].map(classify_family)
    table = table.sort_values(["candidate_key", "source"]).drop_duplicates("candidate_key", keep="first")

    metadata = _read_optional_metadata(root)
    if not metadata.empty:
        table = table.merge(metadata, how="left", on="candidate_key", suffixes=("", "_meta"))

    _coalesce_columns(table, "action_churn", ("action_churn_vs_v306", "action_churn_vs_anchor"))
    _coalesce_columns(table, "point_churn", ("point_churn_vs_v306", "point_churn_vs_anchor"))
    _coalesce_columns(table, "server_churn", ("server_churn_vs_v306", "server_churn_vs_anchor"))
    _coalesce_columns(table, "v338_overlap_proxy", ("v338_changed_overlap", "overlap_with_v338_changed_rows"))
    _coalesce_columns(table, "v341_extra_risk_proxy", ("v341_extra_overlap_rate", "v341_extra_overlap"))
    _coalesce_columns(
        table,
        "point0_additions_proxy",
        ("point0_additions", "point0_additions_vs_v306", "point0_addition_count"),
    )

    table["public_delta_vs_v338"] = table["public_pl"].astype(float) - V338_PL
    table["public_delta_vs_v306"] = table["public_pl"].astype(float) - V306_PL
    table["point0_addition_family"] = table["point0_additions_proxy"].fillna(0).astype(float) > 0
    table["clean_recommendation"] = [
        is_clean_recommendation(family, candidate)
        for family, candidate in zip(table["family"], table["candidate"])
    ]
    table = table.sort_values("public_pl", ascending=False).reset_index(drop=True)
    table["public_rank_desc"] = range(1, len(table) + 1)
    return table


def build_family_response_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "candidate_count",
                "best_public_pl",
                "mean_public_pl",
                "best_delta_vs_v338",
                "clean_recommendation_count",
            ]
        )
    return (
        table.groupby("family", dropna=False)
        .agg(
            candidate_count=("candidate", "count"),
            best_public_pl=("public_pl", "max"),
            mean_public_pl=("public_pl", "mean"),
            best_delta_vs_v338=("public_delta_vs_v338", "max"),
            clean_recommendation_count=("clean_recommendation", "sum"),
        )
        .reset_index()
        .sort_values("best_public_pl", ascending=False)
    )


def write_reports(
    root: str | Path = ".",
    log_text: str | None = None,
    include_fallback: bool = True,
) -> dict[str, Path]:
    root = Path(root)
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_public_response_table(root=root, log_text=log_text, include_fallback=include_fallback)
    summary = build_family_response_summary(table)

    table_path = output_dir / "public_response_table.csv"
    summary_path = output_dir / "family_response_summary.csv"
    report_path = output_dir / "search_report.json"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "rows": int(len(table)),
        "families": sorted(str(family) for family in table["family"].dropna().unique()),
        "metadata_inputs_present": [
            str(path) for path in METADATA_INPUTS if (root / path).exists()
        ],
        "outputs": {
            "public_response_table": str(table_path),
            "family_response_summary": str(summary_path),
            "search_report": str(report_path),
        },
        "policy": {
            "wrote_upload_candidates": False,
            "used_ttmatch": False,
            "used_old_server_branch": False,
            "generated_submissions": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "public_response_table": table_path,
        "family_response_summary": summary_path,
        "search_report": report_path,
    }


def main() -> None:
    outputs = write_reports(Path("."))
    print("V352 public response lab wrote:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
