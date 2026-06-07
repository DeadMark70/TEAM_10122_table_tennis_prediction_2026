"""V413 external license and overlap guard.

Filters the V412 canonical corpus for the clean branch. This is a policy gate:
no TTMATCH, no ND/audit-only first-version rows, and no AICUP-like schema rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
V411_MANIFEST = ROOT / "v411_external_inventory_lockfile" / "external_file_manifest.csv"
V412_CANONICAL = ROOT / "v412_clean_canonical_external" / "canonical_external_events.csv"
OUTDIR = ROOT / "v413_external_license_overlap_guard"

AICUP_SCHEMA_SIGNATURE = {"rally_uid", "actionId", "pointId", "serverGetPoint", "spinId", "strengthId"}


def _parse_columns_json(value: Any) -> set[str]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return set()
    if not isinstance(parsed, list):
        return set()
    return {str(item) for item in parsed}


def _blocked_schema_sources(manifest: pd.DataFrame) -> set[str]:
    if manifest is None or manifest.empty or "columns_json" not in manifest.columns:
        return set()
    blocked: set[str] = set()
    for _, row in manifest.iterrows():
        columns = _parse_columns_json(row.get("columns_json"))
        if AICUP_SCHEMA_SIGNATURE.issubset(columns):
            blocked.add(str(row.get("source_dataset")))
    return blocked


def _row_block_reason(row: pd.Series, schema_blocked_sources: set[str]) -> str | None:
    source = str(row.get("source_dataset", ""))
    license_tag = str(row.get("license_tag", ""))
    risk_tier = str(row.get("risk_tier", ""))
    source_file = str(row.get("source_file", ""))
    if source in schema_blocked_sources:
        return "aicup_like_schema_signature"
    lowered = " ".join([source, source_file, license_tag, risk_tier]).lower()
    if "ttmatch" in lowered:
        return "excluded_ttmatch_overlap_risk"
    if "excluded_overlap_risk" in lowered:
        return "excluded_overlap_risk"
    if "unknown" in license_tag.lower() or "audit_only_unknown" in risk_tier.lower():
        return "unknown_license_audit_only"
    if "cc-by-nc-nd" in license_tag.lower() or "audit_only_nd" in risk_tier.lower():
        return "cc_by_nc_nd_audit_only"
    return None


def apply_guard(
    canonical: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if canonical.empty:
        empty = pd.DataFrame(columns=list(canonical.columns) + ["block_reason"])
        report = {
            "version": "V413",
            "input_rows": 0,
            "clean_rows": 0,
            "blocked_rows": 0,
            "blocked_sources": [],
            "allowed_sources": [],
            "warnings": [],
        }
        return canonical.copy(), report, empty, empty

    schema_blocked_sources = _blocked_schema_sources(manifest)
    guarded = canonical.copy()
    guarded["block_reason"] = [
        _row_block_reason(row, schema_blocked_sources) for _, row in guarded.iterrows()
    ]
    blocked = guarded[guarded["block_reason"].notna()].copy()
    clean = guarded[guarded["block_reason"].isna()].drop(columns=["block_reason"]).copy()

    warnings: list[str] = []
    license_text = " ".join(str(value) for value in clean.get("license_tag", pd.Series(dtype=str)).dropna().unique())
    if "NC-SA" in license_text or "BY-SA" in license_text:
        warnings.append("report_required_share_alike_or_nc")
    if schema_blocked_sources:
        warnings.append("aicup_like_schema_sources_blocked")

    allowed = (
        clean[["source_dataset", "license_tag", "risk_tier"]]
        .drop_duplicates()
        .sort_values(["source_dataset", "license_tag"])
        if not clean.empty
        else pd.DataFrame(columns=["source_dataset", "license_tag", "risk_tier"])
    )
    blocked_sources = (
        blocked[["source_dataset", "license_tag", "risk_tier", "block_reason"]]
        .drop_duplicates()
        .sort_values(["source_dataset", "block_reason"])
        if not blocked.empty
        else pd.DataFrame(columns=["source_dataset", "license_tag", "risk_tier", "block_reason"])
    )
    report = {
        "version": "V413",
        "input_rows": int(len(canonical)),
        "clean_rows": int(len(clean)),
        "blocked_rows": int(len(blocked)),
        "blocked_sources": sorted(set(blocked_sources["source_dataset"])) if not blocked_sources.empty else [],
        "allowed_sources": sorted(set(allowed["source_dataset"])) if not allowed.empty else [],
        "warnings": warnings,
        "schema_blocked_sources": sorted(schema_blocked_sources),
    }
    return clean, report, blocked_sources, allowed


def run_pipeline(
    *,
    canonical_path: Path = V412_CANONICAL,
    manifest_path: Path = V411_MANIFEST,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(canonical_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False) if Path(manifest_path).exists() else pd.DataFrame()
    clean, report, blocked, allowed = apply_guard(canonical, manifest)

    clean_path = outdir / "canonical_clean_events.csv"
    blocked_path = outdir / "blocked_sources.csv"
    allowed_path = outdir / "allowed_sources.csv"
    report_path = outdir / "license_guard_report.json"
    clean.to_csv(clean_path, index=False)
    blocked.to_csv(blocked_path, index=False)
    allowed.to_csv(allowed_path, index=False)
    report["outputs"] = {
        "canonical_clean_events": str(clean_path),
        "blocked_sources": str(blocked_path),
        "allowed_sources": str(allowed_path),
        "license_guard_report": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
