"""V411 external inventory lockfile.

Creates a reproducible manifest for external datasets before any clean
pretraining conversion. This script is read-only with respect to external_data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"
OUTDIR = ROOT / "v411_external_inventory_lockfile"

SOURCE_RULES = {
    "openttgames": ("CC-BY-NC-SA-4.0", "clean_nc_sa", True),
    "DeepMindrobottabletennis": ("CC-BY-4.0-data_Apache-2.0-code", "clean_physics", True),
    "TT3D": ("CC-BY-4.0", "clean_physics", True),
    "CoachAI-Projects-main": ("MIT_repo_citation_required", "clean_cross_sport_coarse", True),
    "AIMY": ("DL-DE-BY-2-0", "clean_physics", True),
    "spindoe": ("CC-BY-SA-4.0", "clean_physics_sa", True),
    "sonytabletennis": ("CC-BY-NC-ND-4.0_audit_only", "audit_only_nd", False),
    "TTMATCH": ("excluded_overlap_risk", "excluded_overlap_risk", False),
    "TT-MatchDynamics": ("unknown_audit_only", "audit_only_unknown", False),
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_dataset(path: Path, external_root: Path = EXTERNAL) -> str:
    try:
        rel = path.resolve().relative_to(external_root.resolve())
    except ValueError:
        return "unknown"
    return rel.parts[0] if rel.parts else "unknown"


def _safe_relative(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _metadata_for_source(source: str) -> tuple[str, str, bool]:
    if source in SOURCE_RULES:
        return SOURCE_RULES[source]
    lowered = source.lower()
    if "aimy" in lowered:
        return SOURCE_RULES["AIMY"]
    if "spin" in lowered:
        return SOURCE_RULES["spindoe"]
    return ("unknown", "audit_only_unknown", False)


def _csv_info(path: Path) -> tuple[int | None, list[str]]:
    try:
        header = pd.read_csv(path, nrows=0)
        columns = list(header.columns)
    except Exception:
        columns = []
    try:
        with path.open("rb") as handle:
            lines = sum(1 for _ in handle)
        row_count = max(lines - 1, 0)
    except Exception:
        row_count = None
    return row_count, columns


def _json_info(path: Path) -> tuple[int | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None, []
    if isinstance(data, list):
        columns = sorted({key for row in data[:100] if isinstance(row, dict) for key in row.keys()})
        return len(data), columns
    if isinstance(data, dict):
        columns = sorted(data.keys())
        for value in data.values():
            if isinstance(value, list):
                return len(value), columns
        return len(data), columns
    return None, []


def _file_info(path: Path) -> tuple[int | None, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_info(path)
    if suffix == ".json":
        return _json_info(path)
    return None, []


def build_manifest(*, external_root: Path = EXTERNAL) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not external_root.exists():
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "path",
                "relative_path",
                "size_bytes",
                "sha256",
                "extension",
                "row_count",
                "columns_json",
                "license_tag",
                "risk_tier",
                "allowed_first_version",
                "notes",
            ]
        )
    for path in sorted(p for p in external_root.rglob("*") if p.is_file()):
        source = _source_dataset(path, external_root)
        license_tag, risk_tier, allowed = _metadata_for_source(source)
        row_count, columns = _file_info(path)
        rows.append(
            {
                "source_dataset": source,
                "path": str(path),
                "relative_path": _safe_relative(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
                "extension": path.suffix.lower(),
                "row_count": row_count,
                "columns_json": json.dumps(columns, ensure_ascii=False),
                "license_tag": license_tag,
                "risk_tier": risk_tier,
                "allowed_first_version": bool(allowed),
                "notes": "",
            }
        )
    manifest = pd.DataFrame(rows)
    if "allowed_first_version" in manifest.columns:
        manifest["allowed_first_version"] = manifest["allowed_first_version"].astype(object)
    return manifest


def summarize(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest.empty:
        empty = pd.DataFrame()
        return empty, empty
    dataset_summary = (
        manifest.groupby(["source_dataset", "license_tag", "risk_tier", "allowed_first_version"], dropna=False)
        .agg(file_count=("path", "count"), total_bytes=("size_bytes", "sum"), rows_known=("row_count", "sum"))
        .reset_index()
    )
    license_summary = (
        manifest.groupby(["license_tag", "risk_tier", "allowed_first_version"], dropna=False)
        .agg(file_count=("path", "count"), total_bytes=("size_bytes", "sum"))
        .reset_index()
    )
    return dataset_summary, license_summary


def run_pipeline(*, outdir: Path = OUTDIR, external_root: Path = EXTERNAL) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(external_root=external_root)
    dataset_summary, license_summary = summarize(manifest)

    manifest_path = outdir / "external_file_manifest.csv"
    dataset_summary_path = outdir / "dataset_summary.csv"
    license_summary_path = outdir / "license_summary.csv"
    report_path = outdir / "search_report.json"
    manifest.to_csv(manifest_path, index=False)
    dataset_summary.to_csv(dataset_summary_path, index=False)
    license_summary.to_csv(license_summary_path, index=False)

    report = {
        "version": "V411",
        "external_root": str(external_root),
        "file_count": int(len(manifest)),
        "allowed_first_version_file_count": int(manifest["allowed_first_version"].sum()) if not manifest.empty else 0,
        "sources": dataset_summary.to_dict(orient="records") if not dataset_summary.empty else [],
        "outputs": {
            "manifest": str(manifest_path),
            "dataset_summary": str(dataset_summary_path),
            "license_summary": str(license_summary_path),
            "search_report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
