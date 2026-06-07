from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_EXTERNAL_ROOT = ROOT / "external_data"
OUTDIR = ROOT / "v438_sony_nd_audit_only"

SONY_DIR_NAMES = ("sonytabletennis", "sony_table_tennis", "sony")
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".license"}
TABLE_EXTENSIONS = {".csv", ".tsv"}
POSSIBLE_COARSE_FEATURE_COLUMNS = [
    "coarse_family",
    "phase",
    "landing_depth_bin",
    "landing_side_bin",
    "speed",
    "spin",
    "terminal",
    "target_terminal",
]


def classify_sony_license_policy(license_text: str = "") -> dict[str, object]:
    text = str(license_text or "").lower()
    has_cc = "creative commons" in text or "cc by" in text
    has_nc = "noncommercial" in text or "non-commercial" in text or "nc" in text
    has_nd = "noderivatives" in text or "no derivatives" in text or "no-derivatives" in text or "nd" in text
    license_evidence_detected = bool((has_cc and has_nc and has_nd) or "cc by-nc-nd" in text)
    return {
        "source_dataset": "sonytabletennis",
        "license_family": "CC BY-NC-ND",
        "license_evidence_detected": license_evidence_detected,
        "tier": "audit_only",
        "audit_only": True,
        "train_allowed_by_default": False,
        "clean_train_allowed": False,
        "requires_explicit_approval_for_feature_learning": True,
        "submission_exports": 0,
        "reason": "Sony table tennis data is treated as CC BY-NC-ND / no-derivatives audit-only by default.",
    }


def scan_sony_sources(root: Path | str = DEFAULT_EXTERNAL_ROOT) -> list[dict[str, object]]:
    base = Path(root)
    summaries: list[dict[str, object]] = []
    for source_dir in _sony_candidate_dirs(base):
        for path in _iter_targeted_sony_files(source_dir):
            summaries.append(_summarize_file(path, base))
    return summaries


def build_sony_audit_report(root: Path | str = DEFAULT_EXTERNAL_ROOT) -> dict[str, object]:
    source_summary = scan_sony_sources(root)
    license_text = _collect_license_text(source_summary)
    policy = classify_sony_license_policy(license_text)
    source_row_count = int(sum(int(row.get("row_count", 0)) for row in source_summary))
    observed_columns = sorted(
        {
            column
            for row in source_summary
            for column in row.get("columns", [])
            if isinstance(column, str)
        }
    )
    possible_coarse_features = [column for column in POSSIBLE_COARSE_FEATURE_COLUMNS if column in observed_columns]

    report = {
        **policy,
        "source_root": str(Path(root)),
        "source_files": source_summary,
        "source_file_count": len(source_summary),
        "source_row_count": source_row_count,
        "trainable_rows": 0,
        "canonical_training_rows_written": 0,
        "clean_training_exports": 0,
        "possible_coarse_features_if_later_approved": possible_coarse_features,
        "blocked_uses": ["clean training rows", "canonical training exports", "submission exports"],
        "allowed_uses": ["local license/file audit", "source row/file inventory"],
    }
    return report


def run_pipeline(
    *,
    root: Path | str = DEFAULT_EXTERNAL_ROOT,
    outdir: Path | str = OUTDIR,
    write_source_summary: bool = True,
) -> dict[str, object]:
    output_path = Path(outdir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = build_sony_audit_report(root)
    (output_path / "sony_nd_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if write_source_summary:
        pd.DataFrame(report["source_files"]).to_csv(output_path / "source_file_summary.csv", index=False)
    return report


def _sony_candidate_dirs(root: Path) -> list[Path]:
    if root.is_dir() and _looks_like_sony_dir(root.name):
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    candidates: list[Path] = []
    for name in SONY_DIR_NAMES:
        path = root / name
        if path.is_dir():
            candidates.append(path)
    for child in root.iterdir():
        if child.is_dir() and _looks_like_sony_dir(child.name) and child not in candidates:
            candidates.append(child)
    return sorted(candidates)


def _iter_targeted_sony_files(source_dir: Path) -> Iterable[Path]:
    allowed_suffixes = TEXT_EXTENSIONS | TABLE_EXTENSIONS
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and (_is_license_like(path) or path.suffix.lower() in allowed_suffixes):
            yield path


def _summarize_file(path: Path, root: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    summary: dict[str, object] = {
        "relative_path": path.relative_to(root).as_posix() if _is_relative_to(path, root) else path.as_posix(),
        "file_name": path.name,
        "file_size_bytes": int(path.stat().st_size),
        "row_count": 0,
        "columns": [],
        "license_text_sample": "",
    }
    if suffix in TABLE_EXTENSIONS:
        table = _read_table_head(path)
        summary["columns"] = [str(column) for column in table.columns]
        summary["row_count"] = _count_table_rows(path, suffix)
    elif _is_license_like(path) or suffix in TEXT_EXTENSIONS:
        text = _read_text_sample(path)
        summary["license_text_sample"] = text[:1000]
    return summary


def _read_table_head(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return pd.read_csv(path, sep=separator, nrows=5, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _count_table_rows(path: Path, suffix: str) -> int:
    separator = "\t" if suffix == ".tsv" else ","
    try:
        return int(sum(len(chunk) for chunk in pd.read_csv(path, sep=separator, chunksize=50_000, low_memory=False)))
    except Exception:
        return 0


def _collect_license_text(source_summary: list[dict[str, object]]) -> str:
    chunks = []
    for row in source_summary:
        sample = row.get("license_text_sample")
        if sample:
            chunks.append(str(sample))
    return "\n".join(chunks)


def _read_text_sample(path: Path, limit: int = 16_384) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _is_license_like(path: Path) -> bool:
    name = path.name.lower()
    return name in {"license", "licence", "copying"} or "license" in name or "licence" in name or "readme" in name


def _looks_like_sony_dir(name: str) -> bool:
    return "sony" in name.lower()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> None:
    report = run_pipeline()
    print(
        "V438 Sony ND audit complete: "
        f"tier={report['tier']} "
        f"clean_train_allowed={report['clean_train_allowed']} "
        f"trainable_rows={report['trainable_rows']} "
        f"submission_exports={report['submission_exports']} "
        f"source_files={report['source_file_count']} "
        f"source_rows={report['source_row_count']}"
    )


if __name__ == "__main__":
    main()
