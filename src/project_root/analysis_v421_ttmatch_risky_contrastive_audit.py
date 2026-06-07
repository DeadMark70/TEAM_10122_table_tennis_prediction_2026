"""V421 risky TTMATCH contrastive audit.

This script is intentionally quarantine-only. It may inspect files under
external_data/TTMATCH, summarize schema and overlap risk, and write reports, but
it must not export upload candidates or exact-label training data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
TTMATCH_DIR = ROOT / "external_data" / "TTMATCH"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
OUTDIR = ROOT / "v421_ttmatch_risky_contrastive_audit"

RISK_REASON = "TTMATCH has AICUP-like schema/overlap risk"
AICUP_LIKE_SIGNATURE = {"rally_uid", "strikeNumber", "actionId", "pointId", "spinId", "strengthId"}
EXACT_LABEL_COLUMNS = {"actionId", "pointId", "serverGetPoint"}
COARSE_CONTEXT_COLUMNS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "handId",
    "positionId",
]
COLUMN_ALIASES = {
    "strickNumber": "strikeNumber",
    "strickId": "strikeId",
}


def _read_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _read_csv_sample(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def _normalized_columns(columns: list[str]) -> list[str]:
    return [COLUMN_ALIASES.get(column, column) for column in columns]


def _normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.rename(columns={old: new for old, new in COLUMN_ALIASES.items() if old in frame.columns})


def _file_rows(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return None


def _safe_relative(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _csv_files_under(ttmatch_dir: Path) -> list[Path]:
    ttmatch_dir = Path(ttmatch_dir)
    if not ttmatch_dir.exists():
        return []
    root = ttmatch_dir.resolve()
    return sorted(path for path in ttmatch_dir.rglob("*.csv") if path.is_file() and path.resolve().is_relative_to(root))


def _row_signatures(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    if frame.empty or not columns:
        return set()
    subset = frame.loc[:, columns].fillna("<NA>").astype(str)
    return {
        hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()
        for values in subset.itertuples(index=False, name=None)
    }


def _count_row_overlaps(ttmatch_frames: list[pd.DataFrame], aicup: pd.DataFrame, columns: list[str]) -> int:
    if aicup.empty or not columns or not set(columns).issubset(aicup.columns):
        return 0
    aicup_signatures = _row_signatures(aicup, columns)
    overlaps = 0
    for frame in ttmatch_frames:
        if set(columns).issubset(frame.columns):
            overlaps += len(_row_signatures(frame, columns) & aicup_signatures)
    return int(overlaps)


def _coarse_context_pair_count(ttmatch_frames: list[pd.DataFrame], train: pd.DataFrame) -> int:
    total = 0
    for frame in ttmatch_frames:
        columns = [col for col in COARSE_CONTEXT_COLUMNS if col in frame.columns and col in train.columns]
        if not columns:
            continue
        total += len(_row_signatures(frame, columns) & _row_signatures(train, columns))
    return int(total)


def audit_ttmatch(
    *,
    ttmatch_dir: Path = TTMATCH_DIR,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
) -> dict[str, Any]:
    """Summarize TTMATCH schema, overlap, and coarse contrastive risk."""

    ttmatch_dir = Path(ttmatch_dir)
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_columns = _read_columns(train_path)
    test_columns = _read_columns(test_path)
    train = _normalize_frame_columns(_read_csv_sample(train_path))
    test = _normalize_frame_columns(_read_csv_sample(test_path))
    normalized_train_columns = _normalized_columns(train_columns)
    normalized_test_columns = _normalized_columns(test_columns)

    files = _csv_files_under(ttmatch_dir)
    file_summaries: list[dict[str, Any]] = []
    ttmatch_frames: list[pd.DataFrame] = []
    max_shared = 0
    aicup_like_count = 0

    for path in files:
        columns = _read_columns(path)
        normalized_columns = _normalized_columns(columns)
        shared_train = sorted(set(normalized_columns) & set(normalized_train_columns))
        shared_test = sorted(set(normalized_columns) & set(normalized_test_columns))
        is_aicup_like = AICUP_LIKE_SIGNATURE.issubset(set(normalized_columns))
        max_shared = max(max_shared, len(shared_train), len(shared_test))
        aicup_like_count += int(is_aicup_like)
        file_summaries.append(
            {
                "path": _safe_relative(path),
                "rows": _file_rows(path),
                "column_count": len(columns),
                "columns": columns,
                "shared_train_columns": shared_train,
                "shared_test_columns": shared_test,
                "aicup_like_schema": bool(is_aicup_like),
            }
        )
        frame = _normalize_frame_columns(_read_csv_sample(path))
        if not frame.empty:
            ttmatch_frames.append(frame)

    row_overlap_columns = sorted(AICUP_LIKE_SIGNATURE)
    summary = {
        "version": "V421",
        "ttmatch_dir": str(ttmatch_dir),
        "ttmatch_present": bool(ttmatch_dir.exists()),
        "file_count": len(files),
        "aicup_like_file_count": int(aicup_like_count),
        "overlap_summary": {
            "train_column_count": len(train_columns),
            "test_column_count": len(test_columns),
            "max_shared_column_count": int(max_shared),
            "files": file_summaries,
        },
        "dedup_summary": {
            "row_signature_columns": row_overlap_columns,
            "row_signature_overlaps_train": _count_row_overlaps(ttmatch_frames, train, row_overlap_columns),
            "row_signature_overlaps_test": _count_row_overlaps(ttmatch_frames, test, row_overlap_columns),
        },
        "contrastive_pair_counts": {
            "coarse_context_pairs": _coarse_context_pair_count(ttmatch_frames, train),
            "exports_exact_label_pairs": 0,
        },
    }
    return summary


def write_risky_report(outdir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "clean_eligible": False,
        "reason": RISK_REASON,
        "submission_exports": 0,
        **summary,
    }
    report_path = outdir / "risky_ttmatch_audit_report.json"
    report["outputs"] = {"risky_ttmatch_audit_report": str(report_path)}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def run_pipeline(
    *,
    ttmatch_dir: Path = TTMATCH_DIR,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    outdir = Path(outdir)
    summary = audit_ttmatch(ttmatch_dir=ttmatch_dir, train_path=train_path, test_path=test_path)
    report = write_risky_report(outdir, summary)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
