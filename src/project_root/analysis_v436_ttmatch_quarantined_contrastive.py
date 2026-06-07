from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v436_ttmatch_quarantined_contrastive"
TTMATCH_ROOT = ROOT / "external_data" / "TTMATCH"
AICUP_TRAIN = ROOT / "train.csv"
AICUP_TEST = ROOT / "test_new.csv"

ALIAS_RENAME = {
    "strickNumber": "strikeNumber",
    "strickId": "strikeId",
}

AICUP_SIGNATURE_COLUMNS = [
    "strikeNumber",
    "sex",
    "numberGame",
    "scoreSelf",
    "scoreOther",
    "serverGetPoint",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]

CONTEXT_COLUMNS = [
    "match",
    "numberGame",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeNumber",
]


def normalize_signature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize AICUP-like TTMATCH aliases without mutating the caller."""
    rename = {src: dst for src, dst in ALIAS_RENAME.items() if src in frame.columns and dst not in frame.columns}
    out = frame.rename(columns=rename).copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def row_signature_set(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> set[str]:
    """Build exact row signatures over available AICUP-like columns."""
    normalized = normalize_signature_columns(frame)
    signature_columns = [col for col in (columns or AICUP_SIGNATURE_COLUMNS) if col in normalized.columns]
    if not signature_columns:
        return set()
    signatures = _signature_series(normalized, signature_columns)
    return set(signatures)


def build_quarantine_report(
    *,
    row_overlap_train: int,
    row_overlap_test: int,
    context_pair_count: int = 0,
    ttmatch_rows: int = 0,
    ttmatch_files: list[str] | None = None,
    repeated_context_pairs: int = 0,
) -> dict[str, object]:
    return {
        "risk_tier": "high_risk_quarantine",
        "clean_eligible": False,
        "submission_exports": 0,
        "row_overlap_train": int(row_overlap_train),
        "row_overlap_test": int(row_overlap_test),
        "context_pair_count": int(context_pair_count),
        "repeated_context_pairs": int(repeated_context_pairs),
        "ttmatch_rows": int(ttmatch_rows),
        "ttmatch_files": list(ttmatch_files or []),
        "allowed_uses": [
            "deduplicate exact overlapping rows vs AICUP train/test",
            "count repeated/coarse context pairs",
            "contrastive research audit only",
        ],
        "blocked_uses": [
            "clean training data",
            "submission exports",
            "upload candidates",
        ],
    }


def run_pipeline(
    *,
    ttmatch_root: Path | str = TTMATCH_ROOT,
    aicup_train: pd.DataFrame | Path | str | None = AICUP_TRAIN,
    aicup_test: pd.DataFrame | Path | str | None = AICUP_TEST,
    outdir: Path | str = OUTDIR,
) -> dict[str, object]:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    ttmatch_frames = load_ttmatch_frames(Path(ttmatch_root))
    ttmatch = _concat_frames(ttmatch_frames)
    train = _load_frame(aicup_train)
    test = _load_frame(aicup_test)

    row_overlap_train = count_row_overlap(ttmatch, train)
    row_overlap_test = count_row_overlap(ttmatch, test)

    context_pairs = count_context_pairs(ttmatch)
    context_pair_count = int(len(context_pairs))
    repeated_context_pairs = int((context_pairs["count"] > 1).sum()) if not context_pairs.empty else 0

    deduped = drop_overlapping_rows(ttmatch, train, test)
    if not context_pairs.empty:
        context_pairs.to_csv(out_path / "context_pair_counts.csv", index=False)
    deduped.to_csv(out_path / "ttmatch_deduped_audit_rows.csv", index=False)

    report = build_quarantine_report(
        row_overlap_train=row_overlap_train,
        row_overlap_test=row_overlap_test,
        context_pair_count=context_pair_count,
        repeated_context_pairs=repeated_context_pairs,
        ttmatch_rows=len(ttmatch),
        ttmatch_files=[str(path) for path in sorted(ttmatch_frames)],
    )
    (out_path / "quarantine_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def load_ttmatch_frames(ttmatch_root: Path) -> dict[Path, pd.DataFrame]:
    if not ttmatch_root.exists():
        return {}
    frames: dict[Path, pd.DataFrame] = {}
    for path in sorted(ttmatch_root.glob("*.csv")):
        if path.name.lower().startswith("sample_submission"):
            continue
        frames[path] = normalize_signature_columns(pd.read_csv(path, low_memory=False))
    return frames


def drop_overlapping_rows(
    ttmatch: pd.DataFrame,
    aicup_train: pd.DataFrame,
    aicup_test: pd.DataFrame,
) -> pd.DataFrame:
    normalized = normalize_signature_columns(ttmatch)
    if normalized.empty:
        return normalized
    train_columns = _common_signature_columns(normalized, aicup_train)
    test_columns = _common_signature_columns(normalized, aicup_test)
    if not train_columns and not test_columns:
        return normalized
    keep = pd.Series(True, index=normalized.index)
    if train_columns:
        keep &= ~_signature_series(normalized, train_columns).isin(row_signature_set(aicup_train, train_columns))
    if test_columns:
        keep &= ~_signature_series(normalized, test_columns).isin(row_signature_set(aicup_test, test_columns))
    return normalized.loc[keep].reset_index(drop=True)


def count_row_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    columns = _common_signature_columns(left, right)
    if not columns:
        return 0
    return len(row_signature_set(left, columns) & row_signature_set(right, columns))


def count_context_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_signature_columns(frame)
    cols = [col for col in CONTEXT_COLUMNS if col in normalized.columns]
    if len(cols) < 2 or normalized.empty:
        return pd.DataFrame(columns=["context_pair", "count"])
    tokens = normalized[cols].map(_normalize_value).agg("|".join, axis=1)
    counts = tokens.value_counts(dropna=False).rename_axis("context_pair").reset_index(name="count")
    return counts.sort_values(["count", "context_pair"], ascending=[False, True]).reset_index(drop=True)


def _concat_frames(frames: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    pieces = []
    for path, frame in frames.items():
        piece = frame.copy()
        piece["__ttmatch_file__"] = path.name
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _common_signature_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    left_cols = set(normalize_signature_columns(left).columns)
    right_cols = set(normalize_signature_columns(right).columns)
    return [col for col in AICUP_SIGNATURE_COLUMNS if col in left_cols and col in right_cols]


def _load_frame(source: pd.DataFrame | Path | str | None) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return normalize_signature_columns(source)
    path = Path(source)
    if not path.exists():
        return pd.DataFrame()
    return normalize_signature_columns(pd.read_csv(path, low_memory=False))


def _signature_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    normalized = pd.DataFrame(index=frame.index)
    for col in columns:
        normalized[col] = frame[col].map(_normalize_value)
    return normalized.agg("\x1f".join, axis=1).map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())


def _normalize_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.8g}"


def main() -> None:
    report = run_pipeline()
    print(
        "V436 quarantine complete: "
        f"clean_eligible={report['clean_eligible']} "
        f"submission_exports={report['submission_exports']} "
        f"row_overlap_train={report['row_overlap_train']} "
        f"row_overlap_test={report['row_overlap_test']} "
        f"context_pair_count={report['context_pair_count']}"
    )


if __name__ == "__main__":
    main()
