"""V440 professor corpus weighting and deduplication lab."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "v430_external_audit_canonical_expander" / "canonical_expanded_events.csv"
OUTDIR = ROOT / "v440_professor_corpus_weighting"

FORBIDDEN_COLUMNS = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
DEDUP_CANDIDATE_KEYS = [
    "source_dataset",
    "sequence_id",
    "event_index",
    "raw_payload_hash",
    "coarse_family",
    "landing_depth_bin",
    "landing_side_bin",
]


def forbid_exact_external_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in frame.columns]).copy()


def deduplicate_external_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = [c for c in DEDUP_CANDIDATE_KEYS if c in frame.columns]
    out = frame.drop_duplicates(keys).reset_index(drop=True)
    return out, {
        "input_rows": int(len(frame)),
        "output_rows": int(len(out)),
        "duplicate_rows_removed": int(len(frame) - len(out)),
        "dedup_keys": keys,
    }


def compute_source_weights(counts: pd.Series, max_weight: float = 2.0, min_weight: float = 0.35) -> dict[str, float]:
    median = float(counts.median()) if len(counts) else 1.0
    raw = (median / counts.clip(lower=1)).pow(0.5)
    return {str(k): float(min(max(v, min_weight), max_weight)) for k, v in raw.items()}


def _blocked_source_reason(source: Any) -> str | None:
    lower = str(source).lower()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    if "sony" in compact:
        return "sony"
    if compact == "ttmatch" or (compact.startswith("ttmatch") and "dynamics" not in compact):
        return "ttmatch"
    return None


def _exclude_blocked_sources(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "source_dataset" not in frame.columns:
        return frame.copy(), {"sony": 0, "ttmatch": 0}

    reasons = frame["source_dataset"].map(_blocked_source_reason)
    blocked = reasons.notna()
    counts = {
        "sony": int(reasons.eq("sony").sum()),
        "ttmatch": int(reasons.eq("ttmatch").sum()),
    }
    return frame.loc[~blocked].copy().reset_index(drop=True), counts


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _source_summary(frame: pd.DataFrame) -> dict[str, int]:
    if "source_dataset" not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame["source_dataset"].value_counts().sort_index().items()}


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    outdir: Path = OUTDIR,
    max_weight: float = 2.0,
    min_weight: float = 0.35,
) -> dict[str, Any]:
    input_path = Path(input_path)
    outdir = Path(outdir)
    frame = pd.read_csv(input_path, low_memory=False)
    input_columns = list(frame.columns)

    clean = forbid_exact_external_columns(frame)
    forbidden_columns_stripped = sorted(set(input_columns) & FORBIDDEN_COLUMNS)
    clean, blocked_source_rows = _exclude_blocked_sources(clean)
    deduped, dedup_report = deduplicate_external_events(clean)

    if "source_dataset" in deduped.columns and len(deduped):
        counts = deduped["source_dataset"].value_counts().sort_index()
    else:
        counts = pd.Series(dtype="int64")
    weights = compute_source_weights(counts, max_weight=max_weight, min_weight=min_weight)

    weighted = deduped.copy()
    if "source_dataset" in weighted.columns:
        weighted["source_weight"] = weighted["source_dataset"].astype(str).map(weights).fillna(1.0)
    else:
        weighted["source_weight"] = 1.0

    source_weight_table = pd.DataFrame(
        {
            "source_dataset": [str(k) for k in counts.index],
            "row_count": [int(v) for v in counts.values],
            "source_weight": [weights[str(k)] for k in counts.index],
        }
    )

    outdir.mkdir(parents=True, exist_ok=True)
    weighted_path = outdir / "v440_weighted_external_events.csv"
    weight_table_path = outdir / "source_weight_table.csv"
    report_path = outdir / "corpus_weighting_report.json"
    weighted.to_csv(weighted_path, index=False)
    source_weight_table.to_csv(weight_table_path, index=False)

    report = {
        "input_path": input_path,
        "output_dir": outdir,
        "input_rows": int(len(frame)),
        "post_forbidden_column_rows": int(len(clean) + sum(blocked_source_rows.values())),
        "trainable_rows": int(len(weighted)),
        "weighted_rows": int(len(weighted)),
        "forbidden_columns_stripped": forbidden_columns_stripped,
        "forbidden_columns_present": sorted(set(weighted.columns) & FORBIDDEN_COLUMNS),
        "blocked_source_rows": blocked_source_rows,
        "source_counts": _source_summary(weighted),
        "source_weights": weights,
        "dedup": dedup_report,
        "tt_matchdynamics_present": bool(
            "source_dataset" in weighted.columns and weighted["source_dataset"].astype(str).eq("TT-MatchDynamics").any()
        ),
        "sony_present": bool(
            "source_dataset" in weighted.columns
            and weighted["source_dataset"].astype(str).str.lower().str.contains("sony", na=False).any()
        ),
        "ttmatch_present": bool(
            "source_dataset" in weighted.columns
            and weighted["source_dataset"].map(_blocked_source_reason).eq("ttmatch").any()
        ),
        "outputs": {
            "weighted_events": weighted_path,
            "source_weight_table": weight_table_path,
            "report": report_path,
        },
    }
    _write_json(report_path, report)
    return _json_safe(report)


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
