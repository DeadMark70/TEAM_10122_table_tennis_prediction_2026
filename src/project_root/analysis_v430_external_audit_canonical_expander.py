"""V430 external audit and canonical expansion.

Expands the V413/V414 clean canonical corpus with only policy-allowed external
clean candidates. Exact AICUP labels are stripped from external rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_V413_CLEAN_PATH = ROOT / "v413_external_license_overlap_guard" / "canonical_clean_events.csv"
DEFAULT_V412_CLEAN_PATH = ROOT / "v412_clean_canonical_external" / "canonical_external_events.csv"
DEFAULT_V414_PRETRAIN_PATH = ROOT / "v414_masked_pretraining_inputs" / "pretrain_sequences.csv"
DEFAULT_EXTERNAL_ROOT = ROOT / "external_data"
OUTDIR = ROOT / "v430_external_audit_canonical_expander"

CLEAN_SOURCES = {
    "openttgames",
    "DeepMindrobottabletennis",
    "TT3D",
    "AIMY",
    "spindoe",
    "CoachAI-Projects-main",
}
CLEAN_CANDIDATE_SOURCES = {"TT-MatchDynamics"}
AUDIT_ONLY_SOURCES = {"sonytabletennis"}
HIGH_RISK_SOURCES = {"TTMATCH"}
FORBIDDEN_AICUP_COLUMNS = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
AICUP_SCHEMA_SIGNATURE = {"rally_uid", "actionId", "pointId", "serverGetPoint"}
COARSE_COLUMNS = [
    "source_dataset",
    "source_file",
    "license_tag",
    "risk_tier",
    "source_tier",
    "clean_train_allowed",
    "sequence_id",
    "match_id",
    "rally_id",
    "event_index",
    "frame",
    "timestamp",
    "phase",
    "event_type",
    "coarse_family",
    "terminal_label",
    "remaining_hint",
    "landing_x",
    "landing_y",
    "landing_z",
    "landing_depth_bin",
    "landing_side_bin",
    "speed_x",
    "speed_y",
    "speed_z",
    "speed_norm",
    "spin_x",
    "spin_y",
    "spin_z",
    "spin_norm",
    "player_x",
    "player_y",
    "opponent_x",
    "opponent_y",
    "raw_label",
    "raw_payload_hash",
]


def classify_external_source(source_name: str, license_text: str = "") -> dict[str, object]:
    lower = f"{source_name} {license_text}".lower()
    if "ttmatch" in lower and "dynamics" not in lower:
        return {
            "tier": "high_risk_quarantine",
            "train_allowed_by_default": False,
            "requires_overlap_audit": True,
        }
    if "tt-matchdynamics" in lower or "matchdynamics" in lower:
        return {
            "tier": "clean_candidate",
            "train_allowed_by_default": "apache" in lower,
            "requires_overlap_audit": True,
        }
    if "sony" in lower or "no derivatives" in lower or "cc by-nc-nd" in lower or "cc-by-nc-nd" in lower:
        return {
            "tier": "audit_only",
            "train_allowed_by_default": False,
            "requires_overlap_audit": True,
        }
    return {"tier": "clean", "train_allowed_by_default": True, "requires_overlap_audit": False}


def canonicalize_allowed_external_rows(
    rows: pd.DataFrame,
    source_policy: dict[str, str] | None = None,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=COARSE_COLUMNS)

    clean = rows.copy()
    if "token_family" in clean.columns and "coarse_family" not in clean.columns:
        clean["coarse_family"] = clean["token_family"]
    if "target_terminal" in clean.columns and "terminal_label" not in clean.columns:
        clean["terminal_label"] = clean["target_terminal"]

    if source_policy and "source_dataset" in clean.columns:
        allowed_sources = {
            source for source, tier in source_policy.items() if tier in {"clean", "clean_candidate"}
        }
        clean = clean[clean["source_dataset"].astype(str).isin(allowed_sources)].copy()

    clean = clean.drop(columns=[col for col in FORBIDDEN_AICUP_COLUMNS if col in clean.columns])
    for col in COARSE_COLUMNS:
        if col not in clean.columns:
            clean[col] = pd.NA
    return clean[COARSE_COLUMNS].reset_index(drop=True)


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


def _read_text_evidence(source_dir: Path) -> str:
    chunks: list[str] = []
    for pattern in ("LICENSE*", "README*", "*.md", "metadata*", "dataset*"):
        for path in source_dir.glob(pattern):
            if not path.is_file() or path.stat().st_size > 250_000:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore")[:50_000])
            except OSError:
                continue
    return "\n".join(chunks)


def _supported_data_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    supported = {".csv", ".json", ".jsonl", ".ndjson"}
    skip_markers = ("license", "readme", "metadata", "copying")
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if any(marker in path.name.lower() for marker in skip_markers):
            continue
        files.append(path)
    return sorted(files)


def _read_supported_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return pd.DataFrame(value)
            return pd.DataFrame([payload])
    return pd.DataFrame()


def _source_data_summary(source_dir: Path) -> dict[str, Any]:
    files = _supported_data_files(source_dir)
    parsed_files = 0
    parsed_rows = 0
    parse_errors: list[str] = []
    aicup_schema_overlap = False
    for path in files:
        try:
            frame = _read_supported_frame(path)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {str(exc)[:160]}")
            continue
        if frame.empty:
            parse_errors.append(f"{path.name}: no_parseable_rows")
            continue
        parsed_files += 1
        parsed_rows += int(len(frame))
        aicup_schema_overlap = aicup_schema_overlap or bool(AICUP_SCHEMA_SIGNATURE & set(frame.columns))
    return {
        "data_files": len(files),
        "csv_files": sum(1 for path in files if path.suffix.lower() == ".csv"),
        "parsed_files": parsed_files,
        "parsed_rows": parsed_rows,
        "parse_errors": "; ".join(parse_errors[:20]),
        "aicup_schema_overlap": aicup_schema_overlap,
    }


def _coarse_depth_from_y(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "unknown"
    if number < 40:
        return "short"
    if number < 80:
        return "half"
    return "long"


def _coarse_side_from_x(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "unknown"
    if number < 90:
        return "left"
    if number < 150:
        return "middle"
    return "right"


def _canonicalize_ttmatchdynamics(frame: pd.DataFrame, source_file: Path, policy: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["source_dataset"] = "TT-MatchDynamics"
    out["source_file"] = str(source_file)
    out["license_tag"] = "Apache-2.0" if policy.get("train_allowed_by_default") else "missing_local_license_evidence"
    out["risk_tier"] = "clean_candidate" if policy.get("train_allowed_by_default") else "audit_only_missing_license"
    out["source_tier"] = policy["tier"]
    out["clean_train_allowed"] = bool(policy.get("train_allowed_by_default"))
    out["sequence_id"] = source_file.stem
    out["event_index"] = np.arange(len(frame), dtype=int)
    out["timestamp"] = frame["date"] if "date" in frame.columns else pd.NA
    out["phase"] = np.where(
        pd.to_numeric(frame.get("Winning in First Three Strokes", pd.Series([0] * len(frame))), errors="coerce").fillna(0)
        > 0,
        "early_three",
        "rally",
    )
    topspin = pd.to_numeric(frame.get("Topspin/Backspin Indicator", pd.Series([np.nan] * len(frame))), errors="coerce")
    forehand = pd.to_numeric(frame.get("Forehand/Backhand Indicator", pd.Series([np.nan] * len(frame))), errors="coerce")
    out["coarse_family"] = np.where(forehand.fillna(0) > 0, "forehand_response", "backhand_response")
    out["event_type"] = np.where(topspin.fillna(0) > 0, "topspin", "backspin")
    out["terminal_label"] = "unknown"
    out["landing_x"] = pd.to_numeric(frame.get("X", pd.Series([np.nan] * len(frame))), errors="coerce")
    out["landing_y"] = pd.to_numeric(frame.get("Y", pd.Series([np.nan] * len(frame))), errors="coerce")
    out["landing_depth_bin"] = out["landing_y"].map(_coarse_depth_from_y)
    out["landing_side_bin"] = out["landing_x"].map(_coarse_side_from_x)
    return canonicalize_allowed_external_rows(out, {"TT-MatchDynamics": "clean_candidate"})


def _scan_ttmatchdynamics(source_dir: Path, policy: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files = sorted(path for path in source_dir.glob("*.csv") if path.is_file())
    for path in files:
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        frames.append(_canonicalize_ttmatchdynamics(raw, path, policy))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COARSE_COLUMNS)
    if not bool(policy.get("train_allowed_by_default")):
        rows = rows.iloc[0:0].copy()
    report = {
        "source_dataset": "TT-MatchDynamics",
        "files": len(files),
        "rows": int(sum(len(frame) for frame in frames)),
        "trainable_rows": int(len(rows)),
        "tier": policy["tier"],
        "train_allowed_by_default": bool(policy.get("train_allowed_by_default")),
    }
    return rows, report


def _source_audit_rows(external_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not external_root.exists():
        return rows
    for source_dir in sorted(path for path in external_root.iterdir() if path.is_dir()):
        evidence = _read_text_evidence(source_dir)
        policy = classify_external_source(source_dir.name, evidence)
        data_summary = _source_data_summary(source_dir)
        rows.append(
            {
                "source_dataset": source_dir.name,
                "tier": policy["tier"],
                "train_allowed_by_default": bool(policy.get("train_allowed_by_default")),
                "requires_overlap_audit": bool(policy.get("requires_overlap_audit")),
                "license_evidence_detected": bool(evidence.strip()),
                "data_files": int(data_summary["data_files"]),
                "csv_files": int(data_summary["csv_files"]),
                "parsed_files": int(data_summary["parsed_files"]),
                "parsed_rows": int(data_summary["parsed_rows"]),
                "trainable_rows": 0,
                "aicup_schema_overlap": bool(data_summary["aicup_schema_overlap"]),
                "parse_errors": data_summary["parse_errors"],
            }
        )
    return rows


def _load_base_canonical(v413_clean_path: Path, v414_pretrain_path: Path) -> pd.DataFrame:
    if v413_clean_path.exists():
        base = pd.read_csv(v413_clean_path, low_memory=False)
    elif DEFAULT_V412_CLEAN_PATH.exists():
        base = pd.read_csv(DEFAULT_V412_CLEAN_PATH, low_memory=False)
    elif v414_pretrain_path.exists():
        base = pd.read_csv(v414_pretrain_path, low_memory=False)
    else:
        return pd.DataFrame(columns=COARSE_COLUMNS)
    if "source_dataset" not in base.columns:
        base["source_dataset"] = "unknown_clean_base"
    source_policy = {str(source): "clean" for source in base["source_dataset"].dropna().astype(str).unique()}
    return canonicalize_allowed_external_rows(base, source_policy)


def _csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0


def run_pipeline(
    *,
    v413_clean_path: str | Path = DEFAULT_V413_CLEAN_PATH,
    v414_pretrain_path: str | Path = DEFAULT_V414_PRETRAIN_PATH,
    external_root: str | Path = DEFAULT_EXTERNAL_ROOT,
    outdir: str | Path = OUTDIR,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    v413_clean_path = Path(v413_clean_path)
    v414_pretrain_path = Path(v414_pretrain_path)
    external_root = Path(external_root)

    base = _load_base_canonical(v413_clean_path, v414_pretrain_path)
    v413_base_rows = _csv_data_row_count(v413_clean_path)
    v414_base_rows = _csv_data_row_count(v414_pretrain_path)
    audit_rows = _source_audit_rows(external_root)
    ttmd_rows = pd.DataFrame(columns=COARSE_COLUMNS)
    ttmd_report: dict[str, Any] = {
        "source_dataset": "TT-MatchDynamics",
        "files": 0,
        "rows": 0,
        "trainable_rows": 0,
        "tier": "missing",
        "train_allowed_by_default": False,
    }
    ttmd_dir = external_root / "TT-MatchDynamics"
    if ttmd_dir.exists():
        evidence = _read_text_evidence(ttmd_dir)
        policy = classify_external_source("TT-MatchDynamics", evidence)
        ttmd_rows, ttmd_report = _scan_ttmatchdynamics(ttmd_dir, policy)
        for row in audit_rows:
            if row.get("source_dataset") == "TT-MatchDynamics":
                row["trainable_rows"] = int(ttmd_report.get("trainable_rows", 0))

    expanded = pd.concat([base, ttmd_rows], ignore_index=True)
    expanded = canonicalize_allowed_external_rows(
        expanded,
        {source: "clean" for source in expanded.get("source_dataset", pd.Series(dtype=str)).dropna().astype(str).unique()},
    )
    expanded.to_csv(outdir / "canonical_expanded_events.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    if audit.empty:
        audit = pd.DataFrame(columns=["source_dataset", "tier", "train_allowed_by_default", "requires_overlap_audit"])
    audit.to_csv(outdir / "external_source_audit.csv", index=False)
    source_counts = {str(k): int(v) for k, v in expanded["source_dataset"].value_counts(dropna=False).sort_index().items()}
    overlap_report = {
        "base_rows": int(len(base)),
        "v413_base_rows": int(v413_base_rows),
        "v414_base_rows": int(v414_base_rows),
        "base_source": "v413" if v413_clean_path.exists() else "v414_fallback",
        "expanded_rows": int(len(expanded)),
        "ttmatchdynamics": ttmd_report,
        "source_counts": source_counts,
        "forbidden_columns_present": sorted(set(expanded.columns) & FORBIDDEN_AICUP_COLUMNS),
        "clean_policy": "TT-MatchDynamics included only with local Apache-2.0 evidence; Sony/TTMATCH excluded.",
    }
    _write_json(outdir / "license_overlap_report.json", overlap_report)
    report = {
        "outdir": str(outdir),
        "base_rows": int(len(base)),
        "v413_base_rows": int(v413_base_rows),
        "v414_base_rows": int(v414_base_rows),
        "base_source": overlap_report["base_source"],
        "expanded_rows": int(len(expanded)),
        "ttmatchdynamics_trainable_rows": int(ttmd_report.get("trainable_rows", 0)),
        "source_counts": source_counts,
        "forbidden_columns_present": overlap_report["forbidden_columns_present"],
    }
    print(json.dumps(_json_safe(report), indent=2))
    return report


if __name__ == "__main__":
    run_pipeline()
