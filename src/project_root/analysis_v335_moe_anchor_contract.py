"""Shared anchor-safety utilities for lightweight MoE experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LEAKY_PREFIXES = ("next_", "y_", "true_", "label_")
LEAKY_EXACT = {"actionId", "pointId", "serverGetPoint", "rally_uid", "rally_id"}
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
SERVE_ACTION_CLASSES = {15, 16, 17, 18}
BANNED_OUTPUT_PARTS = {"upload_candidates_20260519"}


def drop_leaky_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in frame.columns:
        if col in LEAKY_EXACT:
            continue
        if any(str(col).startswith(prefix) for prefix in LEAKY_PREFIXES):
            continue
        keep.append(col)
    return frame.loc[:, keep].copy()


def safe_output_path(outdir: Path, filename: str) -> Path:
    outdir = Path(outdir).resolve()
    path = (outdir / filename).resolve()
    if outdir not in path.parents and path != outdir:
        raise ValueError(f"unsafe output path: {path}")
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & BANNED_OUTPUT_PARTS:
        raise ValueError(f"unsafe output directory: {path}")
    return path


def validate_submission_schema(frame: pd.DataFrame, expected_rows: int | None = 1845) -> None:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"bad columns: {list(frame.columns)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"bad row count: {len(frame)}")
    if not frame["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of range")
    if not frame["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
    server = pd.to_numeric(frame["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    if not server.between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")


def macro_f1(y_true: Iterable[Any], y_pred: Iterable[Any], labels: Iterable[Any] | None = None) -> float:
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred length mismatch")
    if labels is None:
        label_values = sorted(set(true.tolist()) | set(pred.tolist()))
    else:
        label_values = list(labels)
    if not label_values:
        return 0.0

    scores = []
    for label in label_values:
        tp = int(np.sum((true == label) & (pred == label)))
        fp = int(np.sum((true != label) & (pred == label)))
        fn = int(np.sum((true == label) & (pred != label)))
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def _value_counts(values: Iterable[Any]) -> dict[str, int]:
    series = pd.Series(values)
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def action_distribution_report(base_action: Iterable[Any], cand_action: Iterable[Any]) -> dict[str, Any]:
    base = np.asarray(list(base_action), dtype=int)
    cand = np.asarray(list(cand_action), dtype=int)
    if len(base) != len(cand):
        raise ValueError("base_action and cand_action length mismatch")
    base_serve = int(np.isin(base, list(SERVE_ACTION_CLASSES)).sum())
    cand_serve = int(np.isin(cand, list(SERVE_ACTION_CLASSES)).sum())
    changed = int(np.sum(base != cand))
    delta = cand_serve - base_serve
    return {
        "rows": int(len(base)),
        "changed_rows": changed,
        "changed_rate": float(changed / len(base)) if len(base) else 0.0,
        "base_counts": _value_counts(base),
        "candidate_counts": _value_counts(cand),
        "serve_15_18_base": base_serve,
        "serve_15_18_candidate": cand_serve,
        "serve_15_18_delta": int(delta),
        "serve_15_18_explosion": bool(delta >= 3 or (len(base) > 0 and delta / len(base) >= 0.01)),
    }


def point_distribution_report(base_point: Iterable[Any], cand_point: Iterable[Any]) -> dict[str, Any]:
    base = np.asarray(list(base_point), dtype=int)
    cand = np.asarray(list(cand_point), dtype=int)
    if len(base) != len(cand):
        raise ValueError("base_point and cand_point length mismatch")
    changed = int(np.sum(base != cand))
    p0_add = int(np.sum((base != 0) & (cand == 0)))
    p0_remove = int(np.sum((base == 0) & (cand != 0)))
    return {
        "rows": int(len(base)),
        "changed_rows": changed,
        "changed_rate": float(changed / len(base)) if len(base) else 0.0,
        "base_counts": _value_counts(base),
        "candidate_counts": _value_counts(cand),
        "point0_base": int(np.sum(base == 0)),
        "point0_candidate": int(np.sum(cand == 0)),
        "point0_additions": p0_add,
        "point0_removals": p0_remove,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
