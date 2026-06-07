"""V434 anchor-aware residual MoE gate.

Builds conservative action/point candidate proposal tables against the clean
V362 anchor. This stage writes diagnostics only; submission packaging is left
to later versions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v434_anchor_aware_moe_gate"
DEFAULT_SOURCE_DIRS = (
    ROOT / "v432_aicup_exact_model_zoo_finetune",
    ROOT / "v432_aicup_finetune_model_zoo",
    ROOT / "v433_weak_class_expert_bank",
    ROOT / "v420_rare_class_augmented_exact_models",
    ROOT / "v419_intent_first_point_finetune",
)

ACTION_COLUMN_CANDIDATES = (
    "pred_actionId",
    "candidate_actionId",
    "expert_actionId",
    "action_pred",
    "pred_action",
)
POINT_COLUMN_CANDIDATES = (
    "pred_pointId",
    "candidate_pointId",
    "expert_pointId",
    "point_pred",
    "pred_point",
)
ACTION_CONFIDENCE_CANDIDATES = (
    "action_margin",
    "action_confidence",
    "expert_confidence",
    "score",
    "utility",
)
POINT_CONFIDENCE_CANDIDATES = (
    "point_margin",
    "point_confidence",
    "expert_confidence",
    "score",
    "utility",
)
PREDICTION_GLOBS = (
    "test_predictions*.csv",
    "expert_test_scores*.csv",
    "test_action_probs*.csv",
    "test_point_probs*.csv",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _first_existing_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _as_float_series(frame: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    col = _first_existing_column(frame, names)
    if col is None:
        return pd.Series([default] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def _align_to_anchor(anchor: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in source.columns:
        if len(source) != len(anchor):
            raise ValueError("source without rally_uid must match anchor row count")
        aligned = source.copy()
        aligned.insert(0, "rally_uid", anchor["rally_uid"].to_numpy())
        return aligned.reset_index(drop=True)

    anchor_uid = anchor["rally_uid"].astype(str).reset_index(drop=True)
    source = source.copy()
    source["rally_uid"] = source["rally_uid"].astype(str)
    if len(source) == len(anchor) and source["rally_uid"].reset_index(drop=True).equals(anchor_uid):
        return source.reset_index(drop=True)

    reduced = source.drop_duplicates("rally_uid", keep="last")
    missing = sorted(set(anchor_uid) - set(reduced["rally_uid"]))
    if missing:
        raise ValueError(f"source cannot align to anchor; missing rally_uid values: {missing[:5]}")
    return reduced.set_index("rally_uid").loc[anchor_uid].reset_index()


def _risk_tier(rank: int) -> str:
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "top10"
    if rank <= 20:
        return "top20"
    if rank <= 40:
        return "top40"
    return "research"


def moe_change_score(row: dict[str, Any] | pd.Series) -> float:
    """Compute conservative row utility for changing away from the anchor."""

    get = row.get if hasattr(row, "get") else dict(row).get
    anchor_confidence = float(get("anchor_confidence", 0.55) or 0.55)
    expert_confidence = float(get("expert_confidence", get("confidence", 0.0)) or 0.0)
    margin = float(get("margin", 0.0) or 0.0)
    support = int(get("source_support", 0) or 0)

    score = 1.35 * (expert_confidence - 0.62)
    score += 0.85 * margin
    score += 0.12 * max(0, support)
    score -= 1.15 * max(0.0, anchor_confidence - 0.70)
    if support <= 0:
        score -= 0.25

    target = str(get("target", ""))
    anchor_value = get("anchor_value", None)
    candidate_value = get("candidate_value", None)
    try:
        anchor_int = int(anchor_value)
        candidate_int = int(candidate_value)
    except (TypeError, ValueError):
        anchor_int = None
        candidate_int = None

    if target == "point" and candidate_int == 0 and anchor_int not in (None, 0):
        score -= 5.0
    if (
        target == "action"
        and candidate_int in SERVE_ACTION_CLASSES
        and anchor_int not in SERVE_ACTION_CLASSES
    ):
        score -= 5.0
    return float(score)


def _prediction_to_long_candidates(
    anchor: pd.DataFrame,
    source_name: str,
    source: pd.DataFrame,
) -> pd.DataFrame:
    aligned = _align_to_anchor(anchor, source)
    action_col = _first_existing_column(aligned, ACTION_COLUMN_CANDIDATES)
    point_col = _first_existing_column(aligned, POINT_COLUMN_CANDIDATES)
    rows: list[dict[str, Any]] = []

    if action_col is not None:
        confidence = _as_float_series(aligned, ACTION_CONFIDENCE_CANDIDATES, 0.0)
        secondary_conf = _as_float_series(aligned, ("action_confidence",), 0.0)
        candidate_values = pd.to_numeric(aligned[action_col], errors="coerce")
        for idx, candidate in candidate_values.items():
            if pd.isna(candidate):
                continue
            anchor_value = int(anchor.at[idx, "actionId"])
            candidate_value = int(candidate)
            if candidate_value == anchor_value:
                continue
            rows.append(
                {
                    "row_id": int(idx),
                    "rally_uid": anchor.at[idx, "rally_uid"],
                    "target": "action",
                    "anchor_value": anchor_value,
                    "candidate_value": candidate_value,
                    "source": source_name,
                    "expert_confidence": float(max(confidence.at[idx], secondary_conf.at[idx])),
                    "margin": float(confidence.at[idx]),
                    "anchor_confidence": 0.55,
                }
            )

    if point_col is not None:
        confidence = _as_float_series(aligned, POINT_CONFIDENCE_CANDIDATES, 0.0)
        secondary_conf = _as_float_series(aligned, ("point_confidence",), 0.0)
        candidate_values = pd.to_numeric(aligned[point_col], errors="coerce")
        for idx, candidate in candidate_values.items():
            if pd.isna(candidate):
                continue
            anchor_value = int(anchor.at[idx, "pointId"])
            candidate_value = int(candidate)
            if candidate_value == anchor_value:
                continue
            rows.append(
                {
                    "row_id": int(idx),
                    "rally_uid": anchor.at[idx, "rally_uid"],
                    "target": "point",
                    "anchor_value": anchor_value,
                    "candidate_value": candidate_value,
                    "source": source_name,
                    "expert_confidence": float(max(confidence.at[idx], secondary_conf.at[idx])),
                    "margin": float(confidence.at[idx]),
                    "anchor_confidence": 0.55,
                }
            )

    return pd.DataFrame(rows)


def _aggregate_candidates(raw: pd.DataFrame, target: str) -> pd.DataFrame:
    columns = [
        "row_id",
        "rally_uid",
        "target",
        "anchor_value",
        "candidate_value",
        "score",
        "risk_tier",
        "target_changed_rows",
        "source_support",
        "source_agreement",
        "sources",
        "expert_confidence",
        "margin",
        "gate_reason",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    grouped_rows: list[dict[str, Any]] = []
    total_sources = max(1, raw["source"].nunique())
    keys = ["row_id", "rally_uid", "target", "anchor_value", "candidate_value"]
    for key, group in raw.groupby(keys, sort=False):
        source_names = sorted(str(v) for v in group["source"].dropna().unique())
        expert_confidence = float(group["expert_confidence"].max())
        margin = float(group["margin"].max())
        row = {
            "row_id": int(key[0]),
            "rally_uid": key[1],
            "target": key[2],
            "anchor_value": int(key[3]),
            "candidate_value": int(key[4]),
            "source_support": int(len(source_names)),
            "source_agreement": float(len(source_names) / total_sources),
            "sources": "|".join(source_names),
            "expert_confidence": expert_confidence,
            "margin": margin,
            "anchor_confidence": 0.55,
        }
        row["score"] = moe_change_score(row)
        row["gate_reason"] = (
            f"support={row['source_support']};confidence={expert_confidence:.4f};margin={margin:.4f}"
        )
        grouped_rows.append(row)

    out = pd.DataFrame(grouped_rows)
    out = out.loc[(out["target"] == target) & (out["score"] > 0)].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.sort_values(["score", "source_support", "row_id"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(5, "rank", np.arange(1, len(out) + 1, dtype=int))
    out["risk_tier"] = out["rank"].map(_risk_tier)
    out["target_changed_rows"] = out["rank"].map(lambda rank: min(int(rank), 40))
    return out.loc[:, ["rank", *columns[:5], *columns[5:]]].copy()


def build_moe_candidate_tables(
    anchor: pd.DataFrame,
    sources: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    anchor_submission = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(anchor_submission, expected_rows=len(anchor_submission))

    raw_parts = []
    source_errors: list[dict[str, str]] = []
    for source_name, frame in sources:
        try:
            raw_parts.append(_prediction_to_long_candidates(anchor_submission, source_name, frame))
        except ValueError as exc:
            source_errors.append({"source": source_name, "error": str(exc)})

    raw = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    if raw.empty:
        action_candidates = _aggregate_candidates(raw, "action")
        point_candidates = _aggregate_candidates(raw, "point")
        return action_candidates, point_candidates, {
            "source_count": int(len(sources)),
            "source_errors": source_errors,
            "raw_candidate_rows": 0,
            "blocked_serve_15_18_candidates": 0,
            "blocked_point0_candidates": 0,
            "positive_action_candidates": 0,
            "positive_point_candidates": 0,
            "submission_exports": 0,
        }

    blocked_serve = (
        raw["target"].eq("action")
        & raw["candidate_value"].astype(int).isin(SERVE_ACTION_CLASSES)
        & ~raw["anchor_value"].astype(int).isin(SERVE_ACTION_CLASSES)
    )
    blocked_point0 = (
        raw["target"].eq("point")
        & raw["candidate_value"].astype(int).eq(0)
        & raw["anchor_value"].astype(int).ne(0)
    )
    safe_raw = raw.loc[~blocked_serve & ~blocked_point0].copy()
    action_candidates = _aggregate_candidates(safe_raw, "action")
    point_candidates = _aggregate_candidates(safe_raw, "point")
    report = {
        "source_count": int(len(sources)),
        "source_errors": source_errors,
        "raw_candidate_rows": int(len(raw)),
        "safe_candidate_rows": int(len(safe_raw)),
        "blocked_serve_15_18_candidates": int(blocked_serve.sum()),
        "blocked_point0_candidates": int(blocked_point0.sum()),
        "positive_action_candidates": int(len(action_candidates)),
        "positive_point_candidates": int(len(point_candidates)),
        "submission_exports": 0,
    }
    return action_candidates, point_candidates, report


def _load_prediction_sources(source_dirs: Iterable[Path], anchor: pd.DataFrame) -> tuple[list[tuple[str, pd.DataFrame]], dict[str, Any]]:
    sources: list[tuple[str, pd.DataFrame]] = []
    inspected_dirs: list[str] = []
    files_used: list[str] = []
    for source_dir in source_dirs:
        source_dir = Path(source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        inspected_dirs.append(str(source_dir.resolve()))
        seen: set[Path] = set()
        for pattern in PREDICTION_GLOBS:
            for path in sorted(source_dir.glob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    frame = pd.read_csv(path, low_memory=False)
                except (OSError, pd.errors.ParserError):
                    continue
                if "rally_uid" not in frame.columns and len(frame) != len(anchor):
                    continue
                source_name = f"{source_dir.name}/{path.stem}"
                sources.append((source_name, frame))
                files_used.append(str(path.resolve()))

    fallback_anchor_used = False
    if not sources:
        fallback_anchor_used = True
        sources.append(
            (
                "v362_anchor/no_change_fallback",
                pd.DataFrame(
                    {
                        "rally_uid": anchor["rally_uid"],
                        "pred_actionId": anchor["actionId"],
                        "pred_pointId": anchor["pointId"],
                        "action_confidence": [1.0] * len(anchor),
                        "point_confidence": [1.0] * len(anchor),
                    }
                ),
            )
        )

    meta = {
        "inspected_dirs": inspected_dirs,
        "files_used": files_used,
        "fallback_anchor_used": fallback_anchor_used,
    }
    return sources, meta


def _gate_report_frame(action_candidates: pd.DataFrame, point_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, frame in (("action", action_candidates), ("point", point_candidates)):
        for tier, limit in (("top5", 5), ("top10", 10), ("top20", 20), ("top40", 40)):
            subset = frame.head(limit)
            rows.append(
                {
                    "target": target,
                    "risk_tier": tier,
                    "target_changed_rows": int(min(limit, len(frame))),
                    "candidate_count": int(len(subset)),
                    "source_support_min": int(subset["source_support"].min()) if len(subset) else 0,
                    "source_support_max": int(subset["source_support"].max()) if len(subset) else 0,
                    "score_min": float(subset["score"].min()) if len(subset) else 0.0,
                    "score_max": float(subset["score"].max()) if len(subset) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(
    *,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
    source_dirs: list[Path] | None = None,
    expected_rows: int | None = 1845,
    quick: bool = False,
) -> dict[str, Any]:
    anchor = pd.read_csv(anchor_path).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(anchor, expected_rows=expected_rows)
    dirs = source_dirs if source_dirs is not None else list(DEFAULT_SOURCE_DIRS)
    sources, source_meta = _load_prediction_sources(dirs, anchor)
    if quick and len(sources) > 6:
        sources = sources[:6]
        source_meta["quick_source_limit"] = 6

    action_candidates, point_candidates, report = build_moe_candidate_tables(anchor, sources)
    gate_report = _gate_report_frame(action_candidates, point_candidates)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    action_candidates.to_csv(safe_output_path(outdir, "moe_action_candidates.csv"), index=False)
    point_candidates.to_csv(safe_output_path(outdir, "moe_point_candidates.csv"), index=False)
    gate_report.to_csv(safe_output_path(outdir, "moe_gate_report.csv"), index=False)

    summary = {
        "version": "V434",
        "anchor_path": str(Path(anchor_path).resolve()),
        "anchor_rows": int(len(anchor)),
        "quick": bool(quick),
        "source_meta": source_meta,
        "sources_loaded": [name for name, _frame in sources],
        "action_candidate_count": int(len(action_candidates)),
        "point_candidate_count": int(len(point_candidates)),
        "top_candidate_counts": {
            "action_top5": int(min(5, len(action_candidates))),
            "action_top10": int(min(10, len(action_candidates))),
            "action_top20": int(min(20, len(action_candidates))),
            "action_top40": int(min(40, len(action_candidates))),
            "point_top5": int(min(5, len(point_candidates))),
            "point_top10": int(min(10, len(point_candidates))),
            "point_top20": int(min(20, len(point_candidates))),
            "point_top40": int(min(40, len(point_candidates))),
        },
        **report,
    }
    write_json(safe_output_path(outdir, "summary.json"), summary)
    return json_safe(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-path", type=Path, default=ANCHOR_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--source-dir", type=Path, action="append", default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_pipeline(
        anchor_path=args.anchor_path,
        outdir=args.outdir,
        source_dirs=args.source_dir,
        expected_rows=1845,
        quick=args.quick,
    )
    print(json.dumps(summary["top_candidate_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
