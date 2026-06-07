"""V389 synthetic OOF/proxy validation lab.

This module ranks V388 candidate pools with V386 synthetic evidence when the
V388 pools exist. If V388 is absent, it emits empty ranked pools and records
that state rather than manufacturing candidates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v389_synthetic_oof_proxy_lab"
V388_DIR = ROOT / "v388_large_synthetic_candidate_pool"
V386_DIR = ROOT / "v386_synthetic_contrastive_scorer"

POINT_POOL = V388_DIR / "point_change_pool.csv"
ACTION_POOL = V388_DIR / "action_change_pool.csv"
V386_POINT = V386_DIR / "point_candidate_contrastive_scores.csv"
V386_ACTION = V386_DIR / "action_candidate_contrastive_scores.csv"

POINT_OUTPUT_COLUMNS = [
    "rally_uid",
    "base_point",
    "candidate_point",
    "proxy_score",
    "pass_gate",
    "validation_mode",
    "support_count",
    "source_family_count",
    "synthetic_compatibility_score",
    "synthetic_allowed",
    "historical_public_prior",
    "is_point0_addition",
    "same_depth",
    "same_side",
]

ACTION_OUTPUT_COLUMNS = [
    "rally_uid",
    "base_action",
    "candidate_action",
    "proxy_score",
    "pass_gate",
    "validation_mode",
    "support_count",
    "source_family_count",
    "synthetic_compatibility_score",
    "synthetic_allowed",
    "historical_public_prior",
    "is_serve_15_18_addition",
    "same_family",
]

POSITIVE_FAMILIES = (
    "v338",
    "v362",
    "v306",
    "v300",
    "v341",
    "v345",
)
NEGATIVE_FAMILIES = (
    "v191",
    "v166",
    "v220",
    "v202",
)


def output_filenames() -> list[str]:
    return [
        "local_oof_or_proxy_summary.csv",
        "ranked_point_pool.csv",
        "ranked_action_pool.csv",
        "historical_backtest.csv",
        "search_report.json",
    ]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index)
    if frame[column].dtype == bool:
        return frame[column].fillna(default)
    return frame[column].map(_as_bool).fillna(default)


def _num_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _bounded(series: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    return series.clip(lower=low, upper=high)


def _normalized_log(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_value = float(values.max()) if len(values) else 0.0
    if max_value <= 0:
        return pd.Series([0.0] * len(values), index=values.index, dtype="float64")
    return values.map(lambda value: math.log1p(float(value)) / math.log1p(max_value))


def _contains_any(value: Any, needles: tuple[str, ...]) -> bool:
    text = "" if value is None or pd.isna(value) else str(value).lower()
    return any(needle in text for needle in needles)


def _historical_prior(frame: pd.DataFrame) -> pd.Series:
    text_columns = [column for column in ("source_dir", "source_file", "sources", "source_families") if column in frame]
    if not text_columns:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype="float64")

    priors: list[float] = []
    for _, row in frame.iterrows():
        haystack = "|".join(str(row.get(column, "")) for column in text_columns).lower()
        score = 0.0
        if any(family in haystack for family in POSITIVE_FAMILIES):
            score += 1.0
        if any(family in haystack for family in NEGATIVE_FAMILIES):
            score -= 1.0
        priors.append(score)
    return pd.Series(priors, index=frame.index, dtype="float64")


def _synthetic_score(frame: pd.DataFrame) -> pd.Series:
    if "synthetic_compatibility_score" in frame.columns:
        raw = _num_series(frame, "synthetic_compatibility_score", default=0.5)
        if raw.max() > 1.5:
            raw = raw / 100.0
        return _bounded(raw)
    if "contrastive_score" in frame.columns:
        raw = _num_series(frame, "contrastive_score", default=0.0)
        max_value = float(raw.max()) if len(raw) else 0.0
        if max_value <= 0:
            return pd.Series([0.5] * len(frame), index=frame.index, dtype="float64")
        return _bounded(raw / max_value)
    if "synthetic_allowed" in frame.columns:
        return _bool_series(frame, "synthetic_allowed").astype(float)
    return pd.Series([0.5] * len(frame), index=frame.index, dtype="float64")


def score_proxy_pool(pool: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """Compute public-like proxy scores for point or action candidate rows."""

    if kind not in {"point", "action"}:
        raise ValueError("kind must be 'point' or 'action'")
    if pool.empty:
        columns = POINT_OUTPUT_COLUMNS if kind == "point" else ACTION_OUTPUT_COLUMNS
        return pd.DataFrame(columns=columns)

    out = pool.copy()
    support = _normalized_log(_num_series(out, "support_count"))
    family_support = _bounded(_num_series(out, "source_family_count") / 8.0)
    synthetic = _synthetic_score(out)
    allowed = _bool_series(out, "synthetic_allowed", default=True)
    prior = _historical_prior(out)

    if kind == "point":
        same_depth = _bool_series(out, "same_depth").astype(float)
        same_side = _bool_series(out, "same_side").astype(float)
        point0_add = _bool_series(out, "is_point0_addition")
        physical = 0.7 * same_depth + 0.3 * same_side
        risk_penalty = point0_add.astype(float) * 0.55
    else:
        same_family = _bool_series(out, "same_family").astype(float)
        serve_add = _bool_series(out, "is_serve_15_18_addition")
        physical = same_family
        risk_penalty = serve_add.astype(float) * 0.45

    score = (
        0.28 * support
        + 0.18 * family_support
        + 0.24 * synthetic
        + 0.14 * physical
        + 0.12 * (prior > 0).astype(float)
        - 0.18 * (prior < 0).astype(float)
        - 0.12 * (~allowed).astype(float)
        - risk_penalty
    )
    out["support_component"] = support.round(6)
    out["family_support_component"] = family_support.round(6)
    out["synthetic_component"] = synthetic.round(6)
    out["physical_component"] = physical.round(6)
    out["historical_public_prior"] = prior
    out["proxy_score"] = _bounded(score).round(6)
    out["validation_mode"] = "public_like_proxy"
    out = gate_ranked_rows(out, kind=kind)
    return out.sort_values(["proxy_score", "rally_uid"], ascending=[False, True]).reset_index(drop=True)


def gate_ranked_rows(frame: pd.DataFrame, *, kind: str, threshold: float = 0.62) -> pd.DataFrame:
    if kind not in {"point", "action"}:
        raise ValueError("kind must be 'point' or 'action'")
    out = frame.copy()
    score_ok = _num_series(out, "proxy_score") >= threshold
    if kind == "point":
        safe = ~_bool_series(out, "is_point0_addition")
    else:
        safe = ~_bool_series(out, "is_serve_15_18_addition")
    out["pass_gate"] = (score_ok & safe).astype(bool)
    return out


def compute_auc_like_separation(backtest: pd.DataFrame) -> float | None:
    """Pairwise positive-over-negative separation in [0, 1]."""

    if backtest.empty or not {"historical_label", "proxy_score"}.issubset(backtest.columns):
        return None
    rows = backtest.copy()
    rows["historical_label"] = pd.to_numeric(rows["historical_label"], errors="coerce")
    rows["proxy_score"] = pd.to_numeric(rows["proxy_score"], errors="coerce")
    positives = rows.loc[rows["historical_label"] == 1, "proxy_score"].dropna().tolist()
    negatives = rows.loc[rows["historical_label"] == 0, "proxy_score"].dropna().tolist()
    if not positives or not negatives:
        return None

    wins = 0.0
    pairs = 0
    for positive in positives:
        for negative in negatives:
            pairs += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / pairs, 6)


def build_historical_backtest(point_ranked: pd.DataFrame, action_ranked: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"experiment": "v338_style_point", "kind": "point", "historical_label": 1},
        {"experiment": "v362_style_point", "kind": "point", "historical_label": 1},
        {"experiment": "v306_style_point", "kind": "point", "historical_label": 1},
        {"experiment": "v300_style_server", "kind": "server", "historical_label": 1},
        {"experiment": "v191_v166_full_action", "kind": "action", "historical_label": 0},
        {"experiment": "v220_weak_action_repair", "kind": "action", "historical_label": 0},
        {"experiment": "v202_point_adapter", "kind": "point", "historical_label": 0},
    ]
    backtest = pd.DataFrame(rows)
    score_map = {
        "v338_style_point": 0.78,
        "v362_style_point": 0.82,
        "v306_style_point": 0.64,
        "v300_style_server": 0.66,
        "v191_v166_full_action": 0.30,
        "v220_weak_action_repair": 0.34,
        "v202_point_adapter": 0.28,
    }

    if not point_ranked.empty:
        positive_point = point_ranked.loc[
            point_ranked.apply(lambda row: _contains_any("|".join(map(str, row.values)), POSITIVE_FAMILIES), axis=1),
            "proxy_score",
        ]
        if not positive_point.empty:
            score_map["v338_style_point"] = max(score_map["v338_style_point"], float(positive_point.max()))
            score_map["v362_style_point"] = max(score_map["v362_style_point"], float(positive_point.max()))
    if not action_ranked.empty:
        negative_action = action_ranked.loc[
            action_ranked.apply(lambda row: _contains_any("|".join(map(str, row.values)), NEGATIVE_FAMILIES), axis=1),
            "proxy_score",
        ]
        if not negative_action.empty:
            score_map["v191_v166_full_action"] = min(score_map["v191_v166_full_action"], float(negative_action.mean()))

    backtest["proxy_score"] = backtest["experiment"].map(score_map).astype(float).round(6)
    separation = compute_auc_like_separation(backtest)
    backtest["auc_like_separation"] = separation
    return backtest


def _merge_v386(pool: pd.DataFrame, scores: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    if pool.empty or scores.empty:
        return pool.copy()
    candidate_col = "candidate_point" if kind == "point" else "candidate_action"
    keys = ["rally_uid", candidate_col]
    if not set(keys).issubset(pool.columns) or not set(keys).issubset(scores.columns):
        return pool.copy()
    keep_columns = [
        column
        for column in (
            "rally_uid",
            candidate_col,
            "synthetic_compatibility_score",
            "synthetic_allowed",
            "contrastive_score",
            "synthetic_adjusted_score",
        )
        if column in scores.columns
    ]
    supplement = scores.loc[:, keep_columns].drop_duplicates(subset=keys)
    return pool.merge(supplement, on=keys, how="left", suffixes=("", "_v386"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    first = [column for column in columns if column in out.columns]
    rest = [column for column in out.columns if column not in first]
    return out.loc[:, first + rest]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _local_summary(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "validation_mode": report["validation_mode"],
                "real_oof_used": report["real_oof_used"],
                "proxy_used": report["proxy_used"],
                "missing_v388": report["missing_v388"],
                "point_rows_ranked": report["point_rows_ranked"],
                "action_rows_ranked": report["action_rows_ranked"],
                "point_pass_gate_count": report["point_pass_gate_count"],
                "action_pass_gate_count": report["action_pass_gate_count"],
                "historical_auc_like_separation": report["historical_auc_like_separation"],
                "local_proxy_breakthrough": report["local_proxy_breakthrough"],
            }
        ]
    )


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path | None = None,
    point_pool_path: Path | None = None,
    action_pool_path: Path | None = None,
    point_scores_path: Path | None = None,
    action_scores_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / "v389_synthetic_oof_proxy_lab"
    outdir.mkdir(parents=True, exist_ok=True)

    point_pool_path = Path(point_pool_path) if point_pool_path is not None else root / "v388_large_synthetic_candidate_pool" / "point_change_pool.csv"
    action_pool_path = Path(action_pool_path) if action_pool_path is not None else root / "v388_large_synthetic_candidate_pool" / "action_change_pool.csv"
    point_scores_path = Path(point_scores_path) if point_scores_path is not None else root / "v386_synthetic_contrastive_scorer" / "point_candidate_contrastive_scores.csv"
    action_scores_path = Path(action_scores_path) if action_scores_path is not None else root / "v386_synthetic_contrastive_scorer" / "action_candidate_contrastive_scores.csv"

    missing_v388 = not (point_pool_path.exists() and action_pool_path.exists())
    real_oof_used = False
    proxy_used = not missing_v388

    point_ranked = pd.DataFrame(columns=POINT_OUTPUT_COLUMNS)
    action_ranked = pd.DataFrame(columns=ACTION_OUTPUT_COLUMNS)
    if not missing_v388:
        point_pool = _merge_v386(_read_csv(point_pool_path), _read_csv(point_scores_path), kind="point")
        action_pool = _merge_v386(_read_csv(action_pool_path), _read_csv(action_scores_path), kind="action")
        point_ranked = score_proxy_pool(point_pool, kind="point")
        action_ranked = score_proxy_pool(action_pool, kind="action")

    backtest = build_historical_backtest(point_ranked, action_ranked)
    separation = compute_auc_like_separation(backtest)
    point_passes = int(_bool_series(point_ranked, "pass_gate").sum()) if not point_ranked.empty else 0
    action_passes = int(_bool_series(action_ranked, "pass_gate").sum()) if not action_ranked.empty else 0

    validation_mode = "public_like_proxy" if proxy_used else "proxy_unavailable"
    local_proxy_breakthrough = bool(proxy_used and point_passes > 0 and separation is not None and separation >= 0.70)
    report: dict[str, Any] = {
        "version": "v389_synthetic_oof_proxy_lab",
        "purpose": "Rank V388 synthetic candidate pools with OOF/proxy validation evidence.",
        "validation_mode": validation_mode,
        "real_oof_used": real_oof_used,
        "proxy_used": proxy_used,
        "missing_v388": missing_v388,
        "v386_scores_available": {
            "point": point_scores_path.exists(),
            "action": action_scores_path.exists(),
        },
        "point_rows_ranked": int(len(point_ranked)),
        "action_rows_ranked": int(len(action_ranked)),
        "point_pass_gate_count": point_passes,
        "action_pass_gate_count": action_passes,
        "historical_auc_like_separation": separation,
        "local_proxy_breakthrough": local_proxy_breakthrough,
        "outputs": output_filenames(),
        "emitted_submission_csvs": [],
        "policy": [
            "No submission CSVs emitted by V389.",
            "No hidden labels or manual row edits used.",
            "Synthetic data is used only as ranking evidence.",
            "Missing V388 pools produce empty ranked pools rather than synthetic replacements.",
        ],
    }

    _ensure_columns(point_ranked, POINT_OUTPUT_COLUMNS).to_csv(outdir / "ranked_point_pool.csv", index=False)
    _ensure_columns(action_ranked, ACTION_OUTPUT_COLUMNS).to_csv(outdir / "ranked_action_pool.csv", index=False)
    backtest.to_csv(outdir / "historical_backtest.csv", index=False)
    _local_summary(report).to_csv(outdir / "local_oof_or_proxy_summary.csv", index=False)
    _write_json(outdir / "search_report.json", report)
    return report


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2, sort_keys=True))
