"""V469 public-like validation lab for clean server candidates.

This script ranks existing clean server-only candidates before upload. It does
not train new prediction models and does not create modified submissions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v465_clean_server_line import (
    ANCHOR_RELATIVE,
    EXPECTED_ROWS,
    ROOT,
    SUBMISSION_COLUMNS,
    TEST_NEW_RELATIVE,
    TRAIN_RELATIVE,
    clip_prob,
    load_submission,
    no_banned_input_guard,
)

OUT_DIR = ROOT / "v469_server_public_like_validation"
SEARCH_FILES = (
    "v465_clean_server_line/v465_server_search.csv",
    "v466_clean_server_full_sweep/v466_server_search.csv",
    "v467_server_exhaustive_clean_sweep/v467_server_search.csv",
    "v468_server_full_run/v468_server_search.csv",
    "v468_server_full_run_full/v468_server_search.csv",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def _numeric(frame: pd.DataFrame, col: str, default: float = -1.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _action_family(action: pd.Series) -> pd.Series:
    action = pd.to_numeric(action, errors="coerce").fillna(-1).astype(int)
    return pd.Series(
        np.select(
            [
                action.eq(0),
                action.between(1, 7),
                action.between(8, 11),
                action.between(12, 14),
                action.between(15, 18),
            ],
            ["zero", "attack", "control", "defense", "serve"],
            default="other",
        ),
        index=action.index,
    )


def _point_depth(point: pd.Series) -> pd.Series:
    point = pd.to_numeric(point, errors="coerce").fillna(-1).astype(int)
    return pd.Series(
        np.select(
            [
                point.eq(0),
                point.isin([1, 2, 3]),
                point.isin([4, 5, 6]),
                point.isin([7, 8, 9]),
            ],
            ["terminal", "short", "half", "long"],
            default="other",
        ),
        index=point.index,
    )


def make_public_like_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Build observed-feature bins for train/test distribution matching."""
    strike = _numeric(frame, "strikeNumber", default=1.0)
    score_self = _numeric(frame, "scoreSelf", default=0.0)
    score_other = _numeric(frame, "scoreOther", default=0.0)
    score_total = score_self + score_other
    close = (score_self - score_other).abs() <= 1
    pressure = close | ((score_self >= 10) & (score_other >= 9)) | ((score_other >= 10) & (score_self >= 9))
    out = pd.DataFrame(index=frame.index)
    out["prefix_bin"] = pd.cut(strike, bins=[-np.inf, 1, 2, 3, 6, np.inf], labels=["1", "2", "3", "4_6", "7p"]).astype(str)
    out["phase_bin"] = np.select(
        [strike <= 1, strike <= 2, strike <= 3, strike <= 6],
        ["receive", "third_ball", "fourth_ball", "rally_mid"],
        default="rally_long",
    )
    out["score_pressure"] = np.where(pressure, "pressure", "normal")
    out["score_total_bin"] = pd.cut(score_total, bins=[-np.inf, 3, 9, 17, np.inf], labels=["0_3", "4_9", "10_17", "18p"]).astype(str)
    out["lag_action_family"] = _action_family(frame.get("actionId", pd.Series(-1, index=frame.index)))
    out["lag_point_depth"] = _point_depth(frame.get("pointId", pd.Series(-1, index=frame.index)))
    return out.astype(str)


def fit_density_weights(train_bins: pd.DataFrame, test_bins: pd.DataFrame, *, smoothing: float = 5.0, clip: float = 10.0) -> np.ndarray:
    """Estimate p_test(bin)/p_train(bin) on coarse bins."""
    cols = list(train_bins.columns)
    if cols != list(test_bins.columns):
        raise ValueError("train/test bins must have identical columns")
    train_key = train_bins[cols].agg("|".join, axis=1)
    test_key = test_bins[cols].agg("|".join, axis=1)
    train_counts = train_key.value_counts()
    test_counts = test_key.value_counts()
    all_keys = train_counts.index.union(test_counts.index)
    n_train = float(len(train_key))
    n_test = float(len(test_key))
    vocab = float(len(all_keys))
    train_prob = (train_counts.reindex(all_keys, fill_value=0).astype(float) + smoothing) / (n_train + smoothing * vocab)
    test_prob = (test_counts.reindex(all_keys, fill_value=0).astype(float) + smoothing) / (n_test + smoothing * vocab)
    ratio = (test_prob / train_prob).clip(lower=1.0 / clip, upper=clip)
    weights = train_key.map(ratio).to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if mean > 0:
        weights = weights / mean
    return np.clip(weights, 1.0 / clip, clip)


def build_anchor_slices(anchor: pd.DataFrame, test_new: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in anchor.columns or "rally_uid" not in test_new.columns:
        raise ValueError("anchor and test_new must contain rally_uid")
    grouped = test_new.groupby("rally_uid", sort=False)
    ordered = test_new.sort_values(["rally_uid", "strikeNumber"], kind="mergesort") if "strikeNumber" in test_new.columns else test_new
    last = ordered.groupby("rally_uid", sort=False).last(numeric_only=False)
    out = anchor[["rally_uid", "actionId", "pointId", "serverGetPoint"]].copy()
    out["prefix_rows"] = out["rally_uid"].map(grouped.size()).fillna(0).astype(int)
    for col in ["strikeNumber", "scoreSelf", "scoreOther", "actionId", "pointId"]:
        if col in last.columns:
            out[f"{col}_last_obs"] = out["rally_uid"].map(last[col]).fillna(out[col] if col in out.columns else -1)
        else:
            out[f"{col}_last_obs"] = out[col] if col in out.columns else -1
    bins = make_public_like_bins(
        pd.DataFrame(
            {
                "strikeNumber": out["strikeNumber_last_obs"],
                "scoreSelf": out["scoreSelf_last_obs"],
                "scoreOther": out["scoreOther_last_obs"],
                "actionId": out["actionId_last_obs"],
                "pointId": out["pointId_last_obs"],
            }
        )
    )
    for col in bins.columns:
        out[col] = bins[col].to_numpy()
    score_self = _numeric(out, "scoreSelf_last_obs", default=0.0)
    score_other = _numeric(out, "scoreOther_last_obs", default=0.0)
    strike = _numeric(out, "strikeNumber_last_obs", default=1.0)
    out["terminal_like"] = ((out["pointId"].astype(int) == 0) | (out["actionId"].astype(int) == 0) | (strike >= 7)).astype(bool)
    out["score_pressure_bool"] = ((score_self - score_other).abs() <= 1) | ((score_self >= 10) & (score_other >= 9)) | ((score_other >= 10) & (score_self >= 9))
    return out


def load_and_validate_submission(path: Path | str, anchor: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    no_banned_input_guard([path])
    sub = pd.read_csv(path)
    if list(sub.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{path} columns are not {SUBMISSION_COLUMNS}")
    if expected_rows is not None and len(sub) != expected_rows:
        raise ValueError(f"{path} row count mismatch: {len(sub)} != {expected_rows}")
    for col in ["rally_uid", "actionId", "pointId"]:
        if not sub[col].equals(anchor[col]):
            raise ValueError(f"{path} changes {col}")
    if not sub["serverGetPoint"].between(0, 1).all():
        raise ValueError(f"{path} serverGetPoint outside [0, 1]")
    return sub


def load_candidate_boards(root: Path, anchor: pd.DataFrame, *, expected_rows: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seen_paths: set[str] = set()
    for rel in SEARCH_FILES:
        board_path = root / rel
        if not board_path.exists():
            continue
        board = pd.read_csv(board_path)
        if "path" not in board.columns:
            continue
        board = board.copy()
        board["source_board"] = rel
        valid = []
        for _, row in board.iterrows():
            path = Path(str(row["path"]))
            if not path.is_absolute():
                path = root / path
            key = str(path.resolve())
            if key in seen_paths or not path.exists():
                continue
            try:
                load_and_validate_submission(path, anchor, expected_rows=expected_rows)
            except Exception:
                continue
            seen_paths.add(key)
            item = row.copy()
            item["path"] = key
            valid.append(item)
        if valid:
            rows.append(pd.DataFrame(valid))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_candidate_diagnostics(board: pd.DataFrame, anchor: pd.DataFrame, slices: pd.DataFrame) -> pd.DataFrame:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    rows = []
    slice_cols = ["phase_bin", "prefix_bin", "score_pressure", "terminal_like"]
    for _, meta in board.iterrows():
        sub = load_and_validate_submission(meta["path"], anchor, expected_rows=len(anchor))
        server = clip_prob(sub["serverGetPoint"])
        delta = np.abs(server - anchor_server)
        row: dict[str, Any] = meta.to_dict()
        row["server_mad_actual"] = float(delta.mean())
        row["server_corr_actual"] = _corr(server, anchor_server)
        row["server_delta_max"] = float(delta.max())
        row["server_delta_p95"] = float(np.quantile(delta, 0.95))
        total_delta = float(delta.sum())
        top_n = min(20, len(delta))
        row["top20_share"] = float(np.sort(delta)[-top_n:].sum() / total_delta) if total_delta > 0 else 0.0
        ratios = []
        for col in slice_cols:
            for _, idx in slices.groupby(col, sort=False).groups.items():
                mask = np.asarray(list(idx), dtype=int)
                if len(mask) == 0:
                    continue
                slice_mad = float(delta[mask].mean())
                row[f"{col}_max_mad"] = max(float(row.get(f"{col}_max_mad", 0.0)), slice_mad)
                if row["server_mad_actual"] > 0:
                    ratios.append(slice_mad / row["server_mad_actual"])
        row["max_slice_mad_ratio"] = float(max(ratios)) if ratios else 1.0
        row["server_mad"] = float(row.get("server_mad", row["server_mad_actual"]))
        row["server_corr"] = float(row.get("server_corr", row["server_corr_actual"]))
        diversity_raw = row.get("family_diversity", np.nan)
        try:
            diversity = int(diversity_raw) if pd.notna(diversity_raw) else 0
        except Exception:
            diversity = 0
        if diversity <= 0:
            families_text = str(row.get("families", "") if pd.notna(row.get("families", "")) else "")
            diversity = len([part for part in families_text.split("|") if part]) if families_text else 1
        row["family_diversity"] = int(max(diversity, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    mad = pd.to_numeric(out.get("server_mad_actual", out.get("server_mad", 0)), errors="coerce").fillna(0.0)
    corr = pd.to_numeric(out.get("server_corr_actual", out.get("server_corr", 1)), errors="coerce").fillna(0.0)
    diversity = pd.to_numeric(out.get("family_diversity", 1), errors="coerce").fillna(1.0)
    top20 = pd.to_numeric(out.get("top20_share", 1), errors="coerce").fillna(1.0)
    slice_ratio = pd.to_numeric(out.get("max_slice_mad_ratio", 1), errors="coerce").fillna(1.0)
    risk = out.get("risk", pd.Series("safe", index=out.index)).fillna("safe").astype(str)
    decision = out.get("decision", pd.Series("review", index=out.index)).fillna("review").astype(str)
    source = out.get("source_board", pd.Series("", index=out.index)).fillna("").astype(str)
    candidate = out.get("candidate", pd.Series("", index=out.index)).fillna("").astype(str)

    mad_preference = -np.abs(mad - 0.0035) * 180.0
    conservative_bonus = np.where((mad >= 0.0015) & (mad <= 0.0060), 0.8, 0.0)
    ensemble_bonus = candidate.str.contains("ensemble|calibrated|rankmean", case=False, regex=True).astype(float) * 0.4
    newer_bonus = source.str.contains("v468|v467", case=False, regex=True).astype(float) * 0.2
    penalty = (
        (risk.eq("diagnostic") | decision.eq("diagnostic_hold")).astype(float) * 6.0
        + (mad > 0.0100).astype(float) * 4.0
        + np.maximum(top20 - 0.45, 0.0) * 5.0
        + np.maximum(slice_ratio - 2.0, 0.0) * 1.5
        + np.maximum(0.995 - corr, 0.0) * 80.0
    )
    out["v469_score"] = mad_preference + conservative_bonus + ensemble_bonus + newer_bonus + np.log1p(diversity) * 0.15 - penalty
    if "server_mad_actual" not in out.columns:
        out["server_mad_actual"] = mad
    if "server_corr_actual" not in out.columns:
        out["server_corr_actual"] = corr
    out["v469_bucket"] = np.select(
        [
            (mad <= 0.0020) & decision.ne("diagnostic_hold"),
            (mad > 0.0020) & (mad <= 0.0060) & decision.ne("diagnostic_hold"),
            (mad > 0.0060) & (mad <= 0.0100) & decision.ne("diagnostic_hold"),
        ],
        ["conservative", "balanced", "exploratory"],
        default="avoid",
    )
    return out.sort_values(["v469_score", "server_mad_actual"], ascending=[False, True]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    columns = list(frame.columns)
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.to_numpy()]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _write_report_md(path: Path, report: dict[str, Any], ranked: pd.DataFrame) -> None:
    cols = ["candidate", "v469_bucket", "server_mad_actual", "server_corr_actual", "top20_share", "max_slice_mad_ratio", "path"]
    top = ranked.loc[:, [c for c in cols if c in ranked.columns]].head(15)
    lines = [
        "# V469 server public-like validation",
        "",
        f"Candidates ranked: {report['candidate_count']}",
        f"Recommended conservative: {report.get('recommended_conservative')}",
        f"Recommended balanced: {report.get('recommended_balanced')}",
        f"Recommended exploratory: {report.get('recommended_exploratory')}",
        "",
        "This is a no-label validation/ranking layer. It does not guarantee public/private gain.",
        "",
        _markdown_table(top),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(*, root: Path = ROOT, outdir: Path | None = None, expected_rows: int = EXPECTED_ROWS) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / OUT_DIR.name
    no_banned_input_guard([root / ANCHOR_RELATIVE, root / TRAIN_RELATIVE, root / TEST_NEW_RELATIVE, outdir])
    outdir.mkdir(parents=True, exist_ok=True)
    anchor = load_submission(root / ANCHOR_RELATIVE, expected_rows=expected_rows)
    train = pd.read_csv(root / TRAIN_RELATIVE)
    test_new = pd.read_csv(root / TEST_NEW_RELATIVE)

    train_bins = make_public_like_bins(train)
    test_bins = make_public_like_bins(test_new)
    weights = fit_density_weights(train_bins, test_bins)
    weight_profile = train_bins.copy()
    weight_profile["weight"] = weights
    weight_summary = weight_profile.groupby(list(train_bins.columns), sort=False)["weight"].agg(["count", "mean", "max"]).reset_index()
    weight_summary.to_csv(outdir / "v469_weight_profile.csv", index=False)

    slices = build_anchor_slices(anchor, test_new)
    slices.to_csv(outdir / "v469_slice_profile.csv", index=False)
    board = load_candidate_boards(root, anchor, expected_rows=expected_rows)
    diagnostics = compute_candidate_diagnostics(board, anchor, slices) if not board.empty else pd.DataFrame()
    ranked = rank_candidates(diagnostics)
    rank_path = outdir / "v469_candidate_rank.csv"
    ranked.to_csv(rank_path, index=False)

    def best(bucket: str) -> str | None:
        subset = ranked.loc[ranked["v469_bucket"].eq(bucket)] if not ranked.empty else ranked
        return None if subset.empty else str(subset.iloc[0]["path"])

    report = {
        "pipeline": "v469_server_public_like_validation",
        "candidate_count": int(len(ranked)),
        "search_files": SEARCH_FILES,
        "recommended_conservative": best("conservative"),
        "recommended_balanced": best("balanced"),
        "recommended_exploratory": best("exploratory"),
        "weight_mean": float(np.mean(weights)),
        "weight_max": float(np.max(weights)),
        "weight_min": float(np.min(weights)),
        "policy": {
            "no_old_server_labels": True,
            "no_ttmatch": True,
            "no_upload_candidates_20260519": True,
            "no_label_test_new_distribution_only": True,
        },
        "rank_path": str(rank_path.resolve()),
    }
    (outdir / "v469_report.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(outdir / "v469_report.md", report, ranked)
    print(json.dumps(_json_safe({k: report[k] for k in ["candidate_count", "recommended_conservative", "recommended_balanced", "recommended_exploratory"]}), sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
