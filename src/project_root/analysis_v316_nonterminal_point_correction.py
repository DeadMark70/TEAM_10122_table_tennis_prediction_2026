"""V316 nonterminal point correction search.

This branch keeps the public-positive V306 point0 anchor as the base, then
tests only nonzero-to-nonzero point corrections around the 4/5/6 and 7/8/9
confusion groups. Packaging is fixed to V173 action and V300 server from the
V306 submission. Outputs are local-only under v316_nonterminal_point_correction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import POINT_CLASSES
from analysis_v261_action_conditioned_point_residual import (
    EXPECTED_COLUMNS,
    add_foldsafe_proxy_columns,
    build_frames,
    distribution,
    normalize_rows_safe,
    numeric_feature_columns,
    point_depth,
    point_side,
    train_oof_prob,
)
from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column
from analysis_v306_point0_addition_probe import (
    V300_SUBMISSION,
    apply_point0_additions,
    load_artifacts,
    load_submission,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v316_nonterminal_point_correction"
V306_PUBLIC_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V306_SEARCH = ROOT / "v306_point0_addition_probe" / "v306_point0_search.csv"
SEARCH_PATH = OUTDIR / "v316_nonterminal_search.csv"
CHANGED_ROWS_PATH = OUTDIR / "v316_changed_rows.csv"
REPORT_JSON_PATH = OUTDIR / "v316_report.json"
REPORT_MD_PATH = OUTDIR / "v316_report.md"
EXPECTED_ROWS = 1845
NONTERMINAL_TARGETS = set(range(1, 10))
SHORTMID_POINTS = {4, 5, 6}
LONGSIDE_POINTS = {7, 8, 9}
FOCUS_POINTS = SHORTMID_POINTS | LONGSIDE_POINTS
LOCAL_ONLY_BANNED_PARTS = {"upload_candidates_20260519", "selected"}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    submission: str
    budget: int
    family: str


CANDIDATES = [
    CandidateSpec(
        "v316_longside_side_budget12",
        "submission_v316_longside_side_budget12__v173action_v300server.csv",
        12,
        "longside_side",
    ),
    CandidateSpec(
        "v316_longside_side_budget24",
        "submission_v316_longside_side_budget24__v173action_v300server.csv",
        24,
        "longside_side",
    ),
    CandidateSpec(
        "v316_depth_side_agree_budget18",
        "submission_v316_depth_side_agree_budget18__v173action_v300server.csv",
        18,
        "depth_side_agree",
    ),
    CandidateSpec(
        "v316_actioncond_nonterminal_budget24",
        "submission_v316_actioncond_nonterminal_budget24__v173action_v300server.csv",
        24,
        "actioncond_nonterminal",
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def ensure_local_output_path(path: str | Path) -> Path:
    """Return a path only if it stays inside the V316 local output folder."""
    path_obj = Path(path)
    parts = {part.lower() for part in path_obj.parts}
    if parts & LOCAL_ONLY_BANNED_PARTS:
        raise ValueError(f"V316 outputs are local-only; banned path: {path}")
    resolved = path_obj if path_obj.is_absolute() else ROOT / path_obj
    try:
        resolved.resolve().relative_to(OUTDIR.resolve())
    except ValueError as exc:
        raise ValueError(f"V316 outputs are local-only under {OUTDIR}: {path}") from exc
    return resolved


def validate_submission_frame(frame: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"columns={list(frame.columns)} expected={EXPECTED_COLUMNS}")
    if len(frame) != expected_rows:
        raise ValueError(f"rows={len(frame)} expected={expected_rows}")
    if not frame["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of range")
    if not frame["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
    server = pd.to_numeric(frame["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    if not server.between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")
    return frame.loc[:, EXPECTED_COLUMNS].copy()


def count_point0_changes(base: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    base_arr = np.asarray(base, dtype=int)
    pred_arr = np.asarray(pred, dtype=int)
    if len(base_arr) != len(pred_arr):
        raise ValueError("base and pred must have the same length")
    return {
        "point0_additions": int(((base_arr != 0) & (pred_arr == 0)).sum()),
        "point0_removals": int(((base_arr == 0) & (pred_arr != 0)).sum()),
    }


def build_best_nonterminal_candidates(
    base_point: np.ndarray,
    prob: np.ndarray,
    *,
    allowed_targets: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each row's best nonzero replacement and its margin over base."""
    base = np.asarray(base_point, dtype=int)
    p = np.asarray(prob, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    if p.ndim != 2 or len(p) != len(base):
        raise ValueError("prob and base_point row counts must match")
    targets = sorted((set(allowed_targets or NONTERMINAL_TARGETS) & NONTERMINAL_TARGETS) - {0})
    if not targets:
        raise ValueError("allowed_targets must contain at least one nonzero point")

    candidate = base.copy()
    margin = np.full(len(base), -np.inf, dtype=float)
    for i, old in enumerate(base):
        row_targets = [target for target in targets if target != int(old) and target < p.shape[1]]
        if not row_targets:
            continue
        probs = p[i, row_targets]
        best_pos = int(np.argmax(probs))
        new = int(row_targets[best_pos])
        candidate[i] = new
        margin[i] = float(p[i, new] - p[i, np.clip(old, 0, p.shape[1] - 1)])
    return candidate, margin


def select_nonterminal_replacements(
    base_point: np.ndarray,
    candidate_point: np.ndarray,
    score: np.ndarray,
    *,
    budget: int,
    allowed_pairs: set[tuple[int, int]] | None = None,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    """Select top positive nonzero-to-nonzero point replacements."""
    base = np.asarray(base_point, dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    if not (len(base) == len(cand) == len(score_arr)):
        raise ValueError("base_point, candidate_point, and score must have the same length")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    gate_arr = np.ones(len(base), dtype=bool) if gate is None else np.asarray(gate, dtype=bool)
    if len(gate_arr) != len(base):
        raise ValueError("gate must have the same length as base_point")

    eligible = (
        (base != 0)
        & (cand != 0)
        & (base != cand)
        & np.isin(base, sorted(FOCUS_POINTS))
        & np.isin(cand, sorted(FOCUS_POINTS))
        & gate_arr
        & np.isfinite(score_arr)
        & (score_arr > 0)
    )
    if allowed_pairs is not None:
        pair_ok = np.array([(int(old), int(new)) in allowed_pairs for old, new in zip(base, cand)], dtype=bool)
        eligible &= pair_ok

    selected = np.zeros(len(base), dtype=bool)
    if budget == 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score_arr[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected


def add_base_point_columns(frame: pd.DataFrame, base_point: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    base = np.asarray(base_point, dtype=int)
    out["v316_base_point"] = base
    out["v316_base_depth"] = [point_depth(x) for x in base]
    out["v316_base_side"] = [point_side(x) for x in base]
    out["v316_base_is_longside"] = np.isin(base, sorted(LONGSIDE_POINTS)).astype(int)
    return out


def add_action_anchor_columns(frame: pd.DataFrame, action: np.ndarray | None) -> pd.DataFrame:
    out = frame.copy()
    if action is not None:
        out["v316_action_anchor"] = np.asarray(action, dtype=int)
        out["v316_action_family"] = out["v316_action_anchor"].map(lambda x: 0 if x == 0 else 1 if x <= 7 else 2 if x <= 11 else 3 if x <= 14 else 4)
    elif "v261_action_family" in out:
        out["v316_action_family"] = out["v261_action_family"].astype(int)
    else:
        out["v316_action_family"] = 0
    return out


def foldsafe_point_prior(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    key_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build fold-safe P(point | key_cols) tables with Laplace smoothing."""
    target = np.asarray(y, dtype=int)
    global_counts = np.bincount(target, minlength=len(POINT_CLASSES)).astype(float) + 1.0
    global_prob = global_counts / global_counts.sum()
    oof = np.tile(global_prob, (len(train_df), 1))

    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(int(fold)).to_numpy()
        fit = train_df.loc[~valid, key_cols].copy()
        fit["target"] = target[~valid]
        table = point_prior_table(fit, key_cols)
        oof[valid] = lookup_prior(train_df.loc[valid, key_cols], table, key_cols, global_prob)

    fit_all = train_df.loc[:, key_cols].copy()
    fit_all["target"] = target
    table_all = point_prior_table(fit_all, key_cols)
    test_prior = lookup_prior(test_df.loc[:, key_cols], table_all, key_cols, global_prob)
    return normalize_rows_safe(oof), normalize_rows_safe(test_prior), table_all


def point_prior_table(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    counts = frame.groupby(key_cols + ["target"], observed=True).size().unstack("target", fill_value=0)
    for cls in POINT_CLASSES:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[POINT_CLASSES].reset_index()
    total = counts[POINT_CLASSES].sum(axis=1).astype(float)
    out = counts[key_cols].copy()
    out["support"] = total.astype(int)
    denom = total + len(POINT_CLASSES)
    for cls in POINT_CLASSES:
        out[f"p{cls}"] = (counts[cls].astype(float) + 1.0) / denom
        out[f"count_p{cls}"] = counts[cls].astype(int)
    return out


def lookup_prior(frame: pd.DataFrame, table: pd.DataFrame, key_cols: list[str], global_prob: np.ndarray) -> np.ndarray:
    merged = frame.merge(table, on=key_cols, how="left")
    prob_cols = [f"p{cls}" for cls in POINT_CLASSES]
    out = merged[prob_cols].to_numpy(dtype=float)
    missing = ~np.isfinite(out).all(axis=1)
    if missing.any():
        out[missing] = global_prob
    return normalize_rows_safe(out)


def lookup_support(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    key_cols: list[str],
    candidate_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    merged = frame.loc[:, key_cols].merge(table, on=key_cols, how="left")
    support = pd.to_numeric(merged.get("support", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
    counts = np.zeros(len(frame), dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    for cls in POINT_CLASSES:
        mask = cand == int(cls)
        if mask.any() and f"count_p{cls}" in merged:
            counts[mask] = pd.to_numeric(merged.loc[mask, f"count_p{cls}"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return support, counts


def _same_depth_side_change(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    return np.array(
        [
            old != 0
            and new != 0
            and old != new
            and point_depth(old) == point_depth(new)
            and point_side(old) != point_side(new)
            for old, new in zip(base, cand)
        ],
        dtype=bool,
    )


def _confusion_pairs(points: set[int]) -> set[tuple[int, int]]:
    return {(old, new) for old in points for new in points if old != new}


def candidate_arrays(
    family: str,
    base_point: np.ndarray,
    model_prob: np.ndarray,
    prior_prob: np.ndarray,
    v188_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    if family == "longside_side":
        model_cand, model_margin = build_best_nonterminal_candidates(base_point, model_prob, allowed_targets=LONGSIDE_POINTS)
        prior_cand, prior_margin = build_best_nonterminal_candidates(base_point, prior_prob, allowed_targets=LONGSIDE_POINTS)
        v188_cand, v188_margin = build_best_nonterminal_candidates(base_point, v188_prob, allowed_targets=LONGSIDE_POINTS)
        gate = np.isin(base_point, sorted(LONGSIDE_POINTS)) & _same_depth_side_change(base_point, model_cand)
        score = 0.60 * model_margin + 0.25 * v188_margin + 0.15 * prior_margin
    elif family == "depth_side_agree":
        model_cand, model_margin = build_best_nonterminal_candidates(base_point, model_prob, allowed_targets=FOCUS_POINTS)
        prior_cand, prior_margin = build_best_nonterminal_candidates(base_point, prior_prob, allowed_targets=FOCUS_POINTS)
        v188_cand, v188_margin = build_best_nonterminal_candidates(base_point, v188_prob, allowed_targets=FOCUS_POINTS)
        agree = (model_cand == prior_cand) | (model_cand == v188_cand)
        gate = np.isin(base_point, sorted(FOCUS_POINTS)) & _same_depth_side_change(base_point, model_cand) & agree
        score = np.minimum(model_margin, np.maximum(v188_margin, prior_margin)) + 0.25 * model_margin
    elif family == "actioncond_nonterminal":
        prior_cand, prior_margin = build_best_nonterminal_candidates(base_point, prior_prob, allowed_targets=FOCUS_POINTS)
        model_cand, model_margin = build_best_nonterminal_candidates(base_point, model_prob, allowed_targets=FOCUS_POINTS)
        v188_cand, v188_margin = build_best_nonterminal_candidates(base_point, v188_prob, allowed_targets=FOCUS_POINTS)
        gate = np.isin(base_point, sorted(FOCUS_POINTS)) & (prior_cand != 0) & (prior_cand != base_point)
        candidate = prior_cand.copy()
        score = 0.55 * prior_margin + 0.30 * model_margin + 0.15 * v188_margin
        return {
            "candidate": candidate,
            "score": score,
            "model_candidate": model_cand,
            "prior_candidate": prior_cand,
            "v188_candidate": v188_cand,
            "model_margin": model_margin,
            "prior_margin": prior_margin,
            "v188_margin": v188_margin,
            "gate": gate,
        }
    else:
        raise ValueError(f"unknown family: {family}")
    return {
        "candidate": model_cand,
        "score": score,
        "model_candidate": model_cand,
        "prior_candidate": prior_cand,
        "v188_candidate": v188_cand,
        "model_margin": model_margin,
        "prior_margin": prior_margin,
        "v188_margin": v188_margin,
        "gate": gate,
    }


def load_v306_cap0p01_budget(oof_rows: int, test_rows: int) -> tuple[int, int]:
    if V306_SEARCH.exists():
        search = pd.read_csv(V306_SEARCH)
        row = search[search["candidate"].eq("v306_p0_cap0p01")]
        if not row.empty:
            return int(row.iloc[0]["oof_budget"]), int(row.iloc[0]["test_budget"])
    return int(np.floor(oof_rows * 0.01)), int(np.floor(test_rows * 0.01))


def build_bundle() -> dict[str, Any]:
    artifacts = load_artifacts()
    train_df, test_df, _ = build_frames()
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns(train_df, test_df)
    train_df = align_train_to_literal_meta(train_df, artifacts["meta"])
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0

    y = train_df["next_pointId"].astype(int).to_numpy()
    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_cap5_point = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_cap5_point = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    if len(oof_cap5_point) != len(y):
        raise ValueError(f"OOF cap5 length {len(oof_cap5_point)} != y length {len(y)}")

    features = [c for c in numeric_feature_columns(train_df, include_proxy=True) if c in test_df]
    model_oof_prob, model_test_prob, point_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        features,
        seed=31610,
        n_estimators=300,
        min_samples_leaf=4,
    )

    oof_p0_budget, test_p0_budget = load_v306_cap0p01_budget(len(oof_cap5_point), len(test_cap5_point))
    v306_oof_point, oof_p0_selected, _ = apply_point0_additions(oof_cap5_point, model_oof_prob, oof_p0_budget)
    v306_anchor = load_submission(V306_PUBLIC_ANCHOR)
    v300_anchor = load_submission(V300_SUBMISSION)
    if not v306_anchor["rally_uid"].equals(v300_anchor["rally_uid"]):
        raise ValueError("V306 anchor and V300 anchor rally_uid differ")
    v306_test_point = v306_anchor["pointId"].astype(int).to_numpy()
    reconstructed_test_point, _, _ = apply_point0_additions(test_cap5_point, model_test_prob, test_p0_budget)
    if len(v306_test_point) != len(reconstructed_test_point):
        raise ValueError("V306 test anchor row count mismatch")

    train_df = add_base_point_columns(train_df, v306_oof_point)
    test_df = add_base_point_columns(test_df, v306_test_point)
    train_df = add_action_anchor_columns(train_df, None)
    test_df = add_action_anchor_columns(test_df, v306_anchor["actionId"].astype(int).to_numpy())
    key_cols = [
        col
        for col in [
            "v316_action_family",
            "lag0_action_family",
            "lag0_point_depth",
            "lag0_point_side",
            "v316_base_depth",
            "v316_base_side",
        ]
        if col in train_df.columns and col in test_df.columns
    ]
    prior_oof_prob, prior_test_prob, prior_table = foldsafe_point_prior(train_df, test_df, y, key_cols)
    base_score = float(f1_score(y, v306_oof_point, labels=POINT_CLASSES, average="macro", zero_division=0))
    return {
        "artifacts": artifacts,
        "train_df": train_df,
        "test_df": test_df,
        "y": y,
        "base_oof_point": v306_oof_point,
        "base_test_point": v306_test_point,
        "cap5_oof_point": oof_cap5_point,
        "cap5_test_point": test_cap5_point,
        "model_oof_prob": model_oof_prob,
        "model_test_prob": model_test_prob,
        "prior_oof_prob": prior_oof_prob,
        "prior_test_prob": prior_test_prob,
        "prior_table": prior_table,
        "prior_key_cols": key_cols,
        "anchor": v306_anchor,
        "base_score": base_score,
        "folds": proxy_folds + [{"stage": "v316_point_model", **row} for row in point_folds],
        "features_count": len(features),
        "v306_p0_oof_rows": int(oof_p0_selected.sum()),
        "v306_p0_oof_budget": int(oof_p0_budget),
        "v306_p0_test_budget": int(test_p0_budget),
        "v306_reconstructed_test_mismatch_rows": int(np.sum(reconstructed_test_point != v306_test_point)),
    }


def apply_selection(base_point: np.ndarray, candidate_point: np.ndarray, selected: np.ndarray) -> np.ndarray:
    pred = np.asarray(base_point, dtype=int).copy()
    pred[np.asarray(selected, dtype=bool)] = np.asarray(candidate_point, dtype=int)[np.asarray(selected, dtype=bool)]
    return pred


def decision_label(local_delta: float, changed_rows: int, point0_counts: dict[str, int]) -> str:
    if point0_counts["point0_additions"] or point0_counts["point0_removals"]:
        return "REJECT_POINT0_CHANGE"
    churn = changed_rows / EXPECTED_ROWS
    if changed_rows > 0 and churn <= 0.02 and float(local_delta) >= 0.0:
        return "REVIEW"
    return "DIAGNOSTIC"


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = validate_submission_frame(out.loc[:, EXPECTED_COLUMNS])
    path = ensure_local_output_path(OUTDIR / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path.relative_to(ROOT))


def changed_rows_frame(
    spec: CandidateSpec,
    selected: np.ndarray,
    pred: np.ndarray,
    arrays: dict[str, np.ndarray],
    bundle: dict[str, Any],
) -> pd.DataFrame:
    idx = np.where(selected)[0]
    if len(idx) == 0:
        return pd.DataFrame()
    candidate = arrays["candidate"]
    support, target_support = lookup_support(
        bundle["test_df"],
        bundle["prior_table"],
        bundle["prior_key_cols"],
        candidate,
    )
    anchor = bundle["anchor"]
    base = bundle["base_test_point"]
    rows = pd.DataFrame(
        {
            "candidate": spec.name,
            "row_id": idx,
            "rally_uid": anchor.iloc[idx]["rally_uid"].astype(int).to_numpy(),
            "actionId": anchor.iloc[idx]["actionId"].astype(int).to_numpy(),
            "old_pointId": base[idx],
            "new_pointId": pred[idx],
            "change": [f"{int(old)}->{int(new)}" for old, new in zip(base[idx], pred[idx])],
            "score": arrays["score"][idx],
            "model_margin": arrays["model_margin"][idx],
            "prior_margin": arrays["prior_margin"][idx],
            "v188_margin": arrays["v188_margin"][idx],
            "prior_slice_support": support[idx],
            "prior_target_support": target_support[idx],
            "lag0_actionId": bundle["test_df"].iloc[idx]["lag0_actionId"].to_numpy()
            if "lag0_actionId" in bundle["test_df"]
            else np.nan,
            "lag0_pointId": bundle["test_df"].iloc[idx]["lag0_pointId"].to_numpy()
            if "lag0_pointId" in bundle["test_df"]
            else np.nan,
            "prefix_len": bundle["test_df"].iloc[idx]["prefix_len"].to_numpy()
            if "prefix_len" in bundle["test_df"]
            else np.nan,
            "serverGetPoint": anchor.iloc[idx]["serverGetPoint"].to_numpy(dtype=float),
        }
    )
    return rows.sort_values(["score", "model_margin"], ascending=[False, False]).reset_index(drop=True)


def evaluate_candidate(spec: CandidateSpec, bundle: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    oof_arrays = candidate_arrays(
        spec.family,
        bundle["base_oof_point"],
        bundle["model_oof_prob"],
        bundle["prior_oof_prob"],
        bundle["artifacts"]["v188_oof_prob"],
    )
    test_arrays = candidate_arrays(
        spec.family,
        bundle["base_test_point"],
        bundle["model_test_prob"],
        bundle["prior_test_prob"],
        bundle["artifacts"]["v188_test_prob"],
    )

    if spec.family == "longside_side":
        allowed_pairs = _confusion_pairs(LONGSIDE_POINTS)
    elif spec.family in {"depth_side_agree", "actioncond_nonterminal"}:
        allowed_pairs = _confusion_pairs(SHORTMID_POINTS) | _confusion_pairs(LONGSIDE_POINTS)
    else:
        allowed_pairs = None

    cap = spec.budget / len(bundle["base_test_point"])
    oof_budget = int(np.floor(len(bundle["base_oof_point"]) * cap))
    oof_selected = select_nonterminal_replacements(
        bundle["base_oof_point"],
        oof_arrays["candidate"],
        oof_arrays["score"],
        budget=oof_budget,
        allowed_pairs=allowed_pairs,
        gate=oof_arrays["gate"],
    )
    test_selected = select_nonterminal_replacements(
        bundle["base_test_point"],
        test_arrays["candidate"],
        test_arrays["score"],
        budget=spec.budget,
        allowed_pairs=allowed_pairs,
        gate=test_arrays["gate"],
    )
    oof_pred = apply_selection(bundle["base_oof_point"], oof_arrays["candidate"], oof_selected)
    test_pred = apply_selection(bundle["base_test_point"], test_arrays["candidate"], test_selected)
    counts = count_point0_changes(bundle["base_test_point"], test_pred)
    if counts["point0_additions"] or counts["point0_removals"]:
        raise ValueError(f"{spec.name} attempted point0 changes: {counts}")

    score = float(f1_score(bundle["y"], oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    delta = score - float(bundle["base_score"])
    changed = int(test_selected.sum())
    path = write_submission(bundle["anchor"], test_pred, spec.submission)
    changed_rows = changed_rows_frame(spec, test_selected, test_pred, test_arrays, bundle)
    support_mean = float(changed_rows["prior_slice_support"].mean()) if not changed_rows.empty else 0.0
    target_support_mean = float(changed_rows["prior_target_support"].mean()) if not changed_rows.empty else 0.0
    record = {
        "candidate": spec.name,
        "submission": spec.submission,
        "path": path,
        "family": spec.family,
        "budget": spec.budget,
        "oof_budget": oof_budget,
        "point_macro_f1": score,
        "local_delta_vs_v306_point_anchor": delta,
        "base_point_macro_f1": bundle["base_score"],
        "test_changed_rows": changed,
        "oof_changed_rows": int(oof_selected.sum()),
        "test_churn": changed / len(bundle["base_test_point"]),
        "point0_additions": counts["point0_additions"],
        "point0_removals": counts["point0_removals"],
        "changed_456_rows": int(np.isin(bundle["base_test_point"][test_selected], sorted(SHORTMID_POINTS)).sum()) if changed else 0,
        "changed_789_rows": int(np.isin(bundle["base_test_point"][test_selected], sorted(LONGSIDE_POINTS)).sum()) if changed else 0,
        "score_mean_changed": float(test_arrays["score"][test_selected].mean()) if changed else 0.0,
        "model_margin_mean_changed": float(test_arrays["model_margin"][test_selected].mean()) if changed else 0.0,
        "prior_margin_mean_changed": float(test_arrays["prior_margin"][test_selected].mean()) if changed else 0.0,
        "v188_margin_mean_changed": float(test_arrays["v188_margin"][test_selected].mean()) if changed else 0.0,
        "prior_slice_support_mean_changed": support_mean,
        "prior_target_support_mean_changed": target_support_mean,
        "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
        "decision": decision_label(delta, changed, counts),
        "risk_tier": "low" if changed <= 18 and delta >= 0 else "medium" if changed <= 36 else "high",
        "packaging": "V173 action + V300 server from V306 public anchor",
    }
    return record, changed_rows


def write_reports(search: pd.DataFrame, changed_rows: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    search = search.sort_values(
        ["local_delta_vs_v306_point_anchor", "test_changed_rows"],
        ascending=[False, True],
    ).reset_index(drop=True)
    search.to_csv(SEARCH_PATH, index=False)
    if changed_rows.empty:
        changed_rows = pd.DataFrame(
            columns=[
                "candidate",
                "row_id",
                "rally_uid",
                "actionId",
                "old_pointId",
                "new_pointId",
                "change",
                "score",
            ]
        )
    changed_rows.to_csv(CHANGED_ROWS_PATH, index=False)

    review = search[search["decision"].eq("REVIEW")]
    best = search.iloc[0].to_dict() if not search.empty else {}
    report = {
        "version": "V316",
        "verdict": "HAS_REVIEW_CANDIDATE" if not review.empty else "DIAGNOSTIC_ONLY",
        "upload_recommendation": "REVIEW" if not review.empty else "DO_NOT_UPLOAD",
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "policy": {
            "base_point_anchor": str(V306_PUBLIC_ANCHOR.relative_to(ROOT)),
            "fixed_action_server": "V173 action + V300 server",
            "no_point0_additions": True,
            "no_point0_removals": True,
            "no_upload_copy": True,
            "no_ttmatch": True,
            "no_old_server": True,
        },
        "base_point_macro_f1": bundle["base_score"],
        "best_candidate": best,
        "top_candidates": search.head(4).to_dict(orient="records"),
        "review_candidates": review.to_dict(orient="records"),
        "features_count": bundle["features_count"],
        "prior_key_cols": bundle["prior_key_cols"],
        "folds": bundle["folds"],
        "v306_p0_oof_rows": bundle["v306_p0_oof_rows"],
        "v306_p0_oof_budget": bundle["v306_p0_oof_budget"],
        "v306_p0_test_budget": bundle["v306_p0_test_budget"],
        "v306_reconstructed_test_mismatch_rows": bundle["v306_reconstructed_test_mismatch_rows"],
        "notes": [
            "Every candidate starts from the V306 public point anchor and changes only nonzero points to other nonzero points.",
            "Longside candidates focus on 7/8/9 side swaps.",
            "Depth-side agreement requires model agreement with V188 or action-conditioned prior inside 4/5/6 or 7/8/9.",
            "Action-conditioned candidate uses support-gated P(point | action/state/base side-depth) tables.",
            "No files are written to upload_candidates_20260519 or submissions/selected.",
        ],
    }
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    lines = [
        "# V316 Nonterminal Point Correction",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Upload recommendation: `{report['upload_recommendation']}`",
        f"- Base point Macro-F1: `{float(bundle['base_score']):.6f}`",
        f"- Best candidate: `{best.get('candidate', 'none')}`",
        f"- Best local delta: `{float(best.get('local_delta_vs_v306_point_anchor', 0.0)):.6f}`",
        f"- Best changed rows: `{int(best.get('test_changed_rows', 0))}`",
        "",
        "## Candidates",
        "",
        "| candidate | delta | rows | 4/5/6 | 7/8/9 | point0 +/- | support mean | decision |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in search.to_dict(orient="records"):
        lines.append(
            f"| `{row['candidate']}` | {float(row['local_delta_vs_v306_point_anchor']):+.6f} | "
            f"{int(row['test_changed_rows'])} | {int(row['changed_456_rows'])} | {int(row['changed_789_rows'])} | "
            f"{int(row['point0_additions'])}/{int(row['point0_removals'])} | "
            f"{float(row['prior_slice_support_mean_changed']):.1f} | `{row['decision']}` |"
        )
    lines.extend(["", f"Search CSV: `{SEARCH_PATH.relative_to(ROOT).as_posix()}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle()
    records: list[dict[str, Any]] = []
    changed: list[pd.DataFrame] = []
    for spec in CANDIDATES:
        record, rows = evaluate_candidate(spec, bundle)
        records.append(record)
        if not rows.empty:
            changed.append(rows)
    search = pd.DataFrame(records)
    changed_rows = pd.concat(changed, ignore_index=True) if changed else pd.DataFrame()
    return write_reports(search, changed_rows, bundle)


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": report["outdir"],
                "verdict": report["verdict"],
                "best": best.get("candidate"),
                "best_delta": best.get("local_delta_vs_v306_point_anchor"),
                "best_rows": best.get("test_changed_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
