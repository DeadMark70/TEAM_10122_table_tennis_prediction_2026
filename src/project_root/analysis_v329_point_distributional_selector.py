"""V329 distributional point selector.

Clean point-only exploration on top of the packaged V306/V300/V173 anchor.
The script builds fold-safe distributional point signals from AICUP train/test
prefix rows, evaluates local OOF Macro-F1 against the strongest reconstructable
local point anchor, and exports only evidence-cleared submissions under the
local V329 output directory.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)
from analysis_v261_action_conditioned_point_residual import (
    add_foldsafe_proxy_columns,
    add_geometry_columns,
    normalize_rows_safe,
    numeric_feature_columns,
    point_depth,
    point_side,
    train_oof_prob,
)
from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column
from analysis_v306_point0_addition_probe import apply_point0_additions


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "v329_point_distributional_selector"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V306_SEARCH = ROOT / "v306_point0_addition_probe" / "v306_point0_search.csv"
V305_ARTIFACT_DIR = ROOT / "v305_literal_v188_point_artifact"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
NONTERMINAL_POINTS = set(range(1, 10))
DEFAULT_WEAK_POINTS = {1, 3, 4, 7, 8, 9}
LOCAL_ONLY_BANNED_PARTS = {"upload_candidates_20260519", "upload_candidates", "selected", "submissions"}
MAX_EXPORT_ROWS = 24
REVIEW_DELTA = 0.001


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    budget: int
    filename: str


CANDIDATES = [
    CandidateSpec(
        "v329_depth_side_table_b12",
        "depth_side_table",
        12,
        "submission_v329_depth_side_table_b12__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_depth_side_table_b18",
        "depth_side_table",
        18,
        "submission_v329_depth_side_table_b18__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_action_conditioned_table_b12",
        "action_conditioned_table",
        12,
        "submission_v329_action_conditioned_table_b12__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_action_conditioned_table_b18",
        "action_conditioned_table",
        18,
        "submission_v329_action_conditioned_table_b18__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_terminal_suppressed_nonterminal_b12",
        "terminal_suppressed_nonterminal",
        12,
        "submission_v329_terminal_suppressed_nonterminal_b12__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_terminal_suppressed_nonterminal_b18",
        "terminal_suppressed_nonterminal",
        18,
        "submission_v329_terminal_suppressed_nonterminal_b18__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_agreement_only_model_table_b12",
        "agreement_only_model_table",
        12,
        "submission_v329_agreement_only_model_table_b12__v173action_v300server.csv",
    ),
    CandidateSpec(
        "v329_agreement_only_model_table_b18",
        "agreement_only_model_table",
        18,
        "submission_v329_agreement_only_model_table_b18__v173action_v300server.csv",
    ),
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def path_has_banned_input_token(path: Path | str) -> bool:
    text = str(path).lower()
    return "ttmatch" in text or "old_server" in text or "oldserver" in text


def configured_input_paths() -> list[Path]:
    return [
        ROOT / "train.csv",
        ROOT / "test_new.csv",
        ANCHOR_SUBMISSION,
        V306_SEARCH,
        V305_ARTIFACT_DIR / "v305_v188_cap5_oof_pred.csv",
        V305_ARTIFACT_DIR / "v305_v188_cap5_test_pred.csv",
        V305_ARTIFACT_DIR / "v305_v188_oof_meta.csv",
        V305_ARTIFACT_DIR / "v305_v188_r186_w005_oof_proba.npy",
        V305_ARTIFACT_DIR / "v305_v188_r186_w005_test_proba.npy",
    ]


def ensure_clean_input_paths(paths: Iterable[Path]) -> None:
    banned = [relative_path(path) for path in paths if path_has_banned_input_token(path)]
    if banned:
        raise ValueError(f"V329 refuses banned input paths: {banned}")


def protected_output_path(outdir: Path, filename: str) -> Path:
    root = Path(outdir)
    path = root / filename
    parts = {part.lower() for part in path.parts}
    if parts & LOCAL_ONLY_BANNED_PARTS:
        raise ValueError(f"refusing non-local V329 export path: {path}")
    if path.parent != root or Path(filename).name != filename:
        raise ValueError(f"refusing non-local V329 export path: {path}")
    return path


def load_anchor_submission(path: Path = ANCHOR_SUBMISSION) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing fixed V329 anchor submission: {path}")
    frame = pd.read_csv(path)
    return validate_submission_frame(frame)


def validate_submission_frame(frame: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns={list(frame.columns)} expected={SUBMISSION_COLUMNS}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"rows={len(frame)} expected={expected_rows}")
    if not frame["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of range")
    if not frame["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
    server = pd.to_numeric(frame["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def action_family(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return 0
    if 1 <= action <= 7:
        return 1
    if 8 <= action <= 11:
        return 2
    if 12 <= action <= 14:
        return 3
    if 15 <= action <= 18:
        return 4
    return 0


def prefix_bucket(prefix_len: int) -> int:
    value = int(prefix_len)
    if value <= 1:
        return 1
    if value == 2:
        return 2
    if value <= 4:
        return 3
    return 4


def add_prefix_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["v329_prefix_bucket"] = out["prefix_len"].astype(int).map(prefix_bucket)
    return out


def add_anchor_columns(test_df: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    out = test_df.merge(anchor[["rally_uid", "actionId", "pointId"]], on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId"]].isna().any().any():
        raise ValueError("fixed V329 anchor did not align to test prefix rows")
    out = out.rename(columns={"actionId": "v261_action_proxy", "pointId": "v261_anchor_point"})
    out["v261_action_proxy"] = out["v261_action_proxy"].astype(int)
    out["v261_anchor_point"] = out["v261_anchor_point"].astype(int)
    out["v261_action_family"] = out["v261_action_proxy"].map(action_family)
    out["v261_anchor_depth"] = out["v261_anchor_point"].map(point_depth)
    out["v261_anchor_side"] = out["v261_anchor_point"].map(point_side)
    return out


def build_clean_prefix_frames(anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = add_geometry_columns(build_train_prefix_table(train_raw, 6))
    test_df = add_geometry_columns(build_test_prefix_table(test_raw, 6))
    train_df["fold"] = -1
    rally_meta = train_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    for fold, (_train_idx, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"])):
        valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"].astype(int))
        train_df.loc[train_df["rally_uid"].isin(valid_rallies), "fold"] = fold
    if train_df["fold"].lt(0).any():
        raise RuntimeError("V329 fold assignment failed")
    train_df = add_prefix_bucket(train_df)
    test_df = add_prefix_bucket(test_df)
    test_df = add_anchor_columns(test_df, anchor)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_optional_v305_artifacts() -> dict[str, Any]:
    required = {
        "v188_oof_prob": V305_ARTIFACT_DIR / "v305_v188_r186_w005_oof_proba.npy",
        "v188_test_prob": V305_ARTIFACT_DIR / "v305_v188_r186_w005_test_proba.npy",
        "cap5_oof": V305_ARTIFACT_DIR / "v305_v188_cap5_oof_pred.csv",
        "cap5_test": V305_ARTIFACT_DIR / "v305_v188_cap5_test_pred.csv",
        "meta": V305_ARTIFACT_DIR / "v305_v188_oof_meta.csv",
    }
    if any(not path.exists() for path in required.values()):
        return {"status": "missing"}
    return {
        "status": "loaded",
        "v188_oof_prob": normalize_rows_safe(np.load(required["v188_oof_prob"])),
        "v188_test_prob": normalize_rows_safe(np.load(required["v188_test_prob"])),
        "cap5_oof": pd.read_csv(required["cap5_oof"]),
        "cap5_test": pd.read_csv(required["cap5_test"]),
        "meta": pd.read_csv(required["meta"]),
    }


def load_v306_cap0p01_budget(oof_rows: int, test_rows: int) -> tuple[int, int, str]:
    if V306_SEARCH.exists():
        search = pd.read_csv(V306_SEARCH)
        row = search[search["candidate"].astype(str).eq("v306_p0_cap0p01")]
        if not row.empty:
            return int(row.iloc[0]["oof_budget"]), int(row.iloc[0]["test_budget"]), "v306_search"
    return int(np.floor(oof_rows * 0.01)), int(np.floor(test_rows * 0.01)), "fallback_cap0p01"


def macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y_true, pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def class_f1_scores(y_true: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred, dtype=int)
    return {label: float(f1_score(y == label, p == label, zero_division=0)) for label in POINT_CLASSES}


def weak_point_classes(y_true: np.ndarray, base_pred: np.ndarray, count: int = 5) -> list[int]:
    scores = class_f1_scores(y_true, base_pred)
    nonterminal = [(label, score) for label, score in scores.items() if label != 0]
    nonterminal.sort(key=lambda item: (item[1], item[0]))
    weak = {label for label, _score in nonterminal[:count]}
    weak.update(DEFAULT_WEAK_POINTS)
    return sorted(weak & NONTERMINAL_POINTS)


def point_distribution(values: np.ndarray) -> str:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=len(POINT_CLASSES))
    return json.dumps({str(i): int(v) for i, v in enumerate(counts) if v > 0}, sort_keys=True)


def count_point0_changes(base: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    base_arr = np.asarray(base, dtype=int)
    pred_arr = np.asarray(pred, dtype=int)
    return {
        "point0_additions": int(((base_arr != 0) & (pred_arr == 0)).sum()),
        "point0_removals": int(((base_arr == 0) & (pred_arr != 0)).sum()),
    }


def point_prior_table(frame: pd.DataFrame, key_cols: list[str], target_col: str = "target") -> pd.DataFrame:
    counts = frame.groupby(key_cols + [target_col], observed=True).size().unstack(target_col, fill_value=0)
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
    merged = frame.loc[:, key_cols].merge(table, on=key_cols, how="left")
    prob_cols = [f"p{cls}" for cls in POINT_CLASSES]
    out = merged[prob_cols].to_numpy(dtype=float)
    missing = ~np.isfinite(out).all(axis=1)
    if missing.any():
        out[missing] = global_prob
    return normalize_rows_safe(out)


def foldsafe_point_prior(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    key_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    target = np.asarray(y, dtype=int)
    global_counts = np.bincount(target, minlength=len(POINT_CLASSES)).astype(float) + 1.0
    global_prob = global_counts / global_counts.sum()
    oof = np.tile(global_prob, (len(train_df), 1))
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(int(fold)).to_numpy()
        fit = train_df.loc[~valid, key_cols].copy()
        fit["target"] = target[~valid]
        table = point_prior_table(fit, key_cols)
        oof[valid] = lookup_prior(train_df.loc[valid], table, key_cols, global_prob)
    full = train_df.loc[:, key_cols].copy()
    full["target"] = target
    table_all = point_prior_table(full, key_cols)
    test_prob = lookup_prior(test_df.loc[:, key_cols], table_all, key_cols, global_prob)
    return normalize_rows_safe(oof), normalize_rows_safe(test_prob), table_all


def lookup_support(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    key_cols: list[str],
    candidate_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    merged = frame.loc[:, key_cols].merge(table, on=key_cols, how="left")
    support = pd.to_numeric(merged.get("support", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
    counts = np.zeros(len(frame), dtype=int)
    candidate = np.asarray(candidate_point, dtype=int)
    for cls in POINT_CLASSES:
        mask = candidate == int(cls)
        if mask.any() and f"count_p{cls}" in merged:
            counts[mask] = pd.to_numeric(merged.loc[mask, f"count_p{cls}"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return support, counts


def add_base_point_columns(frame: pd.DataFrame, base_point: np.ndarray, action: np.ndarray | None) -> pd.DataFrame:
    out = frame.copy()
    base = np.asarray(base_point, dtype=int)
    out["v329_base_point"] = base
    out["v329_base_depth"] = [point_depth(x) for x in base]
    out["v329_base_side"] = [point_side(x) for x in base]
    if action is None:
        out["v329_action_anchor"] = out["v261_action_proxy"].astype(int)
    else:
        out["v329_action_anchor"] = np.asarray(action, dtype=int)
    out["v329_action_family"] = out["v329_action_anchor"].map(action_family)
    return out


def best_nonterminal_replacement(
    base_point: np.ndarray,
    prob: np.ndarray,
    allowed_targets: Iterable[int] = NONTERMINAL_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_point, dtype=int)
    p = normalize_rows_safe(prob)
    targets = sorted((set(int(x) for x in allowed_targets) & NONTERMINAL_POINTS) - {0})
    if not targets:
        raise ValueError("allowed_targets must include nonterminal points")
    candidate = base.copy()
    margin = np.full(len(base), -np.inf, dtype=float)
    for row_id, old in enumerate(base):
        row_targets = [target for target in targets if target != int(old) and target < p.shape[1]]
        if not row_targets:
            continue
        probs = p[row_id, row_targets]
        pos = int(np.argmax(probs))
        new = int(row_targets[pos])
        candidate[row_id] = new
        margin[row_id] = float(p[row_id, new] - p[row_id, np.clip(int(old), 0, p.shape[1] - 1)])
    return candidate, margin


def select_ranked_replacements(
    base_point: np.ndarray,
    candidate_point: np.ndarray,
    score: np.ndarray,
    *,
    budget: int,
    weak_points: Iterable[int],
    gate: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(base_point, dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    if not (len(base) == len(cand) == len(score_arr)):
        raise ValueError("base_point, candidate_point, and score must have matching lengths")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    gate_arr = np.ones(len(base), dtype=bool) if gate is None else np.asarray(gate, dtype=bool)
    if len(gate_arr) != len(base):
        raise ValueError("gate must have matching length")
    weak = set(int(x) for x in weak_points) & NONTERMINAL_POINTS
    weak_gate = np.array([(int(old) in weak or int(new) in weak) for old, new in zip(base, cand)], dtype=bool)
    eligible = (
        (base != 0)
        & (cand != 0)
        & (base != cand)
        & weak_gate
        & gate_arr
        & np.isfinite(score_arr)
        & (score_arr > 0)
    )
    selected = np.zeros(len(base), dtype=bool)
    if budget == 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score_arr[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected


def apply_selection(base_point: np.ndarray, candidate_point: np.ndarray, selected: np.ndarray) -> np.ndarray:
    out = np.asarray(base_point, dtype=int).copy()
    mask = np.asarray(selected, dtype=bool)
    out[mask] = np.asarray(candidate_point, dtype=int)[mask]
    return out


def candidate_arrays(
    family: str,
    base_point: np.ndarray,
    model_prob: np.ndarray,
    depth_prob: np.ndarray,
    action_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    model_cand, model_margin = best_nonterminal_replacement(base_point, model_prob)
    depth_cand, depth_margin = best_nonterminal_replacement(base_point, depth_prob)
    action_cand, action_margin = best_nonterminal_replacement(base_point, action_prob)
    terminal_prob = normalize_rows_safe(model_prob)[:, 0]

    if family == "depth_side_table":
        candidate = depth_cand
        score = depth_margin + 0.20 * model_margin
        gate = depth_margin > 0
    elif family == "action_conditioned_table":
        candidate = action_cand
        score = action_margin + 0.15 * model_margin
        gate = action_margin > 0
    elif family == "terminal_suppressed_nonterminal":
        candidate = model_cand
        score = model_margin - 0.25 * terminal_prob
        gate = terminal_prob <= 0.45
    elif family == "agreement_only_model_table":
        agree_depth = model_cand == depth_cand
        agree_action = model_cand == action_cand
        table_margin = np.where(agree_action, action_margin, depth_margin)
        table_margin = np.where(agree_depth & agree_action, np.maximum(depth_margin, action_margin), table_margin)
        candidate = model_cand
        score = np.minimum(model_margin, table_margin) + 0.25 * model_margin
        gate = (agree_depth | agree_action) & (table_margin > 0)
    else:
        raise ValueError(f"unknown candidate family: {family}")

    return {
        "candidate": candidate,
        "score": score,
        "gate": gate,
        "model_candidate": model_cand,
        "depth_candidate": depth_cand,
        "action_candidate": action_cand,
        "model_margin": model_margin,
        "depth_margin": depth_margin,
        "action_margin": action_margin,
        "terminal_prob": terminal_prob,
    }


def evidence_passes(row: dict[str, Any] | pd.Series) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    return bool(
        float(data.get("local_delta_vs_anchor", 0.0)) > 0.0
        and 0 < int(data.get("test_changed_rows", 0)) <= MAX_EXPORT_ROWS
        and int(data.get("point0_additions", 0)) == 0
        and int(data.get("point0_removals", 0)) == 0
    )


def upload_worthy(row: dict[str, Any] | pd.Series) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    return bool(evidence_passes(data) and float(data.get("local_delta_vs_anchor", 0.0)) >= REVIEW_DELTA and int(data.get("test_changed_rows", 0)) <= 18)


def build_export_frame(anchor: pd.DataFrame, point_pred: np.ndarray, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    out = anchor.copy()
    out["pointId"] = np.asarray(point_pred, dtype=int)
    return validate_submission_frame(out.loc[:, SUBMISSION_COLUMNS], expected_rows=expected_rows)


def write_submission_if_evidence(
    outdir: Path,
    spec: CandidateSpec,
    anchor: pd.DataFrame,
    point_pred: np.ndarray,
    evidence: dict[str, Any],
) -> str | None:
    if not evidence_passes(evidence):
        return None
    path = protected_output_path(outdir, spec.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_export_frame(anchor, point_pred).to_csv(path, index=False, float_format="%.8f")
    return relative_path(path)


def build_model_probabilities(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0
    features = [col for col in numeric_feature_columns(train_df, include_proxy=True) if col in test_df]
    oof, test, folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        features,
        seed=32910,
        n_estimators=260,
        min_samples_leaf=4,
    )
    return oof, test, [{"stage": "v329_point_model", **row} for row in folds], len(features)


def reconstruct_anchor_oof(
    y: np.ndarray,
    anchor: pd.DataFrame,
    model_oof_prob: np.ndarray,
    model_test_prob: np.ndarray,
    artifacts: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if artifacts.get("status") == "loaded":
        cap5_oof = artifacts["cap5_oof"]
        cap5_test = artifacts["cap5_test"]
        oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
        test_base = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
        if len(oof_base) == len(y) and len(test_base) == len(anchor):
            oof_budget, test_budget, budget_source = load_v306_cap0p01_budget(len(oof_base), len(test_base))
            oof_point, oof_selected, _ = apply_point0_additions(oof_base, model_oof_prob, oof_budget)
            reconstructed_test, test_selected, _ = apply_point0_additions(test_base, model_test_prob, test_budget)
            return (
                oof_point,
                anchor["pointId"].astype(int).to_numpy(),
                {
                    "anchor_source": "v306_reconstructed_cap0p01",
                    "budget_source": budget_source,
                    "oof_budget": int(oof_budget),
                    "test_budget": int(test_budget),
                    "oof_point0_rows": int(oof_selected.sum()),
                    "test_point0_rows": int(test_selected.sum()),
                    "reconstructed_test_mismatch_rows": int(np.sum(reconstructed_test != anchor["pointId"].astype(int).to_numpy())),
                },
            )
    fallback = normalize_rows_safe(model_oof_prob).argmax(axis=1).astype(int)
    return (
        fallback,
        anchor["pointId"].astype(int).to_numpy(),
        {
            "anchor_source": "fallback_model_argmax_oof_anchor",
            "budget_source": "not_available",
            "oof_budget": 0,
            "test_budget": 0,
            "oof_point0_rows": 0,
            "test_point0_rows": 0,
            "reconstructed_test_mismatch_rows": None,
        },
    )


def build_bundle() -> dict[str, Any]:
    ensure_clean_input_paths(configured_input_paths())
    anchor = load_anchor_submission()
    train_df, test_df = build_clean_prefix_frames(anchor)
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns(train_df, test_df)
    artifacts = load_optional_v305_artifacts()
    if artifacts.get("status") == "loaded":
        train_df = align_train_to_literal_meta(train_df, artifacts["meta"])
    y = train_df["next_pointId"].astype(int).to_numpy()
    model_oof_prob, model_test_prob, model_folds, feature_count = build_model_probabilities(train_df, test_df, y)
    base_oof, base_test, anchor_report = reconstruct_anchor_oof(y, anchor, model_oof_prob, model_test_prob, artifacts)
    train_df = add_base_point_columns(train_df, base_oof, None)
    test_df = add_base_point_columns(test_df, base_test, anchor["actionId"].astype(int).to_numpy())

    depth_keys = [
        "v329_base_depth",
        "v329_base_side",
        "lag0_point_depth",
        "lag0_point_side",
        "v329_prefix_bucket",
    ]
    action_keys = [
        "v329_action_family",
        "lag0_action_family",
        "v329_base_depth",
        "v329_base_side",
        "v329_prefix_bucket",
    ]
    depth_oof, depth_test, depth_table = foldsafe_point_prior(train_df, test_df, y, depth_keys)
    action_oof, action_test, action_table = foldsafe_point_prior(train_df, test_df, y, action_keys)
    weak_points = weak_point_classes(y, base_oof)
    return {
        "anchor": anchor,
        "train_df": train_df,
        "test_df": test_df,
        "y": y,
        "base_oof": base_oof,
        "base_test": base_test,
        "base_score": macro_f1(y, base_oof),
        "weak_points": weak_points,
        "model_oof_prob": model_oof_prob,
        "model_test_prob": model_test_prob,
        "depth_oof_prob": depth_oof,
        "depth_test_prob": depth_test,
        "action_oof_prob": action_oof,
        "action_test_prob": action_test,
        "depth_table": depth_table,
        "action_table": action_table,
        "depth_keys": depth_keys,
        "action_keys": action_keys,
        "folds": proxy_folds + model_folds,
        "feature_count": feature_count,
        "artifacts_status": artifacts.get("status", "missing"),
        "anchor_report": anchor_report,
    }


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
    base = bundle["base_test"]
    anchor = bundle["anchor"]
    depth_support, depth_target_support = lookup_support(bundle["test_df"], bundle["depth_table"], bundle["depth_keys"], arrays["candidate"])
    action_support, action_target_support = lookup_support(bundle["test_df"], bundle["action_table"], bundle["action_keys"], arrays["candidate"])
    return pd.DataFrame(
        {
            "candidate": spec.name,
            "row_id": idx,
            "rally_uid": anchor.iloc[idx]["rally_uid"].astype(int).to_numpy(),
            "actionId": anchor.iloc[idx]["actionId"].astype(int).to_numpy(),
            "old_pointId": base[idx],
            "new_pointId": pred[idx],
            "score": arrays["score"][idx],
            "model_margin": arrays["model_margin"][idx],
            "depth_margin": arrays["depth_margin"][idx],
            "action_margin": arrays["action_margin"][idx],
            "terminal_prob": arrays["terminal_prob"][idx],
            "depth_support": depth_support[idx],
            "depth_target_support": depth_target_support[idx],
            "action_support": action_support[idx],
            "action_target_support": action_target_support[idx],
            "prefix_len": bundle["test_df"].iloc[idx]["prefix_len"].astype(int).to_numpy(),
            "lag0_pointId": bundle["test_df"].iloc[idx]["lag0_pointId"].astype(int).to_numpy(),
            "lag0_actionId": bundle["test_df"].iloc[idx]["lag0_actionId"].astype(int).to_numpy(),
            "serverGetPoint": anchor.iloc[idx]["serverGetPoint"].to_numpy(dtype=float),
        }
    ).sort_values(["score", "model_margin"], ascending=[False, False]).reset_index(drop=True)


def evaluate_candidate(spec: CandidateSpec, bundle: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    oof_arrays = candidate_arrays(
        spec.family,
        bundle["base_oof"],
        bundle["model_oof_prob"],
        bundle["depth_oof_prob"],
        bundle["action_oof_prob"],
    )
    test_arrays = candidate_arrays(
        spec.family,
        bundle["base_test"],
        bundle["model_test_prob"],
        bundle["depth_test_prob"],
        bundle["action_test_prob"],
    )
    cap = spec.budget / len(bundle["base_test"])
    oof_budget = int(np.floor(len(bundle["base_oof"]) * cap))
    oof_selected = select_ranked_replacements(
        bundle["base_oof"],
        oof_arrays["candidate"],
        oof_arrays["score"],
        budget=oof_budget,
        weak_points=bundle["weak_points"],
        gate=oof_arrays["gate"],
    )
    test_selected = select_ranked_replacements(
        bundle["base_test"],
        test_arrays["candidate"],
        test_arrays["score"],
        budget=spec.budget,
        weak_points=bundle["weak_points"],
        gate=test_arrays["gate"],
    )
    oof_pred = apply_selection(bundle["base_oof"], oof_arrays["candidate"], oof_selected)
    test_pred = apply_selection(bundle["base_test"], test_arrays["candidate"], test_selected)
    counts = count_point0_changes(bundle["base_test"], test_pred)
    score = macro_f1(bundle["y"], oof_pred)
    delta = score - float(bundle["base_score"])
    changed = int(test_selected.sum())
    record: dict[str, Any] = {
        "candidate": spec.name,
        "family": spec.family,
        "budget": spec.budget,
        "oof_budget": oof_budget,
        "point_macro_f1": score,
        "base_point_macro_f1": bundle["base_score"],
        "local_delta_vs_anchor": delta,
        "test_changed_rows": changed,
        "oof_changed_rows": int(oof_selected.sum()),
        "test_churn": changed / len(bundle["base_test"]),
        "point0_additions": counts["point0_additions"],
        "point0_removals": counts["point0_removals"],
        "score_mean_changed": float(test_arrays["score"][test_selected].mean()) if changed else 0.0,
        "model_margin_mean_changed": float(test_arrays["model_margin"][test_selected].mean()) if changed else 0.0,
        "depth_margin_mean_changed": float(test_arrays["depth_margin"][test_selected].mean()) if changed else 0.0,
        "action_margin_mean_changed": float(test_arrays["action_margin"][test_selected].mean()) if changed else 0.0,
        "terminal_prob_mean_changed": float(test_arrays["terminal_prob"][test_selected].mean()) if changed else 0.0,
        "test_point_distribution": point_distribution(test_pred),
        "submission": spec.filename,
        "decision": "EXPORT_LOCAL" if delta > 0 and counts == {"point0_additions": 0, "point0_removals": 0} and changed <= MAX_EXPORT_ROWS and changed > 0 else "DO_NOT_EXPORT",
        "upload_worthy": False,
        "path": None,
    }
    record["upload_worthy"] = upload_worthy(record)
    path = write_submission_if_evidence(OUTDIR, spec, bundle["anchor"], test_pred, record)
    record["path"] = path
    changed_rows = changed_rows_frame(spec, test_selected, test_pred, test_arrays, bundle)
    return record, test_pred, changed_rows


def write_reports(records: list[dict[str, Any]], changed_rows: list[pd.DataFrame], bundle: dict[str, Any]) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records)
    summary = summary.sort_values(["local_delta_vs_anchor", "test_changed_rows"], ascending=[False, True]).reset_index(drop=True)
    summary.to_csv(OUTDIR / "v329_summary.csv", index=False)
    changed = pd.concat(changed_rows, ignore_index=True) if changed_rows else pd.DataFrame()
    changed.to_csv(OUTDIR / "v329_changed_rows.csv", index=False)
    exported = summary[summary["path"].notna()]
    upload_ready = summary[summary["upload_worthy"].astype(bool)]
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    report = {
        "version": "V329",
        "verdict": "HAS_UPLOAD_WORTHY_CANDIDATE" if not upload_ready.empty else "NO_UPLOAD_WORTHY_CANDIDATE",
        "upload_recommendation": "REVIEW_TOP_V329_EXPORT" if not upload_ready.empty else "DO_NOT_UPLOAD",
        "outdir": relative_path(OUTDIR),
        "policy": {
            "fixed_anchor": relative_path(ANCHOR_SUBMISSION),
            "fixed_action": "V173 from packaged anchor",
            "fixed_server": "V300 server from packaged anchor",
            "point_only": True,
            "no_external_match_inputs": True,
            "no_upload_directory_writes": True,
            "no_manual_row_edits": True,
        },
        "base_anchor": bundle["anchor_report"],
        "base_point_macro_f1": bundle["base_score"],
        "weak_points": bundle["weak_points"],
        "best_candidate": best,
        "top_candidates": summary.head(8).to_dict(orient="records"),
        "exported_submissions": exported[["candidate", "submission", "path", "local_delta_vs_anchor", "test_changed_rows"]].to_dict(orient="records"),
        "upload_worthy_candidates": upload_ready[["candidate", "submission", "path", "local_delta_vs_anchor", "test_changed_rows"]].to_dict(orient="records"),
        "artifacts_status": bundle["artifacts_status"],
        "feature_count": bundle["feature_count"],
        "depth_keys": bundle["depth_keys"],
        "action_keys": bundle["action_keys"],
        "folds": bundle["folds"],
        "notes": [
            "Depth/side and action-conditioned tables are fold-safe for OOF evaluation.",
            "All selectors are nonterminal point replacements; point0 additions/removals block export.",
            "Exports require positive local OOF delta and at most 24 changed test rows.",
            "Upload-worthy additionally requires delta >= 0.001 and at most 18 changed rows.",
        ],
    }
    (OUTDIR / "v329_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    bundle = build_bundle()
    records: list[dict[str, Any]] = []
    changed_rows: list[pd.DataFrame] = []
    for spec in CANDIDATES:
        record, _pred, rows = evaluate_candidate(spec, bundle)
        records.append(record)
        if not rows.empty:
            changed_rows.append(rows)
    return write_reports(records, changed_rows, bundle)


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": report["outdir"],
                "verdict": report["verdict"],
                "best": best.get("candidate"),
                "best_delta": best.get("local_delta_vs_anchor"),
                "best_rows": best.get("test_changed_rows"),
                "exports": [row["submission"] for row in report["exported_submissions"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
