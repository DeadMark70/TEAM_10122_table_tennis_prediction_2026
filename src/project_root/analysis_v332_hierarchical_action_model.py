"""V332 hierarchical action model.

V332 keeps the clean V306 point/server package fixed and tests conservative
action-only edits over a strict rebuilt V173 action anchor.  The action model
factorizes exact action probability as:

    P(actionId) = P(action family) * P(actionId | action family)

Exports are local-only and are written only when the strict evidence gate
passes.
"""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_v292_weak_class_pretraining_action_teacher import numeric_matrix  # noqa: E402
from baseline_lgbm import ACTION_CLASSES  # noqa: E402


OUTDIR = ROOT / "v332_hierarchical_action_model"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]

N_ACTIONS = 19
N_FAMILIES = 5
MIN_ACTION_OOF_DELTA = 0.0015
MIN_CHANGED_ROW_PRECISION = 0.45
MIN_TEST_CHANGED_ROWS = 5
MAX_TEST_CHANGED_ROWS = 80
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
WEAK_ACTIONS = np.array([0, 3, 5, 7, 8, 9, 12, 14], dtype=int)
PROTECTED_ANCHOR_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
LEAKY_FEATURE_PREFIXES = ("next_", "y_", "true_")
LEAKY_FEATURE_EXACT = {
    "actionId",
    "pointId",
    "serverGetPoint",
    "next_actionId",
    "next_pointId",
    "next_spinId",
    "next_strengthId",
    "next_strikeId",
    "rally_uid",
    "rally_id",
}

ACTION_FAMILIES: "OrderedDict[str, tuple[int, ...]]" = OrderedDict(
    [
        ("zero", (0,)),
        ("attack", (1, 2, 3, 4, 5, 6, 7)),
        ("control", (8, 9, 10, 11)),
        ("defensive", (12, 13, 14)),
        ("serve", (15, 16, 17, 18)),
    ]
)
FAMILY_TO_ID = {name: i for i, name in enumerate(ACTION_FAMILIES)}
ACTION_TO_FAMILY = {
    int(action): family_id
    for family_id, actions in enumerate(ACTION_FAMILIES.values())
    for action in actions
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    budget: int
    mode: str
    allowed_actions: tuple[int, ...] | None = None
    require_family_change: bool = False

    @property
    def filename(self) -> str:
        return f"submission_{self.name}__v306point_v300server.csv"


CANDIDATE_SPECS: "OrderedDict[str, CandidateSpec]" = OrderedDict(
    [
        ("v332_soft_route_b10", CandidateSpec("v332_soft_route_b10", 10, "soft_route")),
        ("v332_soft_route_b18", CandidateSpec("v332_soft_route_b18", 18, "soft_route")),
        (
            "v332_family_margin_b10",
            CandidateSpec("v332_family_margin_b10", 10, "family_margin", require_family_change=True),
        ),
        (
            "v332_weak_only_b10",
            CandidateSpec("v332_weak_only_b10", 10, "weak_only", tuple(int(x) for x in WEAK_ACTIONS)),
        ),
        (
            "v332_attack_control_b18",
            CandidateSpec("v332_attack_control_b18", 18, "attack_control", tuple(range(1, 12))),
        ),
    ]
)


def action_family_id(action_id: int) -> int:
    action = int(action_id)
    if action not in ACTION_TO_FAMILY:
        raise ValueError(f"unknown actionId {action}")
    return ACTION_TO_FAMILY[action]


def action_family_name(action_id: int) -> str:
    return list(ACTION_FAMILIES.keys())[action_family_id(action_id)]


def validate_action_hierarchy() -> None:
    seen: list[int] = []
    for actions in ACTION_FAMILIES.values():
        seen.extend(int(a) for a in actions)
    if sorted(seen) != list(range(N_ACTIONS)) or len(seen) != len(set(seen)):
        raise ValueError("action hierarchy must be complete and disjoint for actionId 0..18")


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("matrix must be a non-empty 2D array")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    sums = arr.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= 0.0
    if bad.any():
        arr[bad] = 1.0 / arr.shape[1]
        sums = arr.sum(axis=1, keepdims=True)
    return arr / sums


def predict_proba_aligned(model: Any, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=float)
    class_to_col = {int(cls): i for i, cls in enumerate(classes)}
    for j, cls in enumerate(model.classes_):
        cls_int = int(cls)
        if cls_int in class_to_col:
            out[:, class_to_col[cls_int]] = raw[:, j]
    return normalize_rows_safe(out)


def macro_f1(y: np.ndarray, pred: np.ndarray, labels: list[int] | np.ndarray = ACTION_CLASSES) -> float:
    return float(f1_score(np.asarray(y, dtype=int), np.asarray(pred, dtype=int), labels=list(labels), average="macro", zero_division=0))


def changed_row_precision(y_true: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(anchor, dtype=int)
    pred = np.asarray(candidate, dtype=int)
    if not (len(y) == len(base) == len(pred)):
        raise ValueError("y_true, anchor, and candidate must have matching lengths")
    changed = pred != base
    rows = int(changed.sum())
    correct = int(np.sum(changed & (pred == y)))
    return {
        "changed_rows": rows,
        "changed_correct": correct,
        "changed_precision": float(correct / rows) if rows else 0.0,
    }


def action_distribution(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=int)
    if len(arr) == 0:
        return "{}"
    unique, counts = np.unique(arr, return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(unique, counts)}, sort_keys=True)


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 3:
        return 1
    if 4 <= point <= 6:
        return 2
    if 7 <= point <= 9:
        return 3
    return 0


def point_side(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 9:
        return ((point - 1) % 3) + 1
    return 0


def phase_id(prefix_len: int) -> int:
    value = int(prefix_len)
    if value <= 1:
        return 0
    if value == 2:
        return 1
    if value == 3:
        return 2
    return 3


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown_table(rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str]) -> str:
    records = rows[columns].to_dict(orient="records") if isinstance(rows, pd.DataFrame) else rows

    def cell(value: Any) -> str:
        if isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        return text.replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in records]
    return "\n".join([header, sep, *body])


def protected_output_path(outdir: Path, filename: str) -> Path:
    root = Path(outdir)
    path = root / filename
    lower_parts = [part.lower() for part in path.parts]
    blocked_tokens = ("ttmatch", "old_server", "oldserver", "upload_candidates", "selected", "submissions")
    if any(any(token in part for token in blocked_tokens) for part in lower_parts):
        raise ValueError(f"refusing blocked V332 path: {path}")
    if path.parent != root:
        raise ValueError(f"V332 exports must stay directly under {root}: {path}")
    return path


def load_package_anchor() -> pd.DataFrame:
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing V306 anchor submission: {ANCHOR_SUBMISSION}")
    sub = pd.read_csv(ANCHOR_SUBMISSION)
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{ANCHOR_SUBMISSION} columns {list(sub.columns)} != {EXPECTED_COLUMNS}")
    return sub


def rebuild_strict_v173_anchor() -> dict[str, Any]:
    from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

    state = rebuild_v173_best_actions()
    required = ["rows", "test_rows", "rally_uids", "v173_pred_oof", "v173_pred_test"]
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"rebuilt V173 state missing keys: {missing}")
    return state


def load_strict_anchor_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    validate_action_hierarchy()
    state = rebuild_strict_v173_anchor()
    rows = state["rows"].reset_index(drop=True).copy()
    test_rows = state["test_rows"].reset_index(drop=True).copy()
    y = rows["next_actionId"].astype(int).to_numpy()
    anchor_oof = np.asarray(state["v173_pred_oof"], dtype=int)
    anchor_test = np.asarray(state["v173_pred_test"], dtype=int)
    rally_uids = np.asarray(state["rally_uids"], dtype=int)
    anchor_sub = load_package_anchor()
    if len(rows) != len(y) or len(anchor_oof) != len(y):
        raise ValueError("rebuilt V173 OOF rows, labels, and predictions are not aligned")
    if len(test_rows) != len(anchor_test) or len(anchor_sub) != len(anchor_test):
        raise ValueError("rebuilt V173 test predictions and packaged anchor are not aligned by length")
    if not np.array_equal(anchor_sub["rally_uid"].astype(int).to_numpy(), rally_uids):
        raise ValueError("packaged V306 rally_uid order differs from rebuilt V173 test rally_uids")
    packaged_action = anchor_sub["actionId"].astype(int).to_numpy()
    if not np.array_equal(packaged_action, anchor_test):
        mismatch = int(np.sum(packaged_action != anchor_test))
        raise ValueError(f"packaged V306 actionId does not equal rebuilt strict V173 test prediction ({mismatch} mismatches)")
    rows["anchor_action"] = anchor_oof
    test_rows["anchor_action"] = anchor_test
    return rows, test_rows, y, anchor_oof, anchor_sub, state


def build_feature_frame(rows: pd.DataFrame, *, y: np.ndarray | None = None) -> pd.DataFrame:
    out = rows.copy()
    for col in [
        "prefix_len",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_positionId",
        "scoreSelf",
        "scoreOther",
        "scoreTotal",
        "serverScoreDiff",
        "serverScore",
        "receiverScore",
    ]:
        if col not in out:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["phase"] = out["prefix_len"].astype(int).map(phase_id)
    out["lag0_action_family"] = out["lag0_actionId"].astype(int).map(lambda v: action_family_id(int(np.clip(v, 0, 18))))
    out["lag0_point_depth"] = out["lag0_pointId"].astype(int).map(point_depth)
    out["lag0_point_side"] = out["lag0_pointId"].astype(int).map(point_side)
    out["is_receive"] = out["phase"].eq(0).astype(int)
    out["is_third"] = out["phase"].eq(1).astype(int)
    out["is_rally"] = out["phase"].ge(3).astype(int)
    out["score_abs_diff"] = out["serverScoreDiff"].abs()
    out["is_deuce_like"] = ((out["scoreTotal"] >= 18) & (out["score_abs_diff"] <= 1)).astype(int)
    out["incoming_action_point"] = out["lag0_actionId"].astype(int).astype(str) + "_" + out["lag0_pointId"].astype(int).astype(str)
    out["incoming_spin_strength"] = out["lag0_spinId"].astype(int).astype(str) + "_" + out["lag0_strengthId"].astype(int).astype(str)
    if "anchor_action" in out:
        out["anchor_action_family"] = out["anchor_action"].astype(int).map(action_family_id)
    else:
        out["anchor_action"] = 0
        out["anchor_action_family"] = 0
    return out.reset_index(drop=True)


def drop_leaky_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    for col in frame.columns:
        name = str(col)
        if name in LEAKY_FEATURE_EXACT:
            continue
        if any(name.startswith(prefix) for prefix in LEAKY_FEATURE_PREFIXES):
            continue
        keep.append(name)
    return frame.loc[:, keep].copy()


def build_feature_matrices(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = build_feature_frame(rows, y=y)
    test_frame = build_feature_frame(test_rows)
    train_frame = drop_leaky_feature_columns(train_frame)
    test_frame = drop_leaky_feature_columns(test_frame)
    x_train, x_test = numeric_matrix(train_frame, test_frame)
    leaked = [
        col
        for col in x_train.columns
        if col in LEAKY_FEATURE_EXACT or any(str(col).startswith(prefix) for prefix in LEAKY_FEATURE_PREFIXES)
    ]
    if leaked:
        raise ValueError(f"V332 feature matrix contains leaky columns: {leaked[:10]}")
    return x_train, x_test


def fold_splits(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    if "fold" in rows:
        folds = pd.to_numeric(rows["fold"], errors="coerce").fillna(-1).astype(int).to_numpy()
        uniq = sorted([int(f) for f in np.unique(folds) if int(f) >= 0])
        pairs = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in uniq]
        pairs = [(tr, va) for tr, va in pairs if len(tr) and len(va)]
        if len(pairs) >= 2:
            return pairs
    if "match" in rows and rows["match"].nunique(dropna=False) >= 2:
        from sklearn.model_selection import GroupKFold

        groups = rows["match"].astype(str).to_numpy()
        n_splits = min(5, int(rows["match"].nunique(dropna=False)))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(rows)), groups=groups))
    idx = np.arange(len(rows))
    return [(idx[idx % 5 != fold], idx[idx % 5 == fold]) for fold in range(5)]


def fit_family_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def fit_expert_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def compose_action_probabilities(family_prob: np.ndarray, expert_probs: dict[int, np.ndarray]) -> np.ndarray:
    fam = normalize_rows_safe(family_prob)
    if fam.shape[1] != N_FAMILIES:
        raise ValueError(f"family_prob must have {N_FAMILIES} columns")
    out = np.zeros((len(fam), N_ACTIONS), dtype=float)
    for family_id, actions in enumerate(ACTION_FAMILIES.values()):
        action_list = list(actions)
        if family_id == 0:
            out[:, action_list[0]] = fam[:, family_id]
            continue
        expert = normalize_rows_safe(expert_probs[family_id])
        if expert.shape != (len(fam), len(action_list)):
            raise ValueError(f"expert family {family_id} has shape {expert.shape}, expected {(len(fam), len(action_list))}")
        out[:, action_list] = fam[:, [family_id]] * expert
    return normalize_rows_safe(out)


def _family_prior(y_family: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(y_family, dtype=int), minlength=N_FAMILIES).astype(float) + 1.0
    return counts / counts.sum()


def _expert_prior(y: np.ndarray, actions: tuple[int, ...]) -> np.ndarray:
    counts = np.array([np.sum(np.asarray(y, dtype=int) == int(action)) for action in actions], dtype=float) + 1.0
    return counts / counts.sum()


def train_hierarchical_probabilities(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = np.asarray(y, dtype=int)
    y_family = np.asarray([action_family_id(v) for v in y], dtype=int)
    family_oof = np.zeros((len(x_train), N_FAMILIES), dtype=float)
    family_test_sum = np.zeros((len(x_test), N_FAMILIES), dtype=float)
    expert_oof = {fid: np.zeros((len(x_train), len(actions)), dtype=float) for fid, actions in enumerate(ACTION_FAMILIES.values()) if fid}
    expert_test_sum = {fid: np.zeros((len(x_test), len(actions)), dtype=float) for fid, actions in enumerate(ACTION_FAMILIES.values()) if fid}
    folds = fold_splits(rows)
    fold_reports: list[dict[str, Any]] = []
    fitted_folds = 0
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        if len(np.unique(y_family[train_idx])) >= 2:
            family_model = fit_family_model(33200 + fold_id)
            family_model.fit(x_train.iloc[train_idx], y_family[train_idx])
            family_oof[valid_idx] = predict_proba_aligned(family_model, x_train.iloc[valid_idx], list(range(N_FAMILIES)))
            family_test_sum += predict_proba_aligned(family_model, x_test, list(range(N_FAMILIES)))
        else:
            prior = _family_prior(y_family[train_idx])
            family_oof[valid_idx] = prior
            family_test_sum += np.tile(prior, (len(x_test), 1))

        expert_reports: list[dict[str, Any]] = []
        for family_id, actions in enumerate(ACTION_FAMILIES.values()):
            if family_id == 0:
                continue
            action_tuple = tuple(int(a) for a in actions)
            family_train_idx = train_idx[np.isin(y[train_idx], action_tuple)]
            local_y = y[family_train_idx]
            if len(family_train_idx) and len(np.unique(local_y)) >= 2:
                model = fit_expert_model(33300 + family_id * 31 + fold_id)
                model.fit(x_train.iloc[family_train_idx], local_y)
                expert_oof[family_id][valid_idx] = predict_proba_aligned(model, x_train.iloc[valid_idx], list(action_tuple))
                expert_test_sum[family_id] += predict_proba_aligned(model, x_test, list(action_tuple))
                fitted = True
            else:
                prior = _expert_prior(y[train_idx], action_tuple)
                expert_oof[family_id][valid_idx] = prior
                expert_test_sum[family_id] += np.tile(prior, (len(x_test), 1))
                fitted = False
            expert_reports.append(
                {
                    "family": list(ACTION_FAMILIES.keys())[family_id],
                    "actions": action_tuple,
                    "train_rows": int(len(family_train_idx)),
                    "fitted": bool(fitted),
                }
            )
        fitted_folds += 1
        fold_reports.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "experts": expert_reports,
            }
        )
    if fitted_folds == 0:
        raise RuntimeError("no fold-safe hierarchical models were fitted")
    family_test = family_test_sum / float(fitted_folds)
    expert_test = {fid: arr / float(fitted_folds) for fid, arr in expert_test_sum.items()}
    return compose_action_probabilities(family_oof, expert_oof), compose_action_probabilities(family_test, expert_test), fold_reports


def _target_and_margin(prob: np.ndarray, anchor: np.ndarray, spec: CandidateSpec) -> tuple[np.ndarray, np.ndarray]:
    p = normalize_rows_safe(prob)
    base = np.asarray(anchor, dtype=int)
    scoped = p.copy()
    scoped[:, SERVE_ACTIONS] = 0.0
    if spec.allowed_actions is not None:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[list(spec.allowed_actions)] = True
        scoped[:, ~mask] = 0.0
    target = scoped.argmax(axis=1).astype(int)
    rows = np.arange(len(base))
    anchor_prob = p[rows, np.clip(base, 0, N_ACTIONS - 1)]
    target_prob = scoped[rows, target]
    margin = target_prob - anchor_prob
    if spec.mode == "family_margin":
        base_family = np.asarray([action_family_id(v) for v in base], dtype=int)
        target_family = np.asarray([action_family_id(v) for v in target], dtype=int)
        margin = margin + 0.25 * (p[rows, target] - anchor_prob)
        margin[target_family == base_family] = -np.inf
    return target, margin


def select_candidate_rows(
    anchor: np.ndarray,
    target: np.ndarray,
    margin: np.ndarray,
    spec: CandidateSpec,
) -> np.ndarray:
    base = np.asarray(anchor, dtype=int)
    cand = np.asarray(target, dtype=int)
    score = np.asarray(margin, dtype=float)
    eligible = (
        (cand != base)
        & np.isfinite(score)
        & (score > 0.0)
        & ~np.isin(base, PROTECTED_ANCHOR_ACTIONS)
        & ~np.isin(cand, SERVE_ACTIONS)
    )
    if spec.require_family_change:
        base_family = np.asarray([action_family_id(v) for v in base], dtype=int)
        cand_family = np.asarray([action_family_id(v) for v in cand], dtype=int)
        eligible &= cand_family != base_family
    selected = np.zeros(len(base), dtype=bool)
    if spec.budget <= 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    selected[order[: min(int(spec.budget), len(order))]] = True
    return selected


def apply_selected(anchor: np.ndarray, target: np.ndarray, selected: np.ndarray) -> np.ndarray:
    out = np.asarray(anchor, dtype=int).copy()
    mask = np.asarray(selected, dtype=bool)
    out[mask] = np.asarray(target, dtype=int)[mask]
    return out


def evidence_passes(row: dict[str, Any] | pd.Series) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    return bool(
        float(data.get("action_oof_delta", 0.0)) >= MIN_ACTION_OOF_DELTA
        and float(data.get("changed_row_oof_precision", 0.0)) >= MIN_CHANGED_ROW_PRECISION
        and MIN_TEST_CHANGED_ROWS <= int(data.get("changed_action_rows", 0)) <= MAX_TEST_CHANGED_ROWS
        and int(data.get("serve_action_rows", 0)) <= 0
        and int(data.get("serve_count_delta", 0)) <= 0
    )


def evaluate_candidate(
    spec: CandidateSpec,
    action_prob_oof: np.ndarray,
    action_prob_test: np.ndarray,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    oof_target, oof_margin = _target_and_margin(action_prob_oof, anchor_oof, spec)
    test_target, test_margin = _target_and_margin(action_prob_test, anchor_test, spec)
    scale = len(anchor_oof) / max(len(anchor_test), 1)
    oof_spec = CandidateSpec(spec.name, max(1, int(math.floor(spec.budget * scale))), spec.mode, spec.allowed_actions, spec.require_family_change)
    oof_selected = select_candidate_rows(anchor_oof, oof_target, oof_margin, oof_spec)
    test_selected = select_candidate_rows(anchor_test, test_target, test_margin, spec)
    pred_oof = apply_selected(anchor_oof, oof_target, oof_selected)
    pred_test = apply_selected(anchor_test, test_target, test_selected)
    base_score = macro_f1(y, anchor_oof)
    score = macro_f1(y, pred_oof)
    precision = changed_row_precision(y, anchor_oof, pred_oof)
    changed_actions = pred_test[pred_test != anchor_test]
    serve_rows = int(np.isin(changed_actions, SERVE_ACTIONS).sum())
    serve_count_delta = int(np.isin(pred_test, SERVE_ACTIONS).sum() - np.isin(anchor_test, SERVE_ACTIONS).sum())
    rec = {
        "candidate": spec.name,
        "candidate_file": spec.filename,
        "mode": spec.mode,
        "test_budget": int(spec.budget),
        "oof_budget": int(oof_spec.budget),
        "allowed_actions": "" if spec.allowed_actions is None else "/".join(str(a) for a in spec.allowed_actions),
        "action_macro_f1": float(score),
        "anchor_action_macro_f1": float(base_score),
        "action_oof_delta": float(score - base_score),
        "changed_action_rows": int(np.sum(pred_test != anchor_test)),
        "oof_changed_rows": int(precision["changed_rows"]),
        "changed_correct": int(precision["changed_correct"]),
        "changed_row_oof_precision": float(precision["changed_precision"]),
        "serve_action_rows": int(serve_rows),
        "serve_count_delta": int(serve_count_delta),
        "test_changed_distribution": action_distribution(changed_actions),
        "test_action_distribution": action_distribution(pred_test),
        "min_test_margin_changed": float(test_margin[test_selected].min()) if test_selected.any() else 0.0,
        "mean_test_margin_changed": float(test_margin[test_selected].mean()) if test_selected.any() else 0.0,
        "evidence_pass": 0,
        "decision": "DO_NOT_UPLOAD",
    }
    rec["evidence_pass"] = int(evidence_passes(rec))
    rec["decision"] = "REVIEW_ACTION" if rec["evidence_pass"] else "DO_NOT_UPLOAD"
    return rec, pred_oof, pred_test


def build_search(
    action_prob_oof: np.ndarray,
    action_prob_test: np.ndarray,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for spec in CANDIDATE_SPECS.values():
        rec, pred_oof, pred_test = evaluate_candidate(spec, action_prob_oof, action_prob_test, y, anchor_oof, anchor_test)
        records.append(rec)
        oof_predictions[spec.name] = pred_oof
        test_predictions[spec.name] = pred_test
    search = pd.DataFrame(records)
    if search.empty:
        return search, oof_predictions, test_predictions
    search = search.sort_values(
        ["evidence_pass", "action_oof_delta", "changed_row_oof_precision", "changed_action_rows"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return search, oof_predictions, test_predictions


def build_class_report(y: np.ndarray, anchor_oof: np.ndarray, best_oof: np.ndarray) -> pd.DataFrame:
    rows = []
    for action in ACTION_CLASSES:
        anchor_f1 = macro_f1(y, anchor_oof, [int(action)])
        best_f1 = macro_f1(y, best_oof, [int(action)])
        rows.append(
            {
                "action": int(action),
                "family": action_family_name(int(action)),
                "anchor_f1": float(anchor_f1),
                "v332_best_f1": float(best_f1),
                "delta": float(best_f1 - anchor_f1),
                "support": int(np.sum(np.asarray(y, dtype=int) == int(action))),
            }
        )
    return pd.DataFrame(rows)


def build_export_frame(anchor_sub: pd.DataFrame, action: np.ndarray) -> pd.DataFrame:
    pred = np.asarray(action, dtype=int)
    if len(anchor_sub) != len(pred):
        raise ValueError(f"action rows {len(pred)} != anchor submission rows {len(anchor_sub)}")
    return pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": pred,
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )[EXPECTED_COLUMNS]


def export_submissions(search: pd.DataFrame, test_predictions: dict[str, np.ndarray], anchor_sub: pd.DataFrame) -> list[str]:
    generated: list[str] = []
    if search.empty:
        return generated
    for _, row in search[search["evidence_pass"].astype(int).eq(1)].iterrows():
        spec = CANDIDATE_SPECS[str(row["candidate"])]
        out = build_export_frame(anchor_sub, test_predictions[spec.name])
        path = protected_output_path(OUTDIR, spec.filename)
        out.to_csv(path, index=False, float_format="%.8f")
        generated.append(str(path.relative_to(ROOT)))
    return generated


def write_failure_report(reason: str) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "V332",
        "decision": "ANCHOR_REBUILD_FAILED",
        "error": reason,
        "generated_submissions": [],
        "generated_submission_count": 0,
        "copied_to_upload_or_selected": False,
        "ttmatch_used": False,
        "old_server_used": False,
    }
    pd.DataFrame([{"candidate": "ANCHOR_REBUILD_FAILED", "decision": "NO_EXPORT", "error": reason}]).to_csv(
        OUTDIR / "v332_action_search.csv", index=False
    )
    (OUTDIR / "v332_report.json").write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return report


def write_reports(
    search: pd.DataFrame,
    class_report: pd.DataFrame,
    fold_reports: list[dict[str, Any]],
    generated: list[str],
    anchor_sub: pd.DataFrame,
    state: dict[str, Any],
) -> dict[str, Any]:
    search.to_csv(OUTDIR / "v332_action_search.csv", index=False)
    class_report.to_csv(OUTDIR / "v332_class_report.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    passing = search[search["evidence_pass"].astype(int).eq(1)].copy() if len(search) else pd.DataFrame()
    decision = "REVIEW_ACTION" if not passing.empty else "DO_NOT_UPLOAD"
    report = json_safe(
        {
            "version": "V332",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "action_anchor": "strict rebuilt V173 via analysis_r184_receiver_affordance_refiner.rebuild_v173_best_actions",
            "v173_best_candidate": state.get("best_candidate", ""),
            "v173_schedule": state.get("schedule", ""),
            "v173_alpha": state.get("alpha", None),
            "point_fixed_to": "V306 p0 cap0p01 pointId copied from package anchor",
            "server_fixed_to": "V300 serverGetPoint copied from package anchor",
            "copied_to_upload_or_selected": False,
            "ttmatch_used": False,
            "old_server_used": False,
            "candidate_specs": [spec.__dict__ | {"filename": spec.filename} for spec in CANDIDATE_SPECS.values()],
            "action_hierarchy": ACTION_FAMILIES,
            "evidence_thresholds": {
                "min_action_oof_delta": MIN_ACTION_OOF_DELTA,
                "min_changed_row_oof_precision": MIN_CHANGED_ROW_PRECISION,
                "min_test_changed_rows": MIN_TEST_CHANGED_ROWS,
                "max_test_changed_rows": MAX_TEST_CHANGED_ROWS,
                "max_serve_action_rows": 0,
                "max_serve_count_delta": 0,
            },
            "fold_reports": fold_reports,
            "best_candidate": best,
            "reviewable_candidates": passing.to_dict(orient="records") if not passing.empty else [],
            "decision": decision,
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "anchor_rows": int(len(anchor_sub)),
        }
    )
    (OUTDIR / "v332_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    top = search.head(12)
    md = [
        "# V332 hierarchical action model",
        "",
        f"Anchor submission: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Action anchor: strict rebuilt V173; point/server copied from V306/V300 package anchor.",
        f"Decision: `{decision}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate_file', '')}`",
        f"OOF action delta: {float(best.get('action_oof_delta', 0.0)):.6f}",
        f"Changed test rows: {int(best.get('changed_action_rows', 0))}",
        f"Changed-row OOF precision: {float(best.get('changed_row_oof_precision', 0.0)):.4f}",
        f"Evidence pass: `{bool(best.get('evidence_pass', 0))}`",
        "",
        "## Top search rows",
        "",
        markdown_table(
            top,
            [
                "candidate_file",
                "mode",
                "action_oof_delta",
                "changed_action_rows",
                "changed_row_oof_precision",
                "serve_action_rows",
                "serve_count_delta",
                "decision",
            ],
        ),
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v332_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v332*.csv"):
        stale.unlink()
    try:
        rows, test_rows, y, anchor_oof, anchor_sub, state = load_strict_anchor_frames()
    except Exception as exc:
        write_failure_report(str(exc))
        raise
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    x_train, x_test = build_feature_matrices(rows, test_rows, y)
    action_prob_oof, action_prob_test, fold_reports = train_hierarchical_probabilities(x_train, x_test, rows, y)
    search, oof_predictions, test_predictions = build_search(action_prob_oof, action_prob_test, y, anchor_oof, anchor_test)
    best_key = str(search.iloc[0]["candidate"]) if len(search) else ""
    best_oof = oof_predictions.get(best_key, anchor_oof)
    class_report = build_class_report(y, anchor_oof, best_oof)
    generated = export_submissions(search, test_predictions, anchor_sub)
    return write_reports(search, class_report, fold_reports, generated, anchor_sub, state)


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "decision": report.get("decision", "DO_NOT_UPLOAD"),
                "best_candidate": best.get("candidate_file", ""),
                "best_action_oof_delta": best.get("action_oof_delta", 0.0),
                "best_changed_action_rows": best.get("changed_action_rows", 0),
                "best_changed_row_oof_precision": best.get("changed_row_oof_precision", 0.0),
                "generated_submission_count": report.get("generated_submission_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
