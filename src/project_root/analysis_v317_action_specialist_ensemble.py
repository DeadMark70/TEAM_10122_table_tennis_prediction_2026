"""V317 action specialist ensemble.

V317 revisits action-only edits over the clean V173 action anchor while keeping
the point/server columns fixed to the V306 p0 cap0p01 + V300 submission. It is
intentionally stricter than V312: local CSVs are emitted only when fold-safe OOF
evidence clears the action delta, changed-row precision, and serve-action gates.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_lgbm import ACTION_CLASSES  # noqa: E402
from analysis_v292_weak_class_pretraining_action_teacher import numeric_matrix  # noqa: E402


OUTDIR = ROOT / "v317_action_specialist_ensemble"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V286_OOF = ROOT / "v286_weak_action_specialist_pretraining" / "v286_specialist_oof.csv"
V312_SEARCH = ROOT / "v312_action_weak_complementarity" / "v312_action_search.csv"

N_ACTIONS = 19
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
PROTECTED_ANCHOR_ACTIONS = np.array([1, 10, 13, 15, 16, 17, 18], dtype=int)
MIN_ACTION_OOF_DELTA = 0.0015
MIN_CHANGED_ROW_PRECISION = 0.30
MAX_SERVE_ACTION_ROWS = 0

DEFAULT_SPECIALIST_GROUPS: "OrderedDict[str, tuple[int, ...]]" = OrderedDict(
    [
        ("zero_terminal", (0,)),
        ("attack_finish_control", (3, 4, 5, 7)),
        ("rare_control_defense", (8, 9, 12, 14)),
    ]
)
FOCUS_ACTIONS = np.array(sorted({a for actions in DEFAULT_SPECIALIST_GROUPS.values() for a in actions}), dtype=int)


@dataclass(frozen=True)
class ExportSpec:
    filename: str
    group: str
    budget: int


CANDIDATE_SPECS: "OrderedDict[str, ExportSpec]" = OrderedDict(
    [
        (
            "submission_v317_zero_terminal_action_budget10__v306point_v300server.csv",
            ExportSpec("submission_v317_zero_terminal_action_budget10__v306point_v300server.csv", "zero_terminal", 10),
        ),
        (
            "submission_v317_attack_finish_action_budget20__v306point_v300server.csv",
            ExportSpec(
                "submission_v317_attack_finish_action_budget20__v306point_v300server.csv",
                "attack_finish_control",
                20,
            ),
        ),
        (
            "submission_v317_raredefense_action_budget15__v306point_v300server.csv",
            ExportSpec(
                "submission_v317_raredefense_action_budget15__v306point_v300server.csv",
                "rare_control_defense",
                15,
            ),
        ),
        (
            "submission_v317_specialist_union_safe__v306point_v300server.csv",
            ExportSpec("submission_v317_specialist_union_safe__v306point_v300server.csv", "union_safe", 45),
        ),
    ]
)


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


def macro_f1(y: np.ndarray, pred: np.ndarray, labels: list[int] | np.ndarray = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def class_f1(y: np.ndarray, pred: np.ndarray, action: int) -> float:
    return macro_f1(y, pred, [int(action)])


def action_distribution(values: np.ndarray) -> str:
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(unique, counts)})


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


def evidence_passes(row: dict[str, Any] | pd.Series) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    return bool(
        float(data.get("action_oof_delta", 0.0)) >= MIN_ACTION_OOF_DELTA
        and float(data.get("changed_row_oof_precision", 0.0)) >= MIN_CHANGED_ROW_PRECISION
        and int(data.get("changed_action_rows", 0)) > 0
        and int(data.get("serve_action_rows", 0)) <= MAX_SERVE_ACTION_ROWS
    )


def protected_output_path(outdir: Path, spec: ExportSpec) -> Path:
    root = Path(outdir)
    path = root / spec.filename
    parts = {part.lower() for part in path.parts}
    if any("upload_candidates" in part for part in parts) or "selected" in parts or "submissions" in parts:
        raise ValueError(f"refusing non-local V317 export path: {path}")
    if path.parent != root:
        raise ValueError(f"V317 exports must stay directly under {root}: {path}")
    return path


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
    )


def load_anchor_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF: {V286_OOF}")
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing V306/V300 anchor submission: {ANCHOR_SUBMISSION}")
    from analysis_v290_shortcontrol411_specialist import load_anchor_frames as load_v290_anchor_frames

    rows, test_rows, rebuilt_y, rebuilt_anchor = load_v290_anchor_frames()
    oof = pd.read_csv(V286_OOF)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    rows = rows.reset_index(drop=True).copy()
    test_rows = test_rows.reset_index(drop=True).copy()
    if len(oof) != len(rows):
        raise ValueError(f"V286 OOF length {len(oof)} != rebuilt train rows {len(rows)}")
    if len(anchor_sub) != len(test_rows):
        raise ValueError(f"anchor submission rows {len(anchor_sub)} != test rows {len(test_rows)}")
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    if len(rebuilt_y) == len(y) and not np.array_equal(np.asarray(rebuilt_y, dtype=int), y):
        raise ValueError("rebuilt action labels differ from V286 OOF labels")
    if len(rebuilt_anchor) == len(anchor_oof) and not np.array_equal(np.asarray(rebuilt_anchor, dtype=int), anchor_oof):
        raise ValueError("rebuilt V173 anchor differs from V286 OOF anchor")
    rows["fold"] = oof.get("fold", pd.Series(np.arange(len(rows)) % 5)).astype(int).to_numpy()
    test_rows["anchor_action"] = anchor_sub["actionId"].astype(int).to_numpy()
    return rows, test_rows, y, anchor_oof, anchor_sub


def build_feature_matrices(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    from analysis_v291_weak_class_training_upgrade import build_complete_feature_frame

    oof = pd.read_csv(V286_OOF)
    train_frame = build_complete_feature_frame(rows, oof, y=y).reset_index(drop=True)
    test_frame = build_complete_feature_frame(test_rows, None, y=y, support_rows=rows).reset_index(drop=True)
    x_train, x_test = numeric_matrix(train_frame, test_frame)
    return x_train, x_test


def fold_splits(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = rows["fold"].astype(int).to_numpy() if "fold" in rows else np.arange(len(rows)) % 5
    return [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(np.unique(folds))]


def fit_action_ovr_source(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    oof = np.zeros((len(x_train), N_ACTIONS), dtype=float)
    test = np.zeros((len(x_test), N_ACTIONS), dtype=float)
    metrics: list[dict[str, Any]] = []
    fitted_total = 0
    for action in FOCUS_ACTIONS.tolist():
        target = (np.asarray(y, dtype=int) == int(action)).astype(int)
        test_sum = np.zeros(len(x_test), dtype=float)
        fitted = 0
        for fold_id, (train_idx, valid_idx) in enumerate(fold_splits(rows)):
            if len(np.unique(target[train_idx])) < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    C=0.25,
                    random_state=31700 + int(action) * 11 + fold_id,
                ),
            )
            model.fit(x_train.iloc[train_idx], target[train_idx])
            oof[valid_idx, int(action)] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
            test_sum += model.predict_proba(x_test)[:, 1]
            fitted += 1
        if fitted:
            test[:, int(action)] = test_sum / float(fitted)
        else:
            base = float(target.mean()) if len(target) else 0.0
            oof[:, int(action)] = base
            test[:, int(action)] = base
        fitted_total += fitted
        metrics.append(
            {
                "action": int(action),
                "positive_rows": int(target.sum()),
                "fitted_folds": int(fitted),
                "oof_mean": float(oof[:, int(action)].mean()),
                "test_mean": float(test[:, int(action)].mean()),
            }
        )
    return {
        "name": "ovr_logreg",
        "family": "fold_safe_one_vs_rest_logreg",
        "oof_score": np.clip(oof, 0.0, 1.0),
        "test_score": np.clip(test, 0.0, 1.0),
        "fitted_folds": int(fitted_total),
        "metrics": metrics,
    }


def fit_action_tree_source(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    oof = np.zeros((len(x_train), N_ACTIONS), dtype=float)
    test = np.zeros((len(x_test), N_ACTIONS), dtype=float)
    metrics: list[dict[str, Any]] = []
    fitted_total = 0
    for action in FOCUS_ACTIONS.tolist():
        target = (np.asarray(y, dtype=int) == int(action)).astype(int)
        test_sum = np.zeros(len(x_test), dtype=float)
        fitted = 0
        for fold_id, (train_idx, valid_idx) in enumerate(fold_splits(rows)):
            if len(np.unique(target[train_idx])) < 2:
                continue
            model = ExtraTreesClassifier(
                n_estimators=140,
                min_samples_leaf=9,
                max_features="sqrt",
                class_weight="balanced",
                random_state=31780 + int(action) * 13 + fold_id,
                n_jobs=1,
            )
            model.fit(x_train.iloc[train_idx], target[train_idx])
            oof[valid_idx, int(action)] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
            test_sum += model.predict_proba(x_test)[:, 1]
            fitted += 1
        if fitted:
            test[:, int(action)] = test_sum / float(fitted)
        else:
            base = float(target.mean()) if len(target) else 0.0
            oof[:, int(action)] = base
            test[:, int(action)] = base
        fitted_total += fitted
        metrics.append(
            {
                "action": int(action),
                "positive_rows": int(target.sum()),
                "fitted_folds": int(fitted),
                "oof_mean": float(oof[:, int(action)].mean()),
                "test_mean": float(test[:, int(action)].mean()),
            }
        )
    return {
        "name": "ovr_extratrees",
        "family": "fold_safe_one_vs_rest_extratrees",
        "oof_score": np.clip(oof, 0.0, 1.0),
        "test_score": np.clip(test, 0.0, 1.0),
        "fitted_folds": int(fitted_total),
        "metrics": metrics,
    }


def v286_table_source(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    oof_file = pd.read_csv(V286_OOF)
    oof = np.zeros((len(rows), N_ACTIONS), dtype=float)
    for action in FOCUS_ACTIONS.tolist():
        col = f"specialist_p_{int(action)}"
        if col in oof_file:
            oof[:, int(action)] = pd.to_numeric(oof_file[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    test = support_table_scores(rows, test_rows, y)
    return {
        "name": "v286_support_table",
        "family": "existing_prediction_and_support_table",
        "oof_score": np.clip(oof, 0.0, 1.0),
        "test_score": np.clip(test, 0.0, 1.0),
        "fitted_folds": 0,
        "metrics": [],
    }


def support_table_scores(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    key_cols = [col for col in ["lag0_actionId", "lag0_pointId", "serverScoreDiff"] if col in rows and col in test_rows]
    if not key_cols:
        return np.zeros((len(test_rows), N_ACTIONS), dtype=float)
    train = rows[key_cols].copy()
    train["y"] = np.asarray(y, dtype=int)
    scores = np.zeros((len(test_rows), N_ACTIONS), dtype=float)
    global_prior = np.bincount(np.asarray(y, dtype=int), minlength=N_ACTIONS).astype(float) + 1.0
    global_prior = global_prior / global_prior.sum()
    for action in FOCUS_ACTIONS.tolist():
        grouped = train.assign(hit=(train["y"].astype(int) == int(action)).astype(float)).groupby(key_cols, dropna=False)["hit"].agg(
            ["mean", "count"]
        )
        lookup = grouped.to_dict(orient="index")
        for i, row in test_rows[key_cols].iterrows():
            key = tuple(row[col] for col in key_cols)
            entry = lookup.get(key)
            if entry is None:
                scores[int(i), int(action)] = global_prior[int(action)]
            else:
                support = float(entry["count"])
                shrink = support / (support + 25.0)
                scores[int(i), int(action)] = shrink * float(entry["mean"]) + (1.0 - shrink) * global_prior[int(action)]
    return scores


def blend_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    weights = {"ovr_logreg": 0.40, "ovr_extratrees": 0.40, "v286_support_table": 0.20}
    oof = sum(float(weights.get(src["name"], 1.0)) * np.asarray(src["oof_score"], dtype=float) for src in sources)
    test = sum(float(weights.get(src["name"], 1.0)) * np.asarray(src["test_score"], dtype=float) for src in sources)
    denom = sum(float(weights.get(src["name"], 1.0)) for src in sources)
    return {
        "name": "specialist_blend",
        "family": "weighted_specialist_signal_blend",
        "oof_score": np.clip(oof / denom, 0.0, 1.0),
        "test_score": np.clip(test / denom, 0.0, 1.0),
        "fitted_folds": int(sum(int(src.get("fitted_folds", 0)) for src in sources)),
        "metrics": [],
    }


def targets_and_margin(score: np.ndarray, anchor: np.ndarray, allowed_actions: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(score, dtype=float)
    base = np.asarray(anchor, dtype=int)
    scoped = np.zeros_like(arr)
    scoped[:, list(allowed_actions)] = arr[:, list(allowed_actions)]
    target = scoped.argmax(axis=1).astype(int)
    rows = np.arange(len(base))
    target_score = scoped[rows, target]
    anchor_score = np.zeros(len(base), dtype=float)
    anchor_in_scope = np.isin(base, list(allowed_actions))
    anchor_safe = np.clip(base, 0, arr.shape[1] - 1)
    anchor_score[anchor_in_scope] = arr[rows[anchor_in_scope], anchor_safe[anchor_in_scope]]
    return target, target_score - anchor_score


def select_rows(
    anchor: np.ndarray,
    target: np.ndarray,
    margin: np.ndarray,
    allowed_actions: tuple[int, ...],
    budget: int,
) -> np.ndarray:
    base = np.asarray(anchor, dtype=int)
    cand = np.asarray(target, dtype=int)
    score = np.asarray(margin, dtype=float)
    eligible = (
        (cand != base)
        & np.isin(cand, list(allowed_actions))
        & ~np.isin(base, PROTECTED_ANCHOR_ACTIONS)
        & ~np.isin(cand, SERVE_ACTIONS)
        & np.isfinite(score)
        & (score > 0.0)
    )
    selected = np.zeros(len(base), dtype=bool)
    if budget <= 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected


def apply_selected(anchor: np.ndarray, target: np.ndarray, selected: np.ndarray) -> np.ndarray:
    out = np.asarray(anchor, dtype=int).copy()
    mask = np.asarray(selected, dtype=bool)
    out[mask] = np.asarray(target, dtype=int)[mask]
    return out


def evaluate_group_candidate(
    spec: ExportSpec,
    source: dict[str, Any],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    allowed = FOCUS_ACTIONS.tolist() if spec.group == "union_safe" else DEFAULT_SPECIALIST_GROUPS[spec.group]
    allowed_tuple = tuple(int(a) for a in allowed)
    oof_target, oof_margin = targets_and_margin(source["oof_score"], anchor_oof, allowed_tuple)
    test_target, test_margin = targets_and_margin(source["test_score"], anchor_test, allowed_tuple)
    oof_budget = int(math.floor(len(anchor_oof) * (float(spec.budget) / max(len(anchor_test), 1))))
    oof_selected = select_rows(anchor_oof, oof_target, oof_margin, allowed_tuple, oof_budget)
    test_selected = select_rows(anchor_test, test_target, test_margin, allowed_tuple, spec.budget)
    pred_oof = apply_selected(anchor_oof, oof_target, oof_selected)
    pred_test = apply_selected(anchor_test, test_target, test_selected)
    base_score = macro_f1(y, anchor_oof)
    score = macro_f1(y, pred_oof)
    precision = changed_row_precision(y, anchor_oof, pred_oof)
    changed_actions = pred_test[pred_test != anchor_test]
    serve_rows = int(np.isin(changed_actions, SERVE_ACTIONS).sum())
    rec = {
        "candidate_file": spec.filename,
        "candidate": spec.filename.removesuffix(".csv"),
        "source": source["name"],
        "source_family": source["family"],
        "specialist_group": spec.group,
        "allowed_actions": "/".join(str(a) for a in allowed_tuple),
        "test_budget": int(spec.budget),
        "oof_budget": int(oof_budget),
        "action_macro_f1": float(score),
        "action_oof_delta": float(score - base_score),
        "specialist_group_delta": float(macro_f1(y, pred_oof, list(allowed_tuple)) - macro_f1(y, anchor_oof, list(allowed_tuple))),
        "changed_action_rows": int(np.sum(pred_test != anchor_test)),
        "oof_changed_rows": int(precision["changed_rows"]),
        "changed_correct": int(precision["changed_correct"]),
        "changed_row_oof_precision": float(precision["changed_precision"]),
        "serve_action_rows": serve_rows,
        "evidence_pass": 0,
        "decision": "DO_NOT_UPLOAD",
        "test_changed_distribution": action_distribution(changed_actions) if len(changed_actions) else "{}",
        "test_action_distribution": action_distribution(pred_test),
        "min_test_margin_changed": float(test_margin[test_selected].min()) if test_selected.any() else 0.0,
        "mean_test_margin_changed": float(test_margin[test_selected].mean()) if test_selected.any() else 0.0,
    }
    rec["evidence_pass"] = int(evidence_passes(rec))
    rec["decision"] = "REVIEW_ACTION" if rec["evidence_pass"] else "DO_NOT_UPLOAD"
    return rec, pred_oof, pred_test


def build_search(
    sources: list[dict[str, Any]],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for source in sources:
        for spec in CANDIDATE_SPECS.values():
            rec, pred_oof, pred_test = evaluate_group_candidate(spec, source, y, anchor_oof, anchor_test)
            key = f"{source['name']}::{spec.filename}"
            records.append(rec)
            oof_predictions[key] = pred_oof
            test_predictions[key] = pred_test
            records[-1]["prediction_key"] = key
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
        group = ""
        for name, actions in DEFAULT_SPECIALIST_GROUPS.items():
            if int(action) in actions:
                group = name
                break
        anchor_f1 = class_f1(y, anchor_oof, int(action))
        best_f1 = class_f1(y, best_oof, int(action))
        rows.append(
            {
                "action": int(action),
                "specialist_group": group,
                "is_focus_action": int(group != ""),
                "anchor_f1": float(anchor_f1),
                "v317_best_f1": float(best_f1),
                "delta": float(best_f1 - anchor_f1),
                "support": int(np.sum(np.asarray(y, dtype=int) == int(action))),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_focus_action", "anchor_f1"], ascending=[False, True])


def read_reference_evidence() -> dict[str, Any]:
    ref = {
        "v312_best_delta": 0.0004923976482905656,
        "v312_best_changed_precision": 0.24691358024691357,
        "v220_has_changed_row_precision": False,
    }
    if V312_SEARCH.exists():
        try:
            search = pd.read_csv(V312_SEARCH)
            if "action_oof_delta" in search and len(search):
                ref["v312_best_delta"] = float(search["action_oof_delta"].max())
            if "changed_row_oof_precision" in search and len(search):
                ref["v312_best_changed_precision"] = float(search["changed_row_oof_precision"].max())
        except Exception:
            pass
    return ref


def export_submissions(
    search: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
    anchor_sub: pd.DataFrame,
) -> list[str]:
    generated: list[str] = []
    if search.empty:
        return generated
    for _, row in search[search["evidence_pass"].astype(int).eq(1)].iterrows():
        spec = CANDIDATE_SPECS[str(row["candidate_file"])]
        pred = test_predictions[str(row["prediction_key"])]
        out = build_export_frame(anchor_sub, pred)
        path = protected_output_path(OUTDIR, spec)
        out.to_csv(path, index=False, float_format="%.8f")
        generated.append(str(path.relative_to(ROOT)))
    return generated


def write_reports(
    sources: list[dict[str, Any]],
    search: pd.DataFrame,
    class_report: pd.DataFrame,
    generated: list[str],
    anchor_sub: pd.DataFrame,
) -> dict[str, Any]:
    search.to_csv(OUTDIR / "v317_action_search.csv", index=False)
    class_report.to_csv(OUTDIR / "v317_class_report.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    reviewable = search[search["evidence_pass"].astype(int).eq(1)].copy() if len(search) else pd.DataFrame()
    decision = "REVIEW_ACTION" if not reviewable.empty else "DO_NOT_UPLOAD"
    report = json_safe(
        {
            "version": "V317",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "action_anchor": "V173 action from V306 submission",
            "point_fixed_to": "V306 p0 cap0p01 pointId",
            "server_fixed_to": "V300 serverGetPoint",
            "copied_to_upload_or_selected": False,
            "ttmatch_used": False,
            "old_server_used": False,
            "specialist_groups": DEFAULT_SPECIALIST_GROUPS,
            "candidate_specs": [spec.__dict__ for spec in CANDIDATE_SPECS.values()],
            "evidence_thresholds": {
                "min_action_oof_delta": MIN_ACTION_OOF_DELTA,
                "min_changed_row_oof_precision": MIN_CHANGED_ROW_PRECISION,
                "max_serve_action_rows": MAX_SERVE_ACTION_ROWS,
            },
            "reference_evidence": read_reference_evidence(),
            "sources": [
                {
                    "name": src["name"],
                    "family": src["family"],
                    "fitted_folds": int(src.get("fitted_folds", 0)),
                    "metrics": src.get("metrics", [])[:12],
                }
                for src in sources
            ],
            "best_candidate": best,
            "reviewable_candidates": reviewable.head(8).to_dict(orient="records") if not reviewable.empty else [],
            "decision": decision,
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "anchor_rows": int(len(anchor_sub)),
        }
    )
    (OUTDIR / "v317_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    top = search.head(12)
    md = [
        "# V317 action specialist ensemble",
        "",
        f"Anchor submission: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Point/server: fixed to V306 point line and V300 server.",
        f"Decision: `{decision}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate_file', '')}`",
        f"Source: `{best.get('source', '')}`",
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
                "source_family",
                "specialist_group",
                "action_oof_delta",
                "changed_action_rows",
                "changed_row_oof_precision",
                "serve_action_rows",
                "decision",
            ],
        ),
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v317_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v317*.csv"):
        stale.unlink()
    rows, test_rows, y, anchor_oof, anchor_sub = load_anchor_frames()
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    x_train, x_test = build_feature_matrices(rows, test_rows, y)
    logreg = fit_action_ovr_source(x_train, x_test, rows, y)
    trees = fit_action_tree_source(x_train, x_test, rows, y)
    table = v286_table_source(rows, test_rows, y)
    blend = blend_sources([logreg, trees, table])
    sources = [logreg, trees, table, blend]
    search, oof_predictions, test_predictions = build_search(sources, y, anchor_oof, anchor_test)
    best_key = str(search.iloc[0]["prediction_key"]) if len(search) else ""
    best_oof = oof_predictions.get(best_key, anchor_oof)
    class_report = build_class_report(y, anchor_oof, best_oof)
    generated = export_submissions(search, test_predictions, anchor_sub)
    return write_reports(sources, search, class_report, generated, anchor_sub)


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "decision": report.get("decision", "DO_NOT_UPLOAD"),
                "best_candidate": best.get("candidate_file", ""),
                "best_source": best.get("source", ""),
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
