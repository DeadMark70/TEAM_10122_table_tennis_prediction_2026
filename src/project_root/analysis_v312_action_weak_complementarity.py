"""V312 action weak-class complementarity research.

V312 keeps the V306 point0 cap0p01 + V300 server anchor fixed and studies
small action-only edits over the V173 clean action anchor. The goal is
specialist complementarity for weak action classes, not full action replacement.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_lgbm import ACTION_CLASSES  # noqa: E402
from analysis_v292_weak_class_pretraining_action_teacher import numeric_matrix  # noqa: E402


OUTDIR = ROOT / "v312_action_weak_complementarity"
V286_OOF = ROOT / "v286_weak_action_specialist_pretraining" / "v286_specialist_oof.csv"
V292_CLASS_REPORT = ROOT / "v292_weak_class_pretraining_action_teacher" / "v292_class_report.csv"
V286_CLASS_REPORT = ROOT / "v286_weak_action_specialist_pretraining" / "v286_class_report.csv"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"

N_ACTIONS = 19
DEFAULT_WEAK_GROUPS: "OrderedDict[str, tuple[int, ...]]" = OrderedDict(
    [
        ("terminal_03", (0, 3)),
        ("fast_attack_57", (5, 7)),
        ("style_control_89", (8, 9)),
        ("defensive_1214", (12, 14)),
    ]
)
WEAK_ACTIONS = np.array(sorted({a for group in DEFAULT_WEAK_GROUPS.values() for a in group}), dtype=int)
PROTECTED_ANCHOR_ACTIONS = np.array([1, 10, 13, 15, 16, 17, 18], dtype=int)
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
TEST_BUDGETS = [10, 20, 30, 50, 80]
MIN_CHANGED_PRECISION = 0.55


def weak_group_masks(
    labels: np.ndarray,
    groups: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    group_map = DEFAULT_WEAK_GROUPS if groups is None else groups
    return {name: np.isin(labels, list(actions)) for name, actions in group_map.items()}


def changed_row_precision(y_true: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(anchor, dtype=int)
    pred = np.asarray(candidate, dtype=int)
    if not (len(y) == len(base) == len(pred)):
        raise ValueError("y_true, anchor, and candidate must have matching lengths")
    changed = pred != base
    changed_rows = int(changed.sum())
    changed_correct = int(np.sum(changed & (pred == y)))
    precision = float(changed_correct / changed_rows) if changed_rows else 0.0
    return {
        "changed_rows": changed_rows,
        "changed_correct": changed_correct,
        "changed_precision": precision,
    }


def decision_label(action_oof_delta: float, changed_action_rows: int) -> str:
    delta = float(action_oof_delta)
    rows = int(changed_action_rows)
    if delta >= 0.003 and rows <= 80:
        return "REVIEW_AGGRESSIVE"
    if delta >= 0.0015 and rows <= 30:
        return "REVIEW_ACTION"
    return "DO_NOT_UPLOAD"


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


def token(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


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
    if isinstance(rows, pd.DataFrame):
        records = rows[columns].to_dict(orient="records")
    else:
        records = rows

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
        raise ValueError(f"V286 OOF length {len(oof)} does not match rebuilt rows {len(rows)}")
    if len(anchor_sub) != len(test_rows):
        raise ValueError(f"anchor submission rows {len(anchor_sub)} != test rows {len(test_rows)}")
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    if len(rebuilt_y) == len(y) and not np.array_equal(np.asarray(rebuilt_y, dtype=int), y):
        warnings.warn("rebuilt y differs from V286 OOF labels; using V286 labels", RuntimeWarning)
    if len(rebuilt_anchor) == len(anchor_oof) and not np.array_equal(np.asarray(rebuilt_anchor, dtype=int), anchor_oof):
        warnings.warn("rebuilt anchor differs from V286 OOF anchor; using V286 anchor", RuntimeWarning)
    rows["fold"] = oof.get("fold", pd.Series(np.arange(len(rows)) % 5)).astype(int).to_numpy()
    test_rows["anchor_action"] = anchor_sub["actionId"].astype(int).to_numpy()
    return rows, test_rows, y, anchor_oof, anchor_sub


def build_feature_frames(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from analysis_v291_weak_class_training_upgrade import build_complete_feature_frame

    oof = pd.read_csv(V286_OOF)
    train_frame = build_complete_feature_frame(rows, oof, y=y).reset_index(drop=True)
    test_frame = build_complete_feature_frame(test_rows, None, y=y, support_rows=rows).reset_index(drop=True)
    x_train, x_test = numeric_matrix(train_frame, test_frame)
    return train_frame, x_train, x_test


def fold_splits(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    if "fold" in rows:
        folds = rows["fold"].astype(int).to_numpy()
        return [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(np.unique(folds))]
    fold_ids = np.arange(len(rows)) % 5
    return [(np.where(fold_ids != f)[0], np.where(fold_ids == f)[0]) for f in sorted(np.unique(fold_ids))]


def predict_proba_19(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    out = np.zeros((len(x), N_ACTIONS), dtype=float)
    for col, cls in enumerate(classes):
        if 0 <= int(cls) < N_ACTIONS:
            out[:, int(cls)] = raw[:, col]
    return normalize_rows_safe(out)


def fit_balanced_extratrees_source(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    oof = np.zeros((len(x_train), N_ACTIONS), dtype=float)
    test_sum = np.zeros((len(x_test), N_ACTIONS), dtype=float)
    fitted = 0
    for fold_id, (train_idx, valid_idx) in enumerate(fold_splits(rows)):
        if len(train_idx) == 0 or len(valid_idx) == 0 or len(np.unique(y[train_idx])) < 2:
            continue
        model = ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=7,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=31200 + fold_id,
            n_jobs=1,
        )
        model.fit(x_train.iloc[train_idx], y[train_idx])
        oof[valid_idx] = predict_proba_19(model, x_train.iloc[valid_idx])
        test_sum += predict_proba_19(model, x_test)
        fitted += 1
    if fitted == 0:
        prior = np.bincount(y, minlength=N_ACTIONS).astype(float)
        prior = prior / max(float(prior.sum()), 1.0)
        oof = np.tile(prior, (len(x_train), 1))
        test_prob = np.tile(prior, (len(x_test), 1))
    else:
        missing = oof.sum(axis=1) <= 0.0
        if missing.any():
            oof[missing] = np.eye(N_ACTIONS)[np.clip(y[missing], 0, N_ACTIONS - 1)]
        test_prob = test_sum / float(fitted)
    return {
        "name": "balanced_extratrees",
        "family": "class_balanced_extratrees",
        "oof_prob": normalize_rows_safe(oof),
        "test_prob": normalize_rows_safe(test_prob),
        "fitted_folds": fitted,
    }


def prior_adjusted_source(source: dict[str, Any], y: np.ndarray, tau: float = 0.45) -> dict[str, Any]:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=N_ACTIONS).astype(float) + 1.0
    prior = counts / counts.sum()

    def adjust(prob: np.ndarray) -> np.ndarray:
        p = normalize_rows_safe(prob)
        logits = np.log(np.clip(p, 1e-12, 1.0)) - float(tau) * np.log(prior.reshape(1, -1))
        logits -= logits.max(axis=1, keepdims=True)
        return normalize_rows_safe(np.exp(logits))

    return {
        "name": f"prior_adjusted_tau{token(tau)}",
        "family": "class_prior_adjusted",
        "oof_prob": adjust(source["oof_prob"]),
        "test_prob": adjust(source["test_prob"]),
        "fitted_folds": source.get("fitted_folds", 0),
        "tau": float(tau),
    }


def fit_ovr_weak_source(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    oof = np.zeros((len(x_train), N_ACTIONS), dtype=float)
    test = np.zeros((len(x_test), N_ACTIONS), dtype=float)
    fitted_total = 0
    metrics = []
    for action in WEAK_ACTIONS.tolist():
        target = (np.asarray(y, dtype=int) == int(action)).astype(int)
        action_test_sum = np.zeros(len(x_test), dtype=float)
        fitted = 0
        for fold_id, (train_idx, valid_idx) in enumerate(fold_splits(rows)):
            if len(np.unique(target[train_idx])) < 2:
                continue
            model = ExtraTreesClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=31270 + int(action) * 17 + fold_id,
                n_jobs=1,
            )
            model.fit(x_train.iloc[train_idx], target[train_idx])
            oof[valid_idx, int(action)] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
            action_test_sum += model.predict_proba(x_test)[:, 1]
            fitted += 1
        if fitted:
            test[:, int(action)] = action_test_sum / float(fitted)
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
    # Keep non-weak support tiny so source selection remains weak-action only.
    oof[:, :] = np.clip(oof, 0.0, 1.0)
    test[:, :] = np.clip(test, 0.0, 1.0)
    return {
        "name": "weak_ovr_extratrees",
        "family": "one_vs_rest_weak_specialists",
        "oof_prob": oof,
        "test_prob": test,
        "fitted_folds": fitted_total,
        "metrics": metrics,
    }


def apply_group_cap(prob: np.ndarray, allowed_actions: tuple[int, ...]) -> np.ndarray:
    p = np.asarray(prob, dtype=float).copy()
    blocked = np.ones(p.shape[1], dtype=bool)
    blocked[list(allowed_actions)] = False
    p[:, blocked] = 0.0
    return p


def candidate_targets_and_scores(
    prob: np.ndarray,
    anchor: np.ndarray,
    allowed_actions: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(prob, dtype=float)
    anchor = np.asarray(anchor, dtype=int)
    scoped = apply_group_cap(p, allowed_actions)
    targets = scoped.argmax(axis=1).astype(int)
    rows = np.arange(len(anchor))
    target_score = scoped[rows, targets]
    anchor_safe = np.clip(anchor, 0, p.shape[1] - 1)
    anchor_score = p[rows, anchor_safe]
    weak_anchor = np.isin(anchor, list(allowed_actions))
    score = target_score - np.where(weak_anchor, anchor_score, 0.0)
    return targets, score


def select_rows(
    anchor: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    allowed_actions: tuple[int, ...],
    budget: int,
) -> np.ndarray:
    base = np.asarray(anchor, dtype=int)
    cand = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
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
    pred = np.asarray(anchor, dtype=int).copy()
    pred[np.asarray(selected, dtype=bool)] = np.asarray(target, dtype=int)[np.asarray(selected, dtype=bool)]
    return pred


def distribution_json(values: np.ndarray) -> str:
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(unique, counts)}, sort_keys=True)


def evaluate_candidate(
    name: str,
    source: dict[str, Any],
    group_name: str,
    allowed_actions: tuple[int, ...],
    test_budget: int,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    oof_target, oof_score = candidate_targets_and_scores(source["oof_prob"], anchor_oof, allowed_actions)
    test_target, test_score = candidate_targets_and_scores(source["test_prob"], anchor_test, allowed_actions)
    oof_budget = int(math.floor(len(anchor_oof) * (float(test_budget) / max(len(anchor_test), 1))))
    oof_selected = select_rows(anchor_oof, oof_target, oof_score, allowed_actions, oof_budget)
    test_selected = select_rows(anchor_test, test_target, test_score, allowed_actions, test_budget)
    pred_oof = apply_selected(anchor_oof, oof_target, oof_selected)
    pred_test = apply_selected(anchor_test, test_target, test_selected)
    base_score = macro_f1(y, anchor_oof)
    score = macro_f1(y, pred_oof)
    precision = changed_row_precision(y, anchor_oof, pred_oof)
    changed_actions = pred_test[pred_test != anchor_test]
    precision_pass = bool(precision["changed_rows"] > 0 and precision["changed_precision"] >= MIN_CHANGED_PRECISION)
    raw_decision = decision_label(score - base_score, int(np.sum(pred_test != anchor_test)))
    decision = raw_decision if precision_pass else "DO_NOT_UPLOAD"
    rec = {
        "candidate": name,
        "source": source["name"],
        "source_family": source["family"],
        "weak_group": group_name,
        "allowed_actions": "/".join(str(a) for a in allowed_actions),
        "test_budget": int(test_budget),
        "oof_budget": int(oof_budget),
        "action_macro_f1": float(score),
        "action_oof_delta": float(score - base_score),
        "weak_group_delta": float(macro_f1(y, pred_oof, list(allowed_actions)) - macro_f1(y, anchor_oof, list(allowed_actions))),
        "changed_action_rows": int(np.sum(pred_test != anchor_test)),
        "oof_changed_rows": int(precision["changed_rows"]),
        "changed_correct": int(precision["changed_correct"]),
        "changed_row_oof_precision": float(precision["changed_precision"]),
        "precision_gate_pass": int(precision_pass),
        "decision": decision,
        "raw_threshold_decision": raw_decision,
        "test_changed_distribution": distribution_json(changed_actions) if len(changed_actions) else "{}",
        "test_action_distribution": distribution_json(pred_test),
        "min_test_score_changed": float(test_score[test_selected].min()) if test_selected.any() else 0.0,
        "mean_test_score_changed": float(test_score[test_selected].mean()) if test_selected.any() else 0.0,
    }
    return rec, pred_oof, pred_test


def build_candidate_search(
    sources: list[dict[str, Any]],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    groups: list[tuple[str, tuple[int, ...]]] = [("all_focus_0357891214", tuple(WEAK_ACTIONS.tolist()))]
    groups.extend(DEFAULT_WEAK_GROUPS.items())
    records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for source in sources:
        for group_name, allowed in groups:
            for budget in TEST_BUDGETS:
                name = f"v312_{source['name']}_{group_name}_b{budget}"
                rec, pred_oof, pred_test = evaluate_candidate(
                    name,
                    source,
                    group_name,
                    tuple(int(a) for a in allowed),
                    budget,
                    y,
                    anchor_oof,
                    anchor_test,
                )
                records.append(rec)
                oof_predictions[name] = pred_oof
                test_predictions[name] = pred_test
    search = pd.DataFrame(records).sort_values(
        [
            "decision",
            "precision_gate_pass",
            "action_oof_delta",
            "changed_row_oof_precision",
            "changed_action_rows",
        ],
        ascending=[True, False, False, False, True],
    )
    return search.reset_index(drop=True), oof_predictions, test_predictions


def identify_weak_classes(y: np.ndarray, anchor_oof: np.ndarray) -> tuple[pd.DataFrame, str]:
    source = "computed_from_v286_oof_anchor"
    records = []
    if V292_CLASS_REPORT.exists():
        try:
            report = pd.read_csv(V292_CLASS_REPORT)
            if {"action", "anchor_f1"}.issubset(report.columns):
                for _, row in report.iterrows():
                    action = int(row["action"])
                    records.append({"action": action, "anchor_f1": float(row["anchor_f1"])})
                source = str(V292_CLASS_REPORT.relative_to(ROOT))
        except Exception:
            records = []
    if not records and V286_CLASS_REPORT.exists():
        try:
            report = pd.read_csv(V286_CLASS_REPORT)
            f1_col = "anchor_f1" if "anchor_f1" in report.columns else "v173_f1"
            if {"action", f1_col}.issubset(report.columns):
                for _, row in report.iterrows():
                    records.append({"action": int(row["action"]), "anchor_f1": float(row[f1_col])})
                source = str(V286_CLASS_REPORT.relative_to(ROOT))
        except Exception:
            records = []
    if not records:
        records = [{"action": int(action), "anchor_f1": class_f1(y, anchor_oof, int(action))} for action in ACTION_CLASSES]
    class_report = pd.DataFrame(records)
    class_report["weak_group"] = ""
    for group, actions in DEFAULT_WEAK_GROUPS.items():
        class_report.loc[class_report["action"].astype(int).isin(actions), "weak_group"] = group
    class_report["is_focus_weak"] = class_report["weak_group"].ne("").astype(int)
    return class_report.sort_values(["is_focus_weak", "anchor_f1"], ascending=[False, True]).reset_index(drop=True), source


def build_class_report(
    y: np.ndarray,
    anchor_oof: np.ndarray,
    best_oof: np.ndarray,
    weak_source: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    weak_lookup = weak_source.set_index("action")["anchor_f1"].to_dict() if "action" in weak_source else {}
    for action in ACTION_CLASSES:
        group = ""
        for name, actions in DEFAULT_WEAK_GROUPS.items():
            if int(action) in actions:
                group = name
                break
        anchor_f1 = class_f1(y, anchor_oof, int(action))
        best_f1 = class_f1(y, best_oof, int(action))
        rows.append(
            {
                "action": int(action),
                "weak_group": group,
                "is_focus_weak": int(group != ""),
                "report_anchor_f1": float(weak_lookup.get(int(action), anchor_f1)),
                "computed_anchor_f1": float(anchor_f1),
                "v312_best_f1": float(best_f1),
                "delta": float(best_f1 - anchor_f1),
                "support": int(np.sum(np.asarray(y, dtype=int) == int(action))),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_focus_weak", "computed_anchor_f1"], ascending=[False, True])


def write_submission(name: str, action: np.ndarray, anchor_sub: pd.DataFrame) -> Path:
    out = pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return path


def export_submissions(search: pd.DataFrame, test_predictions: dict[str, np.ndarray], anchor_sub: pd.DataFrame) -> list[str]:
    generated: list[str] = []
    passed = search[(search["precision_gate_pass"].astype(int) == 1) & (search["changed_action_rows"].astype(int) > 0)].copy()
    if passed.empty:
        return generated
    passed = passed.sort_values(["decision", "action_oof_delta", "changed_row_oof_precision"], ascending=[True, False, False])
    # Keep local packaging small; search/report still preserve the full sweep.
    for _, row in passed.head(12).iterrows():
        candidate = str(row["candidate"])
        pred = test_predictions[candidate]
        filename = f"submission_{candidate}__pV306cap0p01_sV300.csv"
        path = write_submission(filename, pred, anchor_sub)
        generated.append(str(path.relative_to(ROOT)))
    return generated


def write_reports(
    sources: list[dict[str, Any]],
    search: pd.DataFrame,
    class_report: pd.DataFrame,
    weak_report: pd.DataFrame,
    weak_report_source: str,
    generated: list[str],
    anchor_sub: pd.DataFrame,
) -> dict[str, Any]:
    search.to_csv(OUTDIR / "v312_action_search.csv", index=False)
    class_report.to_csv(OUTDIR / "v312_class_report.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    reviewable = search[search["decision"].isin(["REVIEW_ACTION", "REVIEW_AGGRESSIVE"])].copy()
    if not reviewable.empty:
        overall_decision = str(reviewable.iloc[0]["decision"])
        best_review = reviewable.iloc[0].to_dict()
    else:
        overall_decision = "DO_NOT_UPLOAD"
        best_review = {}
    weak_focus = weak_report[weak_report["is_focus_weak"].astype(int).eq(1)].sort_values("anchor_f1").head(8)
    source_summary = [
        {
            "name": source["name"],
            "family": source["family"],
            "fitted_folds": int(source.get("fitted_folds", 0)),
            "tau": source.get("tau"),
            "ovr_metrics": source.get("metrics", [])[:8],
        }
        for source in sources
    ]
    report = json_safe(
        {
            "version": "V312",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "action_anchor": "V173 via V286 OOF anchor/test action in V306 submission",
            "point_fixed_to": "V306 p0 cap0p01",
            "server_fixed_to": "V300",
            "copied_to_upload_or_selected": False,
            "weak_class_report_source": weak_report_source,
            "focus_weak_groups": DEFAULT_WEAK_GROUPS,
            "identified_weak_focus_classes": weak_focus.to_dict(orient="records"),
            "specialist_sources": source_summary,
            "min_changed_row_oof_precision": MIN_CHANGED_PRECISION,
            "best_candidate": best,
            "best_review_candidate": best_review,
            "decision": overall_decision,
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "anchor_rows": int(len(anchor_sub)),
        }
    )
    (OUTDIR / "v312_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    top = search.head(10)
    md = [
        "# V312 action weak-class complementarity",
        "",
        f"Anchor submission: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Point/server: fixed to V306 p0 cap0p01 and V300.",
        f"Weak class source: `{weak_report_source}`",
        f"Decision: `{overall_decision}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate', '')}`",
        f"OOF action delta: {float(best.get('action_oof_delta', 0.0)):.6f}",
        f"Changed test rows: {int(best.get('changed_action_rows', 0))}",
        f"Changed-row OOF precision: {float(best.get('changed_row_oof_precision', 0.0)):.4f}",
        f"Candidate decision: `{best.get('decision', '')}`",
        "",
        "## Top search rows",
        "",
        markdown_table(
            top,
            [
                "candidate",
                "source_family",
                "weak_group",
                "action_oof_delta",
                "changed_action_rows",
                "changed_row_oof_precision",
                "decision",
            ],
        ),
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v312_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v312*.csv"):
        stale.unlink()
    rows, test_rows, y, anchor_oof, anchor_sub = load_anchor_frames()
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    weak_report, weak_report_source = identify_weak_classes(y, anchor_oof)
    _train_frame, x_train, x_test = build_feature_frames(rows, test_rows, y)
    balanced = fit_balanced_extratrees_source(x_train, x_test, rows, y)
    prior_adjusted = prior_adjusted_source(balanced, y)
    ovr = fit_ovr_weak_source(x_train, x_test, rows, y)
    sources = [balanced, prior_adjusted, ovr]
    search, oof_predictions, test_predictions = build_candidate_search(sources, y, anchor_oof, anchor_test)
    best_name = str(search.iloc[0]["candidate"]) if len(search) else ""
    best_oof = oof_predictions.get(best_name, anchor_oof)
    class_report = build_class_report(y, anchor_oof, best_oof, weak_report)
    generated = export_submissions(search, test_predictions, anchor_sub)
    return write_reports(sources, search, class_report, weak_report, weak_report_source, generated, anchor_sub)


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "decision": report.get("decision", "DO_NOT_UPLOAD"),
                "best_candidate": best.get("candidate", ""),
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
