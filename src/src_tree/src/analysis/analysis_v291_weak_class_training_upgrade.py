"""V291 weak-class training upgrade diagnostics.

V291 is a local diagnostic layer over the V261/V173/R121 anchor. It upgrades
the weak-class feature audit, hard-negative specialist scoring, and candidate
exports while keeping pointId/serverGetPoint fixed from the V261 anchor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v286_weak_action_specialist_pretraining import (
    FAMILIES,
    add_external_response_features,
    build_external_response_prior,
    build_support_tables,
    class_f1,
    load_clean_external_family_corpus,
)
from analysis_v288_specialist_feature_discovery import (
    SERVE_ACTIONS,
    build_basic_feature_frame,
    changed_row_report,
)
from analysis_v290_shortcontrol411_specialist import build_shortcontrol_feature_frame, load_anchor_frames
from baseline_lgbm import ACTION_CLASSES


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v291_weak_class_training_upgrade"
V286_OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
V286_OOF = V286_OUTDIR / "v286_specialist_oof.csv"
V286_STAGE1 = V286_OUTDIR / "v286_stage1_pretrain_table.csv"
SPECIALIST_GROUPS = {
    "fast_attack_57": [5, 7],
    "terminal_03": [0, 3],
    "style_control_89": [8, 9],
    "short_control_411": [4, 11],
    "defensive_1214": [12, 14],
}

HARD_NEGATIVES = {
    "fast_attack_57": [1, 2, 4, 6, 10, 13],
    "terminal_03": [1, 2, 5, 10, 13],
    "style_control_89": [10, 11, 13],
    "short_control_411": [1, 7, 10, 13],
    "defensive_1214": [0, 1, 3, 5, 13],
}

PROTECTED_ACTIONS = [1, 10, 12, 13]
TEACHER_ACTIONS = [0, 3, 5, 7, 8, 9, 14]
MODEL_BANK = ("logistic_balanced", "extratrees_balanced", "extratrees_conservative")


def group_for_action(action: int) -> str:
    action = int(action)
    for group, actions in SPECIALIST_GROUPS.items():
        if action in actions:
            return group
    return ""


def feature_family_columns() -> dict[str, list[str]]:
    return {
        "phase_prefix": [
            "prefix_len",
            "prefix_len_bin",
            "phase_bin",
            "is_receive",
            "is_third",
            "is_fourth",
            "is_rally",
        ],
        "incoming_ball": [
            "lag0_actionId",
            "lag0_action_family",
            "lag0_pointId",
            "lag0_point_depth",
            "lag0_spin",
            "lag0_strength",
            "lag0_positionId",
            "lag0_action_point_pair",
            "lag0_spin_strength_pair",
        ],
        "score_pressure": [
            "scoreSelf",
            "scoreOther",
            "scoreTotal",
            "serverScoreDiff",
            "score_pressure_bin",
            "is_deuce_like",
            "is_game_point_like",
        ],
        "teacher_specialist": [
            "anchor_action",
            "specialist_p_0",
            "specialist_p_3",
            "specialist_p_5",
            "specialist_p_7",
            "specialist_p_8",
            "specialist_p_9",
            "specialist_p_14",
            "support_0",
            "support_3",
            "support_5",
            "support_7",
            "support_8",
            "support_9",
            "support_14",
        ],
        "support_backoff": [
            "support_exact_0",
            "support_exact_3",
            "support_exact_5",
            "support_exact_7",
            "support_exact_8",
            "support_exact_9",
            "support_exact_14",
            "support_family_depth_0",
            "support_family_depth_3",
            "support_family_depth_5",
            "support_family_depth_7",
            "support_family_depth_8",
            "support_family_depth_9",
            "support_family_depth_14",
            "support_phase_0",
            "support_phase_3",
            "support_phase_5",
            "support_phase_7",
            "support_phase_8",
            "support_phase_9",
            "support_phase_14",
        ],
        "external_clean_prior": [
            "v286_ext_family_Attack",
            "v286_ext_family_Control",
            "v286_ext_family_Defensive",
            "v286_ext_family_Zero",
            "v286_ext_terminal_rate",
        ],
        "style_response": [
            "style_actor_trust",
            "style_receiver_trust",
            "style_pair_familiarity",
            "style_cond_family_match",
            "style_cond_action_match",
            "style_response_entropy",
        ],
    }


def normalize_score01(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, 0.0, 1.0)


def hard_negative_mask(y: np.ndarray, group: str) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    positives = np.isin(y, SPECIALIST_GROUPS[group])
    negatives = np.isin(y, HARD_NEGATIVES[group])
    return positives | negatives


def choose_shortcontrol_action_or_keep(row: pd.Series) -> int:
    anchor = int(row.get("anchor_action", -1))
    if anchor in {10, 12, 13}:
        return anchor
    context = float(row.get("shortcontrol_context_score", 0.0))
    if context < 0.65:
        return anchor
    depth = str(row.get("lag0_point_depth", ""))
    spin = int(row.get("lag0_spin", 0))
    if depth == "short":
        return 11
    if spin in {1, 2, 3}:
        return 4
    return anchor


def _series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(np.full(len(frame), default), index=frame.index)


def _add_teacher_columns(frame: pd.DataFrame, v286: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    if v286 is not None and len(v286) == len(out):
        for col in ["anchor_action"] + [f"specialist_p_{a}" for a in TEACHER_ACTIONS] + [f"support_{a}" for a in TEACHER_ACTIONS]:
            out[col] = _series(v286, col, 0.0).to_numpy()
    else:
        if "anchor_action" not in out:
            out["anchor_action"] = 0
        for action in TEACHER_ACTIONS:
            out[f"specialist_p_{action}"] = 0.0
            out[f"support_{action}"] = 0
    return out


def _add_support_backoff_columns(
    frame: pd.DataFrame,
    rows: pd.DataFrame | None = None,
    y: np.ndarray | None = None,
    support_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    tables: dict[str, pd.DataFrame] | None = None
    source_rows = support_rows if support_rows is not None else rows
    if source_rows is not None and y is not None and len(source_rows) == len(y):
        try:
            basic = build_basic_feature_frame(source_rows)
            tables = build_support_tables(basic, np.asarray(y, dtype=int))
        except Exception:
            tables = None

    for action in TEACHER_ACTIONS:
        fallback = _series(out, f"support_{action}", 0).astype(int)
        for level in ["exact", "family_depth", "phase"]:
            col = f"support_{level}_{action}"
            out[col] = fallback.to_numpy()
            if not tables:
                continue
            tab = tables.get(level)
            if tab is None or tab.empty:
                continue
            key_map = {
                "exact": ["phase_bin", "lag0_actionId", "lag0_pointId", "candidate_action"],
                "family_depth": ["phase_bin", "lag0_action_family", "lag0_point_depth", "candidate_action"],
                "phase": ["phase_bin", "candidate_action"],
            }
            keys = key_map[level]
            left = out[[k for k in keys if k != "candidate_action"]].copy()
            left["_row_id"] = np.arange(len(out))
            left["candidate_action"] = int(action)
            merged = left.merge(tab[keys + ["support_count"]], on=keys, how="left")
            values = pd.to_numeric(merged["support_count"], errors="coerce").fillna(0).astype(int)
            out[col] = values.to_numpy()
    return out


def _add_external_prior_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    family_cols = [f"v286_ext_family_{family}" for family in FAMILIES]
    if all(col in out for col in family_cols + ["v286_ext_terminal_rate"]):
        return out
    try:
        external, _audit = load_clean_external_family_corpus()
        prior = build_external_response_prior(external)
        out = add_external_response_features(out, prior)
    except Exception:
        for col in family_cols:
            out[col] = 0.25
        out["v286_ext_terminal_rate"] = 0.0
    for col in family_cols:
        if col not in out:
            out[col] = 0.25
    if "v286_ext_terminal_rate" not in out:
        out["v286_ext_terminal_rate"] = 0.0
    return out


def _add_style_fallback_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in feature_family_columns()["style_response"]:
        if col not in out:
            out[col] = 0.0
    return out


def build_complete_feature_frame(
    rows: pd.DataFrame,
    v286: pd.DataFrame | None = None,
    y: np.ndarray | None = None,
    support_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = rows.copy()
    for col in ["lag0_positionId", "scoreSelf", "scoreOther", "scoreTotal", "serverScoreDiff"]:
        if col not in base:
            base[col] = 0
    out = build_basic_feature_frame(base)
    out = _add_teacher_columns(out, v286)
    out = _add_support_backoff_columns(out, rows=rows, y=y, support_rows=support_rows)
    out = _add_external_prior_columns(out)
    out = _add_style_fallback_columns(out)
    return out


def feature_coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for family, cols in feature_family_columns().items():
        present = [c for c in cols if c in frame.columns]
        nonconstant = []
        for col in present:
            if frame[col].nunique(dropna=False) > 1:
                nonconstant.append(col)
        records.append(
            {
                "family": family,
                "required_cols": len(cols),
                "present_cols": len(present),
                "nonconstant_cols": len(nonconstant),
                "missing_cols": ",".join(sorted(set(cols) - set(present))),
            }
        )
    return pd.DataFrame(records)


def _numeric_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    wanted: list[str] = []
    for cols in feature_family_columns().values():
        wanted.extend(cols)
    base = frame[[col for col in dict.fromkeys(wanted) if col in frame]].copy()
    numeric = base.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cats = base[[c for c in base.columns if c not in numeric.columns]].astype(str)
    if not cats.empty:
        numeric = pd.concat([numeric, pd.get_dummies(cats, dtype=float)], axis=1)
    return numeric.astype(float)


def _aligned_feature_matrices(train_frame: pd.DataFrame, test_frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    x_train = _numeric_feature_matrix(train_frame)
    if test_frame is None:
        return x_train, None
    x_test = _numeric_feature_matrix(test_frame)
    x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0.0)
    return x_train.astype(float), x_test.astype(float)


def _make_model(name: str, seed: int) -> Any:
    if name == "logistic_balanced":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=seed),
        )
    if name == "extratrees_balanced":
        return ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if name == "extratrees_conservative":
        return ExtraTreesClassifier(
            n_estimators=120,
            min_samples_leaf=16,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"unknown model bank entry: {name}")


def _splits(rows: pd.DataFrame, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if "fold" in rows:
        folds = rows["fold"].astype(int).to_numpy()
        pairs = [(np.where(folds != fold)[0], np.where(folds == fold)[0]) for fold in sorted(np.unique(folds))]
        if len(pairs) >= 2:
            return pairs
    groups = rows["match"].astype(str).to_numpy() if "match" in rows else None
    if groups is not None and len(np.unique(groups)) >= 2:
        n = min(5, len(np.unique(groups)))
        return list(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, groups))
    n = min(5, int(np.bincount(np.asarray(y, dtype=int)).min(initial=2)))
    n = max(2, n)
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=291).split(np.zeros(len(y)), y))


def train_model_bank_scores(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
    test_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, x_test = _aligned_feature_matrices(frame, test_frame)
    y = np.asarray(y, dtype=int)
    scores = pd.DataFrame(index=frame.index)
    test_scores = pd.DataFrame(index=test_frame.index if test_frame is not None else pd.RangeIndex(0))
    records: list[dict[str, Any]] = []
    split_pairs = _splits(rows, y)
    for group, actions in SPECIALIST_GROUPS.items():
        keep = hard_negative_mask(y, group)
        y_group_full = np.isin(y, actions).astype(int)
        for model_name in MODEL_BANK:
            oof = np.zeros(len(frame), dtype=float)
            test_sum = np.zeros(len(test_scores), dtype=float)
            fitted = 0
            for fold_id, (train_idx, valid_idx) in enumerate(split_pairs):
                train_idx = np.asarray([idx for idx in train_idx if keep[idx]], dtype=int)
                valid_idx = np.asarray([idx for idx in valid_idx if keep[idx]], dtype=int)
                if len(train_idx) == 0 or len(valid_idx) == 0:
                    continue
                target = y_group_full[train_idx]
                if len(np.unique(target)) < 2:
                    continue
                model = _make_model(model_name, seed=2910 + fold_id + len(actions) * 31)
                model.fit(x.iloc[train_idx], target)
                oof[valid_idx] = model.predict_proba(x.iloc[valid_idx])[:, 1]
                if x_test is not None and len(x_test):
                    test_sum += model.predict_proba(x_test)[:, 1]
                fitted += 1
            if fitted == 0:
                oof[:] = float(y_group_full[keep].mean()) if keep.any() else 0.0
                if x_test is not None and len(x_test):
                    train_idx = np.where(keep)[0]
                    target = y_group_full[train_idx]
                    if len(train_idx) and len(np.unique(target)) >= 2:
                        model = _make_model(model_name, seed=2910 + len(actions) * 31)
                        model.fit(x.iloc[train_idx], target)
                        test_sum = model.predict_proba(x_test)[:, 1]
                        fitted = 1
                    else:
                        test_sum[:] = float(y_group_full[keep].mean()) if keep.any() else 0.0
            elif x_test is not None and len(x_test):
                test_sum /= float(fitted)
                if len(np.unique(np.round(test_sum, 12))) <= 1:
                    train_idx = np.where(keep)[0]
                    target = y_group_full[train_idx]
                    if len(train_idx) and len(np.unique(target)) >= 2:
                        model = _make_model(model_name, seed=3910 + len(actions) * 31)
                        model.fit(x.iloc[train_idx], target)
                        test_sum = model.predict_proba(x_test)[:, 1]
            oof = normalize_score01(oof)
            test_pred = normalize_score01(test_sum)
            col = f"{group}__{model_name}"
            scores[col] = oof
            if x_test is not None:
                test_scores[col] = test_pred
            y_eval = y_group_full[keep]
            s_eval = oof[keep]
            ap = float(average_precision_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else 0.0
            auc = float(roc_auc_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else 0.5
            records.append(
                {
                    "group": group,
                    "model": model_name,
                    "ap": ap,
                    "auc": auc,
                    "positive_rows": int(y_eval.sum()),
                    "train_rows_after_hard_negative_mask": int(keep.sum()),
                    "oof_score_mean": float(s_eval.mean()) if len(s_eval) else 0.0,
                    "test_score_mean": float(test_pred.mean()) if len(test_pred) else 0.0,
                }
            )
    return scores, test_scores, pd.DataFrame(records)


def _best_score_column(model_comparison: pd.DataFrame, group: str) -> str:
    rows = model_comparison[model_comparison["group"].eq(group)].copy()
    if rows.empty:
        return f"{group}__extratrees_balanced"
    row = rows.sort_values(["ap", "auc", "model"], ascending=[False, False, True]).iloc[0]
    return f"{group}__{row['model']}"


def _candidate_pool(frame: pd.DataFrame, scores: pd.DataFrame, model_comparison: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    anchor = frame["anchor_action"].astype(int).to_numpy()
    for group, actions in SPECIALIST_GROUPS.items():
        score_col = _best_score_column(model_comparison, group)
        group_score = scores[score_col].to_numpy(dtype=float) if score_col in scores else np.zeros(len(frame))
        for action in actions:
            if action not in TEACHER_ACTIONS and group != "short_control_411":
                action_score = group_score
                support = np.zeros(len(frame), dtype=int)
            else:
                prob_col = f"specialist_p_{action}"
                support_col = f"support_family_depth_{action}" if f"support_family_depth_{action}" in frame else f"support_{action}"
                action_prob = _series(frame, prob_col, 0.5).to_numpy(dtype=float)
                action_score = normalize_score01(0.70 * group_score + 0.30 * action_prob)
                support = _series(frame, support_col, 0).astype(int).to_numpy()
            if group == "short_control_411":
                chosen = frame.apply(choose_shortcontrol_action_or_keep, axis=1).astype(int).to_numpy()
                keep_action = chosen == int(action)
                action_score = np.where(keep_action, normalize_score01(group_score), 0.0)
            pieces.append(
                pd.DataFrame(
                    {
                        "row_id": np.arange(len(frame), dtype=int),
                        "group": group,
                        "anchor_action": anchor,
                        "candidate_action": int(action),
                        "group_score": action_score,
                        "support_count": support,
                    }
                )
            )
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["candidate_action"].astype(int).ne(out["anchor_action"].astype(int))]
    out = out[~out["candidate_action"].astype(int).isin(SERVE_ACTIONS)]
    out = out[~out["anchor_action"].astype(int).isin(PROTECTED_ACTIONS + SERVE_ACTIONS)]
    return out.reset_index(drop=True)


def select_group_candidates(frame: pd.DataFrame, min_score: float = 0.05, min_support: int = 5) -> pd.DataFrame:
    filtered = frame[
        (pd.to_numeric(frame["group_score"], errors="coerce").fillna(0.0) >= float(min_score))
        & (pd.to_numeric(frame["support_count"], errors="coerce").fillna(0) >= int(min_support))
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=frame.columns)
    ranked = filtered.sort_values(["row_id", "group_score", "support_count"], ascending=[True, False, False])
    return ranked.groupby("row_id", as_index=False, sort=False).head(1).reset_index(drop=True)


def apply_row_cap(anchor: np.ndarray, row_candidates: pd.DataFrame, max_churn: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(anchor, dtype=int).copy()
    selected = np.zeros(len(pred), dtype=bool)
    max_rows = int(math.floor(len(pred) * float(max_churn)))
    if row_candidates.empty or max_rows <= 0:
        return pred, selected
    ranked = row_candidates.sort_values(["group_score", "support_count"], ascending=[False, False]).head(max_rows)
    ids = ranked["row_id"].astype(int).to_numpy()
    selected[ids] = True
    pred[ids] = ranked["candidate_action"].astype(int).to_numpy()
    return pred, selected


def _macro(y: np.ndarray, pred: np.ndarray, labels: list[int] | np.ndarray = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def _weighted_macro(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    scores = []
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    for label in ACTION_CLASSES:
        yt = y == label
        yp = pred == label
        tp = float(weights[yt & yp].sum())
        fp = float(weights[~yt & yp].sum())
        fn = float(weights[yt & ~yp].sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom <= 0 else 2 * tp / denom)
    return float(np.mean(scores))


def _public_weights(frame: pd.DataFrame) -> np.ndarray:
    try:
        return 1.0 + 0.15 * frame["is_receive"].to_numpy(dtype=float) + 0.10 * frame["is_game_point_like"].to_numpy(dtype=float)
    except Exception:
        return np.ones(len(frame), dtype=float)


def _candidate_tier(rec: dict[str, Any], diagnostic: bool, uses_shortcontrol: bool = False) -> str:
    if diagnostic:
        return "diagnostic_only"
    if uses_shortcontrol:
        class_delta = json.loads(str(rec.get("class_f1_delta_json", "{}")))
        if (
            float(class_delta.get("4", -1.0)) < 0
            or float(class_delta.get("11", -1.0)) < 0
            or float(rec["protected_mean_delta"]) < 0
        ):
            return "diagnostic_only"
    if (
        float(rec["delta_vs_v173"]) >= 0.001
        and float(rec["protected_mean_delta"]) >= 0
        and float(rec["public_like_delta_vs_v173"]) >= 0
        and 3 <= int(rec["test_changed_rows"]) <= 25
    ):
        return "clean_probe"
    return "diagnostic_only"


def evaluate_variant(
    name: str,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    pred_oof: np.ndarray,
    anchor_test: np.ndarray,
    pred_test: np.ndarray,
    allowed: set[int],
    model_family: str,
    diagnostic: bool,
    public_weights: np.ndarray,
    uses_shortcontrol: bool = False,
) -> dict[str, Any]:
    class_delta = {
        str(k): float(class_f1(y, pred_oof, ACTION_CLASSES)[k] - class_f1(y, anchor_oof, ACTION_CLASSES)[k])
        for k in ACTION_CLASSES
    }
    weak = np.array(sorted({a for actions in SPECIALIST_GROUPS.values() for a in actions}), dtype=int)
    rec = {
        "candidate": name,
        "model_family": model_family,
        "allowed_actions": "/".join(str(x) for x in sorted(allowed)),
        "action_macro_f1": _macro(y, pred_oof),
        "delta_vs_v173": _macro(y, pred_oof) - _macro(y, anchor_oof),
        "weak_mean_delta": _macro(y, pred_oof, weak) - _macro(y, anchor_oof, weak),
        "protected_mean_delta": _macro(y, pred_oof, PROTECTED_ACTIONS) - _macro(y, anchor_oof, PROTECTED_ACTIONS),
        "public_like_delta_vs_v173": _weighted_macro(y, pred_oof, public_weights)
        - _weighted_macro(y, anchor_oof, public_weights),
        "test_changed_rows": int(np.sum(np.asarray(pred_test, dtype=int) != np.asarray(anchor_test, dtype=int))),
        "class_f1_delta_json": json.dumps(class_delta, sort_keys=True),
        **changed_row_report(anchor_test, pred_test),
    }
    rec["candidate_tier"] = _candidate_tier(rec, diagnostic=diagnostic, uses_shortcontrol=uses_shortcontrol)
    return rec


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


def _build_feature_inputs(oof: pd.DataFrame, anchor_test: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    try:
        rows, test_rows, y, _anchor_oof = load_anchor_frames()
        if len(rows) == len(oof):
            test_rows = test_rows.reset_index(drop=True).copy()
            if len(test_rows) == len(anchor_test):
                test_rows["anchor_action"] = np.asarray(anchor_test, dtype=int)
            return (
                rows.reset_index(drop=True),
                test_rows,
                np.asarray(y, dtype=int),
                "teacher_specialist columns unavailable for test; V291 test scoring uses train/test-available feature families with teacher columns zero-filled.",
            )
    except Exception:
        pass
    if V286_STAGE1.exists():
        stage = pd.read_csv(V286_STAGE1, low_memory=False)
        stage = stage[stage["next_actionId"].notna()].reset_index(drop=True)
        if len(stage) == len(oof):
            test_rows = pd.DataFrame({"prefix_len": np.zeros(len(anchor_test), dtype=int), "anchor_action": anchor_test})
            return (
                stage,
                test_rows,
                stage["next_actionId"].astype(int).to_numpy(),
                "test rows unavailable from anchor rebuild; using minimal test feature fallback with teacher columns zero-filled.",
            )
    fallback = oof.copy()
    for col in ["prefix_len", "lag0_actionId", "lag0_pointId", "lag0_spinId", "lag0_strengthId"]:
        if col not in fallback:
            fallback[col] = 0
    test_rows = pd.DataFrame({"prefix_len": np.zeros(len(anchor_test), dtype=int), "anchor_action": anchor_test})
    return (
        fallback,
        test_rows,
        oof["y_true_action"].astype(int).to_numpy(),
        "train/test context rows unavailable; using minimal test feature fallback with teacher columns zero-filled.",
    )


def _hard_negative_report(y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": group,
                "positive_actions": "/".join(map(str, SPECIALIST_GROUPS[group])),
                "hard_negative_actions": "/".join(map(str, HARD_NEGATIVES[group])),
                "positive_rows": int(np.isin(y, SPECIALIST_GROUPS[group]).sum()),
                "hard_negative_rows": int(np.isin(y, HARD_NEGATIVES[group]).sum()),
                "training_rows": int(hard_negative_mask(y, group).sum()),
            }
            for group in SPECIALIST_GROUPS
        ]
    )


def build_class_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    anchor_f1 = class_f1(y, anchor, ACTION_CLASSES)
    pred_f1 = class_f1(y, pred, ACTION_CLASSES)
    return pd.DataFrame(
        [
            {
                "action": int(action),
                "group": group_for_action(action),
                "is_protected": int(action in PROTECTED_ACTIONS),
                "anchor_f1": float(anchor_f1[action]),
                "v291_f1": float(pred_f1[action]),
                "delta": float(pred_f1[action] - anchor_f1[action]),
            }
            for action in ACTION_CLASSES
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v291*.csv"):
        stale.unlink()
    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF file: {V286_OOF}")
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing anchor submission: {ANCHOR_SUBMISSION}")

    oof = pd.read_csv(V286_OOF)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    rows, test_rows, _input_y, test_feature_warning = _build_feature_inputs(oof, anchor_test)
    y = oof["y_true_action"].astype(int).to_numpy()

    feature_frame = build_complete_feature_frame(rows, oof, y=y)
    short_frame = build_shortcontrol_feature_frame(rows)
    for col in ["shortcontrol_context_score", "lag0_point_depth", "lag0_spin"]:
        feature_frame[col] = short_frame[col].to_numpy()
    test_feature_frame = build_complete_feature_frame(test_rows, None, y=y, support_rows=rows)
    test_short_frame = build_shortcontrol_feature_frame(test_rows)
    for col in ["shortcontrol_context_score", "lag0_point_depth", "lag0_spin"]:
        test_feature_frame[col] = test_short_frame[col].to_numpy()

    coverage = feature_coverage_report(feature_frame)
    coverage.to_csv(OUTDIR / "v291_feature_coverage.csv", index=False)
    family_report = coverage.copy()
    family_report["family_enabled"] = family_report["present_cols"].gt(0).astype(int)
    family_report.to_csv(OUTDIR / "v291_feature_family_report.csv", index=False)
    _hard_negative_report(y).to_csv(OUTDIR / "v291_hard_negative_report.csv", index=False)

    scores, test_scores, model_comparison = train_model_bank_scores(feature_frame, rows, y, test_frame=test_feature_frame)
    model_comparison.to_csv(OUTDIR / "v291_model_comparison.csv", index=False)
    pool = _candidate_pool(feature_frame, scores, model_comparison)
    test_pool = _candidate_pool(test_feature_frame, test_scores, model_comparison)
    public_weights = _public_weights(feature_frame)

    export_specs = [
        ("v291_fast57_modelbank_c0p005", "modelbank_fast57", {"fast_attack_57"}, {5, 7}, 0.005, False, False, "submission_v291_fast57_modelbank_c0p005__pv261cap1__sr121.csv"),
        ("v291_fast57_modelbank_c0p010", "modelbank_fast57", {"fast_attack_57"}, {5, 7}, 0.010, False, False, "submission_v291_fast57_modelbank_c0p010__pv261cap1__sr121.csv"),
        ("v291_terminal03_modelbank_c0p005", "modelbank_terminal03", {"terminal_03"}, {0, 3}, 0.005, True, False, "submission_v291_terminal03_modelbank_c0p005__pv261cap1__sr121.csv"),
        ("v291_bank_fast_terminal_c0p005", "modelbank_fast_terminal", {"fast_attack_57", "terminal_03"}, {0, 3, 5, 7}, 0.005, False, False, "submission_v291_bank_fast_terminal_c0p005__pv261cap1__sr121.csv"),
        ("v291_bank_fast_terminal_c0p010", "modelbank_fast_terminal", {"fast_attack_57", "terminal_03"}, {0, 3, 5, 7}, 0.010, False, False, "submission_v291_bank_fast_terminal_c0p010__pv261cap1__sr121.csv"),
        ("v291_shortcontrol_diagnostic_c0p005", "two_stage_shortcontrol", {"short_control_411"}, {4, 11}, 0.005, True, True, "submission_v291_shortcontrol_diagnostic_c0p005__pv261cap1__sr121.csv"),
        ("v291_style_defensive_diagnostic_c0p005", "modelbank_style_defensive", {"style_control_89", "defensive_1214"}, {8, 9, 12, 14}, 0.005, True, False, "submission_v291_style_defensive_diagnostic_c0p005__pv261cap1__sr121.csv"),
    ]

    records: list[dict[str, Any]] = []
    generated: list[str] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model_family, groups, allowed, cap, diagnostic, uses_shortcontrol, filename in export_specs:
        cand = pool[pool["group"].isin(groups) & pool["candidate_action"].astype(int).isin(sorted(allowed))].copy()
        cand = select_group_candidates(cand, min_score=0.05, min_support=5)
        pred_oof, _selected = apply_row_cap(anchor_oof, cand, cap)
        test_cand = test_pool[test_pool["group"].isin(groups) & test_pool["candidate_action"].astype(int).isin(sorted(allowed))].copy()
        test_cand = select_group_candidates(test_cand, min_score=0.05, min_support=5)
        pred_test, _test_selected = apply_row_cap(anchor_test, test_cand, cap)
        rec = evaluate_variant(
            name,
            y,
            anchor_oof,
            pred_oof,
            anchor_test,
            pred_test,
            allowed,
            model_family=model_family,
            diagnostic=diagnostic,
            public_weights=public_weights,
            uses_shortcontrol=uses_shortcontrol,
        )
        records.append(rec)
        predictions[name] = pred_oof
        generated.append(str(write_submission(filename, pred_test, anchor_sub).relative_to(ROOT)))

    search = pd.DataFrame(records)
    search["upload_recommendation"] = np.where(
        search["candidate_tier"].eq("clean_probe"),
        "REVIEW_CLEAN_PROBE",
        "DO_NOT_UPLOAD",
    )
    search = search.sort_values(
        ["candidate_tier", "delta_vs_v173", "public_like_delta_vs_v173", "protected_mean_delta", "test_changed_rows"],
        ascending=[True, False, False, False, True],
    )
    search.to_csv(OUTDIR / "v291_candidate_search.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    best_pred = predictions.get(str(best.get("candidate", "")), anchor_oof)
    build_class_report(y, anchor_oof, best_pred).to_csv(OUTDIR / "v291_class_report.csv", index=False)
    search[["candidate", "allowed_actions", "test_changed_rows", "changed_rows", "candidate_tier"]].to_csv(
        OUTDIR / "v291_changed_row_audit.csv", index=False
    )

    upload_recommendation = "DO_NOT_UPLOAD"
    clean = search[search["candidate_tier"].eq("clean_probe")].copy()
    if not clean.empty:
        upload_recommendation = "REVIEW_CLEAN_PROBE"

    report = _json_safe(
        {
            "version": "V291",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "point_server_fixed_to_v261": True,
            "no_ttmatch_no_old_server": True,
            "specialist_groups": SPECIALIST_GROUPS,
            "hard_negatives": HARD_NEGATIVES,
            "model_bank": list(MODEL_BANK),
            "feature_coverage": coverage.to_dict(orient="records"),
            "test_feature_warning": test_feature_warning,
            "best_candidate": best,
            "generated_submissions": generated,
            "upload_recommendation": upload_recommendation,
            "copied_to_upload_or_selected": False,
        }
    )
    (OUTDIR / "v291_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    md = [
        "# V291 weak-class training upgrade",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Point/server: fixed from V261 anchor",
        "TTMATCH/old-server: not used",
        f"Test feature warning: {test_feature_warning}",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Weak mean delta: {float(best.get('weak_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Public-like delta: {float(best.get('public_like_delta_vs_v173', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Upload recommendation: {upload_recommendation}",
        "",
        "## Generated local diagnostic submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v291_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "best_candidate": report["best_candidate"].get("candidate", ""),
                "best_delta_vs_v173": report["best_candidate"].get("delta_vs_v173", 0.0),
                "best_protected_mean_delta": report["best_candidate"].get("protected_mean_delta", 0.0),
                "best_public_like_delta_vs_v173": report["best_candidate"].get("public_like_delta_vs_v173", 0.0),
                "generated_submissions": len(report["generated_submissions"]),
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
