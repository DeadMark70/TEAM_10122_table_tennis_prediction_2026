"""R10/R10B/R11 action macro audit and action-point compatibility.

R10:
  - class-wise action audit for V3/V5/V7/V10B/R7/R8.

R10B:
  - action-only blend search plus conservative class-bias tuning.
  - point fixed to V3 and server fixed to V10B-safe.

R11:
  - action-point compatibility prior for point, using OOF action probabilities.
  - no new point model is trained.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


PREFIX_BINS = [
    ("1", lambda s: s.eq(1).to_numpy()),
    ("2", lambda s: s.eq(2).to_numpy()),
    ("ge3", lambda s: s.ge(3).to_numpy()),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R10/R11 OOF action and joint compatibility analysis.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--action-class-report", default="r10_action_class_report.csv")
    parser.add_argument("--action-summary", default="r10_action_summary.csv")
    parser.add_argument("--action-bias-report", default="r10b_action_bias_report.csv")
    parser.add_argument("--r10-selected", default="r10_selected.json")
    parser.add_argument("--compat-report", default="r11_action_point_compat_report.csv")
    parser.add_argument("--r11-selected", default="r11_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r10_r11.json")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    return prob / prob.sum(axis=1, keepdims=True)


def action_class_report(meta: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for model_name, prob in probs.items():
        pred = np.asarray(ACTION_CLASSES)[np.argmax(prob, axis=1)]
        for cls in ACTION_CLASSES:
            tp = int(((y == cls) & (pred == cls)).sum())
            pred_count = int((pred == cls).sum())
            support = int((y == cls).sum())
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rows.append(
                {
                    "model": model_name,
                    "actionId": cls,
                    "support": support,
                    "pred_count": pred_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def point_score(meta: pd.DataFrame, point_prob: np.ndarray, v3_tuning: V3Tuning) -> tuple[float, np.ndarray]:
    pred = apply_segmented_multipliers(meta, point_prob, v3_tuning.point_multipliers, POINT_CLASSES, v3_tuning.bins_mode)
    score = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    return float(score), pred


def action_pred_with_bias(meta: pd.DataFrame, prob: np.ndarray, biases: dict[str, np.ndarray]) -> np.ndarray:
    pred = np.zeros(len(meta), dtype=int)
    logp = np.log(np.clip(prob, 1e-12, 1.0))
    for label, fn in PREFIX_BINS:
        idx = np.where(fn(meta["prefix_len"]))[0]
        if len(idx) == 0:
            continue
        score = logp[idx] + biases[label][None, :]
        pred[idx] = np.asarray(ACTION_CLASSES)[np.argmax(score, axis=1)]
    return pred


def assign_folds_from_report(meta: pd.DataFrame, fold_report: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    folds = []
    rows = fold_report[fold_report["valid_rows"].gt(0)][["fold", "valid_rows"]]
    for _, row in rows.iterrows():
        folds.extend([int(row["fold"])] * int(row["valid_rows"]))
    if len(folds) != len(out):
        raise ValueError(f"Cannot assign folds: fold rows={len(folds)} meta rows={len(out)}")
    out["fold"] = folds
    return out


def fit_action_bias(
    meta: pd.DataFrame, prob: np.ndarray, fit_mask: np.ndarray
) -> dict[str, np.ndarray]:
    y = meta["next_actionId"].to_numpy(dtype=int)
    candidate_bias = [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8]
    biases = {label: np.zeros(len(ACTION_CLASSES), dtype=float) for label, _ in PREFIX_BINS}
    base_pred = action_pred_with_bias(meta, prob, biases)
    fit_idx = np.where(fit_mask)[0]
    best_global = f1_score(y[fit_idx], base_pred[fit_idx], average="macro", labels=ACTION_CLASSES, zero_division=0)

    # Coordinate descent with low freedom: one class/bin at a time.
    for _ in range(2):
        improved = False
        for label, fn in PREFIX_BINS:
            idx = np.where(fn(meta["prefix_len"]) & fit_mask)[0]
            if len(idx) < 250:
                continue
            for cls_idx, cls in enumerate(ACTION_CLASSES):
                current = biases[label][cls_idx]
                best_value = current
                best_score = best_global
                for value in candidate_bias:
                    biases[label][cls_idx] = value
                    pred = action_pred_with_bias(meta, prob, biases)
                    score = f1_score(y[fit_idx], pred[fit_idx], average="macro", labels=ACTION_CLASSES, zero_division=0)
                    churn = float((pred[fit_idx] != base_pred[fit_idx]).mean())
                    objective = score - 0.002 * churn
                    if objective > best_score + 1e-8:
                        best_score = objective
                        best_value = value
                biases[label][cls_idx] = best_value
                if best_value != current:
                    pred = action_pred_with_bias(meta, prob, biases)
                    best_global = f1_score(
                        y[fit_idx], pred[fit_idx], average="macro", labels=ACTION_CLASSES, zero_division=0
                    )
                    improved = True
        if not improved:
            break
    return biases


def tune_action_bias_nested(meta: pd.DataFrame, prob: np.ndarray) -> tuple[list[dict[str, np.ndarray]], np.ndarray, float]:
    y = meta["next_actionId"].to_numpy(dtype=int)
    pred = np.zeros(len(meta), dtype=int)
    fold_biases = []
    folds = sorted(meta["fold"].unique())
    for fold in folds:
        valid_mask = meta["fold"].eq(fold).to_numpy()
        fit_mask = ~valid_mask
        biases = fit_action_bias(meta, prob, fit_mask)
        fold_biases.append({k: v.copy() for k, v in biases.items()})
        fold_pred = action_pred_with_bias(meta, prob, biases)
        pred[valid_mask] = fold_pred[valid_mask]
    score = f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    return fold_biases, pred, float(score)


def tune_action_bias(meta: pd.DataFrame, prob: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    y = meta["next_actionId"].to_numpy(dtype=int)
    biases = fit_action_bias(meta, prob, np.ones(len(meta), dtype=bool))
    pred = action_pred_with_bias(meta, prob, biases)
    score = f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    return biases, pred, float(score)


def evaluate_overall(action_f1: float, point_f1: float, server_auc: float) -> float:
    return float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc)


def build_action_components(v3, v5, v7, v10, r7, selected_v10, selected_r8):
    v3_action, v3_point, v3_server = compose_v3(v3)
    r7_action, _, _ = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    safe_action = blend_probs(r1_action, v10["v10_action"], float(selected_v10["action_v10_weight"]))
    safe_server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]
    r8_action = blend_probs(safe_action, r7_action, float(selected_r8["r7_weight"]))
    return {
        "v3": v3_action,
        "v5_gru": v5["gru_action"],
        "v7_transformer": v7["tr_action"],
        "v10b": v10["v10_action"],
        "r7_phase": r7_action,
        "r1": r1_action,
        "v10b_safe": safe_action,
        "r8": r8_action,
    }, v3_point, safe_server


def search_action_blends(meta: pd.DataFrame, comps: dict[str, np.ndarray], point_f1: float, server_auc: float):
    rows = []
    candidates = {
        "r1": comps["r1"],
        "v10b_safe": comps["v10b_safe"],
        "r8": comps["r8"],
    }
    # Targeted blend around the useful action experts.
    for w10 in [0.0, 0.2, 0.4, 0.5, 0.6]:
        for wr7 in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15]:
            if w10 + wr7 > 0.8:
                continue
            base = comps["r1"]
            prob = (1.0 - w10 - wr7) * base + w10 * comps["v10b"] + wr7 * comps["r7_phase"]
            prob = normalize_rows(prob)
            candidates[f"r1_v10{w10:g}_r7{wr7:g}"] = prob
    best = None
    for name, prob in candidates.items():
        fold_biases, pred, action_f1 = tune_action_bias_nested(meta, prob)
        no_bias_pred = np.asarray(ACTION_CLASSES)[np.argmax(prob, axis=1)]
        no_bias_f1 = f1_score(meta["next_actionId"], no_bias_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        row = {
            "variant": name,
            "action_macro_f1": float(action_f1),
            "action_no_bias_macro_f1": float(no_bias_f1),
            "point_macro_f1": point_f1,
            "server_auc": server_auc,
            "overall": evaluate_overall(action_f1, point_f1, server_auc),
            "action_churn_vs_r8": float((pred != np.asarray(ACTION_CLASSES)[np.argmax(comps["r8"], axis=1)]).mean()),
        }
        rows.append(row)
        if best is None or row["overall"] > best["row"]["overall"]:
            best = {"row": row, "prob": prob, "fold_biases": fold_biases, "pred": pred}
    return pd.DataFrame(rows).sort_values("overall", ascending=False), best


def fit_action_point_compat(meta: pd.DataFrame, action_prob: np.ndarray, alpha: float) -> np.ndarray:
    y_action = meta["next_actionId"].to_numpy(dtype=int)
    y_point = meta["next_pointId"].to_numpy(dtype=int)
    prefix = meta["prefix_len"].to_numpy(dtype=int)
    bins = np.where(prefix == 1, 1, np.where(prefix == 2, 2, 3))
    compat = np.zeros((len(meta), len(POINT_CLASSES)), dtype=np.float32)
    global_counts = pd.Series(y_point).value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(POINT_CLASSES))
    for bin_id in [1, 2, 3]:
        train_mask = bins == bin_id
        table = np.zeros((len(ACTION_CLASSES), len(POINT_CLASSES)), dtype=float)
        for a, p in zip(y_action[train_mask], y_point[train_mask]):
            table[int(a), int(p)] += 1.0
        table = (table + alpha * global_prior[None, :]) / (table.sum(axis=1, keepdims=True) + alpha)
        idx = np.where(train_mask)[0]
        compat[idx] = action_prob[idx] @ table
    compat = compat / compat.sum(axis=1, keepdims=True)
    return compat


def fit_action_point_compat_foldsafe(meta: pd.DataFrame, action_prob: np.ndarray, alpha: float) -> np.ndarray:
    y_action = meta["next_actionId"].to_numpy(dtype=int)
    y_point = meta["next_pointId"].to_numpy(dtype=int)
    prefix = meta["prefix_len"].to_numpy(dtype=int)
    bins = np.where(prefix == 1, 1, np.where(prefix == 2, 2, 3))
    compat = np.zeros((len(meta), len(POINT_CLASSES)), dtype=np.float32)
    global_counts = pd.Series(y_point).value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(POINT_CLASSES))
    for fold in sorted(meta["fold"].unique()):
        valid_mask = meta["fold"].eq(fold).to_numpy()
        fit_mask = ~valid_mask
        for bin_id in [1, 2, 3]:
            fit_bin = fit_mask & (bins == bin_id)
            valid_bin = valid_mask & (bins == bin_id)
            if not valid_bin.any():
                continue
            table = np.zeros((len(ACTION_CLASSES), len(POINT_CLASSES)), dtype=float)
            for a, p in zip(y_action[fit_bin], y_point[fit_bin]):
                table[int(a), int(p)] += 1.0
            table = (table + alpha * global_prior[None, :]) / (table.sum(axis=1, keepdims=True) + alpha)
            idx = np.where(valid_bin)[0]
            compat[idx] = action_prob[idx] @ table
    compat = compat / compat.sum(axis=1, keepdims=True)
    return compat


def search_action_point_compat(meta, base_point, action_prob, v3_tuning):
    base_point_f1, base_pred = point_score(meta, base_point, v3_tuning)
    rows = []
    best = None
    for alpha in [5.0, 20.0, 50.0, 100.0]:
        compat = fit_action_point_compat_foldsafe(meta, action_prob, alpha)
        for lam in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15]:
            mixed = blend_probs(base_point, compat, lam)
            point_f1, pred = point_score(meta, mixed, v3_tuning)
            row = {
                "alpha": alpha,
                "lambda": lam,
                "point_macro_f1": point_f1,
                "gain_vs_v3": point_f1 - base_point_f1,
                "point_churn_vs_v3": float((pred != base_pred).mean()),
            }
            rows.append(row)
            eligible = row["gain_vs_v3"] >= 0.0015 and row["point_churn_vs_v3"] <= 0.03 and lam > 0
            obj = point_f1 - 0.02 * row["point_churn_vs_v3"]
            if eligible and (best is None or obj > best["objective"]):
                best = {"objective": obj, "row": row, "prob": mixed, "pred": pred}
    report = pd.DataFrame(rows).sort_values(["gain_vs_v3", "point_macro_f1"], ascending=False)
    if best is None:
        return report, {"config": "base_v3", "point_macro_f1": base_point_f1, "gain_vs_v3": 0.0}, base_point, base_pred
    return report, best["row"], best["prob"], best["pred"]


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r7 = load_pickle(args.r7_oof)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    selected_r8 = json.loads(Path(args.r8_selected).read_text(encoding="utf-8"))
    meta = normalize_meta(v3["valid_meta"])
    meta = assign_folds_from_report(meta, v3["fold_report"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7)]:
        assert_aligned(meta, oof["valid_meta"], name)

    action_components, v3_point, safe_server = build_action_components(v3, v5, v7, v10, r7, selected_v10, selected_r8)
    class_df = action_class_report(meta, action_components)
    class_df.to_csv(args.action_class_report, index=False)
    point_f1, point_pred = point_score(meta, v3_point, v3["tuning"])
    server_auc = roc_auc_score(meta["serverGetPoint"], safe_server)
    action_summary_rows = []
    for name, prob in action_components.items():
        pred = np.asarray(ACTION_CLASSES)[np.argmax(prob, axis=1)]
        action_f1 = f1_score(meta["next_actionId"], pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        action_summary_rows.append(
            {
                "variant": name,
                "action_argmax_macro_f1": float(action_f1),
                "overall_with_v3_point_safe_server": evaluate_overall(action_f1, point_f1, server_auc),
            }
        )
    pd.DataFrame(action_summary_rows).sort_values("overall_with_v3_point_safe_server", ascending=False).to_csv(
        args.action_summary, index=False
    )

    action_report, r10_best = search_action_blends(meta, action_components, point_f1, server_auc)
    action_report.to_csv(args.action_bias_report, index=False)
    r10_selected = {
        "selected": r10_best["row"],
        "fold_biases": [
            {k: v.tolist() for k, v in fold_bias.items()} for fold_bias in r10_best["fold_biases"]
        ],
        "point_policy": "fixed_v3_point",
        "server_policy": "fixed_v10b_safe_server",
        "tuning_protocol": "nested_oof_by_outer_fold",
    }
    Path(args.r10_selected).write_text(json.dumps(r10_selected, indent=2), encoding="utf-8")

    compat_report, r11_selected, r11_point, r11_pred = search_action_point_compat(
        meta, v3_point, r10_best["prob"], v3["tuning"]
    )
    compat_report.to_csv(args.compat_report, index=False)
    r11_overall = evaluate_overall(
        float(r10_best["row"]["action_macro_f1"]),
        float(r11_selected["point_macro_f1"]),
        server_auc,
    )
    r11_out = {
        "selected": r11_selected,
        "overall_with_r10_action_safe_server": r11_overall,
        "action_variant": r10_best["row"]["variant"],
        "action_macro_f1": float(r10_best["row"]["action_macro_f1"]),
        "server_auc": float(server_auc),
    }
    Path(args.r11_selected).write_text(json.dumps(r11_out, indent=2), encoding="utf-8")

    metadata = {
        "r10_best": r10_selected["selected"],
        "r11_selected": r11_out,
        "point_v3": point_f1,
        "server_safe_auc": float(server_auc),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("R10 best:")
    print(json.dumps(r10_selected["selected"], indent=2))
    print("R11 selected:")
    print(json.dumps(r11_out, indent=2))
    print(f"wrote {args.action_class_report}, {args.action_bias_report}, {args.compat_report}")


if __name__ == "__main__":
    main()
