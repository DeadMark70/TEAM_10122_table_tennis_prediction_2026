"""R25 action semantic smoothing audit.

R25 tests whether a fixed table-tennis action similarity matrix can improve
Macro-F1 by redistributing a small amount of probability mass among semantically
nearby action classes. This is a post-hoc OOF diagnostic; point/server are kept
fixed to the current action-focused base branch.
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
from analysis_r20_action_candidate_reranker import build_components
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers


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


ACTION_TO_TECHNIQUE = {
    0: "unknown",
    1: "loop",
    2: "loop",
    3: "smash",
    4: "flick",
    5: "loop",
    6: "push",
    7: "flick",
    8: "pips_push",
    9: "pips_push",
    10: "push",
    11: "drop",
    12: "chop",
    13: "block",
    14: "lob",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

ACTION_TO_FAMILY = {
    0: "unknown",
    1: "attack",
    2: "attack",
    3: "attack",
    4: "attack",
    5: "attack",
    6: "attack",
    7: "attack",
    8: "control",
    9: "control",
    10: "control",
    11: "control",
    12: "defensive",
    13: "defensive",
    14: "defensive",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

MANUAL_SIMILAR = {
    1: {2: 0.7, 5: 0.55, 3: 0.20, 4: 0.15},
    2: {1: 0.7, 5: 0.50, 13: 0.20},
    3: {1: 0.20, 2: 0.20, 5: 0.20},
    4: {7: 0.75, 1: 0.15, 10: 0.10, 11: 0.10},
    5: {1: 0.55, 2: 0.50, 13: 0.25},
    6: {10: 0.60, 11: 0.45, 8: 0.35, 9: 0.35},
    7: {4: 0.75, 10: 0.15, 11: 0.15},
    8: {9: 0.85, 10: 0.45, 6: 0.35, 11: 0.25},
    9: {8: 0.85, 10: 0.45, 6: 0.35, 11: 0.25},
    10: {6: 0.60, 11: 0.60, 8: 0.45, 9: 0.45, 12: 0.15},
    11: {10: 0.60, 6: 0.45, 8: 0.25, 9: 0.25, 4: 0.10, 7: 0.10},
    12: {13: 0.35, 10: 0.15, 14: 0.15},
    13: {12: 0.35, 2: 0.20, 5: 0.25, 14: 0.10},
    14: {13: 0.15, 12: 0.15},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R25 action semantic smoothing.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r19-oof", default="oof_proba_r19.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--r19-selected", default="r19_selected.json")
    parser.add_argument("--summary", default="r25_semantic_smoothing_report.csv")
    parser.add_argument("--class-report", default="r25_action_class_report.csv")
    parser.add_argument("--matrix-report", default="r25_action_similarity_matrix.csv")
    parser.add_argument("--selected", default="r25_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r25.json")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, None)
    return prob / prob.sum(axis=1, keepdims=True)


def build_similarity_matrix(kind: str, self_weight: float, same_tech: float, same_family: float) -> np.ndarray:
    n = len(ACTION_CLASSES)
    mat = np.zeros((n, n), dtype=float)
    for i, src in enumerate(ACTION_CLASSES):
        mat[i, i] = self_weight
        for j, dst in enumerate(ACTION_CLASSES):
            if src == dst:
                continue
            if ACTION_TO_TECHNIQUE[src] == "serve" or ACTION_TO_TECHNIQUE[dst] == "serve":
                continue
            if kind in {"technique", "family"} and ACTION_TO_TECHNIQUE[src] == ACTION_TO_TECHNIQUE[dst]:
                mat[i, j] += same_tech
            if kind == "family" and ACTION_TO_FAMILY[src] == ACTION_TO_FAMILY[dst]:
                mat[i, j] += same_family
            if kind == "manual":
                mat[i, j] += MANUAL_SIMILAR.get(src, {}).get(dst, 0.0)
    return normalize_rows(mat)


def semantic_smooth(prob: np.ndarray, mat: np.ndarray, lam: float, method: str) -> np.ndarray:
    prior = normalize_rows(prob @ mat)
    if method == "arith":
        return normalize_rows((1.0 - lam) * prob + lam * prior)
    logp = (1.0 - lam) * np.log(np.clip(prob, 1e-12, 1.0)) + lam * np.log(np.clip(prior, 1e-12, 1.0))
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows(np.exp(logp))


def evaluate(meta, action_prob, point_prob, server_prob, action_mult, point_mult, mode) -> dict[str, float]:
    action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, mode)
    action = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action),
        "point_macro_f1": float(point),
        "server_auc": float(server),
        "overall": float(0.4 * action + 0.4 * point + 0.2 * server),
    }


def class_report(meta: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for name, pred in preds.items():
        for cls in ACTION_CLASSES:
            support = int((y == cls).sum())
            pred_count = int((pred == cls).sum())
            tp = int(((y == cls) & (pred == cls)).sum())
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rows.append({"model": name, "actionId": cls, "support": support, "pred_count": pred_count, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r7 = load_pickle(args.r7_oof)
    r19 = load_pickle(args.r19_oof)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    selected_r8 = json.loads(Path(args.r8_selected).read_text(encoding="utf-8"))
    selected_r19 = json.loads(Path(args.r19_selected).read_text(encoding="utf-8"))["selected"]
    meta = normalize_meta(v3["valid_meta"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7), ("R19", r19)]:
        assert_aligned(meta, oof["valid_meta"], name)
    components, r19_point, server = build_components(v3, v5, v7, v10, r7, r19, selected_v10, selected_r8, selected_r19)
    v3_action, v3_point, v3_server = compose_v3(v3)
    bases = {
        "v3": (v3_action, v3_point, v3_server, v3["tuning"].action_multipliers, v3["tuning"].point_multipliers, v3["tuning"].bins_mode),
        "r8": (components["r8"], v3_point, server, selected_r8["action_multipliers"], v3["tuning"].point_multipliers, "two"),
        "r19": (components["r19"], r19_point, server, selected_r19["action_multipliers"], v3["tuning"].point_multipliers, "two"),
        "v10b_safe": (components["v10b_safe"], v3_point, server, selected_v10["action_multipliers"], v3["tuning"].point_multipliers, "two"),
    }

    rows = []
    preds = {}
    matrices = []
    for base_name, (base_action, point_prob, server_prob, base_action_mult, point_mult, mode) in bases.items():
        base_metrics = evaluate(meta, base_action, point_prob, server_prob, base_action_mult, point_mult, mode)
        base_pred = apply_segmented_multipliers(meta, base_action, base_action_mult, ACTION_CLASSES, mode)
        preds[f"{base_name}_base"] = base_pred
        rows.append(
            {
                "base": base_name,
                "kind": "none",
                "method": "none",
                "lambda": 0.0,
                "self_weight": 1.0,
                "same_tech": 0.0,
                "same_family": 0.0,
                "churn_vs_base": 0.0,
                **base_metrics,
            }
        )
        for kind in ["technique", "family", "manual"]:
            for self_weight in [1.0, 2.0, 4.0]:
                for same_tech in [0.1, 0.25, 0.5]:
                    for same_family in ([0.05, 0.1, 0.2] if kind == "family" else [0.0]):
                        mat = build_similarity_matrix(kind, self_weight, same_tech, same_family)
                        if base_name == "v3":
                            matrix_row = pd.DataFrame(mat, columns=[f"to_{c}" for c in ACTION_CLASSES])
                            matrix_row.insert(0, "from_actionId", ACTION_CLASSES)
                            matrix_row.insert(0, "kind", kind)
                            matrix_row.insert(1, "self_weight", self_weight)
                            matrix_row.insert(2, "same_tech", same_tech)
                            matrix_row.insert(3, "same_family", same_family)
                            matrices.append(matrix_row)
                        for method in ["arith", "geom"]:
                            for lam in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
                                smoothed = semantic_smooth(base_action, mat, lam, method)
                                action_mult = tune_segmented_multipliers(meta, smoothed, ACTION_CLASSES, "action", mode)
                                metrics = evaluate(meta, smoothed, point_prob, server_prob, action_mult, point_mult, mode)
                                pred = apply_segmented_multipliers(meta, smoothed, action_mult, ACTION_CLASSES, mode)
                                churn = float((pred != base_pred).mean())
                                rows.append(
                                    {
                                        "base": base_name,
                                        "kind": kind,
                                        "method": method,
                                        "lambda": lam,
                                        "self_weight": self_weight,
                                        "same_tech": same_tech,
                                        "same_family": same_family,
                                        "churn_vs_base": churn,
                                        **metrics,
                                    }
                                )
                                key = f"{base_name}_{kind}_{method}_lam{lam:g}_self{self_weight:g}_tech{same_tech:g}_fam{same_family:g}"
                                if metrics["overall"] >= base_metrics["overall"] + 0.001:
                                    preds[key] = pred

    report = pd.DataFrame(rows).sort_values("overall", ascending=False)
    report.to_csv(args.summary, index=False)
    if matrices:
        pd.concat(matrices, ignore_index=True).to_csv(args.matrix_report, index=False)
    eligible = report[(report["kind"].ne("none")) & (report["churn_vs_base"].le(0.08))]
    best = (eligible if len(eligible) else report).sort_values("overall", ascending=False).iloc[0].to_dict()
    base_row = report[(report["base"].eq(best["base"])) & (report["kind"].eq("none"))].iloc[0]
    selected = {
        "selected": best,
        "base_overall": float(base_row["overall"]),
        "base_action_macro_f1": float(base_row["action_macro_f1"]),
        "gain_vs_base": float(best["overall"] - base_row["overall"]),
        "submit_recommendation": bool(best["overall"] >= base_row["overall"] + 0.0015 and best["churn_vs_base"] <= 0.06),
        "protocol": "fixed semantic action probability smoothing with retuned action multipliers",
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    Path(args.feature_report).write_text(json.dumps({"selected": selected}, indent=2), encoding="utf-8")
    class_report(meta, preds).to_csv(args.class_report, index=False)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
