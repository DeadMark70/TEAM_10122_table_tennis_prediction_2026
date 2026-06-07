"""R8 action-only audit for R7 phase-aware features.

This uses existing OOF probabilities only. The point branch is fixed to V3/R1
and the server branch is fixed to the V10B-safe setting. Only action
probabilities are blended and retuned.
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

from analysis_r1_oof_ensemble import compose_v3, normalize_meta, prefix_report
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R8 action-only ensemble audit.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--summary", default="r8_action_only_summary.csv")
    parser.add_argument("--prefix-report", default="r8_action_only_prefix_report.csv")
    parser.add_argument("--selected", default="r8_action_only_selected.json")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    return prob / prob.sum(axis=1, keepdims=True)


def evaluate_action_variant(
    name: str,
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    point_mult: dict[str, list[float]],
    server_prob: np.ndarray,
) -> dict[str, object]:
    action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", "two")
    action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, "two")
    point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, "two")
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
    return {
        "variant": name,
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(overall),
        "action_mult": action_mult,
        "action_pred": action_pred,
        "point_pred": point_pred,
    }


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r7 = load_pickle(args.r7_oof)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))

    meta = normalize_meta(v3["valid_meta"])
    assert_aligned(meta, v5["valid_meta"], "V5")
    assert_aligned(meta, v7["valid_meta"], "V7")
    assert_aligned(meta, v10["valid_meta"], "V10B")
    assert_aligned(meta, r7["valid_meta"], "R7")

    v3_action, v3_point, v3_server = compose_v3(v3)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    safe_action = blend_probs(r1_action, v10["v10_action"], float(selected_v10["action_v10_weight"]))
    safe_server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]
    r7_action, _, _ = compose_v3(r7)

    candidates: list[dict[str, object]] = []
    base_variants = {
        "r1_action": r1_action,
        "v10b_safe_action": safe_action,
        "r7_action": r7_action,
    }
    for name, action_prob in base_variants.items():
        candidates.append(evaluate_action_variant(name, meta, action_prob, v3_point, v3["tuning"].point_multipliers, safe_server))

    for weight in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]:
        action_prob = blend_probs(safe_action, r7_action, weight)
        row = evaluate_action_variant(
            f"safe_plus_r7_w{weight:g}",
            meta,
            action_prob,
            v3_point,
            v3["tuning"].point_multipliers,
            safe_server,
        )
        row["r7_weight"] = float(weight)
        candidates.append(row)

    public_rows = []
    for row in candidates:
        public_rows.append(
            {
                "variant": row["variant"],
                "r7_weight": row.get("r7_weight", np.nan),
                "action_macro_f1": row["action_macro_f1"],
                "point_macro_f1": row["point_macro_f1"],
                "server_auc": row["server_auc"],
                "overall": row["overall"],
            }
        )
    summary = pd.DataFrame(public_rows).sort_values("overall", ascending=False).reset_index(drop=True)
    summary.to_csv(args.summary, index=False)

    best_idx = int(summary.index[0])
    best_name = str(summary.iloc[0]["variant"])
    best = next(row for row in candidates if row["variant"] == best_name)
    prefix = prefix_report(meta, best["action_pred"], best["point_pred"], safe_server)
    prefix.to_csv(args.prefix_report, index=False)
    point_churn_vs_v3 = float((best["point_pred"] != apply_segmented_multipliers(meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, "two")).mean())
    action_pred_safe = next(row for row in candidates if row["variant"] == "v10b_safe_action")["action_pred"]
    action_churn_vs_safe = float((best["action_pred"] != action_pred_safe).mean())
    selected = {
        "best_variant": best_name,
        "metrics": {
            "action_macro_f1": best["action_macro_f1"],
            "point_macro_f1": best["point_macro_f1"],
            "server_auc": best["server_auc"],
            "overall": best["overall"],
        },
        "r7_weight": float(summary.iloc[0]["r7_weight"]) if not pd.isna(summary.iloc[0]["r7_weight"]) else None,
        "action_multipliers": best["action_mult"],
        "point_policy": "fixed_v3_point_probabilities_and_v3_point_multipliers",
        "server_policy": "fixed_v10b_safe_server",
        "point_churn_vs_v3": point_churn_vs_v3,
        "action_churn_vs_v10b_safe": action_churn_vs_safe,
        "submit_recommendation": bool(best["overall"] >= 0.316 and point_churn_vs_v3 == 0.0),
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("selected", json.dumps(selected["metrics"] | {"variant": best_name}, indent=2))
    print(f"wrote {args.summary}")
    print(f"wrote {args.prefix_report}")
    print(f"wrote {args.selected}")


if __name__ == "__main__":
    main()
