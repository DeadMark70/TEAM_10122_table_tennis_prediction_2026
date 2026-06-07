"""Analyze whether V10B adds value on top of the submitted R1 ensemble."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, load_pickle, normalize_meta, prefix_report
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, evaluate_v3, tune_segmented_multipliers


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
    parser = argparse.ArgumentParser(description="Search R1 + V10B OOF ensemble.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--summary", default="v10b_r1_ensemble_summary.csv")
    parser.add_argument("--prefix-report", default="v10b_r1_prefix_report.csv")
    parser.add_argument("--selected", default="v10b_r1_selected.json")
    return parser.parse_args()


def assert_aligned(base: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    other = normalize_meta(other)
    cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    mismatch = (base[cols].to_numpy() != other[cols].to_numpy()).any(axis=1)
    if mismatch.any():
        first = int(np.where(mismatch)[0][0])
        raise ValueError(f"{name} not aligned at row {first}.")


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    with open(args.v10b_oof, "rb") as f:
        v10 = pickle.load(f)

    meta = normalize_meta(v3["valid_meta"])
    assert_aligned(meta, v5["valid_meta"], "V5")
    assert_aligned(meta, v7["valid_meta"], "V7")
    assert_aligned(meta, v10["valid_meta"], "V10B")

    v3_action, v3_point, v3_server = compose_v3(v3)
    r1_action = 0.4 * v5["gru_action"] + 0.6 * v7["tr_action"]
    r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
    r1_point = v3_point
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]

    rows = []
    best = None
    for aw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        action_prob = blend_probs(r1_action, v10["v10_action"], aw)
        action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", "two")
        action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, "two")
        action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        for pw in [0.0, 0.05, 0.1, 0.2]:
            point_prob = blend_probs(r1_point, v10["v10_point"], pw)
            point_mult = tune_segmented_multipliers(meta, point_prob, POINT_CLASSES, "point", "two")
            point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, "two")
            point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
            for sw in [0.0, 0.1, 0.2, 0.3]:
                server_prob = (1.0 - sw) * r1_server + sw * v10["v10_server"]
                server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
                overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
                row = {
                    "action_v10_weight": aw,
                    "point_v10_weight": pw,
                    "server_v10_weight": sw,
                    "action_macro_f1": float(action_f1),
                    "point_macro_f1": float(point_f1),
                    "server_auc": float(server_auc),
                    "overall": float(overall),
                }
                rows.append(row)
                if best is None or overall > best["overall"]:
                    best = {
                        **row,
                        "action_prob": action_prob,
                        "point_prob": point_prob,
                        "server_prob": server_prob,
                        "action_mult": action_mult,
                        "point_mult": point_mult,
                        "action_pred": action_pred,
                        "point_pred": point_pred,
                    }

    summary = pd.DataFrame(rows).sort_values("overall", ascending=False)
    summary.to_csv(args.summary, index=False)
    prefix = prefix_report(meta, best["action_pred"], best["point_pred"], best["server_prob"])
    prefix.to_csv(args.prefix_report, index=False)
    selected = {
        k: v
        for k, v in best.items()
        if k
        in {
            "action_v10_weight",
            "point_v10_weight",
            "server_v10_weight",
            "action_macro_f1",
            "point_macro_f1",
            "server_auc",
            "overall",
        }
    }
    selected["action_multipliers"] = best["action_mult"]
    selected["point_multipliers"] = best["point_mult"]
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(summary.head(10).to_string(index=False))
    print("best", json.dumps({k: selected[k] for k in selected if not k.endswith("multipliers")}, indent=2))


if __name__ == "__main__":
    main()
