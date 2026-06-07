"""R26 enhanced action candidate reranker.

R26 extends R20 by adding two semantic/external components to each top-k action
candidate:
- R25 semantic-smoothed action probabilities.
- R22 OpenTTGames canonical transition priors and V3+prior probabilities.

Point and server stay fixed to the same R19/R20 diagnostic branches. This is an
OOF diagnostic only; a separate generator writes test submissions if useful.
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
from analysis_r20_action_candidate_reranker import (
    adjusted_log_scores,
    assign_folds,
    build_components,
    choose_predictions,
    choose_predictions_with_base,
    class_report,
    make_candidate_frame,
    train_meta_oof,
)
from analysis_r22_opentt_transition_prior import (
    arithmetic_blend,
    estimate_external_priors,
    external_action_prior_for_rows,
    geometric_blend,
)
from analysis_r25_action_semantic_smoothing import build_similarity_matrix, semantic_smooth
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, build_train_prefix_table
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers


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
    parser = argparse.ArgumentParser(description="Run R26 semantic/external enhanced action reranker.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--events", default="external_data/openttgames/processed/openttgames_events.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r19-oof", default="oof_proba_r19.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--r19-selected", default="r19_selected.json")
    parser.add_argument("--r22-selected", default="r22_selected.json")
    parser.add_argument("--r25-selected", default="r25_selected.json")
    parser.add_argument("--rerank-report", default="r26_candidate_rerank_report.csv")
    parser.add_argument("--class-report", default="r26_action_class_report.csv")
    parser.add_argument("--selected", default="r26_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r26.json")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, None)
    return prob / prob.sum(axis=1, keepdims=True)


def recover_lag0_meta(train_path: str, meta: pd.DataFrame) -> pd.DataFrame:
    train = add_role_and_score_features(pd.read_csv(train_path))
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
    small = prefix_df[
        ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint", "lag0_actionId"]
    ].copy()
    merged = meta.merge(
        small,
        on=["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"],
        how="left",
        validate="one_to_one",
    )
    if merged["lag0_actionId"].isna().any():
        raise ValueError("Could not recover lag0_actionId for all OOF rows.")
    merged["lag0_actionId"] = merged["lag0_actionId"].astype(int)
    return merged


def make_r25_component(r19_action: np.ndarray, selected_r25: dict) -> np.ndarray:
    cfg = selected_r25.get("selected", selected_r25)
    mat = build_similarity_matrix(
        str(cfg["kind"]),
        float(cfg["self_weight"]),
        float(cfg["same_tech"]),
        float(cfg["same_family"]),
    )
    return semantic_smooth(r19_action, mat, float(cfg["lambda"]), str(cfg["method"]))


def make_r22_components(meta_lag0: pd.DataFrame, v3_action: np.ndarray, events_path: str, selected_r22: dict) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(selected_r22["alpha"])
    mix_tech = float(selected_r22["mix_tech"])
    prior_weight = float(selected_r22["prior_weight"])
    method = str(selected_r22["method"])
    priors = estimate_external_priors(pd.read_csv(events_path), alpha)
    prior = external_action_prior_for_rows(meta_lag0, priors, mix_tech=mix_tech)
    if method == "geom":
        blend = geometric_blend(v3_action, prior, prior_weight)
    else:
        blend = arithmetic_blend(v3_action, prior, prior_weight)
    return prior, blend


def evaluate_rows(meta, point_prob, server_prob, pred, base_pred) -> dict[str, float]:
    point_pred = apply_segmented_multipliers(meta, point_prob, meta.attrs["point_mult"], POINT_CLASSES, meta.attrs["point_bins"])
    action = f1_score(meta["next_actionId"], pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action),
        "point_macro_f1": float(point),
        "server_auc": float(server),
        "overall": float(0.4 * action + 0.4 * point + 0.2 * server),
        "churn_vs_r19": float((pred != base_pred).mean()),
    }


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
    selected_r22 = json.loads(Path(args.r22_selected).read_text(encoding="utf-8"))
    selected_r25 = json.loads(Path(args.r25_selected).read_text(encoding="utf-8"))["selected"]

    meta = assign_folds(normalize_meta(v3["valid_meta"]), v3["fold_report"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7), ("R19", r19)]:
        assert_aligned(meta.drop(columns=["fold"]), oof["valid_meta"], name)
    components, point_prob, server_prob = build_components(v3, v5, v7, v10, r7, r19, selected_v10, selected_r8, selected_r19)
    v3_action, _, _ = compose_v3(v3)
    meta_lag0 = recover_lag0_meta(args.train, meta.drop(columns=["fold"]))
    r22_prior, r22_blend = make_r22_components(meta_lag0, v3_action, args.events, selected_r22)
    r25_action = make_r25_component(components["r19"], selected_r25)
    components = {
        **components,
        "r22_prior": r22_prior,
        "r22_blend": r22_blend,
        "r25": r25_action,
    }
    r19_scores = adjusted_log_scores(meta, components["r19"], selected_r19["action_multipliers"], "two")
    r19_base_pred = np.asarray(ACTION_CLASSES)[np.argmax(r19_scores, axis=1)]
    candidates = make_candidate_frame(meta, r19_scores, components, args.top_k)
    cand_score, fold_report = train_meta_oof(candidates, args.top_k)
    pure_pred = choose_predictions(candidates, cand_score)

    meta.attrs["point_mult"] = v3["tuning"].point_multipliers
    meta.attrs["point_bins"] = v3["tuning"].bins_mode
    rows = []
    base_metrics = evaluate_rows(meta, point_prob, server_prob, r19_base_pred, r19_base_pred)
    rows.append({"variant": "r19_base", "eta": 0.0, **base_metrics})
    rows.append({"variant": "r26_pure_meta", "eta": np.nan, **evaluate_rows(meta, point_prob, server_prob, pure_pred, r19_base_pred)})
    preds = {"r19_base": r19_base_pred, "r26_pure_meta": pure_pred}
    for eta in [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
        pred = choose_predictions_with_base(candidates, cand_score, eta)
        rows.append({"variant": "r26_base_plus_meta", "eta": eta, **evaluate_rows(meta, point_prob, server_prob, pred, r19_base_pred)})
        preds[f"r26_eta_{eta:.3f}"] = pred

    report = pd.DataFrame(rows)
    report.to_csv(args.rerank_report, index=False)
    class_report(meta, preds).to_csv(args.class_report, index=False)
    eligible = report[(report["variant"].ne("r19_base")) & (report["churn_vs_r19"].le(0.08))]
    best = (eligible if len(eligible) else report).sort_values("overall", ascending=False).iloc[0].to_dict()
    selected = {
        "selected": best,
        "base_action_f1": float(base_metrics["action_macro_f1"]),
        "gain_vs_r19": float(best["action_macro_f1"] - base_metrics["action_macro_f1"]),
        "submit_recommendation": bool(
            best["overall"] >= base_metrics["overall"] + 0.0015 and best["churn_vs_r19"] <= 0.06
        ),
        "protocol": "R20 top-k reranker plus R25 semantic and R22 external-prior candidate features",
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    metadata = {
        "selected": selected,
        "r22_selected": selected_r22,
        "r25_selected": selected_r25,
        "candidate_columns": [c for c in candidates.columns if c not in {"row_id", "fold", "label"}],
        "fold_report": fold_report.to_dict(orient="records"),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
