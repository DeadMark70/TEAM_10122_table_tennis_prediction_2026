"""R20 top-k action candidate reranker.

Instead of hard-overriding rare classes, this trains a fold-safe meta-ranker
over the base model's top-k action candidates. The final prediction is
restricted to the top-k candidate set.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, bin_masks


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
    parser = argparse.ArgumentParser(description="Run R20 action top-k candidate reranker.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r19-oof", default="oof_proba_r19.pkl")
    parser.add_argument("--r19-selected", default="r19_selected.json")
    parser.add_argument("--r1-feature-report", default="feature_report_r1.json")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--rerank-report", default="r20_candidate_rerank_report.csv")
    parser.add_argument("--class-report", default="r20_action_class_report.csv")
    parser.add_argument("--selected", default="r20_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r20.json")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def assign_folds(meta: pd.DataFrame, fold_report: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    folds = []
    for _, row in fold_report[fold_report["valid_rows"].gt(0)][["fold", "valid_rows"]].iterrows():
        folds.extend([int(row["fold"])] * int(row["valid_rows"]))
    if len(folds) != len(out):
        raise ValueError("fold assignment mismatch")
    out["fold"] = folds
    return out


def adjusted_log_scores(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict, mode: str) -> np.ndarray:
    score = np.log(np.clip(prob, 1e-12, 1.0))
    for label, mask in bin_masks(meta, mode):
        mult = np.asarray(multipliers[label], dtype=float)
        score[mask] += np.log(np.clip(mult, 1e-6, None))[None, :]
    return score


def build_components(v3, v5, v7, v10, r7, r19, selected_v10, selected_r8, selected_r19):
    v3_action, v3_point, v3_server = compose_v3(v3)
    r7_action, _, _ = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    v10b_safe = blend_probs(r1_action, v10["v10_action"], float(selected_v10["action_v10_weight"]))
    r8_action = blend_probs(v10b_safe, r7_action, float(selected_r8["r7_weight"]))
    r19_action = blend_probs(r1_action, r19["style_action"], float(selected_r19["action_blend"]))
    r19_point = blend_probs(v3_point, r19["style_point"], float(selected_r19["point_blend"]))
    server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]
    return {
        "v3": v3_action,
        "v5": v5["gru_action"],
        "v7": v7["tr_action"],
        "v10": v10["v10_action"],
        "r7": r7_action,
        "r1": r1_action,
        "v10b_safe": v10b_safe,
        "r8": r8_action,
        "r19_style": r19["style_action"],
        "r19": r19_action,
    }, r19_point, server


def action_family(c: int) -> int:
    if c == 0:
        return 0
    if 1 <= c <= 7:
        return 1
    if c in {8, 9, 10, 11}:
        return 2
    if c in {12, 13, 14}:
        return 3
    return 4


def make_candidate_frame(meta: pd.DataFrame, scores: np.ndarray, components: dict[str, np.ndarray], top_k: int) -> pd.DataFrame:
    order = np.argsort(-scores, axis=1)[:, :top_k]
    rows = []
    y = meta["next_actionId"].to_numpy(dtype=int)
    prefix = meta["prefix_len"].to_numpy(dtype=int)
    for i in range(len(meta)):
        top_scores = scores[i, order[i]]
        for rank, cand in enumerate(order[i]):
            row = {
                "row_id": i,
                "fold": int(meta.iloc[i]["fold"]),
                "candidate": int(cand),
                "candidate_family": action_family(int(cand)),
                "rank": rank + 1,
                "prefix_len": int(prefix[i]),
                "prefix_bin": 1 if prefix[i] == 1 else (2 if prefix[i] == 2 else 3),
                "base_score": float(scores[i, cand]),
                "base_score_gap_to_top": float(top_scores[0] - scores[i, cand]),
                "label": int(cand == y[i]),
            }
            for name, prob in components.items():
                row[f"{name}_prob"] = float(prob[i, cand])
                row[f"{name}_rank"] = int(np.where(np.argsort(-prob[i]) == cand)[0][0] + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def train_meta_oof(candidates: pd.DataFrame, top_k: int) -> tuple[np.ndarray, pd.DataFrame]:
    features = [c for c in candidates.columns if c not in {"row_id", "fold", "label"}]
    cand_score = np.zeros(len(candidates), dtype=float)
    rows = []
    for fold in sorted(candidates["fold"].unique()):
        train = candidates[candidates["fold"].ne(fold)]
        valid = candidates[candidates["fold"].eq(fold)]
        pos = max(int(train["label"].sum()), 1)
        neg = max(len(train) - pos, 1)
        weights = np.where(train["label"].eq(1), 0.5 / pos, 0.5 / neg)
        weights = weights * len(train) / weights.sum()
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=1000 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(train[features], train["label"], sample_weight=weights)
        pred = model.predict_proba(valid[features])[:, 1]
        cand_score[valid.index.to_numpy()] = pred
        rows.append({"fold": int(fold), "valid_candidates": len(valid), "valid_positive_rate": float(valid["label"].mean())})
    return cand_score, pd.DataFrame(rows)


def choose_predictions(candidates: pd.DataFrame, cand_score: np.ndarray) -> np.ndarray:
    tmp = candidates[["row_id", "candidate"]].copy()
    tmp["score"] = cand_score
    best = tmp.sort_values(["row_id", "score"], ascending=[True, False]).groupby("row_id", sort=False).head(1)
    pred = np.zeros(int(candidates["row_id"].max()) + 1, dtype=int)
    pred[best["row_id"].to_numpy(dtype=int)] = best["candidate"].to_numpy(dtype=int)
    return pred


def choose_predictions_with_base(candidates: pd.DataFrame, cand_score: np.ndarray, eta: float) -> np.ndarray:
    tmp = candidates[["row_id", "candidate", "base_score"]].copy()
    meta_logit = np.log(np.clip(cand_score, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - cand_score, 1e-6, 1.0))
    tmp["score"] = tmp["base_score"].to_numpy(dtype=float) + eta * meta_logit
    best = tmp.sort_values(["row_id", "score"], ascending=[True, False]).groupby("row_id", sort=False).head(1)
    pred = np.zeros(int(candidates["row_id"].max()) + 1, dtype=int)
    pred[best["row_id"].to_numpy(dtype=int)] = best["candidate"].to_numpy(dtype=int)
    return pred


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
    r1_report = json.loads(Path(args.r1_feature_report).read_text(encoding="utf-8"))
    meta = assign_folds(normalize_meta(v3["valid_meta"]), v3["fold_report"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7), ("R19", r19)]:
        assert_aligned(meta.drop(columns=["fold"]), oof["valid_meta"], name)
    components, point_prob, server_prob = build_components(v3, v5, v7, v10, r7, r19, selected_v10, selected_r8, selected_r19)
    r19_scores = adjusted_log_scores(meta, components["r19"], selected_r19["action_multipliers"], "two")
    r19_base_pred = np.asarray(ACTION_CLASSES)[np.argmax(r19_scores, axis=1)]
    candidates = make_candidate_frame(meta, r19_scores, components, args.top_k)
    cand_score, fold_report = train_meta_oof(candidates, args.top_k)
    pure_pred = choose_predictions(candidates, cand_score)
    base_f1 = f1_score(meta["next_actionId"], r19_base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_pred = apply_segmented_multipliers(meta, point_prob, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    rows = [
        {"variant": "r19_base", "eta": 0.0, "action_macro_f1": base_f1, "point_macro_f1": point_f1, "server_auc": server_auc, "overall": 0.4*base_f1+0.4*point_f1+0.2*server_auc, "churn_vs_r19": 0.0},
    ]
    preds = {"r19_base": r19_base_pred, "r20_pure_meta": pure_pred}
    pure_f1 = f1_score(meta["next_actionId"], pure_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    rows.append({"variant": "r20_pure_meta", "eta": np.nan, "action_macro_f1": pure_f1, "point_macro_f1": point_f1, "server_auc": server_auc, "overall": 0.4*pure_f1+0.4*point_f1+0.2*server_auc, "churn_vs_r19": float((pure_pred != r19_base_pred).mean())})
    for eta in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]:
        eta_pred = choose_predictions_with_base(candidates, cand_score, eta)
        eta_f1 = f1_score(meta["next_actionId"], eta_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        rows.append({"variant": "r20_base_plus_meta", "eta": eta, "action_macro_f1": eta_f1, "point_macro_f1": point_f1, "server_auc": server_auc, "overall": 0.4*eta_f1+0.4*point_f1+0.2*server_auc, "churn_vs_r19": float((eta_pred != r19_base_pred).mean())})
        preds[f"r20_eta_{eta:.2f}"] = eta_pred
    report = pd.DataFrame(rows)
    report.to_csv(args.rerank_report, index=False)
    class_report(meta, preds).to_csv(args.class_report, index=False)
    eligible = report[(report["variant"].ne("r19_base")) & (report["churn_vs_r19"].le(0.12))]
    if len(eligible):
        best = eligible.sort_values("overall", ascending=False).iloc[0].to_dict()
    else:
        best = report.sort_values("overall", ascending=False).iloc[0].to_dict()
    selected = {
        "selected": best,
        "base_action_f1": float(base_f1),
        "gain_vs_r19": float(best["action_macro_f1"] - base_f1),
        "submit_recommendation": bool(best["overall"] >= 0.317 and best["action_macro_f1"] - base_f1 >= 0.001 and best["churn_vs_r19"] <= 0.08),
        "protocol": "fold-safe top-k candidate meta-ranker on OOF probabilities",
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    Path(args.feature_report).write_text(json.dumps({"selected": selected, "fold_report": fold_report.to_dict(orient="records")}, indent=2), encoding="utf-8")
    print("R20 selected:")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
