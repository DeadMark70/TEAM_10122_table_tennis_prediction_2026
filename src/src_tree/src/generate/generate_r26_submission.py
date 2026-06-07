"""Generate R26 semantic/external enhanced reranker submissions."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from analysis_r1_oof_ensemble import normalize_meta
from analysis_r20_action_candidate_reranker import (
    adjusted_log_scores,
    assign_folds,
    build_components,
    choose_predictions_with_base,
    make_candidate_frame,
)
from analysis_r22_opentt_transition_prior import arithmetic_blend, estimate_external_priors, external_action_prior_for_rows, geometric_blend
from analysis_r25_action_semantic_smoothing import build_similarity_matrix, semantic_smooth
from analysis_r26_semantic_external_reranker import make_r22_components, make_r25_component, recover_lag0_meta
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, validate_raw_data
from baseline_v3 import apply_segmented_multipliers
from generate_r1_submission import compose_v3_full
from generate_r19_submission import build_style_full, normalize_rows
from generate_r8_submission import build_r7_full_action


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
    parser = argparse.ArgumentParser(description="Generate R26 submissions.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--events", default="external_data/openttgames/processed/openttgames_events.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r19-oof", default="oof_proba_r19.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--r19-selected", default="r19_selected.json")
    parser.add_argument("--r22-selected", default="r22_selected.json")
    parser.add_argument("--r25-selected", default="r25_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r26_submission.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--etas", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth-k", type=float, default=50.0)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def train_meta_model(candidates: pd.DataFrame, args: argparse.Namespace) -> tuple[lgb.LGBMClassifier, list[str]]:
    features = [c for c in candidates.columns if c not in {"row_id", "fold", "label"}]
    pos = max(int(candidates["label"].sum()), 1)
    neg = max(len(candidates) - pos, 1)
    weights = np.where(candidates["label"].eq(1), 0.5 / pos, 0.5 / neg)
    weights = weights * len(candidates) / weights.sum()
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=2600 + args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(candidates[features], candidates["label"], sample_weight=weights)
    return model, features


def make_test_candidate_frame(test_meta: pd.DataFrame, scores: np.ndarray, components: dict[str, np.ndarray], top_k: int) -> pd.DataFrame:
    tmp = test_meta.copy()
    tmp["next_actionId"] = 0
    tmp["fold"] = 0
    cand = make_candidate_frame(tmp, scores, components, top_k)
    cand["label"] = 0
    return cand


def write_submission(path: str, meta: pd.DataFrame, action_pred: np.ndarray, point_pred: np.ndarray, server_prob: np.ndarray) -> None:
    sub = pd.DataFrame(
        {
            "rally_uid": meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != len(meta):
        raise ValueError("Submission row count mismatch.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    sub.to_csv(path, index=False, float_format="%.8f")


def make_test_r22_components(test_prefix: pd.DataFrame, v3_action: np.ndarray, events_path: str, selected_r22: dict) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(selected_r22["alpha"])
    mix_tech = float(selected_r22["mix_tech"])
    prior_weight = float(selected_r22["prior_weight"])
    method = str(selected_r22["method"])
    priors = estimate_external_priors(pd.read_csv(events_path), alpha)
    prior = external_action_prior_for_rows(test_prefix, priors, mix_tech=mix_tech)
    blend = geometric_blend(v3_action, prior, prior_weight) if method == "geom" else arithmetic_blend(v3_action, prior, prior_weight)
    return prior, blend


def make_test_r25_component(r19_action: np.ndarray, selected_r25: dict) -> np.ndarray:
    cfg = selected_r25.get("selected", selected_r25)
    mat = build_similarity_matrix(str(cfg["kind"]), float(cfg["self_weight"]), float(cfg["same_tech"]), float(cfg["same_family"]))
    return semantic_smooth(r19_action, mat, float(cfg["lambda"]), str(cfg["method"]))


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

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
    components, _, _ = __import__("analysis_r20_action_candidate_reranker").build_components(
        v3, v5, v7, v10, r7, r19, selected_v10, selected_r8, selected_r19
    )
    v3_oof_action = components["v3"]
    meta_lag0 = recover_lag0_meta(args.train, meta.drop(columns=["fold"]))
    r22_prior, r22_blend = make_r22_components(meta_lag0, v3_oof_action, args.events, selected_r22)
    r25_action = make_r25_component(components["r19"], selected_r25)
    oof_components = {**components, "r22_prior": r22_prior, "r22_blend": r22_blend, "r25": r25_action}
    oof_scores = adjusted_log_scores(meta, oof_components["r19"], selected_r19["action_multipliers"], "two")
    oof_candidates = make_candidate_frame(meta, oof_scores, oof_components, args.top_k)
    model, features = train_meta_model(oof_candidates, args)

    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10_full = pickle.load(f)
    test_prefix, v3_action, v3_point, v3_server = compose_v3_full(train, test, v3["tuning"])
    r7_prefix, r7_action = build_r7_full_action(train, test, r7["tuning"])
    style_meta, style_action, style_point, style_features = build_style_full(train, test, args)
    if not test_prefix["rally_uid"].reset_index(drop=True).equals(r7_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("R7 and V3 test rows are not aligned.")
    if not test_prefix["rally_uid"].reset_index(drop=True).equals(style_meta["rally_uid"].reset_index(drop=True)):
        raise ValueError("R19 style and V3 test rows are not aligned.")

    r1_action = normalize_rows(0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]
    v10b_safe = normalize_rows((1.0 - float(selected_v10["action_v10_weight"])) * r1_action + float(selected_v10["action_v10_weight"]) * v10_full["v10_action"])
    r8_action = normalize_rows((1.0 - float(selected_r8["r7_weight"])) * v10b_safe + float(selected_r8["r7_weight"]) * r7_action)
    r19_action = normalize_rows((1.0 - float(selected_r19["action_blend"])) * r1_action + float(selected_r19["action_blend"]) * style_action)
    r22_prior_test, r22_blend_test = make_test_r22_components(test_prefix, v3_action, args.events, selected_r22)
    r25_test = make_test_r25_component(r19_action, selected_r25)
    server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(selected_v10["server_v10_weight"]) * v10_full["v10_server"]
    test_components = {
        "v3": v3_action,
        "v5": r1_seq["gru_action"],
        "v7": r1_seq["tr_action"],
        "v10": v10_full["v10_action"],
        "r7": r7_action,
        "r1": r1_action,
        "v10b_safe": v10b_safe,
        "r8": r8_action,
        "r19_style": style_action,
        "r19": r19_action,
        "r22_prior": r22_prior_test,
        "r22_blend": r22_blend_test,
        "r25": r25_test,
    }
    test_scores = adjusted_log_scores(test_prefix, r19_action, selected_r19["action_multipliers"], "two")
    test_candidates = make_test_candidate_frame(test_prefix, test_scores, test_components, args.top_k)
    cand_score = model.predict_proba(test_candidates[features])[:, 1]

    v3_point_pred = apply_segmented_multipliers(test_prefix, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
    r19_point = normalize_rows((1.0 - float(selected_r19["point_blend"])) * v3_point + float(selected_r19["point_blend"]) * style_point)
    r19_point_pred = apply_segmented_multipliers(test_prefix, r19_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)

    outputs = []
    base_pred = np.asarray(ACTION_CLASSES)[np.argmax(test_scores, axis=1)]
    for eta in args.etas:
        pred = choose_predictions_with_base(test_candidates, cand_score, eta)
        eta_tag = f"{eta:.3f}".rstrip("0").rstrip(".").replace(".", "")
        safe_path = f"submission_r26_eta{eta_tag}_safe_point.csv"
        r19_path = f"submission_r26_eta{eta_tag}_r19_point.csv"
        write_submission(safe_path, test_prefix, pred, v3_point_pred, server)
        write_submission(r19_path, test_prefix, pred, r19_point_pred, server)
        outputs.append(
            {
                "eta": float(eta),
                "safe_point_submission": safe_path,
                "r19_point_submission": r19_path,
                "action_diff_vs_r19_base": float((pred != base_pred).mean()),
                "point_diff_safe_vs_r19": float((v3_point_pred != r19_point_pred).mean()),
            }
        )

    metadata = {
        "source": "R26 semantic/external enhanced top-k reranker",
        "etas": outputs,
        "top_k": args.top_k,
        "candidate_features": features,
        "style_feature_count": len(style_features),
        "rows": int(len(test_prefix)),
        "recommendation": "Prefer safe_point eta=0.05 or eta=0.075 for first public probe; eta=0.10 is higher risk.",
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
