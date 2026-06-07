"""V51/R51 action-conditioned point decision experiments.

V51: point0 terminal gate residual using action/spin context.
R51: point top-k reranker that only chooses among V3 point top-k candidates.

Both keep action/server fixed in submissions and only alter point labels.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r12_rare_action_rescue import assign_folds_from_report
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUT_DIR = Path("v51_r51_point_action_conditioned")
GOLDEN_PATH = Path(r"C:\aicup\tenis_new\submission_golden_blend_probs.csv")
V47_ARTIFACT = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def point_depth(p: int) -> int:
    if p == 0:
        return 0
    if p in {1, 2, 3}:
        return 1
    if p in {4, 5, 6}:
        return 2
    return 3


def point_side(p: int) -> int:
    if p == 0:
        return 0
    return {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}[p]


def action_family(a: int) -> int:
    if a == 0:
        return 0
    if 1 <= a <= 7:
        return 1
    if a in {8, 9, 10, 11}:
        return 2
    if a in {12, 13, 14}:
        return 3
    return 4


def align_prefix_features(meta: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    if is_test:
        raw = add_role_and_score_features(test_raw)
        prefix = build_test_prefix_table(raw, 6)
    else:
        raw = add_role_and_score_features(train_raw)
        prefix = build_train_prefix_table(raw, 6)
    cols = [
        "rally_uid",
        "prefix_len",
        "lag0_actionId",
        "lag0_spinId",
        "lag0_pointId",
        "lag0_handId",
        "lag0_strengthId",
        "lag0_positionId",
        "serverScore",
        "receiverScore",
        "serverScoreDiff",
        "scoreTotal",
        "next_hitter_is_server",
        "next_strikeId_rule",
        "prefix_len_is_odd",
    ]
    merged = meta[["rally_uid", "prefix_len"]].merge(prefix[cols], on=["rally_uid", "prefix_len"], how="left")
    if merged.isna().any().any():
        bad = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Prefix feature alignment failed: {bad}")
    return merged


def build_current_oof_action() -> np.ndarray:
    v5 = load_pickle("oof_proba_v5.pkl")
    v7 = load_pickle("oof_proba_v7.pkl")
    r7 = load_pickle("oof_proba_r7.pkl")
    r7_action, _, _ = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    return normalize_rows(0.85 * r1_action + 0.05 * r7_action + 0.10 * v5["gru_action"])


def add_prob_features(base: pd.DataFrame, point_prob: np.ndarray, action_prob: np.ndarray, rare_scores: np.ndarray | None = None) -> pd.DataFrame:
    out = base.copy()
    p_order = np.argsort(-point_prob, axis=1)
    a_order = np.argsort(-action_prob, axis=1)
    out["v3_point_top1"] = p_order[:, 0]
    out["v3_point_top2"] = p_order[:, 1]
    out["v3_point_top3"] = p_order[:, 2]
    out["v3_point_top1_prob"] = point_prob[np.arange(len(point_prob)), p_order[:, 0]]
    out["v3_point_top2_prob"] = point_prob[np.arange(len(point_prob)), p_order[:, 1]]
    out["v3_point_top3_prob"] = point_prob[np.arange(len(point_prob)), p_order[:, 2]]
    out["v3_point_margin12"] = out["v3_point_top1_prob"] - out["v3_point_top2_prob"]
    out["v3_point0_prob"] = point_prob[:, 0]
    out["action_top1"] = a_order[:, 0]
    out["action_top2"] = a_order[:, 1]
    out["action_top1_prob"] = action_prob[np.arange(len(action_prob)), a_order[:, 0]]
    out["action_top2_prob"] = action_prob[np.arange(len(action_prob)), a_order[:, 1]]
    out["action_margin12"] = out["action_top1_prob"] - out["action_top2_prob"]
    out["action_family_top1"] = out["action_top1"].map(action_family)
    for c in [0, 4, 7, 8, 9, 10, 11, 12, 14]:
        out[f"action_prob_{c}"] = action_prob[:, c]
    out["last_point_depth"] = out["lag0_pointId"].astype(int).map(point_depth)
    out["last_point_side"] = out["lag0_pointId"].astype(int).map(point_side)
    if rare_scores is not None:
        for j, c in enumerate([8, 9, 12, 14]):
            out[f"rare_action_score_{c}"] = rare_scores[:, j]
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    forbidden = {"rally_uid", "prefix_len", "fold", "next_pointId", "label"}
    return [c for c in df.columns if c not in forbidden]


def make_point0_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def binary_weights(y: pd.Series) -> np.ndarray:
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    w = np.where(y.to_numpy(dtype=int) == 1, 0.5 / pos, 0.5 / neg)
    return w * len(y) / w.sum()


def train_point0_oof(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    feats = feature_cols(df)
    for fold in sorted(df["fold"].unique()):
        tr = df[df["fold"].ne(fold)]
        va = df[df["fold"].eq(fold)]
        y = tr["next_pointId"].eq(0).astype(int)
        model = make_point0_model(5100 + int(fold))
        model.fit(tr[feats], y, sample_weight=binary_weights(y))
        out[va.index.to_numpy()] = model.predict_proba(va[feats])[:, 1]
    return out


def train_point0_full(df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    feats = feature_cols(df)
    y = df["next_pointId"].eq(0).astype(int)
    model = make_point0_model(9100)
    model.fit(df[feats], y, sample_weight=binary_weights(y))
    return model.predict_proba(test_df[feats])[:, 1]


def apply_point0_gate(point_prob: np.ndarray, gate_prob: np.ndarray, strength: float) -> np.ndarray:
    out = point_prob.copy()
    # Blend the calibrated point0 gate into point0 only, preserve relative nonzero distribution.
    p0 = np.clip((1.0 - strength) * out[:, 0] + strength * gate_prob, 1e-6, 1 - 1e-6)
    nonzero = out[:, 1:]
    nonzero = nonzero / np.clip(nonzero.sum(axis=1, keepdims=True), 1e-12, None)
    out[:, 0] = p0
    out[:, 1:] = (1.0 - p0[:, None]) * nonzero
    return normalize_rows(out)


def make_candidate_frame(df: pd.DataFrame, point_prob: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    order = np.argsort(-point_prob, axis=1)[:, :top_k]
    rows = []
    for i in range(len(df)):
        for rank, cand in enumerate(order[i], start=1):
            row = {k: df.iloc[i][k] for k in feature_cols(df)}
            row.update(
                {
                    "row_id": i,
                    "fold": int(df.iloc[i]["fold"]) if "fold" in df.columns else -1,
                    "candidate": int(cand),
                    "candidate_rank": rank,
                    "candidate_prob": float(point_prob[i, cand]),
                    "candidate_logprob": float(np.log(np.clip(point_prob[i, cand], 1e-12, 1.0))),
                    "candidate_is_point0": int(cand == 0),
                    "candidate_depth": point_depth(int(cand)),
                    "candidate_side": point_side(int(cand)),
                    "label": int(cand == int(df.iloc[i]["next_pointId"])) if "next_pointId" in df.columns else 0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_ranker(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.25,
        reg_lambda=2.2,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_ranker_oof(candidates: pd.DataFrame) -> np.ndarray:
    feats = [c for c in candidates.columns if c not in {"row_id", "fold", "label"}]
    score = np.zeros(len(candidates), dtype=float)
    for fold in sorted(candidates["fold"].unique()):
        tr = candidates[candidates["fold"].ne(fold)]
        va = candidates[candidates["fold"].eq(fold)]
        y = tr["label"].astype(int)
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        w = np.where(y.to_numpy() == 1, 0.5 / pos, 0.5 / neg)
        w *= len(y) / w.sum()
        model = make_ranker(6100 + int(fold))
        model.fit(tr[feats], y, sample_weight=w)
        score[va.index.to_numpy()] = model.predict_proba(va[feats])[:, 1]
    return score


def fit_ranker_full(candidates: pd.DataFrame) -> tuple[lgb.LGBMClassifier, list[str]]:
    feats = [c for c in candidates.columns if c not in {"row_id", "fold", "label"}]
    y = candidates["label"].astype(int)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    w = np.where(y.to_numpy() == 1, 0.5 / pos, 0.5 / neg)
    w *= len(y) / w.sum()
    model = make_ranker(9600)
    model.fit(candidates[feats], y, sample_weight=w)
    return model, feats


def choose_ranker(candidates: pd.DataFrame, score: np.ndarray, eta: float) -> np.ndarray:
    tmp = candidates[["row_id", "candidate", "candidate_logprob"]].copy()
    logit = np.log(np.clip(score, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - score, 1e-6, 1.0))
    tmp["score"] = tmp["candidate_logprob"].to_numpy(dtype=float) + eta * logit
    best = tmp.sort_values(["row_id", "score"], ascending=[True, False]).groupby("row_id", sort=False).head(1)
    pred = np.zeros(int(candidates["row_id"].max()) + 1, dtype=int)
    pred[best["row_id"].to_numpy(dtype=int)] = best["candidate"].to_numpy(dtype=int)
    return pred


def write_submission(test_meta: pd.DataFrame, point_pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": current_sub["actionId"].astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "point_diff_vs_current": float(np.mean(point_pred != current_sub["pointId"].to_numpy(dtype=int))),
        "point0_count": int((point_pred == 0).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    v3 = load_pickle("oof_proba_v3.pkl")
    meta = assign_folds_from_report(normalize_meta(v3["valid_meta"]), v3["fold_report"]).reset_index(drop=True)
    _, v3_point, _ = compose_v3(v3)
    current_action = build_current_oof_action()
    art = load_pickle(V47_ARTIFACT) if V47_ARTIFACT.exists() else None
    rare_oof = art["rare_oof_scores"] if art is not None else None

    base_point_pred = apply_segmented_multipliers(meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
    base_f1 = f1_score(meta["next_pointId"], base_point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)

    train_feat = align_prefix_features(meta, is_test=False)
    train_df = add_prob_features(train_feat, v3_point, current_action, rare_oof)
    train_df["fold"] = meta["fold"].to_numpy()
    train_df["next_pointId"] = meta["next_pointId"].to_numpy(dtype=int)

    point0_oof = train_point0_oof(train_df)
    v51_rows = []
    v51_oof_probs: dict[float, np.ndarray] = {}
    for strength in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        prob = apply_point0_gate(v3_point, point0_oof, strength)
        pred = apply_segmented_multipliers(meta, prob, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
        v51_oof_probs[strength] = prob
        v51_rows.append(
            {
                "variant": "v51_point0_gate",
                "strength": strength,
                "point_macro_f1": float(f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
                "gain_vs_base": float(f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0) - base_f1),
                "point_churn_vs_base": float(np.mean(pred != base_point_pred)),
                "point0_pred_rate": float(np.mean(pred == 0)),
            }
        )
    v51_report = pd.DataFrame(v51_rows).sort_values("point_macro_f1", ascending=False)
    v51_report.to_csv(OUT_DIR / "v51_point0_gate_oof_search.csv", index=False)

    # R51 top-k reranker on the best V51 probability if it improves, otherwise V3.
    best_strength = float(v51_report.iloc[0]["strength"])
    rerank_base_prob = v51_oof_probs[best_strength] if v51_report.iloc[0]["point_macro_f1"] >= base_f1 else v3_point
    candidates = make_candidate_frame(train_df, rerank_base_prob, top_k=3)
    cand_score = train_ranker_oof(candidates)
    r51_rows = []
    for eta in [0.02, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50, 0.75, 1.00]:
        pred = choose_ranker(candidates, cand_score, eta)
        r51_rows.append(
            {
                "variant": "r51_topk_reranker",
                "eta": eta,
                "point_macro_f1": float(f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
                "gain_vs_base": float(f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0) - base_f1),
                "point_churn_vs_base": float(np.mean(pred != base_point_pred)),
                "point0_pred_rate": float(np.mean(pred == 0)),
            }
        )
    r51_report = pd.DataFrame(r51_rows).sort_values("point_macro_f1", ascending=False)
    r51_report.to_csv(OUT_DIR / "r51_topk_reranker_oof_search.csv", index=False)

    # Full-test submissions.
    test_meta = pd.read_csv("test_new.csv").groupby("rally_uid", sort=False).tail(1)[["rally_uid"]].reset_index(drop=True)
    # Use the same row order as current submission/test prefix table.
    current_sub = pd.read_csv(CURRENT_SUB_PATH)
    test_meta = current_sub[["rally_uid"]].merge(
        pd.read_csv("test_new.csv").groupby("rally_uid", sort=False).size().rename("prefix_len").reset_index(),
        on="rally_uid",
        how="left",
    )
    # Need V3 full test point probability.
    with open("oof_proba_v3.pkl", "rb") as f:
        v3_oof = pickle.load(f)
    from generate_r1_submission import compose_v3_full

    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    test_prefix, _, test_point_prob, _ = compose_v3_full(train_raw, test_raw, v3_oof["tuning"])
    test_meta = test_prefix[["rally_uid", "prefix_len"]].reset_index(drop=True)
    current_sub = test_meta.merge(current_sub, on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current submission did not align.")

    # Build test action prob using current action from V47 artifact if available.
    if art is not None:
        test_action_prob = art["current_test_action"]
        rare_test = art["rare_test_scores"]
    else:
        test_action_prob = np.zeros((len(test_meta), len(ACTION_CLASSES))) + 1.0 / len(ACTION_CLASSES)
        rare_test = None
    test_feat = align_prefix_features(test_meta, is_test=True)
    test_df = add_prob_features(test_feat, test_point_prob, test_action_prob, rare_test)
    point0_test = train_point0_full(train_df, test_df)

    generated = []
    for _, row in v51_report.head(4).iterrows():
        strength = float(row["strength"])
        prob = apply_point0_gate(test_point_prob, point0_test, strength)
        pred = apply_segmented_multipliers(test_meta, prob, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
        name = f"submission_v51_point0_gate_s{str(strength).replace('.', 'p')}_current_action_server.csv"
        info = write_submission(test_meta, pred, current_sub, name)
        info.update({"source_oof_point_f1": float(row["point_macro_f1"]), "source_oof_churn": float(row["point_churn_vs_base"])})
        generated.append(info)

    full_candidates = make_candidate_frame(train_df, rerank_base_prob, top_k=3)
    ranker, feats = fit_ranker_full(full_candidates)
    test_rerank_base = apply_point0_gate(test_point_prob, point0_test, best_strength) if v51_report.iloc[0]["point_macro_f1"] >= base_f1 else test_point_prob
    test_candidates = make_candidate_frame(test_df.assign(fold=-1, next_pointId=0), test_rerank_base, top_k=3)
    test_score = ranker.predict_proba(test_candidates[feats])[:, 1]
    for _, row in r51_report.head(5).iterrows():
        eta = float(row["eta"])
        pred = choose_ranker(test_candidates, test_score, eta)
        name = f"submission_r51_point_topk_eta{str(eta).replace('.', 'p')}_current_action_server.csv"
        info = write_submission(test_meta, pred, current_sub, name)
        info.update({"source_oof_point_f1": float(row["point_macro_f1"]), "source_oof_churn": float(row["point_churn_vs_base"])})
        generated.append(info)

    pd.DataFrame(generated).to_csv(OUT_DIR / "v51_r51_generated_candidates.csv", index=False)
    report = {
        "base_point_macro_f1": base_f1,
        "v51_best": v51_report.head(5).to_dict(orient="records"),
        "r51_best": r51_report.head(5).to_dict(orient="records"),
        "generated": generated,
        "recommendation": "Submit only if OOF gain is material and churn is acceptable; point branch has been fragile historically.",
    }
    (OUT_DIR / "v51_r51_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
