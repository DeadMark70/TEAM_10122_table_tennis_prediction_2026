"""R48 action top-k meta-stacker over V47-V50 experts.

This script consumes `v47_v50_action_experts.pkl`, builds fold-safe OOF
candidate rows, trains a LightGBM binary candidate ranker, and writes action-only
submission candidates with current R34 point/server fixed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from baseline_lgbm import ACTION_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUT_DIR = Path("r48_action_meta_stacker")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_current_oof_action() -> np.ndarray:
    v3 = load_pickle("oof_proba_v3.pkl")
    v5 = load_pickle("oof_proba_v5.pkl")
    v7 = load_pickle("oof_proba_v7.pkl")
    r7 = load_pickle("oof_proba_r7.pkl")
    meta = normalize_meta(v3["valid_meta"]).reset_index(drop=True)
    r7_action, _, _ = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    current = normalize_rows(0.85 * r1_action + 0.05 * r7_action + 0.10 * v5["gru_action"])
    if len(current) != len(meta):
        raise ValueError("Current OOF action length mismatch.")
    return current


def action_family(c: int) -> int:
    if c == 0:
        return 0
    if 1 <= c <= 7:
        return 1
    if c in {6, 8, 9, 10, 11}:
        return 2
    if c in {12, 13, 14}:
        return 3
    return 4


def ranks(prob: np.ndarray) -> np.ndarray:
    order = np.argsort(-prob, axis=1)
    out = np.empty_like(order)
    out[np.arange(len(prob))[:, None], order] = np.arange(prob.shape[1])[None, :] + 1
    return out


def margins(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-prob, axis=1)
    top1 = prob[np.arange(len(prob)), order[:, 0]]
    top2 = prob[np.arange(len(prob)), order[:, 1]]
    return top1, top1 - top2


def make_candidate_frame(
    meta: pd.DataFrame,
    expert_probs: dict[str, np.ndarray],
    rare_scores: np.ndarray,
    rare_classes: list[int],
    base_name: str,
    top_k: int,
    include_rare: bool = True,
) -> pd.DataFrame:
    base = expert_probs[base_name]
    base_order = np.argsort(-base, axis=1)[:, :top_k]
    top1_margin = {name: margins(prob) for name, prob in expert_probs.items()}
    rank_map = {name: ranks(prob) for name, prob in expert_probs.items()}
    rows = []
    y = meta["next_actionId"].to_numpy(dtype=int) if "next_actionId" in meta.columns else None
    for i in range(len(meta)):
        cands = set(int(c) for c in base_order[i])
        for name, prob in expert_probs.items():
            cands.add(int(np.argmax(prob[i])))
        if include_rare:
            for j, cls in enumerate(rare_classes):
                if rare_scores[i, j] >= np.quantile(rare_scores[:, j], 0.90):
                    cands.add(int(cls))
        for cand in sorted(cands):
            row = {
                "row_id": i,
                "fold": int(meta.iloc[i]["fold"]) if "fold" in meta.columns else -1,
                "candidate": cand,
                "candidate_family": action_family(cand),
                "prefix_len": int(meta.iloc[i]["prefix_len"]),
                "prefix_bin": 1 if int(meta.iloc[i]["prefix_len"]) == 1 else (2 if int(meta.iloc[i]["prefix_len"]) == 2 else 3),
                "base_prob": float(base[i, cand]),
                "base_rank": int(rank_map[base_name][i, cand]),
                "label": int(cand == y[i]) if y is not None else 0,
            }
            for name, prob in expert_probs.items():
                row[f"{name}_prob"] = float(prob[i, cand])
                row[f"{name}_logprob"] = float(np.log(np.clip(prob[i, cand], 1e-12, 1.0)))
                row[f"{name}_rank"] = int(rank_map[name][i, cand])
                row[f"{name}_top_prob"] = float(top1_margin[name][0][i])
                row[f"{name}_margin"] = float(top1_margin[name][1][i])
            for j, cls in enumerate(rare_classes):
                row[f"rare_score_{cls}"] = float(rare_scores[i, j])
            row["candidate_rare_score"] = float(rare_scores[i, rare_classes.index(cand)]) if cand in rare_classes else 0.0
            row["candidate_is_rare_target"] = int(cand in rare_classes)
            rows.append(row)
    return pd.DataFrame(rows)


def train_meta_oof(candidates: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    features = [c for c in candidates.columns if c not in {"row_id", "fold", "label"}]
    score = np.zeros(len(candidates), dtype=float)
    rows = []
    for fold in sorted(candidates["fold"].unique()):
        train = candidates[candidates["fold"].ne(fold)]
        valid = candidates[candidates["fold"].eq(fold)]
        pos = max(int(train["label"].sum()), 1)
        neg = max(len(train) - pos, 1)
        weights = np.where(train["label"].eq(1), 0.5 / pos, 0.5 / neg)
        weights *= len(train) / weights.sum()
        # Put mild emphasis on rare true candidates.
        weights *= np.where(train["candidate_is_rare_target"].eq(1), 1.35, 1.0)
        weights *= len(train) / weights.sum()
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=260,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=24,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=12000 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(train[features], train["label"], sample_weight=weights)
        score[valid.index.to_numpy()] = model.predict_proba(valid[features])[:, 1]
        rows.append({"fold": int(fold), "valid_candidates": int(len(valid)), "positive_rate": float(valid["label"].mean())})
    return score, pd.DataFrame(rows)


def fit_meta_full(candidates: pd.DataFrame) -> tuple[lgb.LGBMClassifier, list[str]]:
    features = [c for c in candidates.columns if c not in {"row_id", "fold", "label"}]
    pos = max(int(candidates["label"].sum()), 1)
    neg = max(len(candidates) - pos, 1)
    weights = np.where(candidates["label"].eq(1), 0.5 / pos, 0.5 / neg)
    weights *= np.where(candidates["candidate_is_rare_target"].eq(1), 1.35, 1.0)
    weights *= len(candidates) / weights.sum()
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.028,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=13000,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(candidates[features], candidates["label"], sample_weight=weights)
    return model, features


def choose_predictions(candidates: pd.DataFrame, score: np.ndarray, eta: float = 1.0) -> np.ndarray:
    tmp = candidates[["row_id", "candidate", "base_prob"]].copy()
    meta_logit = np.log(np.clip(score, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - score, 1e-6, 1.0))
    tmp["score"] = np.log(np.clip(tmp["base_prob"].to_numpy(dtype=float), 1e-12, 1.0)) + eta * meta_logit
    best = tmp.sort_values(["row_id", "score"], ascending=[True, False]).groupby("row_id", sort=False).head(1)
    pred = np.zeros(int(candidates["row_id"].max()) + 1, dtype=int)
    pred[best["row_id"].to_numpy(dtype=int)] = best["candidate"].to_numpy(dtype=int)
    return pred


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_PATH, "rb") as f:
        art = pickle.load(f)
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    selected = art["selected"]

    current_oof = build_current_oof_action()
    v64_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * v64_oof)
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    expert_oof = {"r42_base": r42_oof, "current": current_oof, "v47_v64": v64_oof}
    expert_test = {"r42_base": r42_test, "current": current_test, "v47_v64": golden_test}
    for name, prob in art["experts_oof"].items():
        if name == "v47_v64_oof_soft":
            continue
        expert_oof[name] = prob
    for name, prob in art["experts_test"].items():
        if name == "v47_golden_test_soft":
            continue
        expert_test[name] = prob

    candidates = make_candidate_frame(meta, expert_oof, art["rare_oof_scores"], art["rare_classes"], "r42_base", top_k=6)
    cand_score, fold_report = train_meta_oof(candidates)
    y = meta["next_actionId"].to_numpy(dtype=int)

    rows = []
    preds = {}
    # Compare base argmax and base with current action multipliers.
    base_argmax = r42_oof.argmax(axis=1)
    base_mult = apply_segmented_multipliers(meta, r42_oof, selected["action_multipliers"], ACTION_CLASSES, "two")
    rows.append({"variant": "r42_oof_argmax", "eta": 0.0, "action_macro_f1": float(f1_score(y, base_argmax, average="macro", labels=ACTION_CLASSES, zero_division=0)), "churn_vs_base_mult": float(np.mean(base_argmax != base_mult))})
    rows.append({"variant": "r42_oof_mult", "eta": 0.0, "action_macro_f1": float(f1_score(y, base_mult, average="macro", labels=ACTION_CLASSES, zero_division=0)), "churn_vs_base_mult": 0.0})
    for eta in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]:
        pred = choose_predictions(candidates, cand_score, eta=eta)
        preds[f"eta_{eta:g}"] = pred
        rows.append(
            {
                "variant": "r48_meta",
                "eta": eta,
                "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "churn_vs_base_mult": float(np.mean(pred != base_mult)),
                "pred8_count": int((pred == 8).sum()),
                "pred9_count": int((pred == 9).sum()),
                "pred12_count": int((pred == 12).sum()),
                "pred14_count": int((pred == 14).sum()),
            }
        )
    search = pd.DataFrame(rows).sort_values("action_macro_f1", ascending=False)
    search.to_csv(OUT_DIR / "r48_oof_search.csv", index=False)
    fold_report.to_csv(OUT_DIR / "r48_fold_report.csv", index=False)

    # Full meta model and test candidates.
    full_model, features = fit_meta_full(candidates)
    test_candidates = make_candidate_frame(
        test_meta.assign(fold=-1),
        expert_test,
        art["rare_test_scores"],
        art["rare_classes"],
        "r42_base",
        top_k=6,
    )
    test_score = full_model.predict_proba(test_candidates[features])[:, 1]
    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    generated = []
    for eta in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
        pred = choose_predictions(test_candidates, test_score, eta=eta)
        name = f"submission_r48_meta_eta{str(eta).replace('.', 'p')}_current_point_server.csv"
        generated.append(write_submission(test_meta, pred, current_sub, name))
        generated[-1]["eta"] = eta

    pd.DataFrame(generated).to_csv(OUT_DIR / "r48_generated_candidates.csv", index=False)
    report = {
        "artifact": str(ARTIFACT_PATH),
        "oof_search": search.to_dict(orient="records"),
        "generated": generated,
        "expert_names": list(expert_oof),
        "recommendation": "Prefer low-eta candidates first because R42 w=0.2 is already public-positive.",
    }
    (OUT_DIR / "r48_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(12).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
