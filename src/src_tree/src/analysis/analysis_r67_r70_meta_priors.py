"""R67-R70 action meta-stacker v2.

R67:
  R63-aware top-k action meta-stacker. R63 conditional style probabilities are
  used as features/candidates, not direct replacement.

R68:
  Point-conditioned action prior. Build fold-safe P(action | point, phase) and
  project V3 point probabilities into expected action probabilities.

R69:
  Exact motif lookup prior. Fold-safe smoothed priors from recent stroke motifs
  with backoff and support features.

R70:
  Action-0 binary score. Used as a candidate feature, not a hard override.

All generated submissions change action only; point/server are fixed to current
R34.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r7_phase_features import add_phase_features
from analysis_r48_action_meta_stacker import (
    action_family,
    build_current_oof_action,
    choose_predictions,
    fit_meta_full,
    make_candidate_frame as make_r48_candidate_frame,
    ranks,
    train_meta_oof,
)
from analysis_r63_r64_conditional_momentum import (
    ConditionalStyleEncoder,
    add_conditional_style_features,
)
from analysis_r57_player_style_clustering import add_player_id_features
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, full_predict as v3_full_predict
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r67_r70_meta_priors")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R63_OOF_PATH = Path("r63_r64_conditional_momentum/r63_transductive_k8_oof_action.npy")


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


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


def prepare_prefix_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train0 = pd.read_csv("train.csv")
    test0 = pd.read_csv("test_new.csv")
    validate_raw_data(train0, test0)
    train = add_role_and_score_features(train0)
    test = add_role_and_score_features(test0)
    prefix = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    prefix = add_phase_features(prefix, train)
    test_prefix = add_phase_features(test_prefix, test)
    prefix = add_player_id_features(prefix, train)
    test_prefix = add_player_id_features(test_prefix, test)
    player_cols = {"server_id", "receiver_id", "next_hitter_id", "next_receiver_id"}
    features = [c for c in feature_columns(prefix) if c != "remaining_len_bucket" and c not in player_cols]
    return train, test, prefix, test_prefix, features


def align_prefix_meta(meta: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "rally_uid",
        "match",
        "prefix_len",
        "next_actionId",
        "next_pointId",
        "serverGetPoint",
    ]
    merged = meta[cols + ["fold"]].merge(
        prefix,
        on=cols,
        how="left",
        validate="one_to_one",
        suffixes=("", "_prefix"),
    )
    if merged.isna().any().any():
        bad = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Could not align prefix features: {bad[:20]}")
    return merged


def make_lgbm_binary(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def zero_scores_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, features: list[str]) -> np.ndarray:
    score = np.zeros(len(prefix_aligned), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        valid_matches = set(prefix_aligned[prefix_aligned["fold"].eq(fold)]["match"])
        train_df = prefix[~prefix["match"].isin(valid_matches)].copy()
        valid_df = prefix_aligned[prefix_aligned["fold"].eq(fold)].copy()
        y = train_df["next_actionId"].eq(0).astype(int)
        model = make_lgbm_binary(7000 + int(fold))
        counts = y.value_counts().to_dict()
        sw = y.map(lambda v: 1.0 / counts[int(v)]).to_numpy(dtype=float)
        sw = sw / np.mean(sw)
        model.fit(train_df[features], y, sample_weight=sw)
        score[valid_df.index.to_numpy()] = model.predict_proba(valid_df[features])[:, 1]
    return score


def zero_scores_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> np.ndarray:
    y = prefix["next_actionId"].eq(0).astype(int)
    model = make_lgbm_binary(7700)
    counts = y.value_counts().to_dict()
    sw = y.map(lambda v: 1.0 / counts[int(v)]).to_numpy(dtype=float)
    sw = sw / np.mean(sw)
    model.fit(prefix[features], y, sample_weight=sw)
    return model.predict_proba(test_prefix[features])[:, 1]


def phase_point_action_matrix(train_df: pd.DataFrame, alpha: float = 20.0) -> dict[int, np.ndarray]:
    global_counts = train_df["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(ACTION_CLASSES))
    mats: dict[int, np.ndarray] = {}
    for phase in [1, 2, 3, 4]:
        mat = np.zeros((len(POINT_CLASSES), len(ACTION_CLASSES)), dtype=float)
        phase_df = train_df[train_df["phase_id"].astype(int).eq(phase)]
        for p in POINT_CLASSES:
            sub = phase_df[phase_df["next_pointId"].astype(int).eq(p)]
            counts = sub["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
            mat[p] = (counts + alpha * global_prior) / (counts.sum() + alpha)
        mats[phase] = mat
    return mats


def point_conditioned_prior(point_prob: np.ndarray, rows: pd.DataFrame, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    mats = phase_point_action_matrix(train_df)
    support_table = train_df.groupby(["phase_id", "next_pointId"]).size()
    for i, row in enumerate(rows.itertuples(index=False)):
        phase = int(getattr(row, "phase_id"))
        phase = phase if phase in mats else 4
        out[i] = point_prob[i] @ mats[phase]
        supp = 0.0
        for p, pp in enumerate(point_prob[i]):
            supp += float(pp) * float(support_table.get((phase, p), 0))
        support[i] = supp
    return normalize_rows(out), support


def point_prior_oof(v3_point: np.ndarray, prefix_aligned: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(prefix_aligned), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        train_df = prefix[~prefix["match"].isin(valid_matches)].copy()
        p, s = point_conditioned_prior(v3_point[idx], prefix_aligned.loc[idx], train_df)
        out[idx] = p
        support[idx] = s
    return out, support


def make_point_prior_test(v3_point_test: np.ndarray, test_prefix: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return point_conditioned_prior(v3_point_test, test_prefix, prefix)


def motif_keys(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "k3": [
            "phase_id",
            "lag0_actionId",
            "lag0_spinId",
            "lag0_pointId",
            "lag1_actionId",
            "lag1_spinId",
            "lag1_pointId",
        ],
        "k2": ["phase_id", "lag0_actionId", "lag0_spinId", "lag0_pointId"],
        "k1": ["phase_id", "lag0_actionId", "lag0_spinId"],
    }


def build_lookup(df: pd.DataFrame, cols: list[str], alpha: float = 10.0) -> tuple[dict[tuple, np.ndarray], dict[tuple, int], np.ndarray]:
    global_counts = df["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(ACTION_CLASSES))
    lookup: dict[tuple, np.ndarray] = {}
    support: dict[tuple, int] = {}
    for key, g in df.groupby(cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        counts = g["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
        lookup[key] = (counts + alpha * global_prior) / (counts.sum() + alpha)
        support[key] = int(len(g))
    return lookup, support, global_prior


def motif_prior_for_rows(rows: pd.DataFrame, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_defs = motif_keys(rows)
    lookups = {name: build_lookup(train_df, cols) for name, cols in key_defs.items()}
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    level = np.zeros(len(rows), dtype=int)
    for i, (_, row) in enumerate(rows.iterrows()):
        chosen = None
        for li, name in enumerate(["k3", "k2", "k1"], start=3):
            cols = key_defs[name]
            key = tuple(int(row[c]) for c in cols)
            lookup, supp, global_prior = lookups[name]
            if key in lookup and supp[key] >= (20 if name == "k3" else 30 if name == "k2" else 50):
                chosen = (lookup[key], supp[key], li)
                break
        if chosen is None:
            _, _, global_prior = lookups["k1"]
            chosen = (global_prior, 0, 0)
        out[i] = chosen[0]
        support[i] = chosen[1]
        level[i] = chosen[2]
    return normalize_rows(out), support, level


def motif_prior_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(prefix_aligned), dtype=float)
    level = np.zeros(len(prefix_aligned), dtype=int)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        train_df = prefix[~prefix["match"].isin(valid_matches)].copy()
        p, s, l = motif_prior_for_rows(prefix_aligned.loc[idx], train_df)
        out[idx] = p
        support[idx] = s
        level[idx] = l
    return out, support, level


def motif_prior_test(test_prefix: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return motif_prior_for_rows(test_prefix, prefix)


def compose_v3_full_point(train: pd.DataFrame, test: pd.DataFrame, tuning: V3Tuning) -> tuple[pd.DataFrame, np.ndarray]:
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_features = test_prefix[["rally_uid", "match"] + features]
    pred = v3_full_predict(prefix_df, test_features, features, SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0))
    point = blend_probs(pred["lgbm_point"], pred["ngram_point"], tuning.point_ngram_weight)
    return test_prefix, normalize_rows(point)


def rank_feats(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = ranks(prob)
    order = np.argsort(-prob, axis=1)
    top = prob[np.arange(len(prob)), order[:, 0]]
    margin = top - prob[np.arange(len(prob)), order[:, 1]]
    return r, order[:, 0], top, margin


def augment_candidates(
    cand: pd.DataFrame,
    extra_probs: dict[str, np.ndarray],
    extra_scores: dict[str, np.ndarray],
    extra_supports: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = cand.copy()
    for name, prob in extra_probs.items():
        rank, top_cls, top_prob, margin = rank_feats(prob)
        row_ids = out["row_id"].to_numpy(dtype=int)
        cls = out["candidate"].to_numpy(dtype=int)
        out[f"{name}_prob"] = prob[row_ids, cls]
        out[f"{name}_logprob"] = np.log(np.clip(out[f"{name}_prob"].to_numpy(dtype=float), 1e-12, 1.0))
        out[f"{name}_rank"] = rank[row_ids, cls]
        out[f"{name}_top_class"] = top_cls[row_ids]
        out[f"{name}_top_prob"] = top_prob[row_ids]
        out[f"{name}_margin"] = margin[row_ids]
        out[f"{name}_minus_base"] = out[f"{name}_prob"] - out["base_prob"]
    for name, arr in extra_scores.items():
        out[name] = arr[out["row_id"].to_numpy(dtype=int)]
    for name, arr in extra_supports.items():
        out[name] = arr[out["row_id"].to_numpy(dtype=int)]
    out["candidate_is_zero"] = out["candidate"].eq(0).astype(int)
    out["zero_score_if_candidate"] = out["zero_score"] * out["candidate_is_zero"]
    out["nonzero_score_if_candidate"] = (1.0 - out["zero_score"]) * (1 - out["candidate_is_zero"])
    out["r63_agrees_candidate"] = out["r63_prob"].gt(out["r42_base_prob"]).astype(int)
    out["point_prior_agrees_candidate"] = out["point_prior_prob"].gt(out["r42_base_prob"]).astype(int)
    out["motif_agrees_candidate"] = out["motif_prior_prob"].gt(out["r42_base_prob"]).astype(int)
    return out


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "path": str(path),
        "upload_path": str(UPLOAD_DIR / name),
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
    }


def blend_action_prob(base: np.ndarray, expert: np.ndarray, weight: float) -> np.ndarray:
    return normalize_rows((1.0 - weight) * base + weight * expert)


def zero_gate_prob(base: np.ndarray, zero_score: np.ndarray, hi: float, boost: float, lo: float, decay: float) -> np.ndarray:
    out = base.copy()
    out[zero_score >= hi, 0] *= boost
    out[zero_score <= lo, 0] *= decay
    return normalize_rows(out)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    art = load_pickle(ARTIFACT_PATH)
    train, test, prefix, test_prefix, features = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    prefix_aligned = align_prefix_meta(meta, prefix)

    v3_oof = load_pickle("oof_proba_v3.pkl")
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]):
        raise ValueError("V3 OOF is not aligned to action artifact meta.")
    _, v3_point_oof, _ = compose_v3(v3_oof)

    point_prior, point_support = point_prior_oof(v3_point_oof, prefix_aligned, prefix)
    motif_prior, motif_support, motif_level = motif_prior_oof(prefix_aligned, prefix)
    zero_score = zero_scores_oof(prefix_aligned, prefix, features)

    r63_oof = np.load(R63_OOF_PATH)
    if r63_oof.shape != art["experts_oof"]["v47_v64_oof_soft"].shape:
        raise ValueError("R63 OOF shape mismatch.")

    current_oof = build_current_oof_action()
    v64_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * v64_oof)
    expert_oof = {
        "r42_base": r42_oof,
        "current": current_oof,
        "v47_v64": v64_oof,
        "r63": r63_oof,
        "point_prior": point_prior,
        "motif_prior": motif_prior,
    }
    for name, prob in art["experts_oof"].items():
        if name == "v47_v64_oof_soft":
            continue
        expert_oof[name] = prob

    cand = make_r48_candidate_frame(meta, expert_oof, art["rare_oof_scores"], art["rare_classes"], "r42_base", top_k=6)
    cand = augment_candidates(
        cand,
        {"r63": r63_oof, "point_prior": point_prior, "motif_prior": motif_prior},
        {"zero_score": zero_score},
        {"point_prior_support": point_support, "motif_support": motif_support, "motif_level": motif_level.astype(float)},
    )
    score, fold_report = train_meta_oof(cand)
    y = meta["next_actionId"].to_numpy(dtype=int)
    base_pred = apply_action(r42_oof, meta, art["selected"]["action_multipliers"])

    rows = []
    rows.append(
        {
            "variant": "r42_base",
            "eta": 0.0,
            "action_macro_f1": float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
            "churn_vs_r42": 0.0,
        }
    )
    pred_by_eta = {}
    for eta in [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]:
        pred = choose_predictions(cand, score, eta=eta)
        pred_by_eta[eta] = pred
        rows.append(
            {
                "variant": "r67_meta_v2",
                "eta": eta,
                "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "churn_vs_r42": float(np.mean(pred != base_pred)),
                "pred0_count": int((pred == 0).sum()),
                "pred8_count": int((pred == 8).sum()),
                "pred9_count": int((pred == 9).sum()),
                "pred12_count": int((pred == 12).sum()),
                "pred14_count": int((pred == 14).sum()),
            }
        )
    # Diagnostic direct priors.
    for name, prob in {"r63": r63_oof, "point_prior": point_prior, "motif_prior": motif_prior}.items():
        pred = apply_action(prob, meta, art["selected"]["action_multipliers"])
        rows.append(
            {
                "variant": f"{name}_direct",
                "eta": 1.0,
                "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "churn_vs_r42": float(np.mean(pred != base_pred)),
            }
        )
    # Conservative R68/R69/R63 direct blends. These are intentionally low-DoF
    # because direct priors are noisy when used as stand-alone classifiers.
    for name, prob in {"r63_blend": r63_oof, "point_prior_blend": point_prior, "motif_prior_blend": motif_prior}.items():
        for w in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            pred = apply_action(blend_action_prob(r42_oof, prob, w), meta, art["selected"]["action_multipliers"])
            rows.append(
                {
                    "variant": name,
                    "eta": float(w),
                    "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                    "churn_vs_r42": float(np.mean(pred != base_pred)),
                    "pred0_count": int((pred == 0).sum()),
                    "pred8_count": int((pred == 8).sum()),
                    "pred9_count": int((pred == 9).sum()),
                    "pred12_count": int((pred == 12).sum()),
                    "pred14_count": int((pred == 14).sum()),
                }
            )
    # R70 zero-action binary gate. The gate only rescales class 0 and leaves the
    # rest of the probability vector intact.
    for hi in [0.65, 0.75, 0.85, 0.90]:
        for boost in [1.5, 2.0, 3.0, 5.0]:
            for lo in [0.05, 0.10, 0.20]:
                for decay in [0.10, 0.25, 0.50]:
                    prob = zero_gate_prob(r42_oof, zero_score, hi, boost, lo, decay)
                    pred = apply_action(prob, meta, art["selected"]["action_multipliers"])
                    rows.append(
                        {
                            "variant": "r70_zero_gate",
                            "eta": float(boost),
                            "hi": float(hi),
                            "lo": float(lo),
                            "decay": float(decay),
                            "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                            "churn_vs_r42": float(np.mean(pred != base_pred)),
                            "pred0_count": int((pred == 0).sum()),
                            "pred8_count": int((pred == 8).sum()),
                            "pred9_count": int((pred == 9).sum()),
                            "pred12_count": int((pred == 12).sum()),
                            "pred14_count": int((pred == 14).sum()),
                        }
                    )
    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True])
    search.to_csv(OUTDIR / "r67_r70_oof_search.csv", index=False)
    fold_report.to_csv(OUTDIR / "r67_meta_fold_report.csv", index=False)

    # Full test features.
    test_prefix_v3, v3_point_test = compose_v3_full_point(train, test, v3_oof["tuning"])
    if not test_prefix_v3["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 full point test rows are not aligned.")
    point_prior_test_arr, point_support_test = make_point_prior_test(v3_point_test, test_prefix, prefix)
    motif_prior_test_arr, motif_support_test, motif_level_test = motif_prior_test(test_prefix, prefix)
    zero_test = zero_scores_test(prefix, test_prefix, features)

    encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=7300).fit(pd.concat([train, test], ignore_index=True), train)
    train_cond = add_conditional_style_features(prefix, encoder)
    test_cond = add_conditional_style_features(test_prefix, encoder)
    cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=7350,
        n_jobs=-1,
        verbosity=-1,
    )
    cond_features = [c for c in features if c in train_cond.columns] + cond_cols
    model.fit(train_cond[cond_features], train_cond["next_actionId"], sample_weight=class_weight_sample(train_cond["next_actionId"]))
    proba = model.predict_proba(test_cond[cond_features])
    r63_test = np.zeros((len(test_cond), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        r63_test[:, ACTION_CLASSES.index(cls)] = proba[:, i]
    r63_test = normalize_rows(r63_test)

    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    expert_test = {
        "r42_base": r42_test,
        "current": current_test,
        "v47_v64": golden_test,
        "r63": r63_test,
        "point_prior": point_prior_test_arr,
        "motif_prior": motif_prior_test_arr,
    }
    for name, prob in art["experts_test"].items():
        if name == "v47_golden_test_soft":
            continue
        expert_test[name] = prob

    full_model, feature_cols = fit_meta_full(cand)
    test_cand = make_r48_candidate_frame(
        test_meta.assign(fold=-1),
        expert_test,
        art["rare_test_scores"],
        art["rare_classes"],
        "r42_base",
        top_k=6,
    )
    test_cand = augment_candidates(
        test_cand,
        {"r63": r63_test, "point_prior": point_prior_test_arr, "motif_prior": motif_prior_test_arr},
        {"zero_score": zero_test},
        {
            "point_prior_support": point_support_test,
            "motif_support": motif_support_test,
            "motif_level": motif_level_test.astype(float),
        },
    )
    test_score = full_model.predict_proba(test_cand[feature_cols])[:, 1]
    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    base_f1 = float(search[search["variant"].eq("r42_base")].iloc[0]["action_macro_f1"])
    selected = search[
        (search["action_macro_f1"].gt(base_f1))
        & (search["churn_vs_r42"].le(0.12))
        & (~search["variant"].isin(["r42_base", "r63_direct", "point_prior_direct", "motif_prior_direct"]))
    ].head(10)

    generated = []
    for row in selected.itertuples(index=False):
        variant = str(row.variant)
        eta = float(row.eta)
        if variant == "r67_meta_v2":
            pred = choose_predictions(test_cand, test_score, eta=eta)
            name = f"submission_r67_meta_v2_eta{clean_float(eta)}_current_point_server.csv"
        elif variant == "r63_blend":
            pred = apply_action(blend_action_prob(r42_test, r63_test, eta), test_meta, art["selected"]["action_multipliers"])
            name = f"submission_r67_r63_blend_w{clean_float(eta)}_current_point_server.csv"
        elif variant == "point_prior_blend":
            pred = apply_action(blend_action_prob(r42_test, point_prior_test_arr, eta), test_meta, art["selected"]["action_multipliers"])
            name = f"submission_r68_point_prior_blend_w{clean_float(eta)}_current_point_server.csv"
        elif variant == "motif_prior_blend":
            pred = apply_action(blend_action_prob(r42_test, motif_prior_test_arr, eta), test_meta, art["selected"]["action_multipliers"])
            name = f"submission_r69_motif_prior_blend_w{clean_float(eta)}_current_point_server.csv"
        elif variant == "r70_zero_gate":
            hi = float(row.hi)
            lo = float(row.lo)
            decay = float(row.decay)
            pred = apply_action(zero_gate_prob(r42_test, zero_test, hi, eta, lo, decay), test_meta, art["selected"]["action_multipliers"])
            name = (
                f"submission_r70_zero_gate_hi{clean_float(hi)}_b{clean_float(eta)}_"
                f"lo{clean_float(lo)}_d{clean_float(decay)}_current_point_server.csv"
            )
        else:
            continue
        info = write_submission(test_meta, pred, current_sub, name)
        info["source_variant"] = variant
        info["source_oof_action_f1"] = float(row.action_macro_f1)
        info["source_oof_churn"] = float(row.churn_vs_r42)
        info["eta"] = eta
        if hasattr(row, "hi"):
            info["hi"] = None if pd.isna(row.hi) else float(row.hi)
            info["lo"] = None if pd.isna(row.lo) else float(row.lo)
            info["decay"] = None if pd.isna(row.decay) else float(row.decay)
        generated.append(info)
    pd.DataFrame(generated).to_csv(OUTDIR / "r67_r70_generated_candidates.csv", index=False)

    report = {
        "oof_search": search.to_dict(orient="records"),
        "generated": generated,
        "features_added": [
            "R63 conditional-style probabilities",
            "R68 point-conditioned action prior",
            "R69 motif lookup prior",
            "R70 zero-action binary score",
        ],
    }
    (OUTDIR / "r67_r70_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(20).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
