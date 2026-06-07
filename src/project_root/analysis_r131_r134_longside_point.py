"""R131-R134 long-side point calibration.

R131:
  Oracle/error audit for same-depth 7/8/9 fixes.

R132:
  Calibrated long-side refiner v2. It only redistributes P7/P8/P9 and
  preserves all other point probabilities.

R133:
  Pairwise 7<->8 / 8<->9 / 7<->9 swap detector.

R134:
  Generate public-anchor candidates by replacing only the point column in the
  R67 public-validated anchor submission.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from analysis_r1_oof_ensemble import compose_v3
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from analysis_r82_r86_point_style import train_point_style_oof, train_point_style_test
from analysis_r108_r110_r109_transductive import foldsafe_priors, test_priors
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r120_r123_sequence_meta import apply_motif_prior, r120_motif_oof
from analysis_r128_r130_point_refiners import (
    OUTDIR as _R128_UNUSED,
    build_meta_features,
    clean_float,
    depth_mass,
    eval_point_candidate,
    point_pred,
    softmax_log_prob,
    train_long_side_oof,
    train_long_side_test,
)
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR, normalize_rows


OUTDIR = Path("r131_r134_longside_point")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R67_ANCHOR = Path("upload_candidates_20260519/submission_r67_r63_blend_w0p2_current_point_server.csv")


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


def long_normalize(prob: np.ndarray) -> np.ndarray:
    return normalize_rows(prob[:, 7:10] + 1e-9)


def per_class_f1(y: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    return {int(k): float(report[str(k)]["f1-score"]) for k in POINT_CLASSES}


def oracle_audit(meta: pd.DataFrame, base_prob: np.ndarray, experts: dict[str, np.ndarray], tuning: GrUTuning) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = meta["next_pointId"].astype(int).to_numpy()
    base = point_pred(meta, base_prob, tuning)
    base_score = f1_score(y, base, average="macro", labels=POINT_CLASSES, zero_division=0)
    long_mass = base_prob[:, 7:10].sum(axis=1)
    base_top3 = np.argsort(-base_prob, axis=1)[:, :3]
    expert_top1_long = np.zeros(len(base), dtype=bool)
    expert_consensus_long = np.zeros(len(base), dtype=int)
    for p in experts.values():
        top = p.argmax(axis=1)
        expert_top1_long |= np.isin(top, [7, 8, 9])
        expert_consensus_long += np.isin(top, [7, 8, 9]).astype(int)

    gates: dict[str, np.ndarray] = {}
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        gates[f"long_mass_ge_{thr:.2f}"] = long_mass >= thr
    gates["base_top1_long"] = np.isin(base, [7, 8, 9])
    gates["base_top3_contains_long"] = np.any(np.isin(base_top3, [7, 8, 9]), axis=1)
    gates["any_expert_top1_long"] = expert_top1_long
    gates["expert_consensus2_top1_long"] = expert_consensus_long >= 2

    rows = []
    for name, eligible in gates.items():
        oracle = base.copy()
        fix = eligible & np.isin(y, [7, 8, 9])
        oracle[fix] = y[fix]
        score = f1_score(y, oracle, average="macro", labels=POINT_CLASSES, zero_division=0)
        f1s = per_class_f1(y, oracle)
        rows.append(
            {
                "gate": name,
                "eligible_rate": float(eligible.mean()),
                "fixable_true_long_rate": float(fix.mean()),
                "base_point_macro": float(base_score),
                "oracle_point_macro": float(score),
                "oracle_gain": float(score - base_score),
                "oracle_point7_f1": f1s[7],
                "oracle_point8_f1": f1s[8],
                "oracle_point9_f1": f1s[9],
            }
        )
    oracle_df = pd.DataFrame(rows).sort_values("oracle_gain", ascending=False)

    long_mask = np.isin(y, [7, 8, 9]) | np.isin(base, [7, 8, 9])
    cm = confusion_matrix(y[long_mask], base[long_mask], labels=POINT_CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in POINT_CLASSES], columns=[f"pred_{c}" for c in POINT_CLASSES])
    err_rows = []
    for true_k in [7, 8, 9]:
        m = y == true_k
        total = int(m.sum())
        for pred_k in POINT_CLASSES:
            err_rows.append({"true": true_k, "pred": pred_k, "count": int(np.sum(m & (base == pred_k))), "rate_in_true": float(np.sum(m & (base == pred_k)) / max(total, 1))})
    err_df = pd.DataFrame(err_rows)
    return oracle_df, cm_df, err_df


def consensus_masks(base_prob: np.ndarray, experts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    qbase = long_normalize(base_prob)
    p8_gt_base = np.zeros(len(base_prob), dtype=int)
    p8_top2 = np.zeros(len(base_prob), dtype=int)
    for p in experts.values():
        q = long_normalize(p)
        p8_gt_base += (q[:, 1] > qbase[:, 1]).astype(int)
        p8_top2 += (np.argsort(-q, axis=1)[:, :2] == 1).any(axis=1).astype(int)
    return {
        "none": np.ones(len(base_prob), dtype=bool),
        "p8_gt_base_2of3": p8_gt_base >= 2,
        "p8_top2_2of3": p8_top2 >= 2,
    }


def q_expert_for_mode(q_long: np.ndarray, experts: dict[str, np.ndarray], mode: str) -> np.ndarray:
    parts = []
    if "r129" in mode:
        parts.append(q_long)
    for key in ["r119", "r120", "r83"]:
        if key in mode:
            parts.append(long_normalize(experts[key]))
    if not parts:
        parts.append(q_long)
    return normalize_rows(np.mean(np.stack(parts, axis=0), axis=0) + 1e-9)


def apply_calibrated_long(
    base: np.ndarray,
    q_expert: np.ndarray,
    *,
    alpha: float,
    long_thr: float,
    lambda8: float,
    use_mask: np.ndarray,
) -> np.ndarray:
    out = base.copy()
    long = base[:, 7:10].sum(axis=1)
    qbase = long_normalize(base)
    scale = np.array([1.0, 1.0 + lambda8, 1.0], dtype=float)
    logq = np.log(np.clip(qbase, 1e-9, 1.0)) + alpha * scale[None, :] * (
        np.log(np.clip(q_expert, 1e-9, 1.0)) - np.log(np.clip(qbase, 1e-9, 1.0))
    )
    qfinal = softmax_log_prob(logq)
    use = (long >= long_thr) & use_mask
    out[use, 7:10] = long[use, None] * qfinal[use]
    return normalize_rows(out)


def train_pair_model_oof(x: pd.DataFrame, meta: pd.DataFrame, rows: pd.DataFrame, pair: tuple[int, int]) -> tuple[np.ndarray, pd.DataFrame]:
    y = meta["next_pointId"].astype(int).to_numpy()
    out = np.full(len(rows), 0.5, dtype=float)
    folds = []
    lo, hi = pair
    for fold in sorted(rows["fold"].unique()):
        valid_idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        train_idx = rows.index[~rows["fold"].eq(fold)].to_numpy()
        train_idx = train_idx[np.isin(y[train_idx], [lo, hi])]
        if len(train_idx) < 50:
            continue
        target = (y[train_idx] == hi).astype(int)
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=23,
            min_child_samples=16,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=13300 + int(fold) + hi,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x.iloc[train_idx], target, sample_weight=class_weight_sample(pd.Series(target)))
        pred = model.predict_proba(x.iloc[valid_idx])
        classes = [int(c) for c in model.classes_]
        out[valid_idx] = pred[:, classes.index(1)] if 1 in classes else 0.0
        mask = np.isin(y[valid_idx], [lo, hi])
        folds.append({"pair": f"{lo}_{hi}", "fold": int(fold), "n_train": int(len(train_idx)), "n_valid_pair": int(mask.sum())})
    return out, pd.DataFrame(folds)


def train_pair_model_test(x: pd.DataFrame, y: np.ndarray, pair: tuple[int, int], x_test: pd.DataFrame) -> np.ndarray:
    lo, hi = pair
    train_idx = np.flatnonzero(np.isin(y, [lo, hi]))
    target = (y[train_idx] == hi).astype(int)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.035,
        num_leaves=23,
        min_child_samples=16,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=13399 + hi,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x.iloc[train_idx], target, sample_weight=class_weight_sample(pd.Series(target)))
    pred = model.predict_proba(x_test)
    classes = [int(c) for c in model.classes_]
    return pred[:, classes.index(1)] if 1 in classes else np.zeros(len(x_test), dtype=float)


def apply_pairwise_swap(
    meta: pd.DataFrame,
    base: np.ndarray,
    pair_scores: dict[str, np.ndarray],
    tuning: GrUTuning,
    *,
    thr: float,
    move_frac: float,
    long_thr: float,
) -> np.ndarray:
    out = base.copy()
    pred = point_pred(meta, base, tuning)
    long = base[:, 7:10].sum(axis=1)
    use = long >= long_thr
    # score is probability of the higher class in the pair.
    rules = [
        ("78", 7, 8, pair_scores["78"]),
        ("89", 8, 9, pair_scores["89"]),
        ("79", 7, 9, pair_scores["79"]),
    ]
    for _, lo, hi, score in rules:
        to_hi = use & (pred == lo) & (score >= thr)
        to_lo = use & (pred == hi) & (score <= 1.0 - thr)
        mass = out[to_hi, lo] * move_frac
        out[to_hi, lo] -= mass
        out[to_hi, hi] += mass
        mass = out[to_lo, hi] * move_frac
        out[to_lo, hi] -= mass
        out[to_lo, lo] += mass
    return normalize_rows(out)


def write_submission(
    test_meta: pd.DataFrame,
    point_prob: np.ndarray,
    tuning: GrUTuning,
    name: str,
    anchor_sub: pd.DataFrame,
    extra: dict,
) -> dict:
    sub = anchor_sub.copy()
    sub["pointId"] = point_pred(test_meta, point_prob, tuning).astype(int)
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    info.update(extra)
    return info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, features = prepare_prefix_features()
    r101_oof = load_pickle("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
    r101_test = load_pickle("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
    r111_oof = load_pickle("r111_remaining_moe_gru/oof_proba_r111.pkl")
    r111_test = load_pickle("r111_remaining_moe_gru/test_proba_r111.pkl")
    v3_oof = load_pickle("oof_proba_v3.pkl")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])

    meta = art["valid_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix).reset_index(drop=True)
    test_meta = r101_test["test_meta"].copy().reset_index(drop=True)
    tuning = r111_oof["tuning"]

    current_action_oof = build_current_oof_action()
    teacher_action_oof = normalize_rows(0.80 * current_action_oof + 0.20 * art["experts_oof"]["v47_v64_oof_soft"])
    teacher_action_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    base_action_oof = normalize_rows(0.925 * r111_oof["gru_action"] + 0.075 * teacher_action_oof)
    base_action_test = normalize_rows(0.925 * r111_test["gru_action"] + 0.075 * teacher_action_test)
    r101_base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    r101_base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)

    # Rebuild R108/TLP selective base.
    _, tlp_p_oof = foldsafe_priors(rows, prefix, base_action_oof, r101_base_point_oof, mode="tlp", k=100, train_weight=0.50)
    _, tlp_p_test = test_priors(test_prefix, prefix, base_action_test, r101_base_point_test, mode="tlp", k=100, train_weight=0.50)
    ent_oof = -np.sum(np.clip(r101_base_point_oof, 1e-12, 1.0) * np.log(np.clip(r101_base_point_oof, 1e-12, 1.0)), axis=1)
    ent_test = -np.sum(np.clip(r101_base_point_test, 1e-12, 1.0) * np.log(np.clip(r101_base_point_test, 1e-12, 1.0)), axis=1)
    cut = np.quantile(ent_oof, 0.70)
    base_point_oof = r101_base_point_oof.copy()
    base_point_test = r101_base_point_test.copy()
    base_point_oof[ent_oof > cut] = normalize_rows(0.98 * base_point_oof[ent_oof > cut] + 0.02 * tlp_p_oof[ent_oof > cut])
    base_point_test[ent_test > cut] = normalize_rows(0.98 * base_point_test[ent_test > cut] + 0.02 * tlp_p_test[ent_test > cut])

    # Experts.
    r119_oof = r119_oof_prior(rows, prefix, base_action_oof)
    r119_test = action_conditioned_point_prior(test_prefix, prefix, base_action_test)
    _, r120_oof = r120_motif_oof(rows, prefix)
    r120_test = apply_motif_prior(test_prefix, prefix, "next_pointId", 10)
    r83_oof = train_point_style_oof(train_raw, prefix, rows, features)
    r83_test = train_point_style_test(train_raw, test_raw, prefix, test_prefix, features)
    experts_oof = {"r119": r119_oof, "r120": r120_oof, "r83": r83_oof}
    experts_test = {"r119": r119_test, "r120": r120_test, "r83": r83_test}

    oracle_df, cm_df, err_df = oracle_audit(meta, base_point_oof, experts_oof, tuning)
    oracle_df.to_csv(OUTDIR / "r131_longside_oracle_report.csv", index=False)
    cm_df.to_csv(OUTDIR / "r131_longside_confusion_matrix.csv")
    err_df.to_csv(OUTDIR / "r131_point789_error_slices.csv", index=False)

    x_oof = build_meta_features(rows, base_point_oof, experts_oof, base_action_oof)
    x_test = build_meta_features(test_prefix, base_point_test, experts_test, base_action_test)
    q_long_oof, long_fold = train_long_side_oof(meta, rows, x_oof)
    q_long_test = train_long_side_test(prefix, x_oof, meta["next_pointId"].astype(int).to_numpy(), x_test)
    long_fold.to_csv(OUTDIR / "r132_longside_fold_report.csv", index=False)

    search_rows = []
    base_rec = eval_point_candidate(meta, base_point_oof, tuning, "r108_tlp_selective_base", base_point_oof)
    base_f1s = per_class_f1(meta["next_pointId"].astype(int).to_numpy(), point_pred(meta, base_point_oof, tuning))
    search_rows.append({**base_rec, "family": "base"})

    masks = consensus_masks(base_point_oof, experts_oof)
    masks_test = consensus_masks(base_point_test, experts_test)
    modes = ["r129", "r129_r119", "r129_r120", "r129_r119_r120", "r129_r83_r119_r120"]
    for mode in modes:
        q_oof = q_expert_for_mode(q_long_oof, experts_oof, mode)
        q_test = q_expert_for_mode(q_long_test, experts_test, mode)
        for long_thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for alpha in [0.03, 0.05, 0.075, 0.10, 0.15]:
                for lam8 in [0.2, 0.4, 0.6, 0.8]:
                    for mname, mask in masks.items():
                        prob = apply_calibrated_long(base_point_oof, q_oof, alpha=alpha, long_thr=long_thr, lambda8=lam8, use_mask=mask)
                        rec = eval_point_candidate(
                            meta,
                            prob,
                            tuning,
                            f"r132_{mode}_thr{clean_float(long_thr)}_a{clean_float(alpha)}_l8{clean_float(lam8)}_{mname}",
                            base_point_oof,
                            {"family": "r132", "mode": mode, "long_thr": long_thr, "alpha": alpha, "lambda8": lam8, "mask": mname},
                        )
                        f1s = per_class_f1(meta["next_pointId"].astype(int).to_numpy(), point_pred(meta, prob, tuning))
                        rec.update({"point7_delta": f1s[7] - base_f1s[7], "point8_delta": f1s[8] - base_f1s[8], "point9_delta": f1s[9] - base_f1s[9]})
                        # Ranking objective: protect 7/9 while still valuing point8.
                        rec["score_obj"] = rec["point_macro_f1"] + 0.25 * rec["point8_delta"] - 0.15 * max(0.0, -rec["point7_delta"]) - 0.15 * max(0.0, -rec["point9_delta"]) - 0.02 * rec["point_churn_vs_base"]
                        search_rows.append(rec)

    # R133 pairwise swap.
    pair_scores_oof = {}
    pair_scores_test = {}
    pair_folds = []
    y = meta["next_pointId"].astype(int).to_numpy()
    for name, pair in {"78": (7, 8), "89": (8, 9), "79": (7, 9)}.items():
        s, folds = train_pair_model_oof(x_oof, meta, rows, pair)
        pair_scores_oof[name] = s
        pair_scores_test[name] = train_pair_model_test(x_oof, y, pair, x_test)
        pair_folds.append(folds)
    pd.concat(pair_folds, ignore_index=True).to_csv(OUTDIR / "r133_pairwise_fold_report.csv", index=False)

    for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for move in [0.15, 0.25, 0.35, 0.50]:
            for long_thr in [0.35, 0.40, 0.45, 0.50]:
                prob = apply_pairwise_swap(meta, base_point_oof, pair_scores_oof, tuning, thr=thr, move_frac=move, long_thr=long_thr)
                rec = eval_point_candidate(
                    meta,
                    prob,
                    tuning,
                    f"r133_pair_thr{clean_float(thr)}_mv{clean_float(move)}_l{clean_float(long_thr)}",
                    base_point_oof,
                    {"family": "r133", "thr": thr, "move": move, "long_thr": long_thr},
                )
                f1s = per_class_f1(y, point_pred(meta, prob, tuning))
                rec.update({"point7_delta": f1s[7] - base_f1s[7], "point8_delta": f1s[8] - base_f1s[8], "point9_delta": f1s[9] - base_f1s[9]})
                rec["score_obj"] = rec["point_macro_f1"] + 0.25 * rec["point8_delta"] - 0.15 * max(0.0, -rec["point7_delta"]) - 0.15 * max(0.0, -rec["point9_delta"]) - 0.02 * rec["point_churn_vs_base"]
                search_rows.append(rec)

    search = pd.DataFrame(search_rows).sort_values(["point_macro_f1", "score_obj", "point_churn_vs_base"], ascending=[False, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r132_r133_search.csv", index=False)

    def build_candidate(rec: pd.Series, is_test: bool) -> np.ndarray:
        base = base_point_test if is_test else base_point_oof
        if rec["family"] == "r132":
            q_long = q_long_test if is_test else q_long_oof
            exps = experts_test if is_test else experts_oof
            mask = masks_test[str(rec["mask"])] if is_test else masks[str(rec["mask"])]
            q = q_expert_for_mode(q_long, exps, str(rec["mode"]))
            return apply_calibrated_long(base, q, alpha=float(rec["alpha"]), long_thr=float(rec["long_thr"]), lambda8=float(rec["lambda8"]), use_mask=mask)
        if rec["family"] == "r133":
            scores = pair_scores_test if is_test else pair_scores_oof
            meta_like = test_meta if is_test else meta
            return apply_pairwise_swap(meta_like, base, scores, tuning, thr=float(rec["thr"]), move_frac=float(rec["move"]), long_thr=float(rec["long_thr"]))
        return base

    anchor_sub = pd.read_csv(R67_ANCHOR)
    generated = []
    for _, rec in search[search["family"].isin(["r132", "r133"])].head(8).iterrows():
        prob_test = build_candidate(rec, True)
        generated.append(write_submission(test_meta, prob_test, tuning, f"submission_{rec['candidate']}_r67_anchor.csv", anchor_sub, rec.to_dict()))
    # Add pure R129-compatible baseline from the previous best setting for direct comparison.
    prob_r129 = apply_calibrated_long(base_point_test, q_long_test, alpha=0.10, long_thr=0.40, lambda8=0.0, use_mask=np.ones(len(test_meta), dtype=bool))
    rec_r129 = eval_point_candidate(meta, apply_calibrated_long(base_point_oof, q_long_oof, alpha=0.10, long_thr=0.40, lambda8=0.0, use_mask=np.ones(len(meta), dtype=bool)), tuning, "r134_r129_like_a0p1_l0p4", base_point_oof, {"family": "r134"})
    generated.append(write_submission(test_meta, prob_r129, tuning, "submission_r134_r129_like_a0p1_l0p4_r67_anchor.csv", anchor_sub, rec_r129))

    best = search.head(30).to_dict(orient="records")
    report = {
        "base": base_rec,
        "oracle_top": oracle_df.head(12).to_dict(orient="records"),
        "best": best,
        "generated": generated,
        "notes": [
            "R131 oracle directly changes only eligible true long-side rows in prediction space.",
            "R132 preserves total long mass and only redistributes pointId 7/8/9.",
            "R133 moves probability mass only through pairwise swaps among 7/8/9.",
            "R134 generated candidates keep R67 action/server and replace only pointId.",
        ],
    }
    (OUTDIR / "r131_r134_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r131_r134_longside_point.py", "src/analysis/analysis_r131_r134_longside_point.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
