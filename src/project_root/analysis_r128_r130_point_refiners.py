"""R128-R130 point refiners.

R128:
  Low-churn point consensus residual using R108/TLP base plus R119, R120,
  and R83 point experts.

R129:
  Long-depth side refiner for 7/8/9, especially pointId=8.

R130:
  Rare point high-confidence gate for pointId 1/3/4.
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
from sklearn.metrics import classification_report, f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from analysis_r82_r86_point_style import train_point_style_oof, train_point_style_test
from analysis_r108_r110_r109_transductive import foldsafe_priors, test_priors
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r120_r123_sequence_meta import apply_motif_prior, r120_motif_oof
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR, normalize_rows


OUTDIR = Path("r128_r130_point_refiners")
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


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def depth_mass(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    short = prob[:, 1:4].sum(axis=1)
    half = prob[:, 4:7].sum(axis=1)
    long = prob[:, 7:10].sum(axis=1)
    return short, half, long


def softmax_log_prob(logp: np.ndarray) -> np.ndarray:
    z = logp - np.max(logp, axis=1, keepdims=True)
    p = np.exp(z)
    return normalize_rows(p)


def point_pred(meta: pd.DataFrame, point_prob: np.ndarray, tuning: GrUTuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode).astype(int)


def action_pred(meta: pd.DataFrame, action_prob: np.ndarray, tuning: GrUTuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode).astype(int)


def eval_point_candidate(
    meta: pd.DataFrame,
    point_prob: np.ndarray,
    tuning: GrUTuning,
    name: str,
    base_point_prob: np.ndarray,
    extra: dict | None = None,
) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, point_prob, tuning)
    base = point_pred(meta, base_point_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "point8_f1": float(f1_score((y == 8).astype(int), (pred == 8).astype(int), zero_division=0)),
        "point3_pred_count": int(np.sum(pred == 3)),
        "point0_pred_rate": float(np.mean(pred == 0)),
    }
    if extra:
        rec.update(extra)
    return rec


def class_report_csv(meta: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    report = classification_report(
        meta["next_pointId"].astype(int),
        pred.astype(int),
        labels=POINT_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(path)


def r128_consensus_residual(
    base: np.ndarray,
    experts: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    long_thr: float,
    rare_req: int,
    rare_lift: float,
    rare_scale: float,
    common_scale: float,
    point0_scale: float,
) -> np.ndarray:
    eps = 1e-8
    logs = np.log(np.clip(base, eps, 1.0))
    short, half, long = depth_mass(base)
    base_top = base.argmax(axis=1)

    expert_stack = np.stack([p for p in experts.values()], axis=0)
    top3 = np.argsort(-expert_stack, axis=2)[:, :, :3]
    mean_exp = expert_stack.mean(axis=0)

    gate = np.zeros_like(base)
    # Preserve terminal unless the row is close to the nonterminal boundary.
    non0_max = base[:, 1:].max(axis=1)
    gate[:, 0] = ((np.abs(base[:, 0] - non0_max) < 0.08) * point0_scale).astype(float)

    # Same-depth low-risk refinements.
    gate[:, 5] = ((half > 0.35) * common_scale).astype(float)
    for k in [7, 8, 9]:
        gate[:, k] = ((long > long_thr) * common_scale).astype(float)
    # Give point8 a little more room, but still only inside long-depth rows.
    gate[:, 8] *= 1.35

    # Rare point gates: only same-depth compatible and expert consensus.
    for k in [1, 3, 4]:
        consensus = np.zeros(len(base), dtype=int)
        for ei in range(len(experts)):
            consensus += np.any(top3[ei] == k, axis=1)
        compatible = short > 0.25 if k in [1, 3] else half > 0.25
        lift = mean_exp[:, k] / np.clip(base[:, k], eps, None)
        allowed_from = base_top != 0
        if k == 3:
            allowed_from &= np.isin(base_top, [1, 2, 4, 5, 6])
        gate[:, k] = ((consensus >= rare_req) & compatible & allowed_from & (lift > rare_lift)).astype(float) * rare_scale

    out_log = logs.copy()
    for name, expert in experts.items():
        out_log += weights[name] * gate * (np.log(np.clip(expert, eps, 1.0)) - logs)
    return softmax_log_prob(out_log)


def build_meta_features(
    rows: pd.DataFrame,
    base_point: np.ndarray,
    experts: dict[str, np.ndarray],
    action_prob: np.ndarray,
) -> pd.DataFrame:
    cols = [
        "prefix_len",
        "phase_id",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_positionId",
        "lag0_handId",
        "serverScoreDiff",
        "scoreTotal",
        "next_hitter_is_server",
    ]
    out = rows[[c for c in cols if c in rows.columns]].copy()
    for i in range(base_point.shape[1]):
        out[f"base_p{i}"] = base_point[:, i]
    for name, prob in experts.items():
        for i in range(prob.shape[1]):
            out[f"{name}_p{i}"] = prob[:, i]
    for i in range(action_prob.shape[1]):
        out[f"a{i}"] = action_prob[:, i]
    out["short_mass"], out["half_mass"], out["long_mass"] = depth_mass(base_point)
    out["base_point_max"] = base_point.max(axis=1)
    out["base_point_entropy"] = -np.sum(np.clip(base_point, 1e-12, 1.0) * np.log(np.clip(base_point, 1e-12, 1.0)), axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def aligned_long_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        if cls in [7, 8, 9]:
            out[:, cls - 7] = raw[:, i]
    return normalize_rows(out + 1e-9)


def train_long_side_oof(
    meta: pd.DataFrame,
    rows: pd.DataFrame,
    x_all: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    out = np.zeros((len(rows), 3), dtype=float)
    folds = []
    y_all = meta["next_pointId"].astype(int).to_numpy()
    for fold in sorted(rows["fold"].unique()):
        valid_idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        train_idx = rows.index[~rows["fold"].eq(fold)].to_numpy()
        train_idx = train_idx[np.isin(y_all[train_idx], [7, 8, 9])]
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=18,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=12900 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            x_all.iloc[train_idx],
            y_all[train_idx],
            sample_weight=class_weight_sample(pd.Series(y_all[train_idx])),
        )
        out[valid_idx] = aligned_long_proba(model, x_all.iloc[valid_idx])
        pred_long = out[valid_idx].argmax(axis=1) + 7
        mask = np.isin(y_all[valid_idx], [7, 8, 9])
        folds.append(
            {
                "fold": int(fold),
                "n_train_long": int(len(train_idx)),
                "n_valid_long": int(mask.sum()),
                "valid_long_macro_f1": float(f1_score(y_all[valid_idx][mask], pred_long[mask], average="macro", labels=[7, 8, 9], zero_division=0)) if mask.any() else 0.0,
            }
        )
    return normalize_rows(out + 1e-9), pd.DataFrame(folds)


def train_long_side_test(prefix: pd.DataFrame, x_oof: pd.DataFrame, y: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    train_idx = np.flatnonzero(np.isin(y, [7, 8, 9]))
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=18,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.5,
        random_state=12999,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_oof.iloc[train_idx], y[train_idx], sample_weight=class_weight_sample(pd.Series(y[train_idx])))
    return aligned_long_proba(model, x_test)


def apply_long_side_refine(base: np.ndarray, q_long: np.ndarray, *, alpha: float, long_thr: float) -> np.ndarray:
    out = base.copy()
    long = base[:, 7:10].sum(axis=1)
    use = long > long_thr
    refined = long[:, None] * q_long
    out[use, 7:10] = (1.0 - alpha) * out[use, 7:10] + alpha * refined[use]
    return normalize_rows(out)


def r130_rare_gate(
    base: np.ndarray,
    experts: dict[str, np.ndarray],
    *,
    alpha: float,
    consensus_req: int,
    lift_thr: float,
    margin_thr: float,
) -> np.ndarray:
    out = base.copy()
    short, half, _ = depth_mass(base)
    top = base.argmax(axis=1)
    stack = np.stack(list(experts.values()), axis=0)
    top3 = np.argsort(-stack, axis=2)[:, :, :3]
    mean_exp = stack.mean(axis=0)
    for k in [1, 3, 4]:
        consensus = np.zeros(len(base), dtype=int)
        for ei in range(stack.shape[0]):
            consensus += np.any(top3[ei] == k, axis=1)
        compatible = short > 0.30 if k in [1, 3] else half > 0.30
        allowed = top != 0
        if k == 3:
            allowed &= np.isin(top, [1, 2, 4, 5, 6])
        same_depth = [1, 2, 3] if k in [1, 3] else [4, 5, 6]
        others = [j for j in same_depth if j != k]
        margin = mean_exp[:, k] - mean_exp[:, others].max(axis=1)
        lift = mean_exp[:, k] / np.clip(base[:, k], 1e-8, None)
        use = (consensus >= consensus_req) & compatible & allowed & (margin > margin_thr) & (lift > lift_thr)
        target = out[use].copy()
        target[:, k] = (1.0 - alpha) * target[:, k] + alpha * mean_exp[use, k]
        out[use] = normalize_rows(target)
    return normalize_rows(out)


def write_submission_from_point(
    test_meta: pd.DataFrame,
    point_prob: np.ndarray,
    tuning: GrUTuning,
    name: str,
    *,
    base_action_prob: np.ndarray | None = None,
    base_server_prob: np.ndarray | None = None,
    anchor_submission: pd.DataFrame | None = None,
    extra: dict | None = None,
) -> dict:
    point = point_pred(test_meta, point_prob, tuning)
    if anchor_submission is not None:
        sub = anchor_submission.copy()
        sub["pointId"] = point.astype(int)
    else:
        assert base_action_prob is not None and base_server_prob is not None
        sub = pd.DataFrame(
            {
                "rally_uid": test_meta["rally_uid"].astype(int),
                "actionId": action_pred(test_meta, base_action_prob, tuning).astype(int),
                "pointId": point.astype(int),
                "serverGetPoint": np.round(np.clip(base_server_prob, 1e-6, 1.0 - 1e-6), 8),
            }
        )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    if extra:
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

    meta = art["valid_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix).reset_index(drop=True)
    test_meta = r101_test["test_meta"].copy().reset_index(drop=True)
    tuning = r111_oof["tuning"]

    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])

    current_action_oof = build_current_oof_action()
    teacher_action_oof = normalize_rows(0.80 * current_action_oof + 0.20 * art["experts_oof"]["v47_v64_oof_soft"])
    teacher_action_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    base_action_oof = normalize_rows(0.925 * r111_oof["gru_action"] + 0.075 * teacher_action_oof)
    base_action_test = normalize_rows(0.925 * r111_test["gru_action"] + 0.075 * teacher_action_test)
    anchor_server_oof = r101_oof["gru_server"]
    anchor_server_test = r101_test["gru_server"]

    # R108 base: R101 point + tiny V3 teacher, then selective TLP repair.
    r101_base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    r101_base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    _, tlp_p_oof = foldsafe_priors(rows, prefix, base_action_oof, r101_base_point_oof, mode="tlp", k=100, train_weight=0.50)
    _, tlp_p_test = test_priors(test_prefix, prefix, base_action_test, r101_base_point_test, mode="tlp", k=100, train_weight=0.50)
    high_pe_oof = (-np.sum(np.clip(r101_base_point_oof, 1e-12, 1.0) * np.log(np.clip(r101_base_point_oof, 1e-12, 1.0)), axis=1)) > np.quantile(
        -np.sum(np.clip(r101_base_point_oof, 1e-12, 1.0) * np.log(np.clip(r101_base_point_oof, 1e-12, 1.0)), axis=1),
        0.70,
    )
    high_pe_test = (-np.sum(np.clip(r101_base_point_test, 1e-12, 1.0) * np.log(np.clip(r101_base_point_test, 1e-12, 1.0)), axis=1)) > np.quantile(
        -np.sum(np.clip(r101_base_point_oof, 1e-12, 1.0) * np.log(np.clip(r101_base_point_oof, 1e-12, 1.0)), axis=1),
        0.70,
    )
    base_point_oof = r101_base_point_oof.copy()
    base_point_test = r101_base_point_test.copy()
    base_point_oof[high_pe_oof] = normalize_rows(0.98 * base_point_oof[high_pe_oof] + 0.02 * tlp_p_oof[high_pe_oof])
    base_point_test[high_pe_test] = normalize_rows(0.98 * base_point_test[high_pe_test] + 0.02 * tlp_p_test[high_pe_test])

    # Experts.
    r119_oof = r119_oof_prior(rows, prefix, base_action_oof)
    r119_test = action_conditioned_point_prior(test_prefix, prefix, base_action_test)
    _, r120_p_oof = r120_motif_oof(rows, prefix)
    r120_p_test = apply_motif_prior(test_prefix, prefix, "next_pointId", 10)
    r83_oof = train_point_style_oof(train_raw, prefix, rows, features)
    r83_test = train_point_style_test(train_raw, test_raw, prefix, test_prefix, features)

    experts_oof = {"r119": r119_oof, "r120": r120_p_oof, "r83": r83_oof}
    experts_test = {"r119": r119_test, "r120": r120_p_test, "r83": r83_test}

    search_rows: list[dict] = []
    generated: list[dict] = []
    base_rec = eval_point_candidate(meta, base_point_oof, tuning, "r108_tlp_selective_base", base_point_oof)
    search_rows.append({**base_rec, "family": "base"})

    # R128 consensus residual sweep.
    for long_thr in [0.40, 0.45, 0.50]:
        for common_scale in [0.6, 1.0]:
            for rare_req in [2, 3]:
                for w119, w120, w83 in [(0.04, 0.05, 0.02), (0.05, 0.075, 0.025), (0.06, 0.10, 0.03)]:
                    prob = r128_consensus_residual(
                        base_point_oof,
                        experts_oof,
                        {"r119": w119, "r120": w120, "r83": w83},
                        long_thr=long_thr,
                        rare_req=rare_req,
                        rare_lift=1.8,
                        rare_scale=0.55,
                        common_scale=common_scale,
                        point0_scale=0.35,
                    )
                    name = f"r128_cons_l{clean_float(long_thr)}_c{clean_float(common_scale)}_req{rare_req}_w{clean_float(w119)}_{clean_float(w120)}_{clean_float(w83)}"
                    rec = eval_point_candidate(meta, prob, tuning, name, base_point_oof, {"family": "r128", "long_thr": long_thr, "common_scale": common_scale, "rare_req": rare_req, "w119": w119, "w120": w120, "w83": w83})
                    search_rows.append(rec)

    x_oof = build_meta_features(rows, base_point_oof, experts_oof, base_action_oof)
    x_test = build_meta_features(test_prefix, base_point_test, experts_test, base_action_test)
    q_long_oof, long_folds = train_long_side_oof(meta, rows, x_oof)
    q_long_test = train_long_side_test(prefix, x_oof, meta["next_pointId"].astype(int).to_numpy(), x_test)
    long_folds.to_csv(OUTDIR / "r129_long_side_fold_report.csv", index=False)

    # R129 long-side refiner.
    for alpha in [0.10, 0.15, 0.20, 0.30]:
        for long_thr in [0.40, 0.45, 0.50, 0.55]:
            prob = apply_long_side_refine(base_point_oof, q_long_oof, alpha=alpha, long_thr=long_thr)
            name = f"r129_longside_a{clean_float(alpha)}_l{clean_float(long_thr)}"
            rec = eval_point_candidate(meta, prob, tuning, name, base_point_oof, {"family": "r129", "alpha": alpha, "long_thr": long_thr})
            search_rows.append(rec)

    # R130 rare high-confidence gate.
    for alpha in [0.10, 0.20, 0.35]:
        for req in [2, 3]:
            for lift in [1.6, 2.0, 2.5]:
                prob = r130_rare_gate(base_point_oof, experts_oof, alpha=alpha, consensus_req=req, lift_thr=lift, margin_thr=0.005)
                name = f"r130_rare_a{clean_float(alpha)}_req{req}_lift{clean_float(lift)}"
                rec = eval_point_candidate(meta, prob, tuning, name, base_point_oof, {"family": "r130", "alpha": alpha, "rare_req": req, "lift": lift})
                search_rows.append(rec)

    # Combined: best R128 plus best R129/R130 if they improve independently.
    search = pd.DataFrame(search_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    best_r128 = search[search["family"].eq("r128")].iloc[0]
    best_r129 = search[search["family"].eq("r129")].iloc[0]
    best_r130 = search[search["family"].eq("r130")].iloc[0]

    def build_by_rec(rec: pd.Series, base_oof: np.ndarray, base_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        name = str(rec["candidate"])
        if name.startswith("r128"):
            kwargs = {
                "long_thr": float(rec["long_thr"]),
                "rare_req": int(rec["rare_req"]),
                "rare_lift": 1.8,
                "rare_scale": 0.55,
                "common_scale": float(rec["common_scale"]),
                "point0_scale": 0.35,
            }
            weights = {"r119": float(rec["w119"]), "r120": float(rec["w120"]), "r83": float(rec["w83"])}
            return (
                r128_consensus_residual(base_oof, experts_oof, weights, **kwargs),
                r128_consensus_residual(base_test, experts_test, weights, **kwargs),
            )
        if name.startswith("r129"):
            return (
                apply_long_side_refine(base_oof, q_long_oof, alpha=float(rec["alpha"]), long_thr=float(rec["long_thr"])),
                apply_long_side_refine(base_test, q_long_test, alpha=float(rec["alpha"]), long_thr=float(rec["long_thr"])),
            )
        return (
            r130_rare_gate(base_oof, experts_oof, alpha=float(rec["alpha"]), consensus_req=int(rec["rare_req"]), lift_thr=float(rec["lift"]), margin_thr=0.005),
            r130_rare_gate(base_test, experts_test, alpha=float(rec["alpha"]), consensus_req=int(rec["rare_req"]), lift_thr=float(rec["lift"]), margin_thr=0.005),
        )

    combo_oof, combo_test = build_by_rec(best_r128, base_point_oof, base_point_test)
    for rec in [best_r129, best_r130]:
        cand_oof, cand_test = build_by_rec(rec, combo_oof, combo_test)
        cand_rec = eval_point_candidate(meta, cand_oof, tuning, f"combo_after_{rec['candidate']}", base_point_oof, {"family": "combo"})
        if cand_rec["point_macro_f1"] >= eval_point_candidate(meta, combo_oof, tuning, "combo_current", base_point_oof)["point_macro_f1"]:
            combo_oof, combo_test = cand_oof, cand_test
            search_rows.append(cand_rec)
    combo_rec = eval_point_candidate(meta, combo_oof, tuning, "r128_r129_r130_combo", base_point_oof, {"family": "combo"})
    search_rows.append(combo_rec)
    search = pd.DataFrame(search_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r128_r130_search.csv", index=False)

    # Class reports for the most useful variants.
    for rec in [search.iloc[0], best_r128, best_r129, best_r130, pd.Series(combo_rec)]:
        family = str(rec.get("family", ""))
        prob_oof, _ = build_by_rec(rec, base_point_oof, base_point_test) if family in {"r128", "r129", "r130"} else (combo_oof, combo_test)
        class_report_csv(meta, point_pred(meta, prob_oof, tuning), OUTDIR / f"class_report_{str(rec['candidate'])[:80]}.csv")

    anchor_sub = pd.read_csv(R67_ANCHOR) if R67_ANCHOR.exists() else None
    # Generate top point candidates using local action/server and R67 public anchor.
    for _, rec in search.head(6).iterrows():
        name = str(rec["candidate"])
        family = str(rec.get("family", ""))
        if family in {"r128", "r129", "r130"}:
            _, prob_test = build_by_rec(rec, base_point_oof, base_point_test)
        elif family == "combo" or name.startswith("combo"):
            prob_test = combo_test
        else:
            prob_test = base_point_test
        generated.append(
            write_submission_from_point(
                test_meta,
                prob_test,
                tuning,
                f"submission_{name}_local_action_server.csv",
                base_action_prob=base_action_test,
                base_server_prob=anchor_server_test,
                extra=rec.to_dict(),
            )
        )
        if anchor_sub is not None:
            generated.append(
                write_submission_from_point(
                    test_meta,
                    prob_test,
                    tuning,
                    f"submission_{name}_r67_anchor.csv",
                    anchor_submission=anchor_sub,
                    extra=rec.to_dict(),
                )
            )

    report = {
        "base": base_rec,
        "best": search.head(25).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "R128 uses logit residuals from R119/R120/R83 with depth and rare-class gates.",
            "R129 only redistributes point mass inside 7/8/9 when long_mass is high.",
            "R130 only rescues point 1/3/4 under expert consensus.",
            "R67-anchor submissions keep R67 action/server and replace only pointId.",
        ],
    }
    (OUTDIR / "r128_r130_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r128_r130_point_refiners.py", "src/analysis/analysis_r128_r130_point_refiners.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
