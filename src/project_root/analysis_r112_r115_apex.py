"""R112-R115 apex ensemble experiments.

R112:
  Re-run transductive label propagation (TLP) on the stronger R111/R105
  action branch.  This is the "TLP on R111" variant.

R113:
  Joint bivariate action-point calibration.  Apply a smoothed train-domain
  P(action, point) lift to the outer product of the current action/point
  probabilities, then marginalize back to action and point.

R114:
  Confidence/entropy-gated dynamic blend between neural, tabular/golden, and
  TLP branches.  The gate is deterministic and tuned on OOF.

R115:
  Legal scoreboard server specialist.  Uses current/past scoreboard fields and
  fold-safe server-rate features, not old-test direct replacement.
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
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import (
    align_prefix_meta,
    compose_v3_full_point,
    prepare_prefix_features,
)
from analysis_r108_r110_r109_transductive import (
    entropy,
    foldsafe_priors,
    test_priors,
)
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR, normalize_rows


OUTDIR = Path("r112_r115_apex")
SELECTED_DIR = Path("submissions/selected")
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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def normalized_confidence(prob: np.ndarray) -> np.ndarray:
    n = prob.shape[1]
    return 1.0 - entropy(prob) / np.log(n)


def apply_predictions(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, tuning: GrUTuning) -> tuple[np.ndarray, np.ndarray]:
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    return action_pred.astype(int), point_pred.astype(int)


def eval_probs(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: GrUTuning,
    name: str,
    base_action_prob: np.ndarray | None = None,
    base_point_prob: np.ndarray | None = None,
    base_server_prob: np.ndarray | None = None,
) -> dict:
    action_pred, point_pred = apply_predictions(meta, action_prob, point_prob, tuning)
    rec = {
        "candidate": name,
        "action_macro_f1": float(f1_score(meta["next_actionId"].astype(int), action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "point_macro_f1": float(f1_score(meta["next_pointId"].astype(int), point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "server_auc": float(roc_auc_score(meta["serverGetPoint"].astype(int), server_prob)),
    }
    rec["overall_local"] = 0.4 * rec["action_macro_f1"] + 0.4 * rec["point_macro_f1"] + 0.2 * rec["server_auc"]
    if base_action_prob is not None:
        base_action_pred, _ = apply_predictions(meta, base_action_prob, point_prob, tuning)
        rec["action_churn"] = float(np.mean(action_pred != base_action_pred))
    if base_point_prob is not None:
        _, base_point_pred = apply_predictions(meta, action_prob, base_point_prob, tuning)
        rec["point_churn"] = float(np.mean(point_pred != base_point_pred))
    if base_server_prob is not None:
        rec["server_mad"] = float(np.mean(np.abs(server_prob - base_server_prob)))
    return rec


def write_submission(
    test_meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: GrUTuning,
    name: str,
    extra: dict | None = None,
) -> dict:
    action_pred, point_pred = apply_predictions(test_meta, action_prob, point_prob, tuning)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
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


def pair_lift_matrix(pool: pd.DataFrame, alpha: float = 150.0, clip: float = 4.0) -> np.ndarray:
    action = pool["next_actionId"].to_numpy(dtype=int)
    point = pool["next_pointId"].to_numpy(dtype=int)
    joint_counts = np.zeros((19, 10), dtype=float)
    np.add.at(joint_counts, (action, point), 1.0)
    action_prior = np.bincount(action, minlength=19).astype(float) + 1.0
    action_prior /= action_prior.sum()
    point_prior = np.bincount(point, minlength=10).astype(float) + 1.0
    point_prior /= point_prior.sum()
    expected = np.outer(action_prior, point_prior)
    joint_prior = (joint_counts + alpha * expected) / (len(pool) + alpha)
    lift = joint_prior / np.clip(expected, 1e-12, None)
    return np.clip(lift, 1.0 / clip, clip)


def apply_joint_lift(action_prob: np.ndarray, point_prob: np.ndarray, lift: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    adjusted_action = np.zeros_like(action_prob)
    adjusted_point = np.zeros_like(point_prob)
    lift_power = np.power(lift, gamma)
    for i in range(len(action_prob)):
        joint = np.outer(action_prob[i], point_prob[i]) * lift_power
        total = joint.sum()
        if total <= 0:
            adjusted_action[i] = action_prob[i]
            adjusted_point[i] = point_prob[i]
        else:
            joint /= total
            adjusted_action[i] = joint.sum(axis=1)
            adjusted_point[i] = joint.sum(axis=0)
    return normalize_rows(adjusted_action), normalize_rows(adjusted_point)


def joint_calibrate_oof(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    out_a = np.zeros_like(action_prob)
    out_p = np.zeros_like(point_prob)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)]
        lift = pair_lift_matrix(pool)
        out_a[idx], out_p[idx] = apply_joint_lift(action_prob[idx], point_prob[idx], lift, gamma)
    return out_a, out_p


def joint_calibrate_test(prefix: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    lift = pair_lift_matrix(prefix)
    return apply_joint_lift(action_prob, point_prob, lift, gamma)


def dynamic_pair_blend(
    first_a: np.ndarray,
    second_a: np.ndarray,
    first_p: np.ndarray,
    second_p: np.ndarray,
    strength: float,
    floor: float = 0.15,
    ceiling: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    conf_a1 = normalized_confidence(first_a)
    conf_a2 = normalized_confidence(second_a)
    conf_p1 = normalized_confidence(first_p)
    conf_p2 = normalized_confidence(second_p)
    wa = 1.0 / (1.0 + np.exp(-strength * (conf_a1 - conf_a2)))
    wp = 1.0 / (1.0 + np.exp(-strength * (conf_p1 - conf_p2)))
    wa = np.clip(wa, floor, ceiling)
    wp = np.clip(wp, floor, ceiling)
    return normalize_rows(wa[:, None] * first_a + (1.0 - wa[:, None]) * second_a), normalize_rows(wp[:, None] * first_p + (1.0 - wp[:, None]) * second_p)


def add_scoreboard_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["serverScore"].astype(float)
    r = out["receiverScore"].astype(float)
    total = out["scoreTotal"].astype(float)
    diff = out["serverScoreDiff"].astype(float)
    out["r115_points_to_11_server"] = np.maximum(0.0, 11.0 - s)
    out["r115_points_to_11_receiver"] = np.maximum(0.0, 11.0 - r)
    out["r115_abs_diff"] = np.abs(diff)
    out["r115_late_game"] = ((s >= 8) | (r >= 8)).astype(int)
    out["r115_deuce_like"] = ((s >= 10) & (r >= 10)).astype(int)
    out["r115_server_gamepoint"] = ((s >= 10) & (diff >= 1)).astype(int)
    out["r115_receiver_gamepoint"] = ((r >= 10) & (diff <= -1)).astype(int)
    out["r115_pressure_x_diff"] = out["r115_late_game"] * diff
    out["r115_score_total_sqrt"] = np.sqrt(np.maximum(total, 0.0))
    out["r115_serve_pair_idx"] = ((total // 2) % 2).astype(int)
    out["r115_serve_point_in_pair"] = (total % 2).astype(int)
    out["r115_deuce_serve_parity"] = np.where(out["r115_deuce_like"].eq(1), (total % 2).astype(int), -1)
    out["r115_rally_id_log1p"] = np.log1p(out["rally_id"].astype(float)) if "rally_id" in out.columns else 0.0
    return out


def first_rows_with_score_delta(raw: pd.DataFrame) -> pd.DataFrame:
    first = raw.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).head(1).copy()
    first["pmin"] = first[["gamePlayerId", "gamePlayerOtherId"]].min(axis=1)
    first["pmax"] = first[["gamePlayerId", "gamePlayerOtherId"]].max(axis=1)
    rows = []
    group_cols = ["match", "numberGame", "pmin", "pmax"]
    for _, g in first.sort_values(group_cols + ["rally_id"]).groupby(group_cols, sort=False):
        prev_rate = np.nan
        prev_gap = np.nan
        prev_valid = 0
        for i in range(len(g)):
            cur = g.iloc[i]
            rows.append(
                {
                    "rally_uid": int(cur["rally_uid"]),
                    "r115_prev_interval_server_rate": prev_rate,
                    "r115_prev_interval_gap": prev_gap,
                    "r115_prev_interval_valid": prev_valid,
                    "r115_public_game_order": int(i),
                }
            )
            if i < len(g) - 1:
                nxt = g.iloc[i + 1]
                gap = int(nxt["rally_id"] - cur["rally_id"])
                next_score = {
                    int(nxt["gamePlayerId"]): int(nxt["scoreSelf"]),
                    int(nxt["gamePlayerOtherId"]): int(nxt["scoreOther"]),
                }
                server_id = int(cur["gamePlayerId"])
                receiver_id = int(cur["gamePlayerOtherId"])
                ds = next_score.get(server_id, np.nan) - int(cur["scoreSelf"])
                dr = next_score.get(receiver_id, np.nan) - int(cur["scoreOther"])
                valid = bool(gap > 0 and ds >= 0 and dr >= 0 and ds + dr == gap)
                prev_rate = float(ds / gap) if valid else np.nan
                prev_gap = float(gap) if valid else np.nan
                prev_valid = int(valid)
    return pd.DataFrame(rows)


def rate_lookup_features(df: pd.DataFrame, pool: pd.DataFrame, alpha: float = 40.0) -> pd.DataFrame:
    out = df.copy()
    global_mean = float(pool["serverGetPoint"].mean())
    specs = [
        ("r115_server_rate", ["server_id"]),
        ("r115_receiver_rate", ["receiver_id"]),
        ("r115_next_hitter_rate", ["next_hitter_id"]),
        ("r115_next_receiver_rate", ["next_receiver_id"]),
        ("r115_pair_rate", ["server_id", "receiver_id"]),
    ]
    for name, keys in specs:
        grp = pool.groupby(keys, dropna=False)["serverGetPoint"].agg(["sum", "count"])
        rate = (grp["sum"] + alpha * global_mean) / (grp["count"] + alpha)
        idx = pd.MultiIndex.from_frame(out[keys]) if len(keys) > 1 else pd.Index(out[keys[0]])
        out[name] = rate.reindex(idx).fillna(global_mean).to_numpy(dtype=float)
        out[f"{name}_support_log1p"] = np.log1p(grp["count"].reindex(idx).fillna(0.0).to_numpy(dtype=float))
    out["r115_server_minus_receiver_rate"] = out["r115_server_rate"] - out["r115_receiver_rate"]
    out["r115_hitter_minus_receiver_rate"] = out["r115_next_hitter_rate"] - out["r115_next_receiver_rate"]
    return out


def r115_server_oof(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    score_train = first_rows_with_score_delta(train_raw)
    score_test = first_rows_with_score_delta(test_raw)
    rows_base = add_scoreboard_features(rows).merge(score_train, on="rally_uid", how="left")
    prefix_base = add_scoreboard_features(prefix).merge(score_train, on="rally_uid", how="left")
    test_base = add_scoreboard_features(test_prefix).merge(score_test, on="rally_uid", how="left")
    for frame in [rows_base, prefix_base, test_base]:
        frame["r115_prev_interval_server_rate"] = frame["r115_prev_interval_server_rate"].fillna(0.5)
        frame["r115_prev_interval_gap"] = frame["r115_prev_interval_gap"].fillna(0.0)
        frame["r115_prev_interval_valid"] = frame["r115_prev_interval_valid"].fillna(0).astype(int)
        frame["r115_public_game_order"] = frame["r115_public_game_order"].fillna(0).astype(int)

    feature_cols = [
        "sex",
        "numberGame",
        "rally_id",
        "prefix_len",
        "next_hitter_is_server",
        "serverScore",
        "receiverScore",
        "serverScoreDiff",
        "scoreTotal",
        "phase_id",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_positionId",
        "lag0_handId",
        "r115_points_to_11_server",
        "r115_points_to_11_receiver",
        "r115_abs_diff",
        "r115_late_game",
        "r115_deuce_like",
        "r115_server_gamepoint",
        "r115_receiver_gamepoint",
        "r115_pressure_x_diff",
        "r115_score_total_sqrt",
        "r115_serve_pair_idx",
        "r115_serve_point_in_pair",
        "r115_deuce_serve_parity",
        "r115_rally_id_log1p",
        "r115_prev_interval_server_rate",
        "r115_prev_interval_gap",
        "r115_prev_interval_valid",
        "r115_public_game_order",
        "r115_server_rate",
        "r115_server_rate_support_log1p",
        "r115_receiver_rate",
        "r115_receiver_rate_support_log1p",
        "r115_next_hitter_rate",
        "r115_next_hitter_rate_support_log1p",
        "r115_next_receiver_rate",
        "r115_next_receiver_rate_support_log1p",
        "r115_pair_rate",
        "r115_pair_rate_support_log1p",
        "r115_server_minus_receiver_rate",
        "r115_hitter_minus_receiver_rate",
    ]
    oof = np.zeros(len(rows_base), dtype=float)
    fold_rows = []
    for fold in sorted(rows_base["fold"].unique()):
        idx = rows_base.index[rows_base["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows_base.loc[idx, "match"])
        train_pool = prefix_base[~prefix_base["match"].isin(valid_matches)].copy()
        train_fold = rate_lookup_features(train_pool.copy(), train_pool)
        valid_fold = rate_lookup_features(rows_base.loc[idx].copy(), train_pool)
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.2,
            reg_lambda=5.0,
            random_state=9115 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        X = train_fold[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = train_fold["serverGetPoint"].astype(int)
        model.fit(X, y, sample_weight=class_weight_sample(y, 2))
        pred = model.predict_proba(valid_fold[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
        oof[idx] = pred
        fold_rows.append({"fold": int(fold), "auc": float(roc_auc_score(valid_fold["serverGetPoint"].astype(int), pred))})

    full_train = rate_lookup_features(prefix_base.copy(), prefix_base)
    full_test = rate_lookup_features(test_base.copy(), prefix_base)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.84,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=99115,
        n_jobs=-1,
        verbosity=-1,
    )
    y_full = full_train["serverGetPoint"].astype(int)
    model.fit(full_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0), y_full, sample_weight=class_weight_sample(y_full, 2))
    test_pred = model.predict_proba(full_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
    return oof, test_pred, pd.DataFrame(fold_rows)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, _ = prepare_prefix_features()
    r111_oof = load_pickle("r111_remaining_moe_gru/oof_proba_r111.pkl")
    r111_test = load_pickle("r111_remaining_moe_gru/test_proba_r111.pkl")
    r101_oof = load_pickle("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
    r101_test = load_pickle("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
    r111_meta = normalize_meta(r111_oof["valid_meta"]).reset_index(drop=True)
    meta = art["valid_meta"].copy().reset_index(drop=True)
    if not r111_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(
        meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]
    ):
        raise ValueError("R111 OOF does not align with shared OOF meta.")
    test_meta = r111_test["test_meta"].reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)
    tuning = r111_oof["tuning"]

    v3_oof = load_pickle("oof_proba_v3.pkl")
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(
        meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]
    ):
        raise ValueError("V3 OOF does not align with R111.")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])

    current_action_oof = build_current_oof_action()
    teacher_action_oof = normalize_rows(0.80 * current_action_oof + 0.20 * art["experts_oof"]["v47_v64_oof_soft"])
    teacher_action_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])

    # R105-style bases.
    r111_action = normalize_rows(0.925 * r111_oof["gru_action"] + 0.075 * teacher_action_oof)
    r111_point = normalize_rows(r111_oof["gru_point"])
    r111_test_action = normalize_rows(0.925 * r111_test["gru_action"] + 0.075 * teacher_action_test)
    r111_test_point = normalize_rows(r111_test["gru_point"])

    r101_action = normalize_rows(0.97 * r101_oof["gru_action"] + 0.03 * teacher_action_oof)
    r101_point = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    r101_test_action = normalize_rows(0.97 * r101_test["gru_action"] + 0.03 * teacher_action_test)
    r101_test_point = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)

    search_rows: list[dict] = []
    generated: list[dict] = []

    base_r111 = eval_probs(meta, r111_action, r111_point, r111_oof["gru_server"], tuning, "base_r111_distill")
    base_r101 = eval_probs(meta, r101_action, r101_point, r101_oof["gru_server"], tuning, "base_r101_distill")
    search_rows.extend([base_r111, base_r101])

    # R112: TLP on R111-distill.
    r112_priors = {}
    for k in [50, 100, 200]:
        key = f"tlp_k{k}_tw0p5"
        r112_priors[key] = {
            "oof": foldsafe_priors(rows, prefix, r111_action, r111_point, mode="tlp", k=k, train_weight=0.5),
            "test": test_priors(test_prefix, prefix, r111_test_action, r111_test_point, mode="tlp", k=k, train_weight=0.5),
        }
    r112_candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, obj in r112_priors.items():
        pa, pp = obj["oof"]
        pa_test, pp_test = obj["test"]
        for w in [0.01, 0.02, 0.03, 0.05, 0.075]:
            for mode in ["selective", "global"]:
                if mode == "selective":
                    low_a = r111_action.max(axis=1) < 0.42
                    high_p = entropy(r111_point) > np.quantile(entropy(r111_point), 0.70)
                    a = r111_action.copy()
                    p = r111_point.copy()
                    a[low_a] = normalize_rows((1.0 - w) * a[low_a] + w * pa[low_a])
                    p[high_p] = normalize_rows((1.0 - w) * p[high_p] + w * pp[high_p])
                    at = r111_test_action.copy()
                    pt = r111_test_point.copy()
                    low_at = r111_test_action.max(axis=1) < 0.42
                    high_pt = entropy(r111_test_point) > np.quantile(entropy(r111_test_point), 0.70)
                    at[low_at] = normalize_rows((1.0 - w) * at[low_at] + w * pa_test[low_at])
                    pt[high_pt] = normalize_rows((1.0 - w) * pt[high_pt] + w * pp_test[high_pt])
                else:
                    a = normalize_rows((1.0 - w) * r111_action + w * pa)
                    p = normalize_rows((1.0 - w) * r111_point + w * pp)
                    at = normalize_rows((1.0 - w) * r111_test_action + w * pa_test)
                    pt = normalize_rows((1.0 - w) * r111_test_point + w * pp_test)
                name = f"r112_{key}_w{clean_float(w)}_{mode}"
                rec = eval_probs(meta, a, p, r111_oof["gru_server"], tuning, name, r111_action, r111_point, r111_oof["gru_server"])
                rec.update({"family": "r112", "prior": key, "weight": w, "mode": mode})
                search_rows.append(rec)
                r112_candidates[name] = (a, p, at, pt)

    # R113: bivariate joint calibration on the best R112-style and R101-style bases.
    # Keep it low-DoF; joint calibration can easily over-correct point.
    joint_bases = {
        "r111_distill": (r111_action, r111_point, r111_test_action, r111_test_point, r111_oof["gru_server"], r111_test["gru_server"]),
        "r101_distill": (r101_action, r101_point, r101_test_action, r101_test_point, r101_oof["gru_server"], r101_test["gru_server"]),
    }
    top_r112_name = max(
        (r for r in search_rows if str(r["candidate"]).startswith("r112_")),
        key=lambda r: r["action_macro_f1"] + r["point_macro_f1"],
    )["candidate"]
    joint_bases["r112_best"] = (*r112_candidates[top_r112_name], r111_oof["gru_server"], r111_test["gru_server"])
    r113_candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for base_name, (ba, bp, bat, bpt, bs, bst) in joint_bases.items():
        for gamma in [0.10, 0.20, 0.35, 0.50, 0.75]:
            ca, cp = joint_calibrate_oof(rows, prefix, ba, bp, gamma)
            cat, cpt = joint_calibrate_test(prefix, bat, bpt, gamma)
            for w in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
                a = normalize_rows((1.0 - w) * ba + w * ca)
                p = normalize_rows((1.0 - w) * bp + w * cp)
                at = normalize_rows((1.0 - w) * bat + w * cat)
                pt = normalize_rows((1.0 - w) * bpt + w * cpt)
                name = f"r113_{base_name}_g{clean_float(gamma)}_w{clean_float(w)}"
                rec = eval_probs(meta, a, p, bs, tuning, name, ba, bp, bs)
                rec.update({"family": "r113", "base": base_name, "gamma": gamma, "weight": w})
                search_rows.append(rec)
                r113_candidates[name] = (a, p, at, pt, bs, bst)

    # R114: entropy-gated dynamic blend.  Pair strong-action R111 with the
    # better-point R101/TLP branch and with tabular/golden teacher action.
    r112_best_a, r112_best_p, r112_best_at, r112_best_pt = r112_candidates[top_r112_name]
    r114_sources = {
        "r111_vs_r101": (r111_action, r101_action, r111_point, r101_point, r111_test_action, r101_test_action, r111_test_point, r101_test_point, r111_oof["gru_server"], r111_test["gru_server"]),
        "r112_vs_r101": (r112_best_a, r101_action, r112_best_p, r101_point, r112_best_at, r101_test_action, r112_best_pt, r101_test_point, r111_oof["gru_server"], r111_test["gru_server"]),
        "r111_vs_teacher_v3point": (r111_action, teacher_action_oof, r111_point, v3_point_oof, r111_test_action, teacher_action_test, r111_test_point, v3_point_test, r111_oof["gru_server"], r111_test["gru_server"]),
    }
    r114_candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for source_name, vals in r114_sources.items():
        a1, a2, p1, p2, at1, at2, pt1, pt2, bs, bst = vals
        for strength in [2.0, 4.0, 6.0, 8.0]:
            for floor in [0.10, 0.20, 0.30]:
                a, p = dynamic_pair_blend(a1, a2, p1, p2, strength=strength, floor=floor, ceiling=1.0 - floor)
                at, pt = dynamic_pair_blend(at1, at2, pt1, pt2, strength=strength, floor=floor, ceiling=1.0 - floor)
                name = f"r114_{source_name}_s{clean_float(strength)}_f{clean_float(floor)}"
                rec = eval_probs(meta, a, p, bs, tuning, name, a1, p1, bs)
                rec.update({"family": "r114", "source": source_name, "strength": strength, "floor": floor})
                search_rows.append(rec)
                r114_candidates[name] = (a, p, at, pt, bs, bst)

    # R115: legal scoreboard server specialist and server blends.
    r115_oof, r115_test, fold_report = r115_server_oof(rows, prefix, test_prefix, train_raw, test_raw)
    fold_report.to_csv(OUTDIR / "r115_fold_report.csv", index=False)
    r115_single = eval_probs(meta, r111_action, r111_point, r115_oof, tuning, "r115_server_single", r111_action, r111_point, r111_oof["gru_server"])
    r115_single.update({"family": "r115"})
    search_rows.append(r115_single)
    server_bases = {
        "r111": (r111_oof["gru_server"], r111_test["gru_server"]),
        "r101": (r101_oof["gru_server"], r101_test["gru_server"]),
    }
    server_blends: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for base_name, (bo, bt) in server_bases.items():
        for w in [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]:
            so = (1.0 - w) * bo + w * r115_oof
            st = (1.0 - w) * bt + w * r115_test
            name = f"r115_{base_name}_server_w{clean_float(w)}"
            rec = eval_probs(meta, r111_action, r111_point, so, tuning, name, r111_action, r111_point, bo)
            rec.update({"family": "r115", "server_base": base_name, "weight": w})
            search_rows.append(rec)
            server_blends[name] = (so, st)

    search = pd.DataFrame(search_rows).sort_values(["overall_local", "action_macro_f1", "point_macro_f1"], ascending=False).reset_index(drop=True)
    search.to_csv(OUTDIR / "r112_r115_search.csv", index=False)

    # Generate submissions for the strongest distinct families.
    def generate_from(name: str, a: np.ndarray, p: np.ndarray, at: np.ndarray, pt: np.ndarray, so: np.ndarray, st: np.ndarray) -> None:
        rec = search[search["candidate"].eq(name)].iloc[0].to_dict()
        generated.append(
            write_submission(
                test_meta,
                at,
                pt,
                st,
                tuning,
                f"submission_{name}.csv",
                rec,
            )
        )

    for _, rec in search[search["family"].eq("r112")].head(2).iterrows():
        name = str(rec["candidate"])
        a, p, at, pt = r112_candidates[name]
        generate_from(name, a, p, at, pt, r111_oof["gru_server"], r111_test["gru_server"])
    for _, rec in search[search["family"].eq("r113")].head(2).iterrows():
        name = str(rec["candidate"])
        a, p, at, pt, so, st = r113_candidates[name]
        generate_from(name, a, p, at, pt, so, st)
    for _, rec in search[search["family"].eq("r114")].head(2).iterrows():
        name = str(rec["candidate"])
        a, p, at, pt, so, st = r114_candidates[name]
        generate_from(name, a, p, at, pt, so, st)
    for _, rec in search[search["family"].eq("r115") & search["candidate"].str.contains("_server_w", regex=False)].head(4).iterrows():
        name = str(rec["candidate"])
        so, st = server_blends[name]
        generate_from(name, r111_action, r111_point, r111_test_action, r111_test_point, so, st)

    # Combined best action/point with best server if they come from different families.
    best_ap = search[~search["family"].eq("r115")].iloc[0]
    best_server = search[search["family"].eq("r115") & search["candidate"].str.contains("_server_w", regex=False)].iloc[0]
    ap_name = str(best_ap["candidate"])
    if ap_name in r112_candidates:
        a, p, at, pt = r112_candidates[ap_name]
        _, st = server_blends[str(best_server["candidate"])]
        combined_name = f"r112r115_combo_{ap_name}_{best_server['candidate']}"
        rec = eval_probs(meta, a, p, server_blends[str(best_server["candidate"])][0], tuning, combined_name, r111_action, r111_point, r111_oof["gru_server"])
        search = pd.concat([search, pd.DataFrame([rec])], ignore_index=True)
        generated.append(write_submission(test_meta, at, pt, st, tuning, f"submission_{combined_name}.csv", rec))

    report = {
        "base": {"r111_distill": base_r111, "r101_distill": base_r101},
        "best_overall": search.sort_values(["overall_local", "action_macro_f1", "point_macro_f1"], ascending=False).head(30).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "R115 uses current/past scoreboard fields and fold-safe train server rates.",
            "Old-test server direct alignment is intentionally not mixed into these default candidates.",
        ],
    }
    (OUTDIR / "r112_r115_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r112_r115_apex.py", "src/analysis/analysis_r112_r115_apex.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
