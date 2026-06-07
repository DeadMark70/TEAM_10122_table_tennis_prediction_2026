"""R116-R119 point/server feature experiments.

R116:
  Past-only scoreboard momentum and pressure server specialist.

R117:
  Opponent-displacement point expert from recent landing geometry.

R118:
  Matchup cluster interaction server specialist using R57 style clusters.

R119:
  Action-point physical target-encoding prior:
  E[P(point | context, candidate_action)] under the current action posterior.
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
from analysis_r57_player_style_clustering import StyleEncoder, add_style_features, observed_rows_for_prefixes
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from analysis_r112_r115_apex import (
    add_scoreboard_features,
    first_rows_with_score_delta,
    r115_server_oof,
    rate_lookup_features,
)
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR, normalize_rows


OUTDIR = Path("r116_r119_point_server")
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


def apply_predictions(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, tuning: GrUTuning) -> tuple[np.ndarray, np.ndarray]:
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    return action_pred.astype(int), point_pred.astype(int)


def eval_candidate(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    tuning: GrUTuning,
    name: str,
    base_action: np.ndarray | None = None,
    base_point: np.ndarray | None = None,
    base_server: np.ndarray | None = None,
) -> dict:
    ap, pp = apply_predictions(meta, action_prob, point_prob, tuning)
    rec = {
        "candidate": name,
        "action_macro_f1": float(f1_score(meta["next_actionId"].astype(int), ap, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "point_macro_f1": float(f1_score(meta["next_pointId"].astype(int), pp, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "server_auc": float(roc_auc_score(meta["serverGetPoint"].astype(int), server_prob)),
    }
    rec["overall_local"] = 0.4 * rec["action_macro_f1"] + 0.4 * rec["point_macro_f1"] + 0.2 * rec["server_auc"]
    if base_action is not None:
        bp, _ = apply_predictions(meta, base_action, point_prob, tuning)
        rec["action_churn"] = float(np.mean(ap != bp))
    if base_point is not None:
        _, bp = apply_predictions(meta, action_prob, base_point, tuning)
        rec["point_churn"] = float(np.mean(pp != bp))
    if base_server is not None:
        rec["server_mad"] = float(np.mean(np.abs(server_prob - base_server)))
    return rec


def write_submission(test_meta, action_prob, point_prob, server_prob, tuning, name: str, extra: dict | None = None) -> dict:
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


POINT_COORD = {
    0: (0.0, -1.0),
    1: (-1.0, 0.0),
    2: (0.0, 0.0),
    3: (1.0, 0.0),
    4: (-1.0, 1.0),
    5: (0.0, 1.0),
    6: (1.0, 1.0),
    7: (-1.0, 2.0),
    8: (0.0, 2.0),
    9: (1.0, 2.0),
}


def _point_xy(value: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = value.fillna(0).astype(int).to_numpy()
    x = np.array([POINT_COORD.get(int(v), (0.0, 0.0))[0] for v in arr], dtype=float)
    y = np.array([POINT_COORD.get(int(v), (0.0, 0.0))[1] for v in arr], dtype=float)
    return x, y


def add_displacement_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in range(4):
        px, py = _point_xy(out.get(f"lag{lag}_pointId", pd.Series(0, index=out.index)))
        out[f"r117_lag{lag}_point_x"] = px
        out[f"r117_lag{lag}_point_y"] = py
    for lag in range(3):
        dx = out[f"r117_lag{lag}_point_x"] - out[f"r117_lag{lag+1}_point_x"]
        dy = out[f"r117_lag{lag}_point_y"] - out[f"r117_lag{lag+1}_point_y"]
        out[f"r117_dist_lag{lag}_{lag+1}"] = np.sqrt(dx * dx + dy * dy)
        out[f"r117_xmove_lag{lag}_{lag+1}"] = np.abs(dx)
        out[f"r117_ymove_lag{lag}_{lag+1}"] = np.abs(dy)
    out["r117_zigzag_last3"] = out["r117_xmove_lag0_1"] + out["r117_xmove_lag1_2"]
    out["r117_total_dist_last3"] = out["r117_dist_lag0_1"] + out["r117_dist_lag1_2"]
    out["r117_total_dist_last4"] = out["r117_total_dist_last3"] + out["r117_dist_lag2_3"]
    out["r117_direction_reversal_x"] = (
        np.sign(out["r117_lag0_point_x"] - out["r117_lag1_point_x"]) * np.sign(out["r117_lag1_point_x"] - out["r117_lag2_point_x"]) < 0
    ).astype(int)
    out["r117_depth_reversal_y"] = (
        np.sign(out["r117_lag0_point_y"] - out["r117_lag1_point_y"]) * np.sign(out["r117_lag1_point_y"] - out["r117_lag2_point_y"]) < 0
    ).astype(int)
    out["r117_forced_error_pressure"] = out["r117_total_dist_last3"] * (1.0 + 0.3 * out["r117_direction_reversal_x"])
    return out


def add_momentum_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = add_scoreboard_features(df)
    score = first_rows_with_score_delta(raw)
    out = out.merge(score, on="rally_uid", how="left")
    out["r115_prev_interval_server_rate"] = out["r115_prev_interval_server_rate"].fillna(0.5)
    out["r115_prev_interval_gap"] = out["r115_prev_interval_gap"].fillna(0.0)
    out["r115_prev_interval_valid"] = out["r115_prev_interval_valid"].fillna(0).astype(int)
    out["r115_public_game_order"] = out["r115_public_game_order"].fillna(0).astype(int)
    pressure = out["scoreTotal"].astype(float) / (np.abs(out["serverScoreDiff"].astype(float)) + 1.0)
    out["r116_pressure_index"] = pressure
    out["r116_clutch_9_9_plus"] = ((out["serverScore"] >= 9) & (out["receiverScore"] >= 9)).astype(int)
    out["r116_prev_server_win_flag"] = (out["r115_prev_interval_server_rate"] > 0.5).astype(int)
    out["r116_prev_receiver_win_flag"] = (out["r115_prev_interval_server_rate"] < 0.5).astype(int)
    out["r116_prev_interval_margin"] = np.abs(out["r115_prev_interval_server_rate"] - 0.5)
    out["r116_prev_momentum_signed"] = (out["r115_prev_interval_server_rate"] - 0.5) * out["r115_prev_interval_gap"]
    out["r116_pressure_x_prev_momentum"] = out["r116_pressure_index"] * out["r116_prev_momentum_signed"]
    out["r116_late_x_prev_server_win"] = out["r115_late_game"] * out["r116_prev_server_win_flag"]
    return out


def train_point_expert_oof(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    base_features: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows_feat = add_displacement_features(rows)
    prefix_feat = add_displacement_features(prefix)
    test_feat = add_displacement_features(test_prefix)
    r117_cols = [c for c in rows_feat.columns if c.startswith("r117_")]
    features = base_features + [c for c in r117_cols if c not in base_features]
    oof = np.zeros((len(rows_feat), 10), dtype=float)
    folds = []
    for fold in sorted(rows_feat["fold"].unique()):
        idx = rows_feat.index[rows_feat["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows_feat.loc[idx, "match"])
        train_df = prefix_feat[~prefix_feat["match"].isin(valid_matches)].copy()
        valid_df = rows_feat.loc[idx].copy()
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=10,
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=47,
            min_child_samples=28,
            subsample=0.86,
            colsample_bytree=0.82,
            reg_lambda=4.0,
            random_state=11700 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        y = train_df["next_pointId"].astype(int)
        model.fit(train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0), y, sample_weight=class_weight_sample(y, 1.2))
        proba = model.predict_proba(valid_df[features].replace([np.inf, -np.inf], np.nan).fillna(0))
        aligned = np.zeros((len(valid_df), 10), dtype=float)
        for j, cls in enumerate(model.classes_.astype(int)):
            aligned[:, cls] = proba[:, j]
        oof[idx] = normalize_rows(aligned)
        pred = oof[idx].argmax(axis=1)
        folds.append({"fold": int(fold), "point_macro_f1_raw_argmax": float(f1_score(valid_df["next_pointId"].astype(int), pred, average="macro", labels=POINT_CLASSES, zero_division=0))})
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=10,
        n_estimators=520,
        learning_rate=0.025,
        num_leaves=47,
        min_child_samples=28,
        subsample=0.88,
        colsample_bytree=0.84,
        reg_lambda=4.0,
        random_state=11799,
        n_jobs=-1,
        verbosity=-1,
    )
    y_full = prefix_feat["next_pointId"].astype(int)
    model.fit(prefix_feat[features].replace([np.inf, -np.inf], np.nan).fillna(0), y_full, sample_weight=class_weight_sample(y_full, 1.2))
    proba = model.predict_proba(test_feat[features].replace([np.inf, -np.inf], np.nan).fillna(0))
    test_prob = np.zeros((len(test_feat), 10), dtype=float)
    for j, cls in enumerate(model.classes_.astype(int)):
        test_prob[:, cls] = proba[:, j]
    return normalize_rows(oof), normalize_rows(test_prob), pd.DataFrame(folds)


def train_r116_server_oof(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows_base = add_momentum_features(rows, train_raw)
    prefix_base = add_momentum_features(prefix, train_raw)
    test_base = add_momentum_features(test_prefix, test_raw)
    base_cols = [
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
    ]
    extra_cols = [c for c in rows_base.columns if c.startswith("r115_") or c.startswith("r116_")]
    rate_cols = [
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
    features = base_cols + extra_cols + rate_cols
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
            n_estimators=620,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.86,
            colsample_bytree=0.84,
            reg_lambda=5.0,
            random_state=11600 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        y = train_fold["serverGetPoint"].astype(int)
        model.fit(train_fold[features].replace([np.inf, -np.inf], np.nan).fillna(0), y, sample_weight=class_weight_sample(y, 2))
        pred = model.predict_proba(valid_fold[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
        oof[idx] = pred
        fold_rows.append({"fold": int(fold), "auc": float(roc_auc_score(valid_fold["serverGetPoint"].astype(int), pred))})
    full_train = rate_lookup_features(prefix_base.copy(), prefix_base)
    full_test = rate_lookup_features(test_base.copy(), prefix_base)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=720,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.88,
        colsample_bytree=0.84,
        reg_lambda=5.0,
        random_state=11699,
        n_jobs=-1,
        verbosity=-1,
    )
    y_full = full_train["serverGetPoint"].astype(int)
    model.fit(full_train[features].replace([np.inf, -np.inf], np.nan).fillna(0), y_full, sample_weight=class_weight_sample(y_full, 2))
    test_pred = model.predict_proba(full_test[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
    return oof, test_pred, pd.DataFrame(fold_rows)


def train_r118_style_server_oof(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows_base = add_momentum_features(rows, train_raw)
    prefix_base = add_momentum_features(prefix, train_raw)
    test_base = add_momentum_features(test_prefix, test_raw)
    base_cols = [
        "sex",
        "numberGame",
        "rally_id",
        "prefix_len",
        "next_hitter_is_server",
        "serverScoreDiff",
        "scoreTotal",
        "phase_id",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "r116_pressure_index",
        "r116_prev_momentum_signed",
    ]
    oof = np.zeros(len(rows_base), dtype=float)
    fold_rows = []
    for fold in sorted(rows_base["fold"].unique()):
        idx = rows_base.index[rows_base["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows_base.loc[idx, "match"])
        train_pool = prefix_base[~prefix_base["match"].isin(valid_matches)].copy()
        obs = observed_rows_for_prefixes(train_raw, train_pool)
        encoder = StyleEncoder(k=8, alpha=25.0, beta=25.0, seed=11800 + int(fold)).fit(obs, obs)
        train_style = add_style_features(train_pool, encoder)
        valid_style = add_style_features(rows_base.loc[idx].copy(), encoder)
        for df in [train_style, valid_style]:
            df["r118_matchup_cluster"] = df["style_hitter_cluster_top"].astype(int) * 8 + df["style_receiver_cluster_top"].astype(int)
            df["r118_matchup_conf"] = df["style_hitter_confidence"] * df["style_receiver_confidence"]
            df["r118_matchup_trust"] = df["style_hitter_trust"] * df["style_receiver_trust"]
        style_cols = [c for c in train_style.columns if c.startswith("style_")] + ["r118_matchup_cluster", "r118_matchup_conf", "r118_matchup_trust"]
        features = base_cols + style_cols
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=520,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=35,
            subsample=0.86,
            colsample_bytree=0.78,
            reg_lambda=6.0,
            random_state=11800 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        y = train_style["serverGetPoint"].astype(int)
        model.fit(train_style[features].replace([np.inf, -np.inf], np.nan).fillna(0), y, sample_weight=class_weight_sample(y, 2))
        pred = model.predict_proba(valid_style[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
        oof[idx] = pred
        fold_rows.append({"fold": int(fold), "auc": float(roc_auc_score(valid_style["serverGetPoint"].astype(int), pred))})
    obs_full = pd.concat([train_raw, observed_rows_for_prefixes(test_raw, test_base)], ignore_index=True)
    encoder = StyleEncoder(k=8, alpha=25.0, beta=25.0, seed=11899).fit(obs_full, train_raw)
    train_style = add_style_features(prefix_base, encoder)
    test_style = add_style_features(test_base, encoder)
    for df in [train_style, test_style]:
        df["r118_matchup_cluster"] = df["style_hitter_cluster_top"].astype(int) * 8 + df["style_receiver_cluster_top"].astype(int)
        df["r118_matchup_conf"] = df["style_hitter_confidence"] * df["style_receiver_confidence"]
        df["r118_matchup_trust"] = df["style_hitter_trust"] * df["style_receiver_trust"]
    style_cols = [c for c in train_style.columns if c.startswith("style_")] + ["r118_matchup_cluster", "r118_matchup_conf", "r118_matchup_trust"]
    features = base_cols + style_cols
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=620,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.88,
        colsample_bytree=0.78,
        reg_lambda=6.0,
        random_state=11899,
        n_jobs=-1,
        verbosity=-1,
    )
    y_full = train_style["serverGetPoint"].astype(int)
    model.fit(train_style[features].replace([np.inf, -np.inf], np.nan).fillna(0), y_full, sample_weight=class_weight_sample(y_full, 2))
    test_pred = model.predict_proba(test_style[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
    return oof, test_pred, pd.DataFrame(fold_rows)


def make_point_lookup(pool: pd.DataFrame, key_cols: list[str], alpha: float, global_prior: np.ndarray) -> dict[tuple, np.ndarray]:
    out: dict[tuple, np.ndarray] = {}
    for key, g in pool.groupby(key_cols, dropna=False):
        counts = np.bincount(g["next_pointId"].astype(int), minlength=10).astype(float)
        out[key if isinstance(key, tuple) else (key,)] = normalize_rows((counts + alpha * global_prior)[None, :])[0]
    return out


def action_conditioned_point_prior(
    rows: pd.DataFrame,
    pool: pd.DataFrame,
    action_prob: np.ndarray,
    alpha: float = 40.0,
) -> np.ndarray:
    global_prior = np.bincount(pool["next_pointId"].astype(int), minlength=10).astype(float) + 1.0
    global_prior /= global_prior.sum()
    pool = pool.copy()
    pool["target_action_for_prior"] = pool["next_actionId"].astype(int)
    lookups = [
        (["phase_id", "lag0_actionId", "lag0_pointId", "lag0_spinId", "target_action_for_prior"], make_point_lookup(pool, ["phase_id", "lag0_actionId", "lag0_pointId", "lag0_spinId", "target_action_for_prior"], alpha, global_prior)),
        (["phase_id", "lag0_pointId", "lag0_spinId", "target_action_for_prior"], make_point_lookup(pool, ["phase_id", "lag0_pointId", "lag0_spinId", "target_action_for_prior"], alpha, global_prior)),
        (["lag0_pointId", "target_action_for_prior"], make_point_lookup(pool, ["lag0_pointId", "target_action_for_prior"], alpha, global_prior)),
        (["target_action_for_prior"], make_point_lookup(pool, ["target_action_for_prior"], alpha, global_prior)),
    ]
    result = np.zeros((len(rows), 10), dtype=float)
    row_values = rows.reset_index(drop=True)
    for i, row in row_values.iterrows():
        mix = np.zeros(10, dtype=float)
        for a in range(19):
            dist = None
            for cols, lookup in lookups:
                key_vals = []
                for c in cols:
                    key_vals.append(a if c == "target_action_for_prior" else row[c])
                dist = lookup.get(tuple(key_vals))
                if dist is not None:
                    break
            if dist is None:
                dist = global_prior
            mix += action_prob[i, a] * dist
        result[i] = mix
    return normalize_rows(result)


def r119_oof_prior(rows: pd.DataFrame, prefix: pd.DataFrame, action_prob: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), 10), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].copy()
        out[idx] = action_conditioned_point_prior(rows.loc[idx], pool, action_prob[idx])
    return out


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, base_features = prepare_prefix_features()
    r111_oof = load_pickle("r111_remaining_moe_gru/oof_proba_r111.pkl")
    r111_test = load_pickle("r111_remaining_moe_gru/test_proba_r111.pkl")
    r101_oof = load_pickle("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
    r101_test = load_pickle("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
    meta = art["valid_meta"].copy().reset_index(drop=True)
    r111_meta = normalize_meta(r111_oof["valid_meta"]).reset_index(drop=True)
    if not r111_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]):
        raise ValueError("R111 OOF does not align.")
    rows = align_prefix_meta(meta, prefix)
    tuning = r111_oof["tuning"]
    test_meta = r111_test["test_meta"].reset_index(drop=True)

    v3_oof = load_pickle("oof_proba_v3.pkl")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    current_action_oof = build_current_oof_action()
    teacher_action_oof = normalize_rows(0.80 * current_action_oof + 0.20 * art["experts_oof"]["v47_v64_oof_soft"])
    teacher_action_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])

    base_action_oof = normalize_rows(0.925 * r111_oof["gru_action"] + 0.075 * teacher_action_oof)
    base_action_test = normalize_rows(0.925 * r111_test["gru_action"] + 0.075 * teacher_action_test)
    base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)

    # Anchor server: R115/R101 w=0.2 from previous experiment, recomputed here.
    r115_oof, r115_test, r115_folds = r115_server_oof(rows, prefix, test_prefix, train_raw, test_raw)
    anchor_server_oof = 0.8 * r101_oof["gru_server"] + 0.2 * r115_oof
    anchor_server_test = 0.8 * r101_test["gru_server"] + 0.2 * r115_test

    search_rows = []
    generated = []
    base_rec = eval_candidate(meta, base_action_oof, base_point_oof, anchor_server_oof, tuning, "mixed_anchor_r111action_r101point_r115server")
    search_rows.append(base_rec)

    # R116.
    r116_oof, r116_test, r116_folds = train_r116_server_oof(rows, prefix, test_prefix, train_raw, test_raw)
    r116_folds.to_csv(OUTDIR / "r116_fold_report.csv", index=False)
    for w in [0.05, 0.10, 0.20, 0.35, 0.50]:
        server = (1.0 - w) * anchor_server_oof + w * r116_oof
        rec = eval_candidate(meta, base_action_oof, base_point_oof, server, tuning, f"r116_server_w{clean_float(w)}", base_action_oof, base_point_oof, anchor_server_oof)
        rec.update({"family": "r116", "weight": w})
        search_rows.append(rec)

    # R117.
    r117_oof, r117_test, r117_folds = train_point_expert_oof(rows, prefix, test_prefix, base_features)
    r117_folds.to_csv(OUTDIR / "r117_fold_report.csv", index=False)
    for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
        point = normalize_rows((1.0 - w) * base_point_oof + w * r117_oof)
        rec = eval_candidate(meta, base_action_oof, point, anchor_server_oof, tuning, f"r117_point_w{clean_float(w)}", base_action_oof, base_point_oof, anchor_server_oof)
        rec.update({"family": "r117", "weight": w})
        search_rows.append(rec)

    # R118.
    r118_oof, r118_test, r118_folds = train_r118_style_server_oof(rows, prefix, test_prefix, train_raw, test_raw)
    r118_folds.to_csv(OUTDIR / "r118_fold_report.csv", index=False)
    for w in [0.05, 0.10, 0.20, 0.35, 0.50]:
        server = (1.0 - w) * anchor_server_oof + w * r118_oof
        rec = eval_candidate(meta, base_action_oof, base_point_oof, server, tuning, f"r118_server_w{clean_float(w)}", base_action_oof, base_point_oof, anchor_server_oof)
        rec.update({"family": "r118", "weight": w})
        search_rows.append(rec)

    # R119.
    r119_oof = r119_oof_prior(rows, prefix, base_action_oof)
    r119_test = action_conditioned_point_prior(test_prefix, prefix, base_action_test)
    for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
        point = normalize_rows((1.0 - w) * base_point_oof + w * r119_oof)
        rec = eval_candidate(meta, base_action_oof, point, anchor_server_oof, tuning, f"r119_point_w{clean_float(w)}", base_action_oof, base_point_oof, anchor_server_oof)
        rec.update({"family": "r119", "weight": w})
        search_rows.append(rec)

    search = pd.DataFrame(search_rows).sort_values(["overall_local", "point_macro_f1", "server_auc"], ascending=False).reset_index(drop=True)
    search.to_csv(OUTDIR / "r116_r119_search.csv", index=False)

    # Generate family-best candidates.
    for _, rec in search[search["family"].isin(["r116", "r118"])].head(4).iterrows():
        name = str(rec["candidate"])
        if name.startswith("r116"):
            w = float(rec["weight"])
            server_test = (1.0 - w) * anchor_server_test + w * r116_test
        else:
            w = float(rec["weight"])
            server_test = (1.0 - w) * anchor_server_test + w * r118_test
        generated.append(write_submission(test_meta, base_action_test, base_point_test, server_test, tuning, f"submission_{name}.csv", rec.to_dict()))
    for _, rec in search[search["family"].isin(["r117", "r119"])].head(4).iterrows():
        name = str(rec["candidate"])
        w = float(rec["weight"])
        if name.startswith("r117"):
            point_test = normalize_rows((1.0 - w) * base_point_test + w * r117_test)
        else:
            point_test = normalize_rows((1.0 - w) * base_point_test + w * r119_test)
        generated.append(write_submission(test_meta, base_action_test, point_test, anchor_server_test, tuning, f"submission_{name}.csv", rec.to_dict()))

    # Combined best point branch + best server branch.
    best_point = search[search["family"].isin(["r117", "r119"])].iloc[0]
    best_server = search[search["family"].isin(["r116", "r118"])].iloc[0]
    point_w = float(best_point["weight"])
    server_w = float(best_server["weight"])
    point_oof = normalize_rows((1.0 - point_w) * base_point_oof + point_w * (r117_oof if str(best_point["candidate"]).startswith("r117") else r119_oof))
    point_test = normalize_rows((1.0 - point_w) * base_point_test + point_w * (r117_test if str(best_point["candidate"]).startswith("r117") else r119_test))
    server_oof = (1.0 - server_w) * anchor_server_oof + server_w * (r116_oof if str(best_server["candidate"]).startswith("r116") else r118_oof)
    server_test = (1.0 - server_w) * anchor_server_test + server_w * (r116_test if str(best_server["candidate"]).startswith("r116") else r118_test)
    combo_name = f"r116r119_combo_{best_point['candidate']}_{best_server['candidate']}"
    combo_rec = eval_candidate(meta, base_action_oof, point_oof, server_oof, tuning, combo_name, base_action_oof, base_point_oof, anchor_server_oof)
    search = pd.concat([search, pd.DataFrame([combo_rec])], ignore_index=True)
    generated.append(write_submission(test_meta, base_action_test, point_test, server_test, tuning, f"submission_{combo_name}.csv", combo_rec))

    report = {
        "base": base_rec,
        "best": search.sort_values(["overall_local", "point_macro_f1", "server_auc"], ascending=False).head(30).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "R116/R118 only change server.",
            "R117/R119 only change point.",
            "rally_id is used as a scoreboard/order feature; rally_uid is never used for ordering.",
        ],
    }
    (OUTDIR / "r116_r119_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r116_r119_point_server.py", "src/analysis/analysis_r116_r119_point_server.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
