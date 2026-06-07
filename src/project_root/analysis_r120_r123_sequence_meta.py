"""R120-R123 sequence context and meta-learning experiments.

R120:
  N-gram motif target-encoding priors for action/point.

R121:
  Fold-safe server probability trajectory over prefixes within the same rally.

R122:
  Cross-task pseudo-conditioning: use strong server posterior to adjust point0.

R123:
  Player point-entropy profile as a confidence gate for R119 physical point TE.
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
from analysis_r7_phase_features import add_phase_features
from analysis_r57_player_style_clustering import add_player_id_features, observed_rows_for_prefixes
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from analysis_r112_r115_apex import r115_server_oof
from analysis_r116_r119_point_server import (
    action_conditioned_point_prior,
    r119_oof_prior,
    train_r118_style_server_oof,
)
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    _base_prefix_features,
    _empty_counts,
    _increment_counts,
    add_role_and_score_features,
    build_train_prefix_table,
    class_weight_sample,
)
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR, normalize_rows


OUTDIR = Path("r120_r123_sequence_meta")
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


def ensure_role_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"server_id", "receiver_id", "serverScore", "receiverScore", "serverScoreDiff", "scoreTotal"}
    if required.issubset(df.columns):
        return df.copy()
    stale = [
        c
        for c in ["server_id", "receiver_id", "serverScore", "receiverScore", "serverScoreDiff", "scoreTotal", "is_server_hitter"]
        if c in df.columns
    ]
    base = df.drop(columns=stale) if stale else df
    return add_role_and_score_features(base)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def apply_predictions(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, tuning: GrUTuning) -> tuple[np.ndarray, np.ndarray]:
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    return action_pred.astype(int), point_pred.astype(int)


def eval_candidate(meta, action_prob, point_prob, server_prob, tuning, name, base_action=None, base_point=None, base_server=None) -> dict:
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


def build_test_all_prefix_table(test_raw: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    rows: list[dict] = []
    test = ensure_role_features(test_raw)
    for _, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        counts = _empty_counts()
        seen = {field: set() for field in ["actionId", "pointId", "spinId", "handId", "positionId"]}
        for t_index in range(len(group)):
            _increment_counts(counts, group.iloc[t_index])
            for field in seen:
                seen[field].add(int(group.iloc[t_index][field]))
            feats = _base_prefix_features(group, t_index, max_lag, counts, {field: len(values) for field, values in seen.items()})
            feats.update({"rally_uid": int(group.iloc[0]["rally_uid"]), "match": int(group.iloc[0]["match"])})
            rows.append(feats)
    return pd.DataFrame(rows)


def motif_lookup(pool: pd.DataFrame, key_cols: list[str], target: str, n_classes: int, alpha: float = 50.0) -> tuple[dict[tuple, np.ndarray], np.ndarray]:
    global_prior = np.bincount(pool[target].astype(int), minlength=n_classes).astype(float) + 1.0
    global_prior /= global_prior.sum()
    lookup = {}
    for key, g in pool.groupby(key_cols, dropna=False):
        counts = np.bincount(g[target].astype(int), minlength=n_classes).astype(float)
        lookup[key if isinstance(key, tuple) else (key,)] = normalize_rows((counts + alpha * global_prior)[None, :])[0]
    return lookup, global_prior


def apply_motif_prior(rows: pd.DataFrame, pool: pd.DataFrame, target: str, n_classes: int) -> np.ndarray:
    specs = [
        ["phase_id", "lag2_actionId", "lag1_actionId", "lag0_actionId"],
        ["phase_id", "lag1_actionId", "lag0_actionId", "lag0_spinId"],
        ["phase_id", "lag1_pointId", "lag0_pointId", "lag0_actionId"],
        ["lag1_actionId", "lag0_actionId"],
        ["lag0_actionId"],
    ]
    lookups = [(*motif_lookup(pool, cols, target, n_classes), cols) for cols in specs]
    out = np.zeros((len(rows), n_classes), dtype=float)
    for i, (_, row) in enumerate(rows.iterrows()):
        dist = None
        for lookup, global_prior, cols in lookups:
            key = tuple(row[c] for c in cols)
            dist = lookup.get(key)
            if dist is not None:
                break
        if dist is None:
            dist = lookups[-1][1]
        out[i] = dist
    return normalize_rows(out)


def r120_motif_oof(rows: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    action = np.zeros((len(rows), 19), dtype=float)
    point = np.zeros((len(rows), 10), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].copy()
        action[idx] = apply_motif_prior(rows.loc[idx], pool, "next_actionId", 19)
        point[idx] = apply_motif_prior(rows.loc[idx], pool, "next_pointId", 10)
    return action, point


def r122_point0_adjust(point_prob: np.ndarray, rows: pd.DataFrame, server_prob: np.ndarray, beta: float) -> np.ndarray:
    out = point_prob.copy()
    next_strike = rows["prefix_len"].astype(int).to_numpy() + 1
    terminal_server_win = (next_strike % 2 == 0).astype(float)
    compat = np.where(terminal_server_win == 1.0, server_prob, 1.0 - server_prob)
    out[:, 0] *= np.exp(beta * (compat - 0.5))
    return normalize_rows(out)


def point_entropy_profiles_oof(rows: pd.DataFrame, prefix: pd.DataFrame, train_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hitter_entropy = np.zeros(len(rows), dtype=float)
    receiver_entropy = np.zeros(len(rows), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        obs_prefix = prefix[~prefix["match"].isin(valid_matches)]
        obs = observed_rows_for_prefixes(train_raw, obs_prefix)
        ent_map, global_ent = point_entropy_map(obs)
        hitter_entropy[idx] = rows.loc[idx, "next_hitter_id"].map(ent_map).fillna(global_ent).to_numpy(dtype=float)
        receiver_entropy[idx] = rows.loc[idx, "next_receiver_id"].map(ent_map).fillna(global_ent).to_numpy(dtype=float)
    return hitter_entropy, receiver_entropy


def point_entropy_map(observed_raw: pd.DataFrame) -> tuple[dict[int, float], float]:
    eps = 1e-12
    counts_global = np.bincount(observed_raw["pointId"].astype(int), minlength=10).astype(float) + 1.0
    p_global = counts_global / counts_global.sum()
    global_ent = float(-np.sum(p_global * np.log(p_global + eps)) / np.log(10))
    ent = {}
    for pid, g in observed_raw.groupby("gamePlayerId"):
        counts = np.bincount(g["pointId"].astype(int), minlength=10).astype(float) + 3.0 * p_global
        p = counts / counts.sum()
        ent[int(pid)] = float(-np.sum(p * np.log(p + eps)) / np.log(10))
    return ent, global_ent


def server_prefix_model_trajectory(
    prefix: pd.DataFrame,
    test_all: pd.DataFrame,
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    features = [
        "sex",
        "numberGame",
        "rally_id",
        "prefix_len",
        "prefix_len_is_odd",
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
        "lag1_actionId",
        "lag1_pointId",
        "lag2_actionId",
        "lag2_pointId",
    ]
    prefix_pred = np.zeros(len(prefix), dtype=float)
    folds = []
    for fold in sorted(rows["fold"].unique()):
        valid_matches = set(rows.loc[rows["fold"].eq(fold), "match"])
        train_pool = prefix[~prefix["match"].isin(valid_matches)].copy()
        valid_pool = prefix[prefix["match"].isin(valid_matches)].copy()
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=35,
            subsample=0.86,
            colsample_bytree=0.84,
            reg_lambda=4.0,
            random_state=12100 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        y = train_pool["serverGetPoint"].astype(int)
        model.fit(train_pool[features].replace([np.inf, -np.inf], np.nan).fillna(0), y, sample_weight=class_weight_sample(y, 2))
        pred = model.predict_proba(valid_pool[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
        prefix_pred[valid_pool.index.to_numpy()] = pred
        folds.append({"fold": int(fold), "prefix_rows": int(len(valid_pool)), "auc_prefix": float(roc_auc_score(valid_pool["serverGetPoint"].astype(int), pred))})
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=420,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.88,
        colsample_bytree=0.84,
        reg_lambda=4.0,
        random_state=12199,
        n_jobs=-1,
        verbosity=-1,
    )
    y_full = prefix["serverGetPoint"].astype(int)
    model.fit(prefix[features].replace([np.inf, -np.inf], np.nan).fillna(0), y_full, sample_weight=class_weight_sample(y_full, 2))
    test_pred = model.predict_proba(test_all[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
    return prefix_pred, test_pred, {"folds": folds}


def trajectory_from_prefix_predictions(prefix_like: pd.DataFrame, pred: np.ndarray, target_rows: pd.DataFrame) -> np.ndarray:
    tmp = prefix_like[["rally_uid", "prefix_len"]].copy()
    tmp["server_prefix_prob"] = pred
    rows = []
    for _, row in target_rows[["rally_uid", "prefix_len"]].iterrows():
        g = tmp[tmp["rally_uid"].eq(row["rally_uid"]) & tmp["prefix_len"].le(row["prefix_len"])]
        values = g.sort_values("prefix_len")["server_prefix_prob"].to_numpy(dtype=float)
        if len(values) == 0:
            values = np.array([0.5])
        last = values[-1]
        prev = values[-2] if len(values) >= 2 else values[-1]
        rows.append([last, values.mean(), values.max(), values.min(), last - prev, values.max() - values.min()])
    return np.asarray(rows, dtype=float)


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
    meta = art["valid_meta"].copy().reset_index(drop=True)
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

    r115_oof, r115_test, _ = r115_server_oof(rows, prefix, test_prefix, train_raw, test_raw)
    r118_oof, r118_test, r118_fold = train_r118_style_server_oof(rows, prefix, test_prefix, train_raw, test_raw)
    anchor_server_oof = 0.8 * r101_oof["gru_server"] + 0.2 * r118_oof
    anchor_server_test = 0.8 * r101_test["gru_server"] + 0.2 * r118_test

    search_rows = []
    generated = []
    base_rec = eval_candidate(meta, base_action_oof, base_point_oof, anchor_server_oof, tuning, "anchor_r111action_r101point_r118server")
    search_rows.append(base_rec)

    # R120 motif priors.
    motif_a_oof, motif_p_oof = r120_motif_oof(rows, prefix)
    motif_a_test = apply_motif_prior(test_prefix, prefix, "next_actionId", 19)
    motif_p_test = apply_motif_prior(test_prefix, prefix, "next_pointId", 10)
    for aw in [0.01, 0.02, 0.03, 0.05]:
        for pw in [0.03, 0.05, 0.075, 0.10]:
            a = normalize_rows((1 - aw) * base_action_oof + aw * motif_a_oof)
            p = normalize_rows((1 - pw) * base_point_oof + pw * motif_p_oof)
            rec = eval_candidate(meta, a, p, anchor_server_oof, tuning, f"r120_motif_aw{clean_float(aw)}_pw{clean_float(pw)}", base_action_oof, base_point_oof, anchor_server_oof)
            rec.update({"family": "r120", "aw": aw, "pw": pw})
            search_rows.append(rec)

    # R121 server probability trajectory.
    test_all = build_test_all_prefix_table(test_raw, 6)
    test_role = ensure_role_features(test_raw)
    test_all = add_phase_features(test_all, test_role)
    test_all = add_player_id_features(test_all, test_role)
    prefix_pred, test_all_pred, traj_info = server_prefix_model_trajectory(prefix, test_all, rows)
    pd.DataFrame(traj_info["folds"]).to_csv(OUTDIR / "r121_prefix_fold_report.csv", index=False)
    traj_oof = trajectory_from_prefix_predictions(prefix, prefix_pred, rows)
    traj_test = trajectory_from_prefix_predictions(test_all, test_all_pred, test_prefix)
    # Use last/mean/max trajectory signals directly as posteriors.
    for col, label in [(0, "last"), (1, "mean"), (2, "max"), (3, "min")]:
        for w in [0.05, 0.10, 0.20, 0.35]:
            s = (1 - w) * anchor_server_oof + w * traj_oof[:, col]
            rec = eval_candidate(meta, base_action_oof, base_point_oof, s, tuning, f"r121_traj_{label}_w{clean_float(w)}", base_action_oof, base_point_oof, anchor_server_oof)
            rec.update({"family": "r121", "weight": w, "signal": label})
            search_rows.append(rec)

    # R122 point0 cross-task conditioning.
    for beta in [0.15, 0.25, 0.35, 0.50, 0.75, 1.0]:
        p = r122_point0_adjust(base_point_oof, rows, anchor_server_oof, beta)
        rec = eval_candidate(meta, base_action_oof, p, anchor_server_oof, tuning, f"r122_point0_beta{clean_float(beta)}", base_action_oof, base_point_oof, anchor_server_oof)
        rec.update({"family": "r122", "beta": beta})
        search_rows.append(rec)

    # R123 entropy-gated R119.
    r119_oof = r119_oof_prior(rows, prefix, base_action_oof)
    r119_test = action_conditioned_point_prior(test_prefix, prefix, base_action_test)
    ent_h, ent_r = point_entropy_profiles_oof(rows, prefix, train_raw)
    ent_map, global_ent = point_entropy_map(pd.concat([train_raw, observed_rows_for_prefixes(test_raw, test_prefix)], ignore_index=True))
    ent_h_test = test_prefix["next_hitter_id"].map(ent_map).fillna(global_ent).to_numpy(dtype=float)
    ent_r_test = test_prefix["next_receiver_id"].map(ent_map).fillna(global_ent).to_numpy(dtype=float)
    confidence_oof = np.clip(1.0 - 0.5 * (ent_h + ent_r), 0.0, 1.0)
    confidence_test = np.clip(1.0 - 0.5 * (ent_h_test + ent_r_test), 0.0, 1.0)
    for base_w in [0.05, 0.075, 0.10, 0.15]:
        for floor in [0.02, 0.03, 0.05]:
            w_oof = np.clip(floor + base_w * confidence_oof, 0.0, 0.20)
            w_test = np.clip(floor + base_w * confidence_test, 0.0, 0.20)
            p = normalize_rows((1 - w_oof[:, None]) * base_point_oof + w_oof[:, None] * r119_oof)
            rec = eval_candidate(meta, base_action_oof, p, anchor_server_oof, tuning, f"r123_entropy_gate_bw{clean_float(base_w)}_f{clean_float(floor)}", base_action_oof, base_point_oof, anchor_server_oof)
            rec.update({"family": "r123", "base_weight": base_w, "floor": floor, "mean_weight": float(w_oof.mean())})
            search_rows.append(rec)

    search = pd.DataFrame(search_rows).sort_values(["overall_local", "point_macro_f1", "server_auc"], ascending=False).reset_index(drop=True)
    search.to_csv(OUTDIR / "r120_r123_search.csv", index=False)

    # Generate best family candidates.
    for _, rec in search.head(8).iterrows():
        name = str(rec["candidate"])
        a_test = base_action_test
        p_test = base_point_test
        s_test = anchor_server_test
        if name.startswith("r120"):
            aw = float(rec["aw"])
            pw = float(rec["pw"])
            a_test = normalize_rows((1 - aw) * base_action_test + aw * motif_a_test)
            p_test = normalize_rows((1 - pw) * base_point_test + pw * motif_p_test)
        elif name.startswith("r121"):
            w = float(rec["weight"])
            signal = str(rec["signal"])
            col = {"last": 0, "mean": 1, "max": 2, "min": 3}[signal]
            s_test = (1 - w) * anchor_server_test + w * traj_test[:, col]
        elif name.startswith("r122"):
            beta = float(rec["beta"])
            p_test = r122_point0_adjust(base_point_test, test_prefix, anchor_server_test, beta)
        elif name.startswith("r123"):
            bw = float(rec["base_weight"])
            floor = float(rec["floor"])
            w_test = np.clip(floor + bw * confidence_test, 0.0, 0.20)
            p_test = normalize_rows((1 - w_test[:, None]) * base_point_test + w_test[:, None] * r119_test)
        generated.append(write_submission(test_meta, a_test, p_test, s_test, tuning, f"submission_{name}.csv", rec.to_dict()))

    report = {
        "base": base_rec,
        "best": search.head(30).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "R120 uses fold-safe motif TE/backoff.",
            "R121 uses server probability trajectory over prefixes in each rally.",
            "R122 uses server posterior to adjust pointId=0.",
            "R123 gates R119 by player point entropy.",
        ],
    }
    (OUTDIR / "r120_r123_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r120_r123_sequence_meta.py", "src/analysis/analysis_r120_r123_sequence_meta.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
