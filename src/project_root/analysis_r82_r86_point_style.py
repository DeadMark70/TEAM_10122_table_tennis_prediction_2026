"""R82-R86 point/action extensions around the public-validated R67 branch.

R82:
  Project R67 action probabilities through fold-safe P(point | action, phase,
  prefix bin) to build an action-conditioned point prior.

R83:
  Point conditional-style expert. Reuse the R63 conditional player-style
  features, but train a pointId model instead of actionId.

R84:
  Asymmetric R67 action blend. Blend R42 action with R63 by class/group instead
  of a single global weight.

R85:
  Ordinal point auxiliary prior. Train depth and side models separately and
  combine them into a structured 10-way point distribution. It is only blended
  back into V3 point.

R86:
  Joint sweep: combine R67 action branches with R82/R83/R85 point branches and
  generate submission candidates.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r7_phase_features import add_phase_features
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r57_player_style_clustering import add_player_id_features
from analysis_r63_r64_conditional_momentum import ConditionalStyleEncoder, add_conditional_style_features
from analysis_r67_r70_meta_priors import (
    R63_OOF_PATH,
    apply_action,
    clean_float,
    compose_v3_full_point,
    load_pickle,
    prepare_prefix_features,
)
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r82_r86_point_style")
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


def point_depth_arr(values: np.ndarray) -> np.ndarray:
    values = values.astype(int)
    out = np.zeros_like(values, dtype=int)
    mask = values > 0
    out[mask] = ((values[mask] - 1) // 3) + 1
    return out


def point_side_arr(values: np.ndarray) -> np.ndarray:
    values = values.astype(int)
    out = np.zeros_like(values, dtype=int)
    mask = values > 0
    out[mask] = ((values[mask] - 1) % 3) + 1
    return out


def prefix_bin(values: pd.Series | np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=int)
    return np.where(v <= 1, 1, np.where(v == 2, 2, 3)).astype(int)


def align_prefix_meta(meta: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
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


def make_point_model(seed: int, n_estimators: int = 180) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(POINT_CLASSES),
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_binary_model(seed: int, n_estimators: int = 180) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def aligned_multiclass_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        out[:, classes.index(cls)] = proba[:, i]
    return normalize_rows(out)


def aligned_binary_positive(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = [int(c) for c in model.classes_]
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return proba[:, classes.index(1)]


def build_action_point_lookup(train_df: pd.DataFrame, alpha: float = 35.0) -> dict:
    train_df = train_df.copy()
    train_df["prefix_bin"] = prefix_bin(train_df["prefix_len"])
    global_counts = train_df["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(POINT_CLASSES))

    def table_for(cols: list[str]) -> dict[tuple[int, ...], tuple[np.ndarray, float]]:
        out: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
        for key, sub in train_df.groupby(cols, sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            counts = sub["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
            prior = (counts + alpha * global_prior) / (counts.sum() + alpha)
            out[tuple(int(x) for x in key_tuple)] = (prior, float(len(sub)))
        return out

    return {
        "global": global_prior,
        "k3": table_for(["phase_id", "prefix_bin", "next_actionId"]),
        "k2": table_for(["phase_id", "next_actionId"]),
        "k1": table_for(["next_actionId"]),
    }


def action_conditioned_point_for_rows(
    action_prob: np.ndarray,
    rows: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = build_action_point_lookup(train_df)
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    bins = prefix_bin(rows["prefix_len"])
    phases = rows["phase_id"].astype(int).to_numpy()
    for i in range(len(rows)):
        mat = np.zeros((len(ACTION_CLASSES), len(POINT_CLASSES)), dtype=float)
        supp = 0.0
        for ai, action_id in enumerate(ACTION_CLASSES):
            chosen = None
            for name, key in [
                ("k3", (int(phases[i]), int(bins[i]), int(action_id))),
                ("k2", (int(phases[i]), int(action_id))),
                ("k1", (int(action_id),)),
            ]:
                item = lookup[name].get(key)
                if item is not None and item[1] >= (20 if name == "k3" else 35 if name == "k2" else 50):
                    chosen = item
                    break
            if chosen is None:
                chosen = (lookup["global"], 0.0)
            mat[ai] = chosen[0]
            supp += float(action_prob[i, ai]) * float(chosen[1])
        out[i] = action_prob[i] @ mat
        support[i] = supp
    return normalize_rows(out), support


def action_conditioned_point_oof(action_prob: np.ndarray, rows: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        train_df = prefix[~prefix["match"].isin(valid_matches)].copy()
        p, s = action_conditioned_point_for_rows(action_prob[idx], rows.loc[idx], train_df)
        out[idx] = p
        support[idx] = s
    return out, support


def train_point_style_oof(
    train_raw: pd.DataFrame,
    prefix: pd.DataFrame,
    rows: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        train_base = prefix[~prefix["match"].isin(valid_matches)].copy()
        valid_base = rows.loc[idx].copy()
        encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=8200 + int(fold)).fit(
            train_raw[~train_raw["match"].isin(valid_matches)].copy(),
            train_raw[~train_raw["match"].isin(valid_matches)].copy(),
        )
        train_cond = add_conditional_style_features(train_base, encoder)
        valid_cond = add_conditional_style_features(valid_base, encoder)
        cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
        use_cols = [c for c in features if c in train_cond.columns] + cond_cols
        model = make_point_model(8300 + int(fold))
        model.fit(
            train_cond[use_cols],
            train_cond["next_pointId"],
            sample_weight=class_weight_sample(train_cond["next_pointId"]),
        )
        out[idx] = aligned_multiclass_proba(model, valid_cond[use_cols], POINT_CLASSES)
    return normalize_rows(out)


def train_point_style_test(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=8350).fit(
        pd.concat([train_raw, test_raw], ignore_index=True),
        train_raw,
    )
    train_cond = add_conditional_style_features(prefix, encoder)
    test_cond = add_conditional_style_features(test_prefix, encoder)
    cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
    use_cols = [c for c in features if c in train_cond.columns] + cond_cols
    model = make_point_model(8351)
    model.fit(
        train_cond[use_cols],
        train_cond["next_pointId"],
        sample_weight=class_weight_sample(train_cond["next_pointId"]),
    )
    return aligned_multiclass_proba(model, test_cond[use_cols], POINT_CLASSES)


def train_ordinal_oof(prefix: pd.DataFrame, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    y_point = prefix["next_pointId"].to_numpy(dtype=int)
    prefix = prefix.copy()
    prefix["point_is_zero"] = (y_point == 0).astype(int)
    prefix["point_depth_aux"] = point_depth_arr(y_point)
    prefix["point_side_aux"] = point_side_arr(y_point)

    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        tr = prefix[~prefix["match"].isin(valid_matches)].copy()
        va = rows.loc[idx].copy()

        zero = make_binary_model(8500 + int(fold))
        zero.fit(tr[features], tr["point_is_zero"], sample_weight=class_weight_sample(tr["point_is_zero"]))
        p0 = aligned_binary_positive(zero, va[features])

        non = tr[tr["next_pointId"].astype(int) > 0].copy()
        depth = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=170,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=24,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.15,
            reg_lambda=2.0,
            random_state=8520 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        side = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=170,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=24,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.15,
            reg_lambda=2.0,
            random_state=8540 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        depth.fit(non[features], non["point_depth_aux"], sample_weight=class_weight_sample(non["point_depth_aux"]))
        side.fit(non[features], non["point_side_aux"], sample_weight=class_weight_sample(non["point_side_aux"]))
        pdp = aligned_multiclass_proba(depth, va[features], [0, 1, 2, 3])[:, 1:4]
        psp = aligned_multiclass_proba(side, va[features], [0, 1, 2, 3])[:, 1:4]
        grid = np.zeros((len(va), len(POINT_CLASSES)), dtype=float)
        grid[:, 0] = p0
        for d in range(3):
            for s in range(3):
                point_id = d * 3 + s + 1
                grid[:, point_id] = (1.0 - p0) * pdp[:, d] * psp[:, s]
        out[idx] = normalize_rows(grid)
    return normalize_rows(out)


def train_ordinal_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> np.ndarray:
    tr = prefix.copy()
    y_point = tr["next_pointId"].to_numpy(dtype=int)
    tr["point_is_zero"] = (y_point == 0).astype(int)
    tr["point_depth_aux"] = point_depth_arr(y_point)
    tr["point_side_aux"] = point_side_arr(y_point)

    zero = make_binary_model(8560)
    zero.fit(tr[features], tr["point_is_zero"], sample_weight=class_weight_sample(tr["point_is_zero"]))
    p0 = aligned_binary_positive(zero, test_prefix[features])

    non = tr[tr["next_pointId"].astype(int) > 0].copy()
    depth = lgb.LGBMClassifier(objective="multiclass", num_class=4, n_estimators=170, learning_rate=0.04, num_leaves=31, min_child_samples=24, subsample=0.88, subsample_freq=1, colsample_bytree=0.88, reg_alpha=0.15, reg_lambda=2.0, random_state=8561, n_jobs=-1, verbosity=-1)
    side = lgb.LGBMClassifier(objective="multiclass", num_class=4, n_estimators=170, learning_rate=0.04, num_leaves=31, min_child_samples=24, subsample=0.88, subsample_freq=1, colsample_bytree=0.88, reg_alpha=0.15, reg_lambda=2.0, random_state=8562, n_jobs=-1, verbosity=-1)
    depth.fit(non[features], non["point_depth_aux"], sample_weight=class_weight_sample(non["point_depth_aux"]))
    side.fit(non[features], non["point_side_aux"], sample_weight=class_weight_sample(non["point_side_aux"]))
    pdp = aligned_multiclass_proba(depth, test_prefix[features], [0, 1, 2, 3])[:, 1:4]
    psp = aligned_multiclass_proba(side, test_prefix[features], [0, 1, 2, 3])[:, 1:4]
    grid = np.zeros((len(test_prefix), len(POINT_CLASSES)), dtype=float)
    grid[:, 0] = p0
    for d in range(3):
        for s in range(3):
            point_id = d * 3 + s + 1
            grid[:, point_id] = (1.0 - p0) * pdp[:, d] * psp[:, s]
    return normalize_rows(grid)


def blend_point(base: np.ndarray, expert: np.ndarray, w: float) -> np.ndarray:
    return normalize_rows((1.0 - w) * base + w * expert)


def blend_action_classwise(base: np.ndarray, expert: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    w = np.array([float(weights.get(c, 0.0)) for c in ACTION_CLASSES], dtype=float)
    return normalize_rows(base * (1.0 - w[None, :]) + expert * w[None, :])


def write_submission(
    test_meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
    name: str,
    extra: dict | None = None,
) -> dict:
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
    upload_path.write_bytes(path.read_bytes())
    selected_path = SELECTED_DIR / name
    selected_path.write_bytes(path.read_bytes())
    info = {
        "candidate": name,
        "path": str(path),
        "upload_path": str(upload_path),
        "selected_path": str(selected_path),
    }
    if extra:
        info.update(extra)
    return info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, features = prepare_prefix_features()
    # prepare_prefix_features already adds role/score, phase, and player id features.
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)

    v3_oof = load_pickle("oof_proba_v3.pkl")
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(
        meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]
    ):
        raise ValueError("V3 OOF does not align with action artifact meta.")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    test_prefix_v3, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    if not test_prefix_v3["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 test point rows are not aligned.")

    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    r63_oof = np.load(R63_OOF_PATH)
    r67_oof_by_w = {w: normalize_rows((1.0 - w) * r42_oof + w * r63_oof) for w in [0.20, 0.225, 0.25, 0.275, 0.30, 0.35]}

    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    # Recreate the R63 full-test expert exactly like R67 did.
    encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=7350).fit(
        pd.concat([train_raw, test_raw], ignore_index=True),
        train_raw,
    )
    train_cond = add_conditional_style_features(prefix, encoder)
    test_cond = add_conditional_style_features(test_prefix, encoder)
    cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
    cond_features = [c for c in features if c in train_cond.columns] + cond_cols
    r63_model = lgb.LGBMClassifier(
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
    r63_model.fit(train_cond[cond_features], train_cond["next_actionId"], sample_weight=class_weight_sample(train_cond["next_actionId"]))
    r63_test = aligned_multiclass_proba(r63_model, test_cond[cond_features], ACTION_CLASSES)
    r67_test_by_w = {w: normalize_rows((1.0 - w) * r42_test + w * r63_test) for w in [0.20, 0.225, 0.25, 0.275, 0.30, 0.35]}

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")
    server_test = current_sub["serverGetPoint"].to_numpy(dtype=float)

    # R82.
    r82_oof, r82_support = action_conditioned_point_oof(r67_oof_by_w[0.20], rows, prefix)
    r82_test, r82_support_test = action_conditioned_point_for_rows(r67_test_by_w[0.20], test_prefix, prefix)

    # R83.
    r83_oof = train_point_style_oof(train_raw, prefix, rows, features)
    r83_test = train_point_style_test(train_raw, test_raw, prefix, test_prefix, features)

    # R85.
    r85_oof = train_ordinal_oof(prefix, rows, features)
    r85_test = train_ordinal_test(prefix, test_prefix, features)

    y_action = meta["next_actionId"].to_numpy(dtype=int)
    y_point = meta["next_pointId"].to_numpy(dtype=int)
    base_action_pred = apply_action(r67_oof_by_w[0.20], meta, art["selected"]["action_multipliers"])
    base_point_pred = apply_segmented_multipliers(meta, v3_point_oof, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
    base_action_f1 = f1_score(y_action, base_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    base_point_f1 = f1_score(y_point, base_point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)

    rows_out: list[dict] = [
        {
            "variant": "r67_w0p2_action_v3_point",
            "action_w": 0.20,
            "point_source": "v3",
            "point_w": 0.0,
            "action_macro_f1": float(base_action_f1),
            "point_macro_f1": float(base_point_f1),
            "action_churn_vs_r67_w0p2": 0.0,
            "point_churn_vs_v3": 0.0,
        }
    ]

    point_experts = {"r82_actioncond": r82_oof, "r83_pointstyle": r83_oof, "r85_ordinal": r85_oof}
    point_test_experts = {"r82_actioncond": r82_test, "r83_pointstyle": r83_test, "r85_ordinal": r85_test}
    best_point_by_source: dict[str, tuple[float, float, float]] = {}
    for source, prob in point_experts.items():
        best = (-1.0, 0.0, 1.0)
        for w in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            p = blend_point(v3_point_oof, prob, w)
            pred = apply_segmented_multipliers(meta, p, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
            f1 = f1_score(y_point, pred, average="macro", labels=POINT_CLASSES, zero_division=0)
            churn = float(np.mean(pred != base_point_pred))
            rows_out.append(
                {
                    "variant": source,
                    "action_w": 0.20,
                    "point_source": source,
                    "point_w": float(w),
                    "action_macro_f1": float(base_action_f1),
                    "point_macro_f1": float(f1),
                    "action_churn_vs_r67_w0p2": 0.0,
                    "point_churn_vs_v3": churn,
                    "point0_count": int((pred == 0).sum()),
                    "point3_count": int((pred == 3).sum()),
                    "point6_count": int((pred == 6).sum()),
                    "point8_count": int((pred == 8).sum()),
                }
            )
            if f1 > best[0] and churn <= 0.08:
                best = (float(f1), float(w), churn)
        best_point_by_source[source] = best

    # R84 low-DoF asymmetric class blends.
    templates: dict[str, dict[int, float]] = {}
    for low in [0.05, 0.10, 0.15]:
        for rare in [0.25, 0.35, 0.45, 0.55]:
            weights = {c: low for c in ACTION_CLASSES}
            for c in [0, 3, 4, 7, 8, 9, 11, 12, 14]:
                weights[c] = rare
            for c in [1, 2, 5, 10, 13]:
                weights[c] = low
            templates[f"r84_asym_low{clean_float(low)}_rare{clean_float(rare)}"] = weights
    for defw in [0.30, 0.40, 0.50]:
        weights = {c: 0.10 for c in ACTION_CLASSES}
        for c in [8, 9, 12, 14]:
            weights[c] = defw
        for c in [0, 3, 4, 7, 11]:
            weights[c] = 0.20
        templates[f"r84_asym_raredef{clean_float(defw)}"] = weights

    r84_test_probs: dict[str, np.ndarray] = {}
    for name, weights in templates.items():
        prob = blend_action_classwise(r42_oof, r63_oof, weights)
        pred = apply_action(prob, meta, art["selected"]["action_multipliers"])
        f1 = f1_score(y_action, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        churn = float(np.mean(pred != base_action_pred))
        rows_out.append(
            {
                "variant": name,
                "action_w": np.nan,
                "point_source": "v3",
                "point_w": 0.0,
                "action_macro_f1": float(f1),
                "point_macro_f1": float(base_point_f1),
                "action_churn_vs_r67_w0p2": churn,
                "point_churn_vs_v3": 0.0,
            }
        )
        r84_test_probs[name] = blend_action_classwise(r42_test, r63_test, weights)

    # R67 weight sweep.
    for aw, prob in r67_oof_by_w.items():
        pred = apply_action(prob, meta, art["selected"]["action_multipliers"])
        f1 = f1_score(y_action, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        churn = float(np.mean(pred != base_action_pred))
        rows_out.append(
            {
                "variant": "r67_global_weight",
                "action_w": float(aw),
                "point_source": "v3",
                "point_w": 0.0,
                "action_macro_f1": float(f1),
                "point_macro_f1": float(base_point_f1),
                "action_churn_vs_r67_w0p2": churn,
                "point_churn_vs_v3": 0.0,
            }
        )

    search = pd.DataFrame(rows_out).sort_values(["point_macro_f1", "action_macro_f1"], ascending=[False, False])
    search.to_csv(OUTDIR / "r82_r86_oof_search.csv", index=False)

    # Generate action-only R67 weight continuation and R84 asymmetric bests.
    generated: list[dict] = []
    action_candidates: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for aw, prob in r67_test_by_w.items():
        name_key = f"r67_w{clean_float(aw)}"
        test_pred = apply_action(prob, test_meta, art["selected"]["action_multipliers"])
        oof_pred = apply_action(r67_oof_by_w[aw], meta, art["selected"]["action_multipliers"])
        action_candidates[name_key] = (
            test_pred,
            oof_pred,
            {
                "action_variant": name_key,
                "action_w": float(aw),
                "oof_action_f1": float(f1_score(y_action, oof_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "oof_action_churn_vs_r67_w0p2": float(np.mean(oof_pred != base_action_pred)),
            },
        )
    for row in (
        search[search["variant"].astype(str).str.startswith("r84_asym")]
        .sort_values(["action_macro_f1", "action_churn_vs_r67_w0p2"], ascending=[False, True])
        .head(4)
        .itertuples(index=False)
    ):
        variant = str(row.variant)
        prob = r84_test_probs[variant]
        pred = apply_action(prob, test_meta, art["selected"]["action_multipliers"])
        action_candidates[variant] = (
            pred,
            apply_action(blend_action_classwise(r42_oof, r63_oof, templates[variant]), meta, art["selected"]["action_multipliers"]),
            {
                "action_variant": variant,
                "oof_action_f1": float(row.action_macro_f1),
                "oof_action_churn_vs_r67_w0p2": float(row.action_churn_vs_r67_w0p2),
            },
        )

    point_candidates: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    # include V3 as base.
    point_candidates["v3"] = (
        apply_segmented_multipliers(test_meta, v3_point_test, art["selected"]["point_multipliers"], POINT_CLASSES, "two"),
        base_point_pred,
        {"point_variant": "v3", "point_w": 0.0, "oof_point_f1": float(base_point_f1), "oof_point_churn_vs_v3": 0.0},
    )
    for source, (best_f1, best_w, best_churn) in best_point_by_source.items():
        if best_w <= 0:
            continue
        # Generate best and one conservative neighbor when available.
        for w in sorted(set([best_w, min(best_w, 0.05), 0.02])):
            if w <= 0:
                continue
            test_prob = blend_point(v3_point_test, point_test_experts[source], w)
            oof_prob = blend_point(v3_point_oof, point_experts[source], w)
            test_pred = apply_segmented_multipliers(test_meta, test_prob, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
            oof_pred = apply_segmented_multipliers(meta, oof_prob, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
            key = f"{source}_w{clean_float(w)}"
            point_candidates[key] = (
                test_pred,
                oof_pred,
                {
                    "point_variant": key,
                    "point_w": float(w),
                    "oof_point_f1": float(f1_score(y_point, oof_pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
                    "oof_point_churn_vs_v3": float(np.mean(oof_pred != base_point_pred)),
                },
            )

    # Choose a compact set of submissions: action sweep + best point branch combos.
    for action_key, (action_test_pred, action_oof_pred, action_info) in action_candidates.items():
        # Always emit action-only for the important global R67 continuation and top asymmetric variants.
        if action_key in {"r67_w0p225", "r67_w0p25", "r67_w0p275", "r67_w0p3"} or action_key.startswith("r84_asym"):
            point_test_pred, point_oof_pred, point_info = point_candidates["v3"]
            name = f"submission_r86_{action_key}_v3point_current_server.csv"
            generated.append(
                write_submission(
                    test_meta,
                    action_test_pred,
                    point_test_pred,
                    server_test,
                    name,
                    {**action_info, **point_info},
                )
            )

    # Best current action with point candidates.
    for point_key, (point_test_pred, point_oof_pred, point_info) in point_candidates.items():
        if point_key == "v3":
            continue
        action_test_pred, _, action_info = action_candidates["r67_w0p2"]
        name = f"submission_r86_r67_w0p2_{point_key}_current_server.csv"
        generated.append(
            write_submission(
                test_meta,
                action_test_pred,
                point_test_pred,
                server_test,
                name,
                {**action_info, **point_info},
            )
        )

    # A few joint combos if point branch is not too churny.
    best_point_keys = [
        k
        for k, (_, _, point_info) in sorted(point_candidates.items(), key=lambda kv: kv[1][2].get("oof_point_f1", 0), reverse=True)
        if k != "v3" and kv_safe(point_info)
    ][:3]
    for action_key in ["r67_w0p225", "r67_w0p25", "r67_w0p275"]:
        if action_key not in action_candidates:
            continue
        for point_key in best_point_keys:
            action_test_pred, _, action_info = action_candidates[action_key]
            point_test_pred, _, point_info = point_candidates[point_key]
            name = f"submission_r86_{action_key}_{point_key}_current_server.csv"
            generated.append(write_submission(test_meta, action_test_pred, point_test_pred, server_test, name, {**action_info, **point_info}))

    pd.DataFrame(generated).to_csv(OUTDIR / "r82_r86_generated_candidates.csv", index=False)

    report = {
        "base": {
            "r67_w0p2_oof_action_f1": float(base_action_f1),
            "v3_oof_point_f1": float(base_point_f1),
        },
        "best_point_by_source": {
            k: {"oof_point_f1": v[0], "weight": v[1], "churn": v[2]} for k, v in best_point_by_source.items()
        },
        "generated": generated,
        "notes": [
            "R82/R83/R85 point branches are blended into V3 point only.",
            "R84 class-asymmetric action branches are low-DoF templates.",
            "R86 submissions keep current R34 server fixed.",
        ],
    }
    (OUTDIR / "r82_r86_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(30).to_string(index=False))
    print(pd.DataFrame(generated).sort_values(["oof_point_f1", "oof_action_f1"], ascending=[False, False]).head(30).to_string(index=False))


def kv_safe(info: dict) -> bool:
    return float(info.get("oof_point_churn_vs_v3", 1.0)) <= 0.08


if __name__ == "__main__":
    main()
