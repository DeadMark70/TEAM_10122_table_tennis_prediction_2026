"""R55/V55 structured point follow-up.

R55: joint residual matrix on top of terminal/depth/side heads.
V55: structured heads as auxiliary probability features for a direct point model.

Both are diagnostics on the same sampled match-group CV protocol used by V3.
No submission is generated unless a branch beats V3 materially.
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

from baseline_lgbm import (
    POINT_CLASSES,
    POINT_NONTERMINAL_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v3 import apply_segmented_multipliers, blend_probs, tune_segmented_multipliers


OUTDIR = Path("r55_v55_structured_residual_aux")
DEPTH_CLASSES = [1, 2, 3]
SIDE_CLASSES = [1, 2, 3]
POINT_LABELS = list(range(10))


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def make_lgbm(objective: str, seed: int, num_class: int | None = None, n_estimators: int = 110) -> lgb.LGBMClassifier:
    params: dict[str, int | float | str] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.045,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if num_class is not None:
        params["num_class"] = num_class
    return lgb.LGBMClassifier(**params)


def aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    out = np.zeros((len(x), len(classes)), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    for src_idx, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = proba[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0
    if zero.any():
        out[zero, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def point_depth(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) // 3 + 1).astype(np.int16)
    return out


def point_side(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) % 3 + 1).astype(np.int16)
    return out


def action_group(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.full(len(arr), -1, dtype=np.int16)
    out[arr == 0] = 0
    out[(arr >= 1) & (arr <= 7)] = 1
    out[(arr >= 8) & (arr <= 11)] = 2
    out[(arr >= 12) & (arr <= 14)] = 3
    out[(arr >= 15) & (arr <= 18)] = 4
    return out


def add_struct_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["point_depth_target"] = point_depth(out["next_pointId"])
    out["point_side_target"] = point_side(out["next_pointId"])
    out["prefix_bin"] = np.where(out["prefix_len"] <= 2, 0, 1).astype(np.int16)
    out["lag0_action_group"] = action_group(out["lag0_actionId"])
    return out


def train_structured_heads(train_df: pd.DataFrame, features: list[str], seed: int):
    terminal = make_lgbm("binary", seed=seed)
    terminal.fit(train_df[features], train_df["next_pointId"].eq(0).astype(int))

    nt = train_df[train_df["next_pointId"].gt(0)].copy()
    depth = make_lgbm("multiclass", seed=seed + 1, num_class=3)
    depth.fit(nt[features], nt["point_depth_target"], sample_weight=class_weight_sample(nt["point_depth_target"]))
    side = make_lgbm("multiclass", seed=seed + 2, num_class=3)
    side.fit(nt[features], nt["point_side_target"], sample_weight=class_weight_sample(nt["point_side_target"]))
    return terminal, depth, side


def predict_structured(models, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    terminal, depth, side = models
    p0_raw = terminal.predict_proba(df[features])
    p0 = p0_raw[:, 1] if p0_raw.ndim == 2 else p0_raw
    depth_prob = aligned_proba(depth, df[features], DEPTH_CLASSES)
    side_prob = aligned_proba(side, df[features], SIDE_CLASSES)
    return np.clip(p0, 1e-6, 1 - 1e-6), depth_prob, side_prob


def factor_point(p0: np.ndarray, depth_prob: np.ndarray, side_prob: np.ndarray, lift: np.ndarray | None = None) -> np.ndarray:
    joint = np.zeros((len(p0), 9), dtype=float)
    idx = 0
    for d in range(3):
        for s in range(3):
            joint[:, idx] = depth_prob[:, d] * side_prob[:, s]
            idx += 1
    if lift is not None:
        joint *= lift
    joint = joint / np.clip(joint.sum(axis=1, keepdims=True), 1e-12, None)
    out = np.zeros((len(p0), 10), dtype=float)
    out[:, 0] = p0
    out[:, 1:] = (1.0 - p0[:, None]) * joint
    return out / out.sum(axis=1, keepdims=True)


def lift_table(train_df: pd.DataFrame, key_cols: list[str], alpha: float = 20.0, clip: float = 4.0):
    nt = train_df[train_df["next_pointId"].gt(0)].copy()
    global_counts = np.bincount(nt["next_pointId"].to_numpy(dtype=int) - 1, minlength=9).astype(float)
    table: dict[tuple, tuple[int, np.ndarray]] = {}

    def counts_to_lift(counts: np.ndarray) -> np.ndarray:
        joint = (counts + alpha * (global_counts + 1.0) / (global_counts.sum() + 9.0))
        joint = joint / joint.sum()
        depth = joint.reshape(3, 3).sum(axis=1)
        side = joint.reshape(3, 3).sum(axis=0)
        indep = np.outer(depth, side).reshape(9)
        lift = joint / np.clip(indep, 1e-12, None)
        return np.clip(lift, 1.0 / clip, clip)

    table[()] = (int(global_counts.sum()), counts_to_lift(global_counts))
    if key_cols:
        for key, part in nt.groupby(key_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            counts = np.bincount(part["next_pointId"].to_numpy(dtype=int) - 1, minlength=9).astype(float)
            table[tuple(int(v) for v in key)] = (int(counts.sum()), counts_to_lift(counts))
    return table


def lift_for_rows(df: pd.DataFrame, table, key_cols: list[str], min_support: int) -> np.ndarray:
    out = np.zeros((len(df), 9), dtype=float)
    global_lift = table[()][1]
    for i, row in enumerate(df.itertuples(index=False)):
        if not key_cols:
            out[i] = global_lift
            continue
        row_dict = row._asdict()
        key = tuple(int(row_dict[c]) for c in key_cols)
        support, lift = table.get(key, (0, global_lift))
        out[i] = lift if support >= min_support else global_lift
    return out


def add_structured_feature_columns(df: pd.DataFrame, p0: np.ndarray, depth: np.ndarray, side: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["aux_p_point0"] = p0
    for i in range(3):
        out[f"aux_depth_{i+1}"] = depth[:, i]
        out[f"aux_side_{i+1}"] = side[:, i]
    fac = factor_point(np.zeros(len(df)), depth, side)[:, 1:]
    for j in range(9):
        out[f"aux_factor_point_{j+1}"] = fac[:, j]
    return out


def train_direct_point(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    terminal = make_lgbm("binary", seed=seed)
    terminal.fit(train_df[features], train_df["next_pointId"].eq(0).astype(int))
    nt = train_df[train_df["next_pointId"].gt(0)].copy()
    point = make_lgbm("multiclass", seed=seed + 1, num_class=9)
    point.fit(nt[features], nt["next_pointId"], sample_weight=class_weight_sample(nt["next_pointId"]))
    p0_raw = terminal.predict_proba(valid_df[features])
    p0 = p0_raw[:, 1] if p0_raw.ndim == 2 else p0_raw
    nt_prob = aligned_proba(point, valid_df[features], POINT_NONTERMINAL_CLASSES)
    out = np.zeros((len(valid_df), 10), dtype=float)
    out[:, 0] = p0
    out[:, 1:] = (1.0 - p0[:, None]) * nt_prob
    return out / out.sum(axis=1, keepdims=True)


def inner_structured_oof(train_df: pd.DataFrame, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = np.zeros(len(train_df), dtype=float)
    depth = np.zeros((len(train_df), 3), dtype=float)
    side = np.zeros((len(train_df), 3), dtype=float)
    train_df = train_df.reset_index(drop=True)
    splitter = GroupKFold(n_splits=3)
    for k, (tr_idx, va_idx) in enumerate(splitter.split(train_df, groups=train_df["match"]), start=1):
        models = train_structured_heads(train_df.iloc[tr_idx], features, seed + k * 10)
        a, b, c = predict_structured(models, train_df.iloc[va_idx], features)
        p0[va_idx], depth[va_idx], side[va_idx] = a, b, c
    return p0, depth, side


def load_v3_point(meta: pd.DataFrame) -> tuple[np.ndarray, object]:
    with open("oof_proba_v3.pkl", "rb") as f:
        oof = pickle.load(f)
    tuning = oof["tuning"]
    v3_point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    src = oof["valid_meta"].reset_index(drop=True).copy()
    src["_row"] = np.arange(len(src))
    merged = meta[["rally_uid", "prefix_len", "next_pointId"]].merge(
        src[["rally_uid", "prefix_len", "next_pointId", "_row"]],
        on=["rally_uid", "prefix_len", "next_pointId"],
        how="left",
        validate="one_to_one",
    )
    if merged["_row"].isna().any():
        raise ValueError("Could not align V3 OOF point to R55 validation rows.")
    return v3_point[merged["_row"].to_numpy(dtype=int)], tuning


def score_point(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict | None = None, mode: str = "two") -> float:
    if multipliers is None:
        pred = np.asarray(POINT_CLASSES)[np.argmax(prob, axis=1)]
    else:
        pred = apply_segmented_multipliers(meta, prob, multipliers, POINT_CLASSES, mode)
    return float(f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_LABELS, zero_division=0))


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_df = add_struct_context(build_train_prefix_table(train, max_lag=6))
    test_prefix = build_test_prefix_table(test, max_lag=6)
    test_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    features = feature_columns(prefix_df)
    features = [c for c in features if c not in {"point_depth_target", "point_side_target", "prefix_bin", "lag0_action_group"}]

    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates().reset_index(drop=True)
    sampled_parts = []
    r55_variants = {
        "global": [],
        "prefix": [],
        "action_group": [],
        "prefix_action_group": [],
        "prefix_action_spin": [],
    }
    v55_parts = []

    for fold, (tr_rally_idx, va_rally_idx) in enumerate(GroupKFold(n_splits=5).split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[tr_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[va_rally_idx]["rally_uid"])
        tr = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy().reset_index(drop=True)
        valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_lengths, 42 + fold)
        va = valid_pool.loc[sampled_idx].copy().reset_index(drop=True)
        sampled_parts.append(va[["rally_uid", "match", "prefix_len", "next_pointId"]].copy())

        models = train_structured_heads(tr, features, 5500 + fold * 100)
        p0, depth, side = predict_structured(models, va, features)
        contexts = {
            "global": [],
            "prefix": ["prefix_bin"],
            "action_group": ["lag0_action_group"],
            "prefix_action_group": ["prefix_bin", "lag0_action_group"],
            "prefix_action_spin": ["prefix_bin", "lag0_action_group", "lag0_spinId"],
        }
        for name, cols in contexts.items():
            table = lift_table(tr, cols)
            lift = lift_for_rows(va, table, cols, min_support=80 if len(cols) >= 2 else 40)
            r55_variants[name].append(factor_point(p0, depth, side, lift))

        # V55 auxiliary structured features with inner OOF on train rows.
        tr_p0, tr_depth, tr_side = inner_structured_oof(tr, features, 6500 + fold * 100)
        va_p0, va_depth, va_side = p0, depth, side
        tr_aug = add_structured_feature_columns(tr, tr_p0, tr_depth, tr_side)
        va_aug = add_structured_feature_columns(va, va_p0, va_depth, va_side)
        aux_features = features + [c for c in tr_aug.columns if c.startswith("aux_")]
        v55_parts.append(train_direct_point(tr_aug, va_aug, aux_features, 7500 + fold * 100))
        print(f"fold {fold} done")

    meta = pd.concat(sampled_parts, ignore_index=True)
    v3_point, v3_tuning = load_v3_point(meta)
    v3_fixed = score_point(meta, v3_point, v3_tuning.point_multipliers, v3_tuning.bins_mode)
    v3_raw = score_point(meta, v3_point)

    rows = [
        {"variant": "v3_raw", "point_macro_f1": v3_raw, "blend_weight": 0.0, "retuned": False},
        {"variant": "v3_fixed_multipliers", "point_macro_f1": v3_fixed, "blend_weight": 0.0, "retuned": False},
    ]

    all_probs: dict[str, np.ndarray] = {}
    for name, parts in r55_variants.items():
        all_probs[f"r55_{name}"] = np.vstack(parts)
    all_probs["v55_aux_direct"] = np.vstack(v55_parts)

    blend_grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    for name, prob in all_probs.items():
        rows.append({"variant": name, "point_macro_f1": score_point(meta, prob), "blend_weight": 1.0, "retuned": False})
        for w in blend_grid:
            blended = (1.0 - w) * v3_point + w * prob
            rows.append(
                {
                    "variant": f"blend_v3_{name}",
                    "point_macro_f1": score_point(meta, blended, v3_tuning.point_multipliers, v3_tuning.bins_mode),
                    "blend_weight": w,
                    "retuned": False,
                }
            )
            mult = tune_segmented_multipliers(meta, blended, POINT_CLASSES, "point", v3_tuning.bins_mode)
            rows.append(
                {
                    "variant": f"blend_v3_{name}",
                    "point_macro_f1": score_point(meta, blended, mult, v3_tuning.bins_mode),
                    "blend_weight": w,
                    "retuned": True,
                }
            )

    result = pd.DataFrame(rows).sort_values("point_macro_f1", ascending=False)
    result.to_csv(OUTDIR / "r55_v55_point_scores.csv", index=False)
    for name, prob in all_probs.items():
        np.save(OUTDIR / f"{name}_oof.npy", prob)
    meta.to_csv(OUTDIR / "r55_v55_valid_meta.csv", index=False)
    report = {
        "v3_raw": v3_raw,
        "v3_fixed_multipliers": v3_fixed,
        "best": result.head(20).to_dict(orient="records"),
        "features": len(features),
    }
    (OUTDIR / "r55_v55_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
